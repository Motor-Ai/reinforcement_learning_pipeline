import carla # pylint: disable=no-member
import random
import pygame
import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from gymnasium import spaces
import os
import sys
import yaml  # Import YAML parser
from typing import Optional, Tuple, Any

# Expand the CARLA_ROOT environment variable correctly:
carla_root = os.environ.get("CARLA_ROOT")
if carla_root is None:
    raise EnvironmentError("CARLA_ROOT environment variable is not set.")
sys.path.append(os.path.join(carla_root, "PythonAPI", "carla"))

# Import the default VehiclePIDController from CARLA
from agents.navigation.controller import VehiclePIDController
from agents.navigation.global_route_planner import GlobalRoutePlanner

# MAI imports
from src.envs.observation.vector_BEV_observer import Vector_BEV_observer
from src.envs.carla_env_render import MatplotlibAnimationRenderer
#from models.dipp_predictor_py.dipp_carla import Predictor
from src.envs.observation.observation_manager import ObservationManager
from src.envs.actions.action_manager import ActionManager
from src.envs.reward.base_reward import Reward
# pyright: reportAttributeAccessIssue=none

#TODO: config should be passed as an argument to the environment, not hard-coded.
# Load configurations from YAML
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs/config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

# Assign configurations to variables
N_VEHICLES = config["N_VEHICLES"]
SCENE_DURATION = config["SCENE_DURATION"]
SLOWDOWN_PERCENTAGE = config["SLOWDOWN_PERCENTAGE"]
EGO_AUTOPILOT = config["EGO_AUTOPILOT"]
FOLLOW_POINT_DIST = config["FOLLOW_POINT_DIST"]
REQ_TIME = config["REQ_TIME"]
FREQUENCY = config["FREQUENCY"]
USE_CUSTOM_MAP = config["USE_CUSTOM_MAP"]
NUM_ACTIONS = config["NUM_ACTIONS"]
N_ACTION_PER_MANEUVER = config["N_ACTION_PER_MANEUVER"]
SHOW_ROUTE = config["SHOW_ROUTE"]
PREPROCESS_OBSERVATION = config["PREPROCESS_OBSERVATION"]
display_width = config["display_width"]
display_height = config["display_height"]

MAI_ACTION_SPACE = True

class CarlaGymEnv(gym.Env):
    """
    A Gym environment wrapping a CARLA simulation. Observations are a dictionary 
    with keys "ego", "neighbors", and "map". The action is a 2D offset (in the 
    ego coordinate system) that is rotated using the current ego yaw.
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, render_enabled=False):
        """
        Initialize the environment.
        
        Parameters:
            render_enabled (bool): If True, the camera sensor is spawned and
                                   its output rendered via Pygame.
        """
        super(CarlaGymEnv, self).__init__()
        self.render_enabled = render_enabled

        # Simulation parameters
        self.scene_duration = SCENE_DURATION
        self.slowdown_percentage = SLOWDOWN_PERCENTAGE
        self.ego_autopilot = EGO_AUTOPILOT
        #self.follow_point_dist = FOLLOW_POINT_DIST
        self.req_time = REQ_TIME
        self.frequency = FREQUENCY
        self.use_custom_map = USE_CUSTOM_MAP
        self.show_route = SHOW_ROUTE
        self.sim_time = 0.0
        self.timestep = 0
        self.goal_threshold = 0.5  # meters
        self.map_path = '/home/ratul/Downloads/Tegel_map_for_Decision_1302.xodr'
        self.prev_distance: Optional[float] = None
        self.matplotlib_renderer: Optional[MatplotlibAnimationRenderer] = None
        self.preprocess_observation = PREPROCESS_OBSERVATION
        self.obs: Optional[dict] = None
        self.global_route: np.ndarray = None
        self.ego_vehicle: carla.Actor = None
        self.spawn_points: list[carla.Transform] = []
        self.global_route_start: carla.Location = None
        self.global_route_destination: carla.Location = None
        self.reward_func = Reward(scene_duration=self.scene_duration, step_frequecny=self.frequency)

        # Pygame setup for camera display (only if rendering enabled)
        if self.render_enabled:
            pygame.init()  # pylint: disable=no-member
            self.screen = pygame.display.set_mode((display_width, display_height))
            pygame.display.set_caption("Carla Gym Environment")
        self.display_width = display_width
        self.display_height = display_height

        # Connect to CARLA server and get world
        self.client = carla.Client('localhost', 2000)  #
        self.client.set_timeout(10.0)
        if self.use_custom_map:
            # Load the custom OpenDRIVE (.xodr) map
            with open(self.map_path, 'r') as f:
                opendrive_data = f.read()

            # Generate the world using the OpenDRIVE map
            self.world = self.client.generate_opendrive_world(
                opendrive_data,  # commented out, fix if needed
                carla.OpendriveGenerationParameters(
                    wall_height=0.0,
                    additional_width=0.0,
                    smooth_junctions=True,
                    enable_mesh_visibility=True
                ),
                reset_settings=True
            )
            print("Successfully loaded custom map:", self.map_path)
        else:
            # load default world
            self.world = self.client.get_world()

        self.blueprint_library = self.world.get_blueprint_library()

        # Set synchronous mode and fixed delta time
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.frequency
        self.world.apply_settings(settings)

        # Traffic manager
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.global_percentage_speed_difference(self.slowdown_percentage)

        # Initialize BEV observer (your original code uses FUTURE_LEN=1)
        self.bev_info = Vector_BEV_observer(FUTURE_LEN=1) #TODO: this was moved from observation_manager and only used here for constants. replace it

        # For cleanup, track spawned actors
        self.actor_list: list[carla.Actor] = []

        # Define the action space early on:
        # self.action_space = spaces.Box(low=5, high=10, shape=(2,), dtype=np.float32)

        if not MAI_ACTION_SPACE:
            self.num_maneuvers = NUM_ACTIONS  # Possible actions: 0, 1, 2, 3, 4
            self.n_action_per_maneuver = N_ACTION_PER_MANEUVER
            self.action_space = spaces.MultiDiscrete([self.num_maneuvers, self.n_action_per_maneuver])
        else:
            self.action_manager = ActionManager( # Not used, Just a stub 
                action_fields=["acc", "lat_shift", "manuver"] ,
                n_samples = NUM_ACTIONS)
            self.action_space = self.action_manager.action_space 

        # Define the possible values for each dimension
        self.index_map = {1: 1, 2: 5, 3: 10, 4: 15}

        self.observation_manager = ObservationManager(
            obs_keys=["ego", "neighbors", "map", "global_route"],
            preprocess=self.preprocess_observation,
        )
        self.observation_space = self.observation_manager.observation_space

        # Initialize collision flag
        self.collision_detected = False

        # Call reset to start the simulation.
        self.reset()

    #TODO: parts of this method should be moved to a higher scope including model and before env.__init__.
    # def seed(self, seed: Optional[float]=None):
    #     """
    #     Set the random seed for Python, NumPy, and Torch for reproducibility.
    #     Returns a list with the seed used.
    #     """
    #     self.np_random, seed = gym.utils.seeding.np_random(seed)
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     return [seed]

    def process_image(self, image, screen):
        """
        Convert the CARLA image to a numpy array, then to a pygame surface,
        and display it.
        """
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        array = array[:, :, :3][:, :, ::-1]  # Convert from BGRA to RGB
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        screen.blit(surface, (0, 0))
        pygame.display.flip()

    def ego_to_global(self, action: NDArray[np.float64], ego_position: NDArray[np.float64], ego_yaw: float):
        """
        Convert a 2D action (offset in ego frame) into a global coordinate.
        """
        cos_theta = np.cos(ego_yaw)
        sin_theta = np.sin(ego_yaw)
        R = np.array([[cos_theta, -sin_theta],
                      [sin_theta, cos_theta]])
        global_point = (R @ action.reshape(2, 1)).reshape(2,) + ego_position
        return global_point

    def _on_collision(self, event):
        """
        Callback for collision events. Sets the collision flag.
        """
        self.collision_detected = True

    def _cleanup(self):
        """
        Destroy all actors (vehicles, sensors, etc.) that were spawned.
        """
        #print("Destroying actors...")
        for actor in self.actor_list:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception: #TODO: replace with specific exception
                    print("Error destroying actor:", actor)
                    pass
        self.actor_list = []

    def reset(self, *, seed: int | None = None, options=None) -> tuple[dict, dict[str, Any]]:
        """
        Reset the simulation: clean up previous actors, spawn the ego vehicle,
        attach sensors (including collision sensor and, if enabled, camera), spawn 
        other vehicles, and reset time.
        Returns an initial observation.
        """
        self._cleanup()
        self.observation_manager.reset()
        self.sim_time = 0.0
        self.timestep = 0
        self.collision_detected = False

        # TODO Compute distance to the goal and use it to initialise self.prev_distance

        # Spawn Ego Vehicle
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        vehicle_bp.set_attribute("role_name", "hero")
        self.ego_vehicle = None
        while self.ego_vehicle is None:
            self.spawn_points = self.world.get_map().get_spawn_points()
            self.ego_vehicle = self.world.try_spawn_actor(vehicle_bp, random.choice(self.spawn_points))
        assert isinstance(self.ego_vehicle, carla.Vehicle), "Ego vehicle is not a Vehicle."
        self.actor_list.append(self.ego_vehicle)
        if not self.ego_autopilot:
            # Initialize the PID controller for the ego vehicle
            args_lateral = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.2, 'dt': 1.0 / 20.0}
            args_longitudinal = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.2, 'dt': 1.0 / 20.0}
            self.pid_controller = VehiclePIDController(self.ego_vehicle,
                                                       args_lateral=args_lateral,
                                                       args_longitudinal=args_longitudinal)
        else:
            self.ego_vehicle.set_autopilot(self.ego_autopilot, self.tm.get_port())

        # Attach Camera to Ego Vehicle if rendering is enabled
        if self.render_enabled:
            camera_bp = self.blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.display_width))
            camera_bp.set_attribute('image_size_y', str(self.display_height))
            camera_bp.set_attribute('fov', '90')
            camera_transform = carla.Transform(carla.Location(x=0, z=35.0), carla.Rotation(pitch=-90))
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
            assert isinstance(self.camera, carla.Sensor), "Camera is not a Sensor."
            self.actor_list.append(self.camera)
            self.camera.listen(lambda image: self.process_image(image, self.screen)) # type: ignore[list-item, return-value]

        # Attach Collision Sensor to Ego Vehicle
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        collision_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
        self.collision_sensor = self.world.spawn_actor(collision_bp, collision_transform, attach_to=self.ego_vehicle)
        assert isinstance(self.collision_sensor, carla.Sensor), "Collision sensor is not a Sensor."
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self._on_collision(event)) # type: ignore[list-item, return-value]

        # Spawn other vehicles on autopilot
        self.vehicles = []
        for _ in range(N_VEHICLES):
            vehicle_bp = random.choice(self.blueprint_library.filter('vehicle.*'))
            spawn_point = random.choice(self.spawn_points)
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle is not None:
                vehicle.set_autopilot(True, self.tm.get_port())
                self.tm.vehicle_percentage_speed_difference(vehicle, self.slowdown_percentage)
                self.vehicles.append(vehicle)
                self.actor_list.append(vehicle)

        # Tick the world once to initialize everything
        self.world.tick()

        # Compute the global route only once.
        self._generate_global_route()
        if self.show_route:
            for point in self.global_route:
                point = carla.Location(x=float(point[0]), y=float(point[1]))
                self.world.debug.draw_point(
                    point,
                    size=0.1,
                    color=carla.Color(255, 255, 0),
                    life_time=self.scene_duration
                )
        
        observation = self.observation_manager.get_obs(self.world)

        info = {}
        return observation, info

    def _generate_global_route(self):
        """
        Generate the global route using the GlobalRoutePlanner and convert it into the ego frame.
        Returns:
            A numpy array of shape (N, 3) with columns [x, y, relative_yaw] in the ego coordinate frame.
        """
        # assert ego_vehicle is Actor with an assertion
        assert isinstance(self.ego_vehicle, carla.Actor), "Ego vehicle is not an Actor instance."
        # Get the ego vehicle's current location.
        start_location = self.ego_vehicle.get_transform().location

        # Choose a destination from spawn_points that is at least 100m away.
        destination = random.choice(self.spawn_points)
        while destination.location.distance(start_location) < 100:
            destination = random.choice(self.spawn_points)

        # Set up the global route planner.
        grp = GlobalRoutePlanner(self.world.get_map(), 2.0)
        # trace_route returns a list of tuples (waypoint, road_option).
        route = grp.trace_route(start_location, destination.location)

        # Convert the global route into an array with columns [x, y, yaw].
        global_route_list = [
            [wp.transform.location.x, wp.transform.location.y, wp.transform.rotation.yaw]
            for wp, _ in route
        ]
        self.global_route = np.array(global_route_list)
        # Optionally store the original start and destination too.
        self.global_route_start = start_location
        self.global_route_destination = destination

    def _transform_to_ego_frame(self):
        # Now, convert the global route into the ego frame.
        ego_transform = self.ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_rotation = ego_transform.rotation
        ego_yaw_rad = np.deg2rad(ego_rotation.yaw)
        cos_yaw = np.cos(ego_yaw_rad)
        sin_yaw = np.sin(ego_yaw_rad)

        global_route_ego_frame = []
        for global_point in self.global_route:
            global_x, global_y, global_yaw = global_point
            # Translate: shift coordinates by subtracting the ego's position.
            dx = global_x - ego_location.x
            dy = global_y - ego_location.y
            # Rotate: apply the inverse rotation so that ego heading aligns with the x-axis.
            x_ego = dx * cos_yaw + dy * sin_yaw
            y_ego = -dx * sin_yaw + dy * cos_yaw
            # Relative yaw is the difference between the waypoint's yaw and the ego yaw.
            relative_yaw = global_yaw - ego_rotation.yaw
            global_route_ego_frame.append([x_ego, y_ego, relative_yaw])

        global_route_ego_frame = np.array(global_route_ego_frame)
        global_route_ego_frame = global_route_ego_frame[global_route_ego_frame[:, 0] >= 0] # keep only the positive route
        global_route_ego_frame_no_padding = global_route_ego_frame[:self.bev_info.MAX_LANE_LEN].copy()
        # make the route fixed size.
        global_route_ego_frame = global_route_ego_frame[:self.bev_info.MAX_LANE_LEN]
        pad_len = self.bev_info.MAX_LANE_LEN - len(global_route_ego_frame)
        if pad_len > 0:
            global_route_ego_frame = np.pad(global_route_ego_frame, ((0, pad_len), (0, 0)))
        return global_route_ego_frame, global_route_ego_frame_no_padding

    def step(self, action: NDArray[np.float64]|spaces.Box) -> Tuple[dict, float, bool, bool, dict]:
        info = {}
        # print("Action received:", action)
        # transform global route to ego frame
        global_route_ego_frame, global_route_ego_frame_no_padding = self._transform_to_ego_frame()

        # Get current ego transform information
        assert isinstance(self.ego_vehicle, carla.Vehicle), "Ego vehicle is not a Vehicle."
        ego_transform = self.ego_vehicle.get_transform()
        current_location = ego_transform.location
        ego_position_global = np.array([current_location.x, current_location.y])
        ego_yaw_global = np.deg2rad(ego_transform.rotation.yaw)
        ego_velocity = self.ego_vehicle.get_velocity()
        ego_state = [current_location.x, current_location.y, ego_yaw_global, ego_velocity.x, ego_velocity.y]
        if not self.ego_autopilot:
            if not MAI_ACTION_SPACE:
                action_point = action.copy()
                if len(global_route_ego_frame_no_padding):
                    if action[0] in {0, 1, 2} and 1 <= action[1] <= 4:
                        self.index_map = {1: 1, 2: 5, 3: 10, 4: 15} #distnace in meters?
                        chosen_index = self.index_map[action[1]]

                        # Clamp: 0 <= chosen_index < len(global_route_ego_frame_no_padding) 
                        chosen_index = max(0, min(chosen_index, len(global_route_ego_frame_no_padding) - 1))

                        # If the last index has a negative value, choose index 0
                        if global_route_ego_frame_no_padding[chosen_index, 0] < 0:
                            chosen_index = 0

                        action_point = global_route_ego_frame_no_padding[chosen_index, :2].copy()

                        yaw = global_route_ego_frame_no_padding[chosen_index, 2]  # Extract yaw (in radians)

                        # Compute perpendicular displacement (90-degree rotation)
                        perpendicular = np.array([-np.sin(yaw), np.cos(yaw)])

                        if action[0] == 1: #left?
                            action_point = action_point + (-5.0 * perpendicular)
                        elif action[0] == 2: #right?
                            action_point = action_point + (5.0 * perpendicular)
                else:
                    action_point = np.array([0.0, 0.0])

            else: 
                if len(global_route_ego_frame_no_padding) > 1:
                    # Action from Policy converted to Frenet to determine the target point
                    # ref_path = np.unique(global_route_ego_frame_no_padding[:,:2], axis = -1)
                    path_x, path_y, path_yaw, path_vel, path_time = self.action_manager.get_path(action, global_route_ego_frame_no_padding[:,:2], 
                                                                                                ego_state, plan_time_range=3.0, plan_dt=0.2)
                    TARGET_PT_IDX = 2 # NOTE: PARAM to set which point to use from the planned path
                    action_point = np.array([path_x[TARGET_PT_IDX], path_y[TARGET_PT_IDX]]) # Take the second point in the planned path
                else:
                    action_point = np.array([0.0, 0.0])

            target_global = self.ego_to_global(np.array(action_point), ego_position_global, ego_yaw_global)
            target_location = carla.Location(x=target_global[0], y=target_global[1], z=current_location.z)

            # Draw the target point in CARLA for debugging
            self.world.debug.draw_point(
                target_location,
                size=0.1,
                color=carla.Color(255, 0, 0),
                life_time=self.frequency * 2
            )

            # Compute the target speed based on the distance (and req_time)
            distance = np.linalg.norm(np.array([target_location.x, target_location.y]) - ego_position_global)
            target_speed = distance / self.req_time * 3.6  # converting to km/h
            info["target_speed"] = target_speed

            # Get the nearest waypoint corresponding to the target location
            target_waypoint = self.world.get_map().get_waypoint(target_location)

            # Calculate control command using the PID controller and apply it
            control = self.pid_controller.run_step(target_speed, target_waypoint)
            self.ego_vehicle.apply_control(control)
        else:
            # Autopilot is enabled, so ignore the provided action.
            pass

        # Advance the simulation by one tick
        self.world.tick()
        self.timestep += 1
        self.sim_time = self.timestep * self.frequency
        info["sim_time"] = self.sim_time

        observation = self.observation_manager.get_obs(self.world, global_route_ego_frame)

        self.obs = observation  # Save the latest observation for rendering


        ######################### Reward and termination ############################
        # Compute distance to goal
        distance_to_goal = self.compute_distance_to_goal(target_location)
        reward = self.reward_func(distance_to_goal, self.prev_distance, self.collision_detected, self.timestep)
        self.prev_distance = distance_to_goal
        info["distance_to_goal"] = distance_to_goal

        # Check if goal is reached
        goal_reached = distance_to_goal < self.goal_threshold
        info["goal_reached"] = goal_reached
        terminated = goal_reached # End episode if goal is reached
        truncated = self.timestep >= self.scene_duration / self.frequency # End episode if time is up

        # Check for collision penalty
        if self.collision_detected:
            terminated = True  # End episode on collision
        info["crash"] = self.collision_detected

        return observation, reward, terminated, truncated, info

    def _compute_route_error(self, target_global):
        """
        Given a target point in global coordinates, compute the lateral error and 
        longitudinal progress along the stored global route.
        
        Returns:
            lateral_error (float): Distance from target_global to nearest route point.
            longitudinal_progress (float): Cumulative distance along the route up to that point.
        """
        # Extract the route's (x, y) coordinates.
        route_points = self.global_route[:, :2]  # shape (N, 2)

        # Compute the Euclidean distances from the target to each route point.
        target_xy = np.array([target_global.x, target_global.y]).reshape(1, 2)
        distances = np.linalg.norm(route_points - target_xy, axis=1)
        idx = np.argmin(distances)
        lateral_error = distances[idx]

        # Compute cumulative distance along the route up to the nearest point.
        if idx == 0:
            longitudinal_progress = 0.0
        else:
            # Sum the distances between successive points along the route.
            longitudinal_progress = np.sum(np.linalg.norm(np.diff(route_points[:idx+1], axis=0), axis=1))
        return lateral_error, longitudinal_progress

    def _get_desired_lane_info(self):
        """
        Determine the desired lane information from the global route based on the ego's current location.
        
        Returns:
            desired_lane_yaw (float): The desired lane direction (yaw in degrees) from the global route.
            desired_lane_point (np.array): The (x, y) coordinate of the closest point on the global route.
        """
        # Get ego's current (x, y) location.
        ego_location = self.ego_vehicle.get_location()
        ego_xy = np.array([ego_location.x, ego_location.y])

        assert self.global_route is not None, "Global route is not set. Call _generate_global_route first."
        # Extract (x, y) coordinates from the global route.
        route_xy = self.global_route[:, :2]  # shape (N, 2)

        # Compute distances from ego to each route point.
        distances = np.linalg.norm(route_xy - ego_xy.reshape(1, 2), axis=1)
        idx = np.argmin(distances)

        # Get the desired lane yaw from the global route.
        desired_lane_yaw = self.global_route[idx, 2]
        desired_lane_point = route_xy[idx]
        return desired_lane_yaw, desired_lane_point

    def compute_distance_to_goal(self, goal_location: carla.Location) -> float:
        """
        Compute the remaining distance to the goal by summing the route distances
        from the closest point to the goal.

        Args:
            action_xy: The agent's current position as (x, y).

        Returns:
            distance_to_goal (float): The accumulated distance along the global route.
        """
        # Compute distance to goal
        action_xy = np.array([goal_location.x, goal_location.y])

        # Extract (x, y) positions of the route
        route_xy = self.global_route[:, :2]

        # Find the closest point on the global route
        distances = np.linalg.norm(route_xy - action_xy, axis=1)  # Distance to all route points
        closest_idx = np.argmin(distances)  # Index of the closest point

        # Compute cumulative distance from the closest point to the goal
        remaining_distances = np.linalg.norm(np.diff(route_xy[closest_idx:], axis=0), axis=1)
        distance_to_goal = np.sum(remaining_distances)  # Sum up all distances

        return distance_to_goal

    def render(self, mode: str = "human"):
        # Create the renderer if it doesn't already exist.
        if self.matplotlib_renderer is None:
            self.matplotlib_renderer = MatplotlibAnimationRenderer()

        # Use the stored observation data (if available) to update the renderer.
        if self.obs is not None:
            ego_obs = self.obs["ego"]
            neighbors_obs = self.obs["neighbors"]
            map_obs = self.obs["map"]
            self.matplotlib_renderer.update_data(ego_obs, neighbors_obs, map_obs)

    def custom_sample_action(self):
        """
        Returns a random valid action from the defined MultiDiscrete action space.
        The first value corresponds to a maneuver (0 to NUM_MANEUVERS - 1),
        and the second value corresponds to a sub-action within that maneuver (0 to N_ACTION_PER_MANEUVER - 1).
        """
        maneuver = np.random.randint(0, self.num_maneuvers)  # Random maneuver index
        sub_action = np.random.randint(0, self.n_action_per_maneuver)  # Random sub-action index

        return np.array([maneuver, sub_action])

    def close(self):
        """
        Clean up all actors and close the environment.
        """
        self._cleanup()
        if self.render_enabled:
            pygame.quit() # pylint: disable=no-member
