import carla
import numpy as np
import pickle
import random
import yaml
import os
import sys
import time
import h5py
from src.envs.observation.observation_manager import ObservationManager 
from src.envs.actions.action_manager import ActionManager
from src.envs.observation.vector_BEV_observer import Vector_BEV_observer

# Expand the CARLA_ROOT environment variable correctly:
carla_root = os.environ.get("CARLA_ROOT")
if carla_root is None:
    raise EnvironmentError("CARLA_ROOT environment variable is not set.")
sys.path.append(os.path.join(carla_root, "PythonAPI", "carla"))

from agents.navigation.global_route_planner import GlobalRoutePlanner

import torch

# CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs/config.yaml")
CONFIG_PATH = "/home/ubuntu/Workstation/mangal/reinforcement_learning_pipeline/src/envs/configs/config.yaml"
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

# Connect to CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

# traffic manager
tm = client.get_trafficmanager()
tm_port = tm.get_port()


blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle_bp.set_attribute('role_name', 'ego_vehicle')
ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)

observation_manager = ObservationManager(obs_keys=["ego", "neighbors", "map", "global_route"], preprocess=PREPROCESS_OBSERVATION)
action_manager = ActionManager( # Not used, Just a stub 
                action_fields=["acc", "lat_shift", "manuver"] ,
                n_samples = NUM_ACTIONS)
action_space = action_manager.action_space
# Enable autopilot (CARLA built-in expert)
ego_vehicle.set_autopilot(True, tm_port)

spawn_points = world.get_map().get_spawn_points()
# Storage for expert data
observations, actions = [], []
ego, neighbors, maps, global_routes = [], [], [], []

assert isinstance(ego_vehicle, carla.Actor), "Ego vehicle is not an Actor instance."
# Get the ego vehicle's current location.
start_location = ego_vehicle.get_transform().location

# Choose a destination from spawn_points that is at least 100m away.
destination = random.choice(spawn_points)
while destination.location.distance(start_location) < 100:
    destination = random.choice(spawn_points)

# Set up the global route planner.
grp = GlobalRoutePlanner(world.get_map(), 2.0)
# trace_route returns a list of tuples (waypoint, road_option).
route = grp.trace_route(start_location, destination.location)

# Convert the global route into an array with columns [x, y, yaw].
global_route_list = [
    [wp.transform.location.x, wp.transform.location.y, wp.transform.rotation.yaw]
    for wp, _ in route
]
global_route = np.array(global_route_list)

route_location = [r[0].transform.location for r in route]
tm.ignore_lights_percentage(ego_vehicle, 100)
tm.set_path(ego_vehicle, route_location)
tm.auto_lane_change(ego_vehicle, True)

def transform_to_ego_frame():
    # Now, convert the global route into the ego frame.
    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location
    ego_rotation = ego_transform.rotation
    ego_yaw_rad = np.deg2rad(ego_rotation.yaw)
    cos_yaw = np.cos(ego_yaw_rad)
    sin_yaw = np.sin(ego_yaw_rad)

    global_route_ego_frame = []
    for global_point in global_route:
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
    global_route_ego_frame_no_padding = global_route_ego_frame[:observation_manager.bev_info.MAX_LANE_LEN].copy()
    # make the route fixed size.
    global_route_ego_frame = global_route_ego_frame[:observation_manager.bev_info.MAX_LANE_LEN]
    pad_len = observation_manager.bev_info.MAX_LANE_LEN - len(global_route_ego_frame)
    if pad_len > 0:
        global_route_ego_frame = np.pad(global_route_ego_frame, ((0, pad_len), (0, 0)))
    return global_route_ego_frame, global_route_ego_frame_no_padding

# Initialize the Storage for HDF5

h5file = h5py.File("carla_data.h5", "w")
obs_group = h5file.create_group("observations")

# Datasets for each dict entry
ego_ds   = obs_group.create_dataset("ego",   shape=(0, 20,16),  maxshape=(None, 20,16),  dtype=np.uint8,   chunks=True)
neighbors_ds = obs_group.create_dataset("neighbors", shape=(0, 10, 20, 16),   maxshape=(None, 10, 20, 16),   dtype=np.float32, chunks=True)
map_ds = obs_group.create_dataset("map", shape=(0, 20, 50, 50),   maxshape=(None, 20, 50, 50),   dtype=np.float32, chunks=True)
global_route_ds = obs_group.create_dataset("global_route", shape=(0, 50,3), maxshape=(None, 50,3), dtype=np.float32, chunks=True)

act_ds = h5file.create_dataset("actions", shape=(0, 3), maxshape=(None, 3), dtype=np.float32, chunks=True)
next_ds= h5file.create_dataset("next_location", shape=(0, 3), maxshape=(None, 3), dtype=np.float32, chunks=True)

def bicycle_model(control, current_state):
    dt = 1 # discrete time period [s]
    max_delta = 0.6 # vehicle's steering limits [rad]
    max_a = 5 # vehicle's accleration limits [m/s^2]

    x_0 = current_state[:, 0] # vehicle's x-coordinate [m]
    y_0 = current_state[:, 1] # vehicle's y-coordinate [m]
    theta_0 = current_state[:, 2] # vehicle's heading [rad]
    v_0 = torch.hypot(current_state[:, 3], current_state[:, 4]) # vehicle's velocity [m/s]
    L = 3.089 # vehicle's wheelbase [m]
    a = control[:, :, 0].clamp(-max_a, max_a) # vehicle's accleration [m/s^2]
    delta = control[:, :, 1].clamp(-max_delta, max_delta) # vehicle's steering [rad]

    # speed
    v = v_0.unsqueeze(1) + torch.cumsum(a * dt, dim=1)
    v = torch.clamp(v, min=0)

    # angle
    d_theta = v * delta / L # use delta to approximate tan(delta)
    theta = theta_0.unsqueeze(1) + torch.cumsum(d_theta * dt, dim=-1)
    theta = torch.fmod(theta, 2*torch.pi)
    
    # x and y coordniate
    x = x_0.unsqueeze(1) + torch.cumsum(v * torch.cos(theta) * dt, dim=-1)
    y = y_0.unsqueeze(1) + torch.cumsum(v * torch.sin(theta) * dt, dim=-1)
    
    # output trajectory
    traj = torch.nn.functional.pad(torch.stack([ x, y, theta, v], dim=-1), (1,0))

    return traj

def append_step(obs, action, next_location):
    N = ego_ds.shape[0]

    # Resize datasets to make room
    ego_ds.resize(N+1, axis=0)
    neighbors_ds.resize(N+1, axis=0)
    map_ds.resize(N+1, axis=0)
    global_route_ds.resize(N+1, axis=0)
    act_ds.resize(N+1, axis=0)

    next_ds.resize(N+1, axis=0)

    # Store data
    ego_ds[N]   = obs["ego"]
    neighbors_ds[N] = obs["neighbors"]
    map_ds[N] = obs["map"]
    global_route_ds[N] = obs["global_route"]
    act_ds[N]   = action
    next_ds[N] = next_location

try:
    for i in range(2000):  # number of steps to collect
        world.tick()
        # Get state (simplified: speed + location)
        velocity = ego_vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        transform = ego_vehicle.get_transform()
        location = transform.location
        global_route_ego_frame, global_route_ego_frame_no_padding = transform_to_ego_frame()

        obs = observation_manager.get_obs(world, global_route_ego_frame)

        # Get autopilot control (expert action)
        control = ego_vehicle.get_control()
        # act = np.array([control.steer, control.throttle, control.brake], dtype=np.float32)
        
        #  Apply bicycle model
        next_target_point = bicycle_model(torch.tensor([[[control.throttle, control.steer]]]),
                                 torch.tensor([[location.x, location.y, np.deg2rad(transform.rotation.yaw), velocity.x, velocity.y]]))

        # transform next_target_point to carla location object
        next_location = carla.Location(x=next_target_point[0, 0, 1].item(), y=next_target_point[0, 0, 2].item(), z=location.z)

        # transform to ego frame
        delta_x = next_location.x - location.x
        delta_y = next_location.y - location.y

        next_target_point = observation_manager.bev_info.global_to_egocentric(next_target_point, [location.x, location.y, np.deg2rad(transform.rotation.yaw)])
        # yaw = np.deg2rad(transform.rotation.yaw)
        # next_x_ego = delta_x * np.cos(yaw) + delta_y * np.sin(yaw)
        # next_y_ego = -delta_x * np.sin(yaw) + delta_y * np.cos(yaw)
        next_target_point = observation_manager.bev_info.carla_to_MAI_coordinates(data=next_target_point, is_map=False)
        # next_target_point_ego = np.array([next_x_ego, next_y_ego, 0])

        # TODO: Get MAI action space
        longitudinal_accel = (next_target_point[0, 0, 3] - np.hypot(velocity.x, velocity.y)).item()
        # longitudinal_accel = np.hypot(accel_vector[0], accel_vector[1])

        next_wp = world.get_map().get_waypoint(next_location, project_to_road=True)
        curr_wp = world.get_map().get_waypoint(location, project_to_road=True)

        # lane_center = curr_wp.transform.location
        # right_vec = curr_wp.transform.get_right_vector()
        # delta = next_location - location
        # lateral_shift = delta.x * right_vec.x + delta.y * right_vec.y
        
        left_wp = next_wp.get_left_lane() if next_wp.get_left_lane() is not None else None
        right_wp = next_wp.get_right_lane() if next_wp.get_right_lane() is not None else None
        
        lateral_shift = next_wp.transform.location.y - next_location.y
        # next_wp = next_wp.next(2.0)[0]   # 2 meters ahead

        if next_target_point[0,0,2] < -curr_wp.lane_width / 2 and left_wp is not None:
            maneuver = -1  # left
        elif next_target_point[0,0,2] > curr_wp.lane_width / 2 and right_wp is not None:
            maneuver = 1   # right
        else:
            maneuver = 0  # straight 

        act = np.array([maneuver, longitudinal_accel, lateral_shift], dtype=np.float32)
        next_loc = np.array([next_target_point[0,0,1], next_target_point[0,0,2], next_target_point[0,0,3]], dtype=np.float32)
        
        # # Plotting for debugging , TODO: remove later
        # import matplotlib.pyplot as plt
        # plt.scatter( obs['map'][0,:,:, 0], obs['map'][0,:,:, 1], c='red', s=1)
        # plt.scatter(obs['global_route'][:, 0], obs['global_route'][:, 1], c='purple', s=1)


        # # plt.figure()

        # plt.scatter(obs['ego'][0,:, 1], obs['ego'][0,:, 2], c='blue', s=10, marker='x')
        # plt.scatter(next_loc[0], next_loc[1], c='orange', s=10, marker='x')

        # # plt.scatter(neighbors[i][:, :, 1], neighbors[i][:, :, 2], c='green', s=1)


        # plt.xlim(-50, 50)
        # plt.ylim(-50, 50)
        # plt.savefig("plots/plot_{:03d}.png".format(i))
        # plt.cla()

        append_step(obs, act, next_loc)

finally:
    ego_vehicle.destroy()
    print("Data collection finished, vehicle destroyed.")




# # Save dataset
# dataset = {"obs": np.array(observations), "acts": np.array(actions)}
# with open("expert_carla_OG.pkl", "wb") as f:
#     pickle.dump(dataset, f)

print("Expert dataset saved to carla_data.h5")
