import matplotlib
import math
import numpy as np
import carla
import heapq
import torch
import time

from importlib.metadata import metadata

from packaging.version import Version
from time import sleep

# PATH = os.path.dirname(os.path.realpath(__file__)).split("train")[0]
# sys.path.append(PATH)

# from carla import ad
from src.envs.observation.decision_traffic_rules.feature_indices import agent_feat_id
from src.envs.observation.decision_traffic_rules.traffic_sign_db import traffic_feat_idx, regulatory_ts_encoding
# from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from src.envs.observation.decision_traffic_rules.lanelet_data_extractor import LaneletDataExtractor

from src.envs.observation.tr_costs import get_rel_prio_intersection

matplotlib.use("Agg")

RSS_NOT_FOUND_VALUE = -100.0

class Vector_BEV_observer:
    def __init__(
        self,
        HISTORY=20,
        FUTURE_LEN=30,
        FOV=40,
        MAX_LANES=20,
        MAX_LANE_LEN=50,
        MAX_NEIGHBORS=10,
        LANEPOINT_DIST=2,
        MAX_POINTS_CROSSWALK = 10,
        MAX_CROSSWALKS = 10,
    ):
        # initialise buffer - no refernce
        self.ego_data = []
        self.neighbor_data = []
        self.map_data = []
        self.crosswalk_data = []
        self.ego_transform_data = []

        # initialise buffer - rereferenced
        self.ego_hist = []
        self.neighbor_hist = []
        self.map_hist = []
        self.crosswalk_hist = []
        self.ground_truth = []

        self.attempts_to_collect = 0
        self.HISTORY = HISTORY  # overall length of buffer
        self.FUTURE_LEN = FUTURE_LEN  # amount of buffer length to treat as future
        self.FOV = FOV  # fold of view in m
        self.MAX_NEIGHBORS = MAX_NEIGHBORS
        self.MAX_LANES = MAX_LANES
        self.MAX_LANE_LEN = MAX_LANE_LEN
        self.MAX_POINTS_CROSSWALK = MAX_POINTS_CROSSWALK
        self.MAX_CROSSWALKS = MAX_CROSSWALKS
        self.LANEPOINT_DIST = LANEPOINT_DIST  # diatnce between lane points in m
        self.file_name = 0
        try:
            self.lanelet_data = LaneletDataExtractor(
                file_path="/home/ubuntu/workstation/CARLA_0.9.14_RSS/CarlaUE4/Content/Carla/Maps/Town04.osm",
                origin=(0, 0),
                proj_type="utm",
            )
        except Exception as e:
            # print("Lanelets cant be loaded")
            # print(e)
            self.lanelet_data = None

    def create_client(self):
        print("Creating client")
        # Tunable parameters
        client_timeout = 10.0  # in seconds
        wait_for_world = 20.0  # in seconds
        MIN_CARLA_VERSION = "0.9.14"

        host = "127.0.0.1"
        port = 2000
        client = carla.Client(host, int(port))
        client.set_timeout(client_timeout)
        carla_version = Version(metadata("carla")["Version"])
        if carla_version < Version(MIN_CARLA_VERSION):
            raise ImportError(
                "CARLA version {} or newer required. CARLA version found: {}".format(
                    MIN_CARLA_VERSION, carla_version
                )
            )
        world = client.get_world()

        # Set to synchronous with fixed timestep
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1  # NOTE: Should not exceed 0.1
        world.apply_settings(settings)
        return world

    # def rss_callback(self, response):
    #     self.rss_states = None
    #     if len(response.rss_state_snapshot.individualResponses):
    #         self.rss_states = []
    #     for individual_states in response.rss_state_snapshot.individualResponses:
    #         indv_state = []
    #         indv_state.append(individual_states.objectId)
    #         indv_state.append(ad.rss.state.isDangerous(individual_states))
    #         indv_state.append(
    #             float(
    #                 individual_states.longitudinalState.rssStateInformation.currentDistance
    #             )
    #         )
    #         indv_state.append(
    #             float(
    #                 individual_states.longitudinalState.rssStateInformation.safeDistance
    #             )
    #         )
    #         indv_state.append(
    #             float(
    #                 individual_states.lateralStateRight.rssStateInformation.currentDistance
    #             )
    #         )
    #         indv_state.append(
    #             float(
    #                 individual_states.lateralStateRight.rssStateInformation.safeDistance
    #             )
    #         )
    #         indv_state.append(
    #             float(
    #                 individual_states.lateralStateLeft.rssStateInformation.currentDistance
    #             )
    #         )
    #         indv_state.append(
    #             float(
    #                 individual_states.lateralStateLeft.rssStateInformation.safeDistance
    #             )
    #         )

    #         self.rss_states.append(indv_state)
    #     return self.rss_states

    def client_init(
        self,
        world,
        maprenderer=None,
    ):
        self.world = world
        self.ego = None

        # get actors from the world
        carla_vehicles = list(world.get_actors().filter("vehicle.*"))+list(world.get_actors().filter("walker.*")) # TODO split up and cover remaing classes, such as debris
        #print("-----")
        for i, veh in enumerate(carla_vehicles):
            if veh.attributes["role_name"] == ("hero") or veh.attributes["role_name"] == ("ego_vehicle"): # TODO this role name depends on the scenario generation tool
                try:
                    # self.rss = (
                    #     list(world.get_actors().filter("sensor.other.rs*"))[0]
                    # )
                    self.col = (
                         list(world.get_actors().filter("sensor.other.col*"))[0]
                    )
                except:
                    # self.rss = None
                    self.col = None
                self.ego = carla_vehicles.pop(i)

                self.neighs = carla_vehicles
                self.carla_map = world.get_map()
                self.maprenderer = maprenderer

                # self.extract_map_features()
                self.attempts_to_collect = 0
                return True
        if self.ego is None:
            if self.attempts_to_collect > 3:
                print(
                    f"Looking for ego vehicle attempt {self.attempts_to_collect} failed"
                )
                return False
            sleep(3)
            self.attempts_to_collect += 1
            self.client_init(world)


    def carla_to_MAI_coordinates(self, data, is_map=False):
        if not is_map:
            idx_x = agent_feat_id["x"]
            idx_y = agent_feat_id["y"]
            idx_yaw = agent_feat_id["yaw"]
        else: # TODO extract from map config
            idx_x = traffic_feat_idx["cl_x"]
            idx_y = traffic_feat_idx["cl_y"]
            idx_yaw = traffic_feat_idx["cl_yaw"]

        # flip y axis and angle
        try:
            data[..., idx_y] = -data[..., idx_y]
            data[..., idx_yaw] = -data[..., idx_yaw]
        except:
            # TODO explicitely handle rereferencing without provided yaw
            pass

        return data

    def get_turning_intention(self, vehicle):
        """
        Get the turning intention of vehicle at intersection
        """
        turnleft = False
        turnRight = False
        if vehicle.get_light_state() == carla.VehicleLightState.LeftBlinker:
            turnleft = True
        if vehicle.get_light_state() == carla.VehicleLightState.RightBlinker:
            turnRight = True
        return turnleft, turnRight

    def global_to_egocentric(self, data, ego, is_map=False):
        if not is_map:
            idx_x = agent_feat_id["x"]
            idx_y = agent_feat_id["y"]
            idx_yaw = agent_feat_id["yaw"]
        else: # TODO extract from map config
            idx_x = traffic_feat_idx["cl_x"]
            idx_y = traffic_feat_idx["cl_y"]
            idx_yaw = traffic_feat_idx["cl_yaw"]

        data = np.asarray(data)
        data_mask = np.where(data == 0, 0, 1)
        x_e, y_e, yaw_ego = ego
        # Translate the global position relative to the ego
        dx = data[..., idx_x] - x_e
        dy = data[..., idx_y] - y_e
        # Rotation matrix for the ego's heading (convert to local coordinates)
        cos_ego = np.cos(-yaw_ego)
        sin_ego = np.sin(-yaw_ego)
        # Apply the rotation to the translated position
        data[..., idx_x] = cos_ego * dx - sin_ego * dy
        data[..., idx_y] = sin_ego * dx + cos_ego * dy
        
        try:
            # Adjust the heading to be relative to the ego vehicle's heading
            data[..., idx_yaw] = data[..., idx_yaw] - yaw_ego
            # Ensure the heading is within -pi to pi
            data[..., idx_yaw] = np.arctan2(np.sin(data[..., idx_yaw]), np.cos(data[..., idx_yaw]))
        except:
            # TODO explicitely handle location rereferencing without provided yaw
            pass 
        return data * data_mask

    def get_ego_current_pos(self):
        return self.ego_current, self.ego_transform

    def extract_map_features(self):
        keys = {
            "maxspeed",
            "traffic_sign_warning",
            "traffic_sign_directional",
            "speed_limit",
            "priority",
            "stop_sign",
            "traffic_light",
            "pedestrian_crossing",
            "yield_right_of_way",
        }
        # collect lane topology around ego
        ego_location = self.ego.get_location()
        self.waypoints = []
        # Dictionary to store the distances to each lane
        lane_distances = []
        for wp in self.carla_map.get_topology():
            start_distance = wp[0].transform.location.distance(ego_location)
            end_distance = wp[1].transform.location.distance(ego_location)
            # if start_distance < self.FOV or end_distance < self.FOV:
            # Add the starting waypoint of the lane and its distance to the list
            lane_distances.append((min(start_distance, end_distance), wp[0])) #.next_until_lane_end(self.LANEPOINT_DIST)))

        # Get n closest lanes
        n_closest_lanes = heapq.nsmallest(self.MAX_LANES, lane_distances, key=lambda x: x[0])
        # Append the waypoints for the n closest lanes to self.waypoints
        # TODO: this for loop can be avoided
        for _, closest_wp in n_closest_lanes:
            self.waypoints.append(closest_wp.next_until_lane_end(self.LANEPOINT_DIST))

        # make cardreamer visualize the extracted data
        if self.maprenderer is not None:
            self.maprenderer._surface *= 0
            self.maprenderer._topology = self.waypoints
            self.maprenderer.render_topology()

        # collect lanes [x,y,heading, lane/junction, left_boundary_type, right_boundary_type]
        map_lanes = np.zeros(
            [self.MAX_LANES, self.MAX_LANE_LEN, max(traffic_feat_idx.values()) + 1]
        )

        for l, lane in enumerate(self.waypoints):
            if l < self.MAX_LANES:
                # location and yaw
                points_x = np.asarray([wp.transform.location.x for wp in lane])
                points_y = np.asarray([wp.transform.location.y for wp in lane])
                points_yaw = np.deg2rad(
                    np.asarray([wp.transform.rotation.yaw for wp in lane])
                )
                # check if intersection
                points_type = np.asarray(
                    [0 if lane[0].is_junction else 1 for wp in lane]
                )
                # lane markings
                left_marking_type = (
                    0
                    if lane[0].left_lane_marking.type
                    == carla.libcarla.LaneMarkingType.SolidSolid
                    else 1
                )
                right_marking_type = (
                    0
                    if lane[0].right_lane_marking.type
                    == carla.libcarla.LaneMarkingType.SolidSolid
                    else 1
                )
                left_marking = np.asarray([left_marking_type for wp in lane])
                right_marking = np.asarray([right_marking_type for wp in lane])

                # longitudinal progress, road id, lane id
                points_s = np.asarray([wp.s for wp in lane])
                points_road_id = np.asarray([wp.road_id for wp in lane])
                points_lane_id =  np.asarray([wp.lane_id for wp in lane])
                points_is_route = [0]*len(points_road_id)

                # assign collected information
                map_lanes[l, : points_x.shape[0], traffic_feat_idx["cl_x"]] = points_x[
                    : self.MAX_LANE_LEN
                ]
                map_lanes[l, : points_y.shape[0], traffic_feat_idx["cl_y"]] = points_y[
                    : self.MAX_LANE_LEN
                ]
                map_lanes[l, : points_yaw.shape[0], traffic_feat_idx["cl_yaw"]] = (
                    points_yaw[: self.MAX_LANE_LEN]
                )
                map_lanes[l, : points_type.shape[0], traffic_feat_idx["cl_type"]] = (
                    points_type[: self.MAX_LANE_LEN]
                )
                map_lanes[l, : left_marking.shape[0], traffic_feat_idx["ll_type"]] = (
                    left_marking[: self.MAX_LANE_LEN]
                )
                map_lanes[l, : right_marking.shape[0], traffic_feat_idx["rl_type"]] = (
                    right_marking[: self.MAX_LANE_LEN]
                )
                map_lanes[l, : points_s.shape[0], traffic_feat_idx["s"]] = (
                    points_s[: self.MAX_LANE_LEN]
                )
                map_lanes[l, : points_road_id.shape[0], traffic_feat_idx["road_id"]] = (
                    points_road_id[: self.MAX_LANE_LEN]
                )
                map_lanes[l, : points_lane_id.shape[0], traffic_feat_idx["lane_id"]] = (
                    points_lane_id[: self.MAX_LANE_LEN]
                )

                if (self.world.get_map().get_waypoint(ego_location).road_id == points_road_id[0]):
                     map_lanes[l, :points_road_id.shape[0], traffic_feat_idx["is_route"]] = 1

        if self.lanelet_data is not None:
            map_lanes = self.lanelet_data.extract_route_rules(
                torch.tensor(map_lanes),
                [ego_location.x, ego_location.y],
                keys,
            ).tolist()

        # Repeat it for all vehicles
        # map_lanes_features = map_lanes_features.repeat(self.MAX_NEIGHBORS + 1, 1, 1, 1)

        # def get_signs(world):
        #     speed_limit = list(world.get_actors().filter("*traffic.speed_limit*"))
        #     stop = list(world.get_actors().filter("*traffic.stop*")) 
        #     yields = list(world.get_actors().filter("*traffic.yield*")) 
        #     warning = list(world.get_actors().filter('*warning*'))
        #     dirtdebris = list(world.get_actors().filter('*dirtdebris*'))
        #     cone = list(world.get_actors().filter('*cone*'))
        #     traffic_lights = list(world.get_actors().filter("*traffic_light*"))
        #     return speed_limit, stop, yields, warning, dirtdebris, cone, traffic_lights

        def sign_features(sign):
            traffic_light_state = { # TODO map to config file
                carla.libcarla.TrafficLightState.Red: 1,
                carla.libcarla.TrafficLightState.Yellow: 2,
                carla.libcarla.TrafficLightState.Green: 3,
                carla.libcarla.TrafficLightState.Off: 4,
                carla.libcarla.TrafficLightState.Unknown: 4,
            }
            pos_x = sign.get_location().x
            pos_y = sign.get_location().y
            pos_yaw = sign.trigger_volume.rotation.yaw
            extent_x = sign.trigger_volume.extent.x
            extent_y = sign.trigger_volume.extent.y
            state = traffic_light_state[sign.state] if hasattr(sign, "state") else 0
            return pos_x, pos_y, pos_yaw, extent_x, extent_y, state 

        # run through regulatory elements in FOV
        ego_wp = self.carla_map.get_waypoint(self.ego.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
        landmarks = ego_wp.get_landmarks(self.FOV)
        for landmark in landmarks:
            try:
                sign = self.world.get_traffic_sign(landmark)
                pos_x, pos_y, pos_yaw, extent_x, extent_y, state = sign_features(sign)

                # get regulatory element features
                pos_x, pos_y, pos_yaw, extent_x, extent_y, state = sign_features(sign)
                sign_name = str(sign.type_id).split(".")

                # find lane with closest waypoint
                closest_lane_id = np.linalg.norm(np.asarray([pos_x, pos_y])-map_lanes[...,:2], axis=2).min(-1).argmin(0)

                # find closest waypoint within closest lane
                closest_wp_id = np.linalg.norm(np.asarray([pos_x, pos_y])-map_lanes[closest_lane_id][...,:2], axis=-1).argmin(0)

                # assign regulatory element type and its features to map lane features
                if "speed_limit" in sign_name:
                    map_lanes[closest_lane_id][closest_wp_id][traffic_feat_idx["speed_limit"]] = float(sign_name[-1])
                elif "traffic_light" in sign_name:
                    map_lanes[closest_lane_id][closest_wp_id][traffic_feat_idx["traffic_light"]] = float(state)
                elif "stop" in sign_name:
                    map_lanes[closest_lane_id][closest_wp_id][traffic_feat_idx["traffic_sign_regulatory"]] = regulatory_ts_encoding["stop_sign"]
                    map_lanes[closest_lane_id][closest_wp_id][traffic_feat_idx["stop_sign"]] = 1.
                elif "yield" in sign_name:
                    map_lanes[closest_lane_id][closest_wp_id][traffic_feat_idx["traffic_sign_regulatory"]] = regulatory_ts_encoding["yield_right_of_way"]

                # stop line. since carla does not have them properly defined, we use the last point in the lane # TODO improve
                last_waypoint_idx = -1 # TODO verify this is the last point in the lane in driving direction
                if "stop" in sign_name or "traffic_light" in sign_name:
                    # print("adding stop point")
                    map_lanes[closest_lane_id][-1][traffic_feat_idx["stop_point"]] = 1.
            except Exception as e:
                # print(f"Error in sign_features(): {e}")
                continue
        
        self.map_lanes = map_lanes

        # get crosswalks in FOV 
        # TODO add other features, right now is only x,y position
        crosswalks = np.zeros((self.MAX_CROSSWALKS, self.MAX_POINTS_CROSSWALK, 2))
        crosswalks_list = self.carla_map.get_crosswalks()
        list_of_groups = zip(*(iter(crosswalks_list),) * 5)
        cw_polygons = [list(i) for i in list_of_groups]
        count = len(crosswalks_list) % 2
        cw_polygons.append(crosswalks_list[-count:]) if count != 0 else cw_polygons
        for i in range(len(cw_polygons)):
            cw_polygons[i] = [[c.x for c in cw_polygons[i]], [c.y for c in cw_polygons[i]]]
            cw_polygons[i] = np.asarray(cw_polygons[i]).T 
            # interpolate to desired length
            cw_polygons[i] = cw_polygons[i][np.linspace(0, cw_polygons[i].shape[0], num=self.MAX_POINTS_CROSSWALK, endpoint=False, dtype=np.int32)]
        cw_polygons = np.asarray(cw_polygons)
        # keep only crosswalks in FOV
        if cw_polygons.shape[0] > 0:
            distances = np.linalg.norm(cw_polygons-np.asarray([ego_location.x, ego_location.y]), axis=-1).min(-1)
            cw_polygons = cw_polygons[distances<self.FOV]
            crosswalks[:cw_polygons.shape[0]] = cw_polygons

        self.crosswalks = crosswalks

    def lon_rss(self, agent):
        # longitudinal RSS min distance versus distance between center points of agents
        # based on https://github.com/nellro/rgt/
        # TODO add lateral
        # TODO requires check if actually in front and same lane?
        # TODO use bounding box of cars instead of center points, like here: https://github.com/nellro/rgt/blob/d961897e4c0a8f9a1732283417b7a24ddccd1eea/code/tools/dist_aux.py#L79
        # TODO compare with Carla's RSS package sensor, which should include yielding rights and unstructured, like here: https://github.com/nellro/rgt/blob/d961897e4c0a8f9a1732283417b7a24ddccd1eea/code/scenario_runner_extension/rss_aux.py#L113

        lon_a_max = 3.5
        lon_b_max = 8.0
        lon_b_min = 4.0
        lat_a_max = 0.2
        lat_b_min = 0.8
        t_resp = 1.0

        v_ego = self.ego.get_velocity()
        v_ego = math.sqrt(v_ego.x**2 + v_ego.y**2 + v_ego.z**2)

        agent_v = agent.get_velocity()
        v_agent = math.sqrt(agent_v.x**2 + agent_v.y**2 + agent_v.z**2)

        loc_ego = self.ego.get_location()
        true_dist = loc_ego.distance(agent.get_location())

        d_rss = max(
            0,
            v_ego * t_resp
            + 0.5 * lon_a_max * t_resp**2
            + (v_ego + lon_a_max * t_resp) ** 2 / (2 * lon_b_min)
            - v_agent / (2 * lon_b_max),
        )

        # print(
        #     "d_rss",
        #     "true_dist - d_rss",
        #     true_dist - d_rss,
        #     "safe",
        #     true_dist - d_rss > 0,
        # )

        return d_rss, true_dist - d_rss

    def update(self, plot=True, re_reference=False, rss_states=None):
        self.extract_map_features()
        # if self.rss is not None:
        #     rss_states = self.rss.listen(self.rss_callback)

        map_lanes = self.map_lanes

        # collect ego
        ego_location = self.ego.get_location()
        ego = np.zeros(max(agent_feat_id.values()) + 1)
        ego[agent_feat_id["x"]] = self.ego.get_location().x
        ego[agent_feat_id["y"]] = self.ego.get_location().y
        ego[agent_feat_id["yaw"]] = np.deg2rad(self.ego.get_transform().rotation.yaw)
        ego[agent_feat_id["vx"]] = self.ego.get_velocity().x
        ego[agent_feat_id["vy"]] = self.ego.get_velocity().y
        ego[agent_feat_id["length"]] = self.ego.bounding_box.extent.x
        ego[agent_feat_id["width"]] = self.ego.bounding_box.extent.y
        ego[agent_feat_id["height"]] = self.ego.bounding_box.extent.z
        ego_closest_wp = self.carla_map.get_waypoint(self.ego.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving)) # TODO improve filtering
        ego[agent_feat_id["road_id"]] = ego_closest_wp.road_id
        ego[agent_feat_id["lane_id"]] = ego_closest_wp.lane_id
        ego[agent_feat_id["is_junction"]] = ego_closest_wp.is_junction
        ego[agent_feat_id["s"]] = ego_closest_wp.s

        try:
            # print("Getting ego turning intention")
            ego_turnleft, ego_turnright = self.get_turning_intention(self.ego)
            ego[agent_feat_id["turning_intention_left"]] = float(ego_turnleft)
            ego[agent_feat_id["turning_intention_right"]] = float(ego_turnright)
        except:
            pass

        # print("EGO:")
        # print("road_id", ego[agent_feat_id["road_id"]])
        # print("lane_id", ego[agent_feat_id["lane_id"]])
        # print("is_junction", ego[agent_feat_id["is_junction"]])
        # print("s", ego[agent_feat_id["s"]])

        # collect neighbor
        neighbor_location = np.zeros(
            (self.MAX_NEIGHBORS, max(agent_feat_id.values()) + 1)
        )

        # find smallest actor id to allow sorting them by id # TODO improve
        # min_actor_id = np.min([actor.id for actor in self.neighs])
        # for j, actor in enumerate(self.neighs):
        #     print("self neighs", j, actor.id)
        # print("min_actor_id", min_actor_id)
        i = 0
        # print("neighbor_location", neighbor_location.shape)
        for j, actor in enumerate(self.neighs):
            if j < self.MAX_NEIGHBORS:
                # i = actor.id - min_actor_id # to prevent assigning different cars to one array index we assign it using actor's id # TODO improve
                # print("actor", actor.id, i)

                # actor = task[0]  # task.unwrapped._world.actor_dict[neighbor_id]
                if actor.is_alive and (actor.id != self.ego.id):
                    neighbor_location[i, agent_feat_id["id"]] = actor.id
                    neighbor_location[i, agent_feat_id["x"]] = actor.get_location().x
                    neighbor_location[i, agent_feat_id["y"]] = actor.get_location().y
                    neighbor_location[i, agent_feat_id["yaw"]] = np.deg2rad(
                        actor.get_transform().rotation.yaw
                    )
                    neighbor_location[i, agent_feat_id["vx"]] = actor.get_velocity().x
                    neighbor_location[i, agent_feat_id["vy"]] = actor.get_velocity().y
                    neighbor_location[i, agent_feat_id["length"]] = (
                        actor.bounding_box.extent.x
                    )
                    neighbor_location[i, agent_feat_id["width"]] = (
                        actor.bounding_box.extent.y
                    )
                    neighbor_location[i, agent_feat_id["height"]] = (
                        actor.bounding_box.extent.z
                    )
                    agent_class = 0
                    # CARLA base classes: car, truck, van, bus, motorcycle, bicycle
                    # Waymo DIPP: Unset=0, Vehicle=1, Pedestrian=2, Cyclist=3, Other=4
                    if "base_type" in actor.attributes:
                        # print("actor.attributes[base_type]", actor.attributes["base_type"] )
                        if actor.attributes["base_type"] in ["car", "truck", "van", "bus", "motorcycle"]:
                            agent_class = 1

                            # turning intention for vehicles based on turning lights
                            try:
                                # print("Getting actor turning intention", actor.id)
                                actor_turnleft, actor_turnright = self.get_turning_intention(actor)
                                neighbor_location[agent_feat_id["turning_intention_left"]] = float(actor_turnleft)
                                neighbor_location[agent_feat_id["turning_intention_right"]] = float(actor_turnright)
                            except:
                                pass

                        elif actor.attributes["base_type"] == "bicycle":  # TODO check
                            agent_class = 3
                    else: # TODO check if it is pedestrian explicitely
                        print("adding pedestrian")
                        agent_class = 2

                    neighbor_location[i, agent_feat_id["class"]] = agent_class
                    if rss_states is not None:
                        added = False
                        for state in rss_states:
                            if state[0] == actor.id:
                                neighbor_location[i, agent_feat_id['rss_obj_id']] = state[0]
                                neighbor_location[i, agent_feat_id['rss_status']] = state[1]
                                neighbor_location[i, agent_feat_id['rss_long_current_dist']] = state[2]
                                neighbor_location[i, agent_feat_id['rss_long_safe_dist']] = state[3]
                                neighbor_location[i, agent_feat_id['rss_lat_current_right_dist']] = state[4]
                                neighbor_location[i, agent_feat_id['rss_lat_safe_right_dist']] = state[5]
                                neighbor_location[i, agent_feat_id['rss_lat_current_left_dist']] = state[6]
                                neighbor_location[i, agent_feat_id['rss_lat_safe_left_dist']] = state[7]
                                added = True
                                break
                        if not added:
                            print(
                                f"\n{'='*50}\n🚨 RSS states not added for neighbor with id: ", actor.id
                            )
                            neighbor_location[i, agent_feat_id['rss_obj_id']] = RSS_NOT_FOUND_VALUE
                            neighbor_location[i, agent_feat_id['rss_status']] = RSS_NOT_FOUND_VALUE
                            neighbor_location[i, agent_feat_id['rss_long_current_dist']] = RSS_NOT_FOUND_VALUE
                            neighbor_location[i, agent_feat_id['rss_long_safe_dist']] = RSS_NOT_FOUND_VALUE
                            neighbor_location[i, agent_feat_id['rss_lat_current_right_dist']] = RSS_NOT_FOUND_VALUE
                            neighbor_location[i, agent_feat_id['rss_lat_safe_right_dist']] = RSS_NOT_FOUND_VALUE
                            neighbor_location[i, agent_feat_id['rss_lat_current_left_dist']] = RSS_NOT_FOUND_VALUE
                            neighbor_location[i, agent_feat_id['rss_lat_safe_left_dist']] = RSS_NOT_FOUND_VALUE
                    else:
                        # print("\n{'='*50}\n🚨 RSS states not added for neighbor with id: ", actor.id)
                        neighbor_location[i, agent_feat_id['rss_obj_id']] = RSS_NOT_FOUND_VALUE
                        neighbor_location[i, agent_feat_id['rss_status']] = RSS_NOT_FOUND_VALUE
                        neighbor_location[i, agent_feat_id['rss_long_current_dist']] = RSS_NOT_FOUND_VALUE
                        neighbor_location[i, agent_feat_id['rss_long_safe_dist']] = RSS_NOT_FOUND_VALUE
                        neighbor_location[i, agent_feat_id['rss_lat_current_right_dist']] = RSS_NOT_FOUND_VALUE
                        neighbor_location[i, agent_feat_id['rss_lat_safe_right_dist']] = RSS_NOT_FOUND_VALUE
                        neighbor_location[i, agent_feat_id['rss_lat_current_left_dist']] = RSS_NOT_FOUND_VALUE
                        neighbor_location[i, agent_feat_id['rss_lat_safe_left_dist']] = RSS_NOT_FOUND_VALUE

                    neigh_closest_wp = self.carla_map.get_waypoint(actor.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving)) # TODO improve filtering
                    neighbor_location[i, agent_feat_id["road_id"]] = neigh_closest_wp.road_id
                    neighbor_location[i, agent_feat_id["lane_id"]] = neigh_closest_wp.lane_id
                    neighbor_location[i, agent_feat_id["is_junction"]] = neigh_closest_wp.is_junction
                    neighbor_location[i, agent_feat_id["s"]] = neigh_closest_wp.s

                    # get the relative priority at intersections
                    # TODO first we need to check if any actor was inside a intersection, otherwise we can skip this
                    # TODO add traffic signs to priority computation
                    # TODO skip if traffic light applies
                    try:
                        other_relative_priority = get_rel_prio_intersection(self.ego, actor, self.carla_map)
                        neighbor_location[i, agent_feat_id["relative_priority"]] = other_relative_priority
                    except:
                        print("ERROR IN RELATIVE PRIORITY")
                    i +=1
                    # import time
                    # time.sleep(1.0)

        # print(neighbor_location[:,agent_feat_id["id"]])
        sorted_neighbor_ids = np.argsort(np.array(neighbor_location[:,agent_feat_id["id"]]))
        # print(neighbor_location[sorted_neighbor_ids][:,agent_feat_id["id"]])
        sorted_neighbors = neighbor_location[sorted_neighbor_ids]

        # add to buffer
        self.ego_data.append(ego)
        self.ego_transform_data.append(self.ego.get_transform())
        self.neighbor_data.append(sorted_neighbors)
        self.map_data.append(map_lanes)
        self.crosswalk_data.append(self.crosswalks)

        if re_reference:
            self.ego_data_l = self.ego_data[-(self.HISTORY + self.FUTURE_LEN) :]
            self.neighbor_data_l = self.neighbor_data[
                -(self.HISTORY + self.FUTURE_LEN) :
            ]
            self.map_data_l = self.map_data[-(self.HISTORY + self.FUTURE_LEN) :]
            self.crosswalk_data_l = self.crosswalk_data[-(self.HISTORY + self.FUTURE_LEN) :]

            if len(self.ego_data_l) == (self.HISTORY + self.FUTURE_LEN):
                ego_hist, neighbor_hist, map_hist, ground_truth, crosswalk_hist = (
                    self.get_rereferenced_chunk(plot=plot)
                )
                self.ego_hist.append(ego_hist)
                self.neighbor_hist.append(neighbor_hist)
                self.map_hist.append(map_hist)
                self.crosswalk_hist.append(crosswalk_hist)
                self.ground_truth.append(ground_truth)
                return ego_hist, neighbor_hist, map_hist, ground_truth, crosswalk_hist
            else:
                return None, None, None, None, None
        else:
                return self.ego_data[-self.HISTORY:], self.neighbor_data[-self.HISTORY:], self.map_data[-self.HISTORY:], self.crosswalk_data[-self.HISTORY:], None
            # return (
            #     np.asarray(ego_location),
            #     np.asarray(neighbor_location),
            #     np.asarray(map_lanes),
            # )

    def get_rereferenced_chunk(self, plot=True):
        ego_hist = np.array(self.ego_data_l[: self.HISTORY])
        neighbor_hist = np.array(self.neighbor_data_l[: self.HISTORY])
        map_hist = self.map_data_l[: self.HISTORY]
        crosswalk_hist = self.crosswalk_data_l[: self.HISTORY]

        # rereference buffer to current ego position
        self.ego_current = ego_hist[-1, agent_feat_id["x"]:agent_feat_id["yaw"]+1] # TODO extract properly from config
        self.ego_transform = self.ego_transform_data[-1]

        map_hist = map_hist[-1]
        crosswalk_hist = crosswalk_hist[-1]
        ego_hist = torch.tensor(ego_hist)
        neighbor_hist = torch.tensor(neighbor_hist)
        neighbor_hist = neighbor_hist.swapaxes(0, 1)
        actors_hist = np.vstack(
            [ego_hist[..., :3].unsqueeze(0), neighbor_hist[..., :3]]
        )
        for l in range(len(map_hist)):
            map_hist[l] = torch.tensor(
                self.global_to_egocentric(np.asarray(map_hist[l]), self.ego_current, is_map=True)
            )
        for l in range(len(crosswalk_hist)):
            crosswalk_hist[l] = torch.tensor(
                self.global_to_egocentric(np.asarray(crosswalk_hist[l]), self.ego_current, is_map=True)
            )

        # ego_current[2] *= 0  # set ego global yaw to zero TODO check
        ego_hist = self.global_to_egocentric(ego_hist, self.ego_current)
        neighbor_hist = self.global_to_egocentric(neighbor_hist, self.ego_current)

        ego_future = self.ego_data_l[self.HISTORY :]
        ego_future = self.global_to_egocentric(ego_future, self.ego_current)
        neigh_future = self.neighbor_data_l[self.HISTORY :]
        neigh_future = self.global_to_egocentric(neigh_future, self.ego_current)
        neigh_future = neigh_future.swapaxes(0, 1)
        # Pad EGO with zeros for ground truth 
        temp = np.zeros_like(neigh_future)[0]
        temp[:,:ego_future.shape[1]] = ego_future
        ground_truth = np.vstack(
            [np.expand_dims(temp, 0), neigh_future]
        )

        # if plot:
        #     self.plot(
        #         (neighbor_hist),
        #         (ego_hist),
        #         (neigh_future),
        #         (ego_future),
        #         np.concatenate(map_hist),
        #     )

        ego_hist = torch.tensor(ego_hist).unsqueeze(0).type(torch.float32)
        neighbor_hist = torch.tensor(neighbor_hist).unsqueeze(0).type(torch.float32)
        ground_truth = torch.tensor(ground_truth).unsqueeze(0).type(torch.float32)
        map_hist = torch.tensor(np.stack(map_hist)).unsqueeze(0).unsqueeze(0).type(torch.float32)
        crosswalk_hist = torch.tensor(np.stack(crosswalk_hist)).unsqueeze(0).unsqueeze(0).type(torch.float32)
        return ego_hist, neighbor_hist, map_hist, ground_truth, crosswalk_hist

    def save_file(self, file_name=None, selected_frame=None): # TODO selected_frame should match history length!
        
        # stack data into buffer and rerefernce to egocentric frame
        ego = np.vstack(self.ego_data)
        neighbors  = np.concatenate(np.expand_dims(self.neighbor_data, axis=0))
        map = np.concatenate(np.expand_dims(self.map_data, axis=0))
        crosswalk = np.concatenate(np.expand_dims(self.crosswalk_data, axis=0))
        ego, neighbors, map, ground_truth, crosswalk = self.create_buffer_and_transform_frame(ego, neighbors, map, crosswalk)
        
        # select desired parts of data
        if selected_frame is not None:
            print("selecting frame", selected_frame)
            ego=ego[selected_frame:selected_frame+1],
            neighbors=neighbors[selected_frame:selected_frame+1],
            map=map[selected_frame:selected_frame+1],
            crosswalk=crosswalk[selected_frame:selected_frame+1],
            ground_truth=ground_truth[selected_frame:selected_frame+1]

        # convert from carla (flipped y axis) to MAI coordinates
        ego = self.carla_to_MAI_coordinates(data=ego, is_map=False)
        neighbors = self.carla_to_MAI_coordinates(data=neighbors, is_map=False)
        ground_truth = self.carla_to_MAI_coordinates(data=ground_truth, is_map=False)
        map = self.carla_to_MAI_coordinates(data=map, is_map=True)
        crosswalk = self.carla_to_MAI_coordinates(data=crosswalk, is_map=True)

        # save to disk
        file_name = f"{file_name}{time.time()}.npz"
        np.savez(
            file_name,
            ego=ego,
            neighbors=neighbors,
            map=map,
            ground_truth=ground_truth,
            crosswalk=crosswalk,
        )

        print(f"Ego shape: {ego.shape}")
        print(f"Neighbors shape: {neighbors.shape}")
        print(f"Map shape: {map.shape}")
        print(f"crosswalk shape: {crosswalk.shape}")
        print(f"Ground Truth shape: {ground_truth.shape}")
        print(f"Data saved as {file_name}")

        self.file_name += 1

    def run_and_save_data(self, file_name=None, world=None, re_reference=False):
        if file_name is None:
            file_name = f"data_{self.file_name}.npz"

        if world is None:
            world = self.create_client()
        self.ego_hist = []
        self.neighbor_hist = []
        self.map_hist = []
        self.ground_truth = []

        print("Data recording.....")

        while self.client_init(world):
            if not re_reference:
                self.update(re_reference=re_reference)
            else:
                ego_hist, neighbor_hist, map_hist, ground_truth, crosswalk_hist = self.update(
                    re_reference=re_reference
                )
                if ego_hist is not None:
                    self.ego_hist.append(np.array(ego_hist))
                    self.neighbor_hist.append(np.array(neighbor_hist))
                    self.map_hist.append(np.array(map_hist))
                    self.crosswalk_hist.append(np.array(crosswalk_hist))
                    self.ground_truth.append(np.array(ground_truth))

        self.save_file(
            #re_reference=re_reference, # TODO double check and remove have rereferenced before, if requested
            file_name=file_name,
        )

    def create_buffer(self, ego, neighbors):  # create the buffer
        ego_buffer = np.zeros(
            [ego.shape[0], self.HISTORY + self.FUTURE_LEN, ego.shape[1]]
        )
        neighbors_buffer = np.zeros(
            [
                neighbors.shape[0],
                self.HISTORY + self.FUTURE_LEN,
                neighbors.shape[1],
                neighbors.shape[2],
            ]
        )
        for frame_id in range(len(ego)):
            for time_id in range(self.HISTORY):
                if frame_id - time_id >= 0:
                    ego_buffer[frame_id, self.HISTORY - time_id - 1] = ego[
                        frame_id - time_id
                    ]
                    neighbors_buffer[frame_id, self.HISTORY - time_id - 1] = neighbors[
                        frame_id - time_id
                    ]
            ego_buffer[frame_id, self.HISTORY] = ego[frame_id]
            neighbors_buffer[frame_id, self.HISTORY] = neighbors[frame_id]
            for time_id in range(self.FUTURE_LEN):
                if frame_id + time_id < len(ego):
                    ego_buffer[frame_id, self.HISTORY + time_id - 1] = ego[
                        frame_id + time_id
                    ]
                    neighbors_buffer[frame_id, self.HISTORY + time_id - 1] = neighbors[
                        frame_id + time_id
                    ]
        return ego_buffer, np.transpose(neighbors_buffer, (0, 2, 1, 3))

    def transform_global2ego_frame(self, ego_buffer, neighbors_buffer, map, crosswalk):
        # rereference the buffer, all features excluding rss and features afterward
        for frame_id in range(len(ego_buffer)):
            ego_current = ego_buffer[frame_id, self.HISTORY - 1][..., agent_feat_id["x"]:agent_feat_id["yaw"]+1]
            neighbors_buffer[frame_id][..., :9] = self.global_to_egocentric(
                neighbors_buffer[frame_id][..., :9],
                ego_current
            )
            map[frame_id] = self.global_to_egocentric(
                map[frame_id], ego_current,
                is_map=True
            )
            # important to transform ego after others
            ego_buffer[frame_id][..., :9] = self.global_to_egocentric(
                ego_buffer[frame_id][..., :9], ego_current
            )
            crosswalk[frame_id] = self.global_to_egocentric(
                crosswalk[frame_id], ego_current,
                is_map=True
            )
        return ego_buffer, neighbors_buffer, map, crosswalk

    def create_buffer_and_transform_frame(self, ego, neighbors, map, crosswalk):
        ego_buffer, neighbors_buffer = self.create_buffer(ego, neighbors)
        ego_buffer, neighbors_buffer, map, crosswalk = self.transform_global2ego_frame(
            ego_buffer, neighbors_buffer, map, crosswalk
        )

        # pad with zeros to match neighbors features size
        padding_width = neighbors_buffer.shape[-1] - ego_buffer.shape[-1]
        ego_buffer = np.pad(
            ego_buffer,
            ((0, 0), (0, 0), (0, padding_width)),
            mode="constant",
            constant_values=0,
        )

        ground_truth = np.concatenate(
            (ego_buffer[:, None, ...], neighbors_buffer), axis=1
        )

        # return ego_buffer, neighbors_buffer, map, ground_truth
        return (
            ego_buffer[:, : self.HISTORY, :],
            neighbors_buffer[:, :, : self.HISTORY],
            map,
            ground_truth[
                :,
                :,
                self.HISTORY :,
            ],
            crosswalk
        )


# if __name__ == "__main__":
#     vbo = Vector_BEV_observer()
#     vbo.run_and_save_data(re_reference=False)
