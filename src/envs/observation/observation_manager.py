import numpy as np
import torch
from gymnasium import spaces
from typing import Optional

from src.envs.observation.decision_traffic_rules.feature_indices import agent_feat_id
from src.envs.observation.decision_traffic_rules.traffic_sign_db import traffic_feat_idx
from src.envs.observation.vector_BEV_observer import Vector_BEV_observer


class ObservationManager:
    """Defines the observation space and handles observation creation and preprocessing."""
    def __init__(self, obs_keys: list[str], preprocess = False) -> None :
        """
        obs_keys: list of observations to include.
        """
        self.obs_keys = obs_keys
        self.preprocess_observations = preprocess
        self.preprocessor = Preprocessor()
        # Initialize BEV observer (your original code uses FUTURE_LEN=1)
        self.bev_info = Vector_BEV_observer(FUTURE_LEN=1)

        self.observation_space = self._build_space()

    def _build_space(self) -> spaces.Dict:
        obs = {}
        # --------------------------------------------------------
        # Hard-code your observation space to match the shapes you expect.
        # Example shapes (leading dimension 1 for ego, 5 for neighbors, etc.).
        # Adjust these as needed to match your real data.
        # --------------------------------------------------------
        for obs_name in self.obs_keys:
            if obs_name == "ego":
                if self.preprocess_observations:
                    obs["ego"] = spaces.Box(low=-1, high=1, shape=(1, self.bev_info.HISTORY, 7), dtype=np.float32)
                else:
                    obs["ego"] = spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.bev_info.HISTORY, 24), dtype=np.float32)
            elif obs_name == "neighbors":
                if self.preprocess_observations:
                    obs["neighbors"] = spaces.Box(low=-1, high=1, shape=(1, self.bev_info.MAX_NEIGHBORS, self.bev_info.HISTORY, 7), dtype=np.float32)
                else:
                    obs["neighbors"] = spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.bev_info.MAX_NEIGHBORS, self.bev_info.HISTORY, 24), dtype=np.float32)
            elif obs_name == "map":
                if self.preprocess_observations:
                    obs["map"] = spaces.Box(low=-1, high=1, shape=(1, self.bev_info.MAX_LANES, self.bev_info.MAX_LANE_LEN, 10), dtype=np.float32)
                else:
                    obs["map"] = spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.bev_info.MAX_LANES, self.bev_info.MAX_LANE_LEN, 46), dtype=np.float32)
            elif obs_name == "global_route":
                if self.preprocess_observations:
                    obs["global_route"] = spaces.Box(low=-1, high=1, shape=(self.bev_info.MAX_LANE_LEN, 3), dtype=np.float32)
                else:
                    obs["global_route"] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.bev_info.MAX_LANE_LEN, 3), dtype=np.float32)
            else:
                raise ValueError(f"Unknown observation key: {obs_name}")
        return spaces.Dict(obs)
    
    def reset(self):
        # reset observation history
        self.bev_info = Vector_BEV_observer(FUTURE_LEN=1)
    
    #TODO: types? global_route_ego_frame has different types in env
    def get_obs(self, world, global_route_ego_frame: Optional[np.ndarray | torch.Tensor]=None) -> dict:
        # Get new observation from BEV observer
        if self.bev_info.client_init(world=world):
            ego_hist, neighbor_hist, map_hist, crosswalk_hist, _ = self.bev_info.update(re_reference=False)
            ego_hist, neighbor_hist, map_hist, ground_truth, crosswalk_hist = \
                self.bev_info.create_buffer_and_transform_frame(np.array(ego_hist),
                                                                np.array(neighbor_hist),
                                                                np.array(map_hist),
                                                                np.array(crosswalk_hist))
            ego_obs = self.bev_info.carla_to_MAI_coordinates(data=ego_hist, is_map=False)
            neighbors_obs = self.bev_info.carla_to_MAI_coordinates(data=neighbor_hist, is_map=False)
            map_obs = self.bev_info.carla_to_MAI_coordinates(data=map_hist, is_map=True)
            if global_route_ego_frame is None:
                global_route_ego_frame = torch.zeros(size=(self.bev_info.MAX_LANE_LEN, 3))
            else:
                global_route_ego_frame = self.bev_info.carla_to_MAI_coordinates(data=global_route_ego_frame,
                                                                                    is_map=True)
            if len(ego_obs) > 1:
                ego_obs = ego_obs[-2:-1]
                neighbors_obs = neighbors_obs[-2:-1]
                map_obs = map_obs[-2:-1]

        else: #TODO: understand this line
            ego_obs, neighbors_obs, map_obs = None, None, None

        observation = {}
        for obs_name in self.obs_keys:
            if obs_name == "ego":
                observation["ego"] = ego_obs
            elif obs_name == "neighbors":
                observation["neighbors"] = neighbors_obs
            elif obs_name == "map":
                observation["map"] = map_obs
            elif obs_name == "global_route":
                observation["global_route"] = global_route_ego_frame
            else:
                raise ValueError(f"Unknown observation key: {obs_name}")

        if self.preprocess_observations:
            observation = self.preprocessor.preprocess_observation(observation)

        return observation


class Preprocessor:
    def __init__(self):
        # Add your initialization logic here
        self.ego_attr_keep = np.array([agent_feat_id["x"], agent_feat_id["y"], agent_feat_id['yaw'],
                                       agent_feat_id["vx"], agent_feat_id["vy"],
                                       agent_feat_id["length"], agent_feat_id["width"]])
        self.neighbor_attr_keep = np.array([agent_feat_id["x"], agent_feat_id["y"], agent_feat_id['yaw'],
                                            agent_feat_id["vx"], agent_feat_id["vy"],
                                            agent_feat_id["length"], agent_feat_id["width"], 
                                            agent_feat_id["class"]])
        self.map_attr_keep = np.array([traffic_feat_idx["cl_x"], traffic_feat_idx["cl_y"], traffic_feat_idx["cl_yaw"], 
                                       traffic_feat_idx['ll_x'], traffic_feat_idx["ll_y"], traffic_feat_idx["ll_yaw"],
                                       traffic_feat_idx['ll_x'], traffic_feat_idx["ll_y"], traffic_feat_idx["ll_yaw"],
                                       traffic_feat_idx["speed_limit"]])
        self.FOV = 50
        self.max_speed = 80 / 3.6  # Convert km/h to m/s
        self.R_min = -350
        self.R_max = 150

    def set_reward_range(self, r_min, r_max):
        self.R_min = r_min
        self.R_max = r_max
        
    def preprocess_observation(self, observation: dict) -> dict:
        # remove unnecessary attributes
        ego_data = observation['ego'][..., self.ego_attr_keep]
        neighbors_data = observation['neighbors'][..., self.ego_attr_keep] # keep the same attributes as ego
        map_data = observation['map'][..., self.map_attr_keep]

        #TODO: modularize
        processed_observation = {
            'ego': ego_data,
            'neighbors': neighbors_data,
            'map': map_data,
            'global_route': observation['global_route']
        }

        # processed_observation = self.normalize(processed_observation)

        return processed_observation

    def preprocess_action(self, action):
        # Add your action preprocessing logic here
        pass

    def preprocess_reward(self, reward):
        # Normalize the reward
        reward = 2 * (reward - self.R_min) / (self.R_max - self.R_min) - 1
        return reward

    def normalize(self, observation: dict) -> dict:
        for key in observation:
            # Normalize the dimensions of the observation
            if key == 'ego' or key == 'neighbors':
                observation[key][..., [0, 1, 5, 6]] /= self.FOV
                # Remove points outside the FOV
                observation[key][..., [0, 1, 5, 6]] = np.where(observation[key][..., [0, 1, 5, 6]] > self.FOV, 0, observation[key][..., [0, 1, 5, 6]])
                observation[key][..., 2] /= np.pi
                observation[key][..., [3, 4]] /= self.max_speed
                observation[key][..., [3, 4]] = (observation[key][..., [3, 4]] - 0.5) * 2
            elif key == 'map':
                observation[key][..., [0, 1, 3, 4, 6, 7]] /= self.FOV
                observation[key][..., [2, 5, 8]] /= np.pi
                observation[key][..., 9] /= self.max_speed
                observation[key][..., 9] = (observation[key][..., 9] - 0.5) * 2
            elif key == 'global_route':
                observation[key][..., [0, 1]] /= self.FOV
                observation[key][..., 2] /= np.pi
        return observation
