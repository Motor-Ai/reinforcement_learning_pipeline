import numpy as np
from typing import Any
from typing_extensions import TypedDict

from envs.observation.decision_traffic_rules.feature_indices import agent_feat_id
from envs.observation.decision_traffic_rules.traffic_sign_db import traffic_feat_idx


class Observation(TypedDict):  #TODO: set types for the keys
    ego: Any
    neighbors: Any
    map: Any
    global_route: Any

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
        
    def preprocess_observation(self, observation: Observation) -> Observation:
        self.observations_before_preprocessing = observation
        # remove unnecessary attributes
        ego_data = observation.get('ego')[..., self.ego_attr_keep]
        neighbors_data = observation.get('neighbors')[..., self.ego_attr_keep] # keep the same attributes as ego
        map_data = observation.get('map')[..., self.map_attr_keep]

        processed_observation: Observation = {
            'ego': ego_data,
            'neighbors': neighbors_data,
            'map': map_data,
            'global_route': observation.get('global_route')
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

    def normalize(self, observation: Observation) -> Observation:
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