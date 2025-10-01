import math
from functools import partial
from typing import Any

import carla


class RewardData:
    """
    A class that provide access to variables that can be used in the reward sub-function.
    """

    def __init__(self, env):
        self.env = env
        self.data = {}
        self.getters = {
            "prev_distance": self.get_previous_distance,
            "distance_to_goal": self.compute_distance_to_goal,
            "collision": self.get_collision_detected,
            "episode_length": self.get_episode_length,
            "timestep": self.get_timestep,
            "lane_invasions": self.get_lane_invasions,
            "ego_speed": self.get_ego_speed,
            "current_waypoint": self.get_current_waypoint,
            "ego_vehicle": self.get_ego_vehicle,
            "blocked_at_red_light": self.get_blocked_at_red_light,
            "every_timestep_penalty": self.get_every_timestep_penalty
        }

    def clear(self) -> None:
        self.data = {}

    def __getitem__(self, key: str) -> Any:
        if key not in self.data.keys():
            if key not in self.getters.keys():
                return None
            self.data[key] = self.getters[key]()
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def compute_distance_to_goal(self) -> float:
        return self.env.compute_distance_to_goal(self.env.ego_vehicle.get_location())

    def get_previous_distance(self):
        return self.env.prev_distance

    def get_collision_detected(self):
        return self.env.collision_detected

    def get_episode_length(self):
        return int(self.env.scene_duration / self.env.frequency)

    def get_timestep(self):
        return self.env.timestep

    def get_lane_invasions(self):
        return self.env.lane_invasions

    def get_ego_speed(self):
        ego_velocity = self.env.ego_vehicle.get_velocity()
        return math.sqrt(ego_velocity.x ** 2 + ego_velocity.y ** 2)

    def get_current_waypoint(self):
        return self.env.world.get_map().get_waypoint(self.env.ego_vehicle.get_location())

    def get_ego_vehicle(self):
        return self.env.ego_vehicle

    def get_blocked_at_red_light(self):
        if self.env.ego_vehicle.is_at_traffic_light():
            traffic_light = self.env.ego_vehicle.get_traffic_light()
            if traffic_light.state == carla.TrafficLightState.Red:
                return True
        return False

    @staticmethod
    def get_every_timestep_penalty():
        # By default, there is no penalty at each time step.
        return 0.0


class RewardFunction:
    """
    A class representing a reward function.
    """

    def __init__(self, env):
        """
        Create a reward function.
        :param env: the environment for which the reward function is created.
        """

        self.reward_functions = {
            "every_timestep_penalty": self.every_timestep_penalty,
            "goal_improvement": self.goal_improvement,
            "goal_reached": self.goal_reached,
            "collision": self.collision,
            "illegal_lane_invasions": self.illegal_lane_invasions,
            "red_light_violation": self.red_light_violation,
            "ego_is_too_fast": self.ego_is_too_fast,
            "ego_is_too_slow": self.ego_is_too_slow,
            "driving_on_sidewalks": self.driving_on_sidewalks,
        }
        self.reward_chain = []
        self.data = RewardData(env)

    def add(self, reward_name, **kwargs):
        """
        Add an element to the chain of reward functions.
        :param reward_name: name of the reward sub-function
        :param kwargs: keyword arguments passed to the reward function
        :return: self
        """
        function = partial(self.reward_functions[reward_name], **kwargs)
        self.reward_chain.append(function)
        return self

    def compute(self):
        """
        Compute the reward.
        :return: the reward
        """
        self.data.clear()
        reward = 0.0
        for element in self.reward_chain:
            reward += element(self.data)
        return reward

    @staticmethod
    def every_timestep_penalty(cache, penalty=1.0):
        """
        Apply a small penalty at each time step.
        :param cache: the cache allowing access to various data entries
        :param penalty: the penalty to apply at each time step
        :return: the reward
        """
        cache["every_timestep_penalty"] = penalty
        return -penalty

    @staticmethod
    def goal_improvement(cache, reward=1.0):
        """
        Add a reward when moving closer to the goal (+reward per meter).
        :param cache: the cache allowing access to various data entries
        :param reward: the reward to give for each meter the agent progressed towards the goal
        :return: the reward
        """

        # Check that all inputs are available.
        assert cache["distance_to_goal"] is not None, "No 'distance_to_goal' provided."

        # Compute reward based on the progress made by the agent.
        if cache["prev_distance"] is None:
            return 0.0
        return reward * (cache["prev_distance"] - cache["distance_to_goal"])

    @staticmethod
    def goal_reached(cache, reward=50, goal_threshold=0.5):
        """
        Add a reward when the agent reaches the goal location.
        :param cache: the cache allowing access to various data entries
        :param reward: the reward to apply when the ego vehicle reaches the goal
        :param goal_threshold: the radius in meters of the goal location
        :return: the reward
        """

        # Check that all inputs are available.
        assert cache["distance_to_goal"] is not None, "No 'distance_to_goal' provided."

        # Compute the reward.
        cache["goal_reached"] = cache["distance_to_goal"] < goal_threshold
        return reward if cache["goal_reached"] else 0.0

    @staticmethod
    def collision(cache, penalty=50, time_penalty=1.0):
        """
        Add a penalty for colliding with another object.
        IMPORTANT: Make sure to call this function after every_timestep_penalty.
        :param cache: the cache allowing access to various data entries
        :param penalty: the penalty to apply when colliding with another object
        :param time_penalty: the time penalty (-time_penalty per step until the goal is reached)
        :return: the reward
        """

        # Check that all inputs are available.
        assert cache["collision"] is not None, "No 'collision' provided."
        assert cache["episode_length"] is not None, "No 'episode_length' provided."
        assert cache["timestep"] is not None, "No 'timestep' provided."

        # Compute the collision reward.
        reward = 0.0
        if cache["collision"]:
            reward -= penalty

            # Ensure that the car does not learn to crash to avoid small penalty at each time step,
            reward -= cache["every_timestep_penalty"] * (cache["episode_length"] - cache["timestep"])

        return reward

    @staticmethod
    def illegal_lane_invasions(cache, penalty=25.0):
        """
        Add a penalty for each illegal lane invasion.
        :param cache: the cache allowing access to various data entries
        :param penalty: the penalty to apply when the ego vehicle is performing a lane invasion
        :return: the reward
        """

        # Check that all inputs are available.
        assert cache["lane_invasions"] is not None, "No 'lane_invasions' provided."

        # Compute a penalty for illegal lane invasions.
        reward = 0.0
        for lane_invasion in cache["lane_invasions"]:
            for crossed_lane_marking in lane_invasion.crossed_lane_markings:
                if crossed_lane_marking.lane_change == carla.LaneChange.NONE:
                    reward -= penalty
        return reward

    @staticmethod
    def red_light_violation(cache, penalty=1.0):
        """
        Add a penalty for driving through a red light (-penalty per speed unit over 0 km/h).
        :param cache: the cache allowing access to various data entries
        :param penalty: the penalty to apply when the ego vehicle is driving through a red light
        :return: the reward
        """

        # Check that all inputs are available.
        assert cache["ego_vehicle"] is not None, "No 'ego_vehicle' provided."
        assert cache["ego_speed"] is not None, "No 'ego_speed' provided."

        # Add penalty if the agent is moving, while at a red traffic light.
        reward = 0.0
        if cache["ego_vehicle"].is_at_traffic_light():
            traffic_light = cache["ego_vehicle"].get_traffic_light()
            if traffic_light.state == carla.TrafficLightState.Red:
                reward -= penalty * cache["ego_speed"]
        return reward

    @staticmethod
    def ego_is_too_fast(cache, penalty=1.0):
        """
        Add a penalty for driving over the speed limit (-penalty per speed unit over 30 km/h).
        :param cache: the cache allowing access to various data entries
        :param penalty: the penalty to apply when the ego vehicle is driving too fast
        :return: the reward
        """

        # Check that all inputs are available.
        assert cache["ego_speed"] is not None, "No 'ego_speed' provided."

        # Penalise the agent for driving too fast (over 30 km/h = 8.333 m/s).
        reward = 0.0
        if 8.333 < cache["ego_speed"]:
            reward -= penalty * (cache["ego_speed"] - 8.333)
        return reward

    @staticmethod
    def ego_is_too_slow(cache, penalty=1.0):
        """
        Add a penalty for driving under the speed limit without reason (-penalty per speed unit under 25 km/h).
        :param cache: the cache allowing access to various data entries
        :param penalty: the penalty to apply when the ego vehicle is driving too slow
        :return: the reward
        """

        # Check that all inputs are available.
        assert cache["ego_speed"] is not None, "No 'ego_speed' provided."

        # Penalise the agent for driving too slowly (below 25 km/h = 6.944 m/s).
        reward = 0.0
        if not cache["blocked_at_red_light"] and not cache["goal_reached"] and cache["ego_speed"] < 6.944:
            reward -= penalty * (6.944 - cache["ego_speed"])
        return reward

    @staticmethod
    def driving_on_sidewalks(cache, penalty=25.0):
        """
        Add a penalty for driving on the sidewalks.
        :param cache: the cache allowing access to various data entries
        :param penalty: the penalty to apply when the ego vehicle is driving on the sidewalks
        :return: the reward
        """

        # Check that all inputs are available.
        assert cache["current_waypoint"] is not None, "No 'current_waypoint' provided."

        # Penalise the agent for driving on the sidewalks.
        reward = 0.0
        if cache["current_waypoint"].lane_type == carla.LaneType.Sidewalk:
            reward -= penalty
        return reward
