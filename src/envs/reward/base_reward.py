import math
from typing import List

import carla


class Reward:
    """Base class for computing rewards in a CARLA environment."""
    def __init__(self, scene_duration: float, step_frequecny: float):
        self.episode_length = int(scene_duration / step_frequecny)
        self.time_penalty = -1.0
        self.velocity_penalty = 1.0
        self.goal_threshold = 0.5 # in meters

    def __call__(
        self,
        distance_to_goal: float,
        prev_distance: float,
        collision: bool,
        timestep: int,
        lane_invasions: List[carla.LaneInvasionEvent],
        ego_vehicle: carla.Vehicle,
        world_map: carla.Map
    ) -> float:
        """
        Compute the reward based on:
        - Progress towards the goal (+1 per meter)
        - Reaching the goal (+50)
        - Collision (-50)
        - Time penalty (-1 per step until goal is reached)
        - Lane invasion (-25 per illegal lane invasion)
        - Red light violation (-1 per speed unit)
        - Driving over the speed limit (-1 per speed unit over 30 km/h)
        - Driving under the speed limit without reason (-1 per speed unit under 25 km/h)
        - Driving on the sidewalks (-25)

        Args:
            distance_to_goal (float): Current distance to the goal.
            prev_distance (float): Previous distance to the goal.
            collision (bool): Whether a collision has occurred.
            timestep (int): Current timestep in the episode.
            lane_invasions (list of lane invasion events): one event for each lane invasion.
            ego_vehicle (carla.Vehicle): the ego vehicle.
            world_map (carla.Map): The world map.
        """
        reward = 0.0

        # Reduced progress-based reward
        if prev_distance is not None:
            reward += 1.0 * (prev_distance - distance_to_goal)

        # Check if goal is reached. Implicitly rewarded by terminating the episode.
        goal_reached = distance_to_goal < self.goal_threshold
        if goal_reached:
            reward += 50.0

        # Collision adds the remaining time penalty, since the episode will terminate
        if collision:
            reward -= 50.0
            reward += self.time_penalty * (self.episode_length - timestep)

        # Add penalty for illegal lane invasion.
        for lane_invasion in lane_invasions:
            for crossed_lane_marking in lane_invasion.crossed_lane_markings:
                if crossed_lane_marking.lane_change == carla.LaneChange.NONE:
                    reward -= 25

        # Add penalty if the agent is moving, while at a red traffic light.
        ego_velocity = ego_vehicle.get_velocity()
        ego_speed = math.sqrt(ego_velocity.x ** 2 + ego_velocity.y ** 2)
        blocked_at_red_light = False
        if ego_vehicle.is_at_traffic_light():
            traffic_light = ego_vehicle.get_traffic_light()
            if traffic_light.state == carla.TrafficLightState.Red:
                blocked_at_red_light = True
                reward -= self.velocity_penalty * ego_speed

        # Penalise the agent for driving too fast (over 30 km/h = 8.333 m/s).
        if 8.333 < ego_speed:
            reward -= self.velocity_penalty * (ego_speed - 8.333)

        # Penalise the agent for driving too slowly (below 25 km/h = 6.944 m/s).
        if not blocked_at_red_light and not goal_reached and ego_speed < 6.944:
            reward -= self.velocity_penalty * (6.944 - ego_speed)

        # Penalise the agent for driving on the sidewalks.
        current_waypoint = world_map.get_waypoint(ego_vehicle.get_location())
        if current_waypoint.lane_type == carla.LaneType.Sidewalk:
            reward -= 25

        return reward
