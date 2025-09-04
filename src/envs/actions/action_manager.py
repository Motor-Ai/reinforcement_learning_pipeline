import numpy as np
import torch
from gymnasium import spaces
from typing import Optional

from src.envs.observation.decision_traffic_rules.feature_indices import agent_feat_id
from src.envs.observation.decision_traffic_rules.traffic_sign_db import traffic_feat_idx
from src.envs.observation.vector_BEV_observer import Vector_BEV_observer

from src.envs.actions.act_integration import rss_utils_poly_calc
import src.envs.actions.act_integration.spline as sp

from numpy.typing import NDArray
class ActionManager:
    """Defines the action space and handles action creation and preprocessing."""
    def __init__(self, action_fields: list[str], n_samples: int) -> None :
        """
        action_fields: list of actions to include.
        """
        self.action_fields = action_fields
        self.n_samples = n_samples
        self.action_space = spaces.Box(low=np.repeat(np.array([-1, -1, -1]), repeats=n_samples, axis=0), 
                                        high=np.repeat(np.array([-1, -1, -1]), repeats=n_samples, axis=0), 
                                        dtype=np.float32)

    # def _build_space(self) -> spaces.Dict:
    #     actions = {}
    #     # --------------------------------------------------------
    #     # Hard-code your observation space to match the shapes you expect.
    #     # Example shapes (leading dimension 1 for ego, 5 for neighbors, etc.).
    #     # Adjust these as needed to match your real data.
    #     # --------------------------------------------------------
    #     for action_name in self.action_fields:
    #         if action_name == "acc":
    #             actions["acc"] = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    #         elif action_name == "lat_shift":
    #             actions["lat_shift"] = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
    #         elif action_name == "manuver":
    #             actions["manuver"] = spaces.Discrete(3, start=-1)  # -1: keep lane, 0: left, 1: right
    #         else:
    #             raise ValueError(f"Unknown observation key: {action_name}")
    #     return spaces.Box(low=np.array([0, -1, -1]), 
    #                         high=np.array([2,  1,  1]), 
    #                         dtype=np.float32)

    def action_space_to_fp_params(self,action_space: NDArray[np.float64], ego_vel_lon, plan_time_range, plan_dt):
        """
        Converts the action_space dict into a list of parameter tuples for frenet path generation.

        Reference:
        ActionSpace.msg
            std_msgs/Header header
            float64[] acceleration # linear acceleration at each dt
            float64[] lateral_shift # meters with respect to each target lane 
            int8[] manuever # target lane that will be mapped to ["left" (-1), "current" (0), "right" (1)]
            float64[] plan_time_range # time horizon for the plan in seconds [time_min, time_max, time_step]
            float64 plan_time_dt # Time resolution (1/predictor fps) in seconds
            string rl_id # Current RL ID of the ego vehicle

        Parameters:
        action_space (matrix): Contains info in the order 'manuever', 'linear_acceleration', 'lateral_shift'.
        ego_vel_lon (float): ego current longitudinal speed.
        plan_time_range (list): [min_time, max_time, time_step] for candidate plan horizons (sec).
        plan_dt (float): path time step (sec).

        Returns:
        list: List of tuples (target_speed, target_d, plan_time, plan_dt) for each candidate combination.
        """
        fp_params = []
        action_space = action_space.reshape(self.n_samples,-1)
        t_min, t_max, t_step = plan_time_range

        for i in range(self.n_samples):
            lon_acc = action_space[i][1]
            target_d = action_space[i][2]
            # For each candidate, sweep over plan times in range
            plan_times = [round(t, 3) for t in np.arange(t_min, t_max + t_step, t_step)]
            for plan_time in plan_times:
                target_speed = ego_vel_lon + lon_acc * plan_time
                fp_params.append((target_speed, target_d, plan_time, plan_dt))
        return fp_params
    
    def compute_frenet_projection(self, ego_x, ego_y, centerline_x, centerline_y):
        """
        Computes the longitudinal (arc-length) and lateral (offset) distance
        of the ego agent from a lane's centerline.
        
        Returns:
            long_position (float): Distance along the centerline from the start.
            lateral_distance (float): Signed perpendicular distance to the centerline.
        """
        # Stack centerline points
        centerline = np.column_stack((centerline_x, centerline_y))
        ego_pos = np.array([ego_x, ego_y])

        # Compute distances to all centerline points
        dists = np.linalg.norm(centerline - ego_pos, axis=1)
        closest_idx = np.argmin(dists)

        # Clip index for neighbor points
        prev_idx = max(closest_idx - 1, 0)
        next_idx = min(closest_idx + 1, len(centerline_x) - 1)

        # Local centerline direction
        dir_vec = centerline[next_idx] - centerline[prev_idx]
        dir_vec /= np.linalg.norm(dir_vec) + 1e-8

        # Vector from centerline to ego
        to_ego_vec = ego_pos - centerline[closest_idx]

        # Longitudinal distance: sum of segment distances up to closest point
        long_position = sum(
            np.linalg.norm(centerline[i+1] - centerline[i])
            for i in range(closest_idx)
        )

        # Lateral distance: signed based on cross product direction
        lateral_distance = np.cross(dir_vec, to_ego_vec)

        return long_position, lateral_distance    # Clip index for neighbor points

    def preprocess_path(self, ref_x, ref_y, factor=2.0, max_angle_change=np.pi/4.0):
        """
        Preprocess path points:
        1. Insert extra points if spacing between consecutive points is too large.
        2. Remove points that create sharp heading transitions.
        
        Args:
            ref_x (np.array): X coordinates of path.
            ref_y (np.array): Y coordinates of path.
            factor (float): Threshold multiplier. If gap > factor * avg_gap, insert points.
            max_angle_change (float): Max allowed heading change (radians) between segments.
        
        Returns:
            (np.array, np.array): Processed ref_x, ref_y with adjusted points.
        """
        # Step 1: Fill large gaps
        dists = np.sqrt(np.diff(ref_x)**2 + np.diff(ref_y)**2)
        avg_dist = np.mean(dists)
        threshold = factor * avg_dist

        new_x, new_y = [ref_x[0]], [ref_y[0]]

        for i in range(1, len(ref_x)):
            dx = ref_x[i] - ref_x[i-1]
            dy = ref_y[i] - ref_y[i-1]
            dist = np.sqrt(dx**2 + dy**2)

            if dist > threshold:
                num_new = int(np.ceil(dist / avg_dist)) - 1
                for j in range(1, num_new + 1):
                    factor_j = j / (num_new + 1)
                    new_x.append(ref_x[i-1] + dx * factor_j)
                    new_y.append(ref_y[i-1] + dy * factor_j)

            new_x.append(ref_x[i])
            new_y.append(ref_y[i])

        new_x = np.array(new_x)
        new_y = np.array(new_y)

        # Step 2: Remove sharp heading transitions
        keep_idx = [0]  # always keep the first point

        for i in range(1, len(new_x)-1):
            # vectors before and after current point
            v1 = np.array([new_x[i] - new_x[i-1], new_y[i] - new_y[i-1]])
            v2 = np.array([new_x[i+1] - new_x[i], new_y[i+1] - new_y[i]])

            # normalize
            if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                continue
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)

            # angle between vectors
            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

            if angle < max_angle_change:
                keep_idx.append(i)

        keep_idx.append(len(new_x)-1)  # always keep last point

        return new_x[keep_idx], new_y[keep_idx]

    def get_path(self, action, ref_path, ego_state, plan_time_range, plan_dt):
        plan_time_range = [1,10,0.5] # [time_min, time_max, time_step]
        plan_dt = 0.1 # seconds

        ref_x, ref_y = ref_path[:,0], ref_path[:,1]

        # === Compute path length as sum of Euclidean distances between centerline points ===
        path_len = sum(
            np.linalg.norm(np.array([ref_x[i+1], ref_y[i+1]]) - np.array([ref_x[i], ref_y[i]]))
            for i in range(len(ref_x) - 1)
        )

        ref_x, ref_y = self.preprocess_path(ref_x, ref_y, factor=0.5)

        # === Generate a B-spline path with 0.1m resolution ===
        x_arr, y_arr, yaw_arr, _ = sp.calc_bspline_course_2(ref_x, ref_y, path_len, 0.1)
        print(f"Processed path to {len(x_arr)} points.")

        ego_vel_lon, ego_vel_lat = ego_state[-2], ego_state[-1]

        long_position, lateral_distance = self.compute_frenet_projection(ego_state[0], ego_state[1], x_arr, y_arr)
    
        fp_params = self.action_space_to_fp_params(action, ego_vel_lon, plan_time_range, plan_dt)
        # For each candidate, generate frenet path and pick the first valid (for demo, can expand to all candidates)
        for fp_param in fp_params:
            # Set up params for frenet path generation
            lon_params = [long_position, ego_vel_lon]  # Start position, velocity
            lat_params = [lateral_distance, ego_vel_lat, 0.0]
            target_speed, target_d, plan_time, plan_dt = fp_param

            # Overwrite target lateral offset and speed from ActionSpace
            lat_params[0] = target_d
            time_params = [plan_time, plan_dt]

            _, f_path = rss_utils_poly_calc.frenet_path_gen_for_decision_targets(
                lon_params, lat_params, time_params, target_speed,
                x_arr, y_arr, yaw_arr
            )

            path_x, path_y, path_yaw, path_vel, path_time = f_path.x, f_path.y, f_path.yaw, f_path.s_d, f_path.t
            # If you want all candidates, you can append to a list instead of overwrite

            break  # only use first candidate for now
        
        return path_x, path_y, path_yaw, path_vel, path_time
