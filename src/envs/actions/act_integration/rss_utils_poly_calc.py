"""
1. Takes the following inputs
     - lane info of all lanes (lane_id, center line x,y points, left boundary & right boundary x, y points), 
     - Neighbor obstacles (x,y,yaw, width, height, velocity, Class)
     - Ego vehicle (x,y,yaw,width, height, velocity)
     
Python functions to perform 
- Convert the lane info & neighbor and ego poses to a frenet coordinate frame (to nullify the curvature of the lane)
- From converted ego pose Identify which lane id the ego vehicle is present in
- From converted neighbor pose Identify which lane id the neighbor obstacle is in, and also check if the neighbor is in Ego Lane or Non-Ego Lane, and also calculate the directions of travel (same direction as ego or opposite directions as ego) of the neighbor.
- Derive longitudinal and lateral components of the velocity based on the yaw of ego and neighboring actors. 
- 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import src.envs.actions.act_integration.fp_helper as fp_helper
import os
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate


def fit_spline(centerline):
    """Fits a spline to the centerline points."""
    x, y = zip(*list(dict.fromkeys(centerline)))  # Remove duplicate points
    if len(x) < 2:
        raise ValueError("Insufficient unique points for spline fitting")
    
    assert all(np.isfinite(x)), "Non-finite values in x!"
    assert all(np.isfinite(y)), "Non-finite values in y!"
    
    try:
        tck, _ = splprep([x, y], s=0, k=min(3, len(x)-1))  # Adjust degree if necessary
    except Exception as e:
        raise RuntimeError(f"Spline fitting failed: {e}")
    
    return tck


def predict_future_state(x0, y0, yaw0, v0, a, t):
    """
    Predicts future position and yaw using basic longitudinal kinematics.
    
    Args:
        x0, y0: Initial position
        yaw0: Initial heading in radians
        v0: Initial velocity (m/s)
        a: Acceleration (m/s²)
        t: Time horizon (s)
    
    Returns:
        x, y, yaw: Predicted position and orientation
    """
    # Compute displacement along heading
    s = max(v0 * t + 0.5 * a * t**2, 0.0)
    
    # New position
    x = x0 + s * np.cos(yaw0)
    y = y0 + s * np.sin(yaw0)
    
    # Yaw remains unchanged (no turning modeled)
    yaw = yaw0
    
    return x, y, yaw


def frenet_path_gen_for_decision_targets(lon_params, lat_params, time_params, target_speed, tx, ty, tyaw):
    """
    Generate Frenet paths for decision targets.
    
    Returns:
        list: List of Frenet paths for decision targets.
    """
    # Extract current state variables from input parameters
    s0 = lon_params[0]  # current course position
    c_speed = lon_params[1]  # current speed [m/s]
    c_d = lat_params[0]  # current lateral position [m]
    c_d_d = lat_params[1]  # current lateral speed [m/s]
    c_d_dd = lat_params[2]  # current lateral acceleration [m/s]

    # Initialize the target speeds for Frenet paths (set to a constant value for now)
    fp_target_speeds = np.ones(1) * target_speed

    # Initialize lateral distances (set to zero for now)
    fp_lateral_dists = np.zeros(1)

    # Set planning time and time step for Frenet paths
    fp_plan_time = np.ones(1) * time_params[0]
    fp_plan_dt = np.ones(1) * time_params[1]


    # Combine target speeds, lateral distances, planning time, and time step into a single parameter array
    fp_params = np.transpose(
        [fp_target_speeds, fp_lateral_dists, fp_plan_time, fp_plan_dt]
    )
    fp_params = np.unique(fp_params, axis=0)  # Remove duplicate parameter sets

    # Calculate Frenet paths based on the given parameters
    fplist = fp_helper.calc_frenet_paths(
        c_speed, c_d, c_d_d, c_d_dd, s0, fp_params
    )

    # Convert the calculated Frenet paths to Cartesian coordinates (global frame)
    fplist = fp_helper.calc_global_paths(fplist, tx, ty, tyaw)            

    # Check each path for potential collisions with dynamic obstacles if any
    # fplist = fp_helper.check_collision_dynamic(
    #     fplist, dynamic_ob, self.ego_x, self.ego_y,
    # )

    # Filter out paths that exceed curvature limits or collide with obstacles
    # fplist = fp_helper.check_paths(
    #     self, fplist, self.frenet_path_time, self.frenet_path_time_tick
    # )

    # Find the path with the minimum cost from the remaining paths
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return fplist, best_path  # Return all feasible paths and the best path
    

def calculate_distances(x_arr, y_arr):
    # Ensure that x_arr and y_arr have the same length
    if len(x_arr) != len(y_arr):
        raise ValueError("x_arr and y_arr must have the same length")
    # Create a 2D array of points by combining x_arr and y_arr
    points = np.column_stack((x_arr, y_arr))
    # Compute differences between successive points
    diffs = np.diff(points, axis=0)
    # Compute Euclidean distances
    distances = np.linalg.norm(diffs, axis=1)
    return distances


def plot_vehicle(x, y, yaw, width, length, color):
        """Plots a vehicle with its bounding box and yaw arrow."""
        # Compute the four corners of the bounding box
        corners = np.array([
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2]  # Closing the box
        ])
        
        # Rotation matrix for yaw
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        
        rotated_corners = (rotation_matrix @ corners.T).T
        translated_corners = rotated_corners + np.array([x, y])

        # Plot bounding box
        plt.plot(translated_corners[:, 0], translated_corners[:, 1], color=color, linewidth=2)

        # Plot the yaw arrow
        arrow_dx = np.cos(yaw) * length * 0.5  # Scaled arrow length
        arrow_dy = np.sin(yaw) * length * 0.5
        plt.arrow(x, y, arrow_dx, arrow_dy, head_width=0.5, head_length=1, fc=color, ec=color)



def get_vehicle_corners(s, d, yaw, length, width):
    """Compute the four corners of a tilted vehicle in Frenet frame based on yaw angle."""
    half_length = length / 2
    half_width = width / 2

    # Compute rotation matrix for yaw angle
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # Define vehicle corners relative to center (before rotation)
    corners = [
        (+half_length, +half_width),  # Front Right
        (+half_length, -half_width),  # Front Left
        (-half_length, +half_width),  # Rear Right
        (-half_length, -half_width)   # Rear Left
    ]

    # Rotate and translate corners
    rotated_corners = [
        (s + x * cos_yaw - y * sin_yaw, d + x * sin_yaw + y * cos_yaw) for x, y in corners
    ]

    return rotated_corners


def rss_longitudinal_distance_same_dir(v_lon_rear, v_lon_front, rss_params):
    """
    Calculate the required safe longitudinal distance between a rear and a front vehicle 
    travelling in same direction according to RSS (Responsibility-Sensitive Safety) parameters.

    Parameters:
        v_lon_rear (float): Speed of the rear vehicle (m/s)
        v_lon_front (float): Speed of the front vehicle (m/s)
        rss_params (dict): Dictionary containing the following keys:
            - "reaction_time" (float): Reaction time of the rear vehicle (s)
            - "lon_accel_max" (float): Maximum longitudinal acceleration of the rear vehicle (m/s^2)
            - "lon_decel_min" (float): Minimum longitudinal deceleration of the rear vehicle (m/s^2)
            - "lon_decel_max" (float): Maximum longitudinal deceleration of the front vehicle (m/s^2)

    Returns:
        float: The minimum required safe longitudinal distance (m)
    """

    # Distance covered by the rear vehicle during the reaction time.
    d_reaction = rss_params["reaction_time"] * v_lon_rear

    # Additional distance assuming the rear vehicle accelerates at maximum rate during the reaction time.
    d_acceleration = 0.5 * rss_params["lon_accel_max"] * (rss_params["reaction_time"] ** 2)

    # Speed of the rear vehicle after reaction time (assuming max acceleration).
    v_lon_rear_after_reaction = v_lon_rear + rss_params["reaction_time"] * rss_params["lon_accel_max"]

    # Distance required for the rear vehicle to decelerate (after reaction) 
    # using at least the minimum deceleration.
    d_braking_rear = 0.5 * (v_lon_rear_after_reaction ** 2) / rss_params["lon_decel_min"]

    # Distance the front vehicle requires to decelerate to a stop 
    # using its maximum deceleration.
    d_braking_front = 0.5 * (v_lon_front ** 2) / rss_params["lon_decel_max"]

    # The minimum safe longitudinal distance for same direction vehicles.
    d_min = d_reaction + d_acceleration + d_braking_rear - d_braking_front

    return d_min


def rss_longitudinal_distance_opp_dir(v_ego_lon, v_opp_lon, rss_params):
    """
    Calculate the required safe longitudinal distance for vehicles traveling in opposite directions
    based on Responsibility-Sensitive Safety (RSS) parameters.

    Parameters:
        v_ego_lon (float): Ego vehicle longitudinal velocity (m/s)
        v_opp_lon (float): Opposing vehicle longitudinal velocity (m/s)
        rss_params (dict): Dictionary containing:
            - "reaction_time" (float): Reaction time (s)
            - "lon_accel_max" (float): Maximum longitudinal acceleration (m/s²)
            - "lon_decel_min_corr" (float): Corrected minimum longitudinal deceleration for ego (m/s²)
            - "lon_decel_min" (float): Minimum longitudinal deceleration for the opposing vehicle (m/s²)

    Returns:
        float: Minimum required safe longitudinal distance (m)
    """
    # Compute velocities after reaction time
    v_ego_lon_after = v_ego_lon + rss_params["reaction_time"] * rss_params["lon_accel_max"]
    v_opp_lon_after = abs(v_opp_lon) + rss_params["reaction_time"] * rss_params["lon_accel_max"]

    # Distance covered during reaction time
    d_ego_reaction = 0.5 * (v_ego_lon + v_ego_lon_after) * rss_params["reaction_time"]
    d_opp_reaction = 0.5 * (v_opp_lon + v_opp_lon_after) * rss_params["reaction_time"]

    # Additional braking distance after reaction time
    d_ego_decel = 0.5 * (v_ego_lon_after ** 2) / rss_params["lon_decel_min_corr"]
    d_opp_decel = 0.5 * (v_opp_lon_after ** 2) / rss_params["lon_decel_min"]

    # The minimum safe longitudinal distance for opposite direction vehicles.
    d_min = d_ego_reaction + d_ego_decel + d_opp_reaction + d_opp_decel

    return d_min


def rss_lateral_distance_left_right(left_vehicle_lat_vel, right_vehicle_lat_vel, rss_params):
    """
    Calculate the required safe lateral distance between a left vehicle and a right vehicle 
    according to RSS (Responsibility-Sensitive Safety) parameters, assuming the left vehicle 
    is on the left side relative to the right vehicle.
    
    Parameters:
        left_vehicle_lat_vel (float): Lateral velocity of the left vehicle (m/s)
        right_vehicle_lat_vel (float): Lateral velocity of the right vehicle (m/s)
        rss_params (dict): Dictionary containing:
            - "reaction_time" (float): Reaction time (s)
            - "lat_accel_max" (float): Maximum lateral acceleration (m/s^2)
            - "lat_decel_min_corr" (float): Corrected minimum lateral deceleration for the left vehicle (m/s^2)
            - "lat_decel_min" (float): Minimum lateral deceleration for the right vehicle (m/s^2)
            - "lat_buffer_dist" (float): Lateral buffer distance (m)
            
    Returns:
        float: Minimum required safe lateral distance (m)
    """
    # Compute lateral velocity after reaction time
    left_vehicle_lat_after = left_vehicle_lat_vel + rss_params["reaction_time"] * rss_params["lat_accel_max"]
    right_vehicle_lat_after = right_vehicle_lat_vel - rss_params["reaction_time"] * rss_params["lat_accel_max"]

    # Buffer distance
    d_buffer = rss_params["lat_buffer_dist"]

    # Distance covered by the left vehicle during reaction time
    d_left_reaction = 0.5 * (left_vehicle_lat_vel + left_vehicle_lat_after) * rss_params["reaction_time"]
    # Additional distance for the left vehicle to decelerate laterally after the reaction time
    d_left_decel = 0.5 * (left_vehicle_lat_after ** 2) / rss_params["lat_decel_min"]

    # Distance covered by the right vehicle during reaction time
    d_right_reaction = 0.5 * (right_vehicle_lat_vel + right_vehicle_lat_after) * rss_params["reaction_time"]
    # Additional distance for the right vehicle to decelerate laterally after the reaction time
    d_right_decel = 0.5 * (right_vehicle_lat_after ** 2) / rss_params["lat_decel_min"]

    # Total required safe lateral distance
    d_min = d_buffer + d_left_reaction + d_left_decel - (d_right_reaction - d_right_decel)

    return d_min


def create_vehicle_polygon(x, y, yaw, length, width,
                           rear_extension=0, front_extension=0,
                           left_extension=0, right_extension=0):
    """
    Create a rectangular polygon for a vehicle given position, yaw, length, and width.
    Applies extensions in the vehicle-local frame (before rotation), then rotates and positions.
    """

    # Compute total size
    total_length = length + front_extension + rear_extension
    total_width = width + left_extension + right_extension

    # Compute shift of the polygon center due to asymmetrical extensions
    x_offset = (front_extension - rear_extension)/2.0
    y_offset = (left_extension - right_extension)/2.0

    # Half-dimensions
    half_l = total_length / 2
    half_w = total_width / 2

    # Create the base rectangle centered at origin, with extensions applied
    base_poly = Polygon([
        (-half_l, -half_w),  # rear-right
        (-half_l, half_w),   # rear-left
        (half_l, half_w),    # front-left
        (half_l, -half_w)    # front-right
    ])

    # Apply center shift due to asymmetric extensions
    shifted = translate(base_poly, xoff=x_offset, yoff=y_offset)

    # Rotate around (0, 0) and then translate to (x, y)
    #rotated = rotate(shifted, yaw, use_radians=True)
    rotated = rotate(shifted, yaw, use_radians=True, origin=(0, 0))
    positioned = translate(rotated, xoff=x, yoff=y)

    return positioned


def ext_rss_calc_adjacent_behind_same_dir(
    rss_params,
    vel_comps_ego,
    vel_comps_neighbor,
    ego_dimen,
    neighbor_dimen,
    ego_pos,
    ego_yaw,
    neighbor_pos,
    neighbor_yaw,
    right_of_ego,
    path_x,
    path_y,
    path_yaw,
    path_vel,
    path_t
):    
    v_lon_rear = vel_comps_neighbor[0]
    v_lon_front = vel_comps_ego[0]

    if right_of_ego:
        v_lat_left = vel_comps_ego[1]
        v_lat_right = vel_comps_neighbor[1]
    else:
        v_lat_right = vel_comps_ego[1]
        v_lat_left = vel_comps_neighbor[1]

    d_lon_rss = rss_longitudinal_distance_same_dir(v_lon_rear, v_lon_front, rss_params)
    d_lat_rss = rss_lateral_distance_left_right(v_lat_left, v_lat_right, rss_params)

    ego_polys = []
    neighbor_polys = []
    risk_values = []

    for i in range(len(path_t)):
        t = path_t[i]
        x = path_x[i]
        y = path_y[i]
        yaw = path_yaw[i]

        ego_length, ego_width = ego_dimen
        neighbor_length, neighbor_width = neighbor_dimen

        # Predict future position of neighbor
        neighbor_x, neighbor_y, neighbor_yaw_pred = predict_future_state(
            neighbor_pos[0], neighbor_pos[1], neighbor_yaw,
            vel_comps_neighbor[0], rss_params["lon_accel_max"], t
        )

        # Create neighbor polygon
        neighbor_poly = create_vehicle_polygon(
            neighbor_x, neighbor_y, neighbor_yaw_pred,
            neighbor_length, neighbor_width
        )

        # RSS margins
        lat_safety = rss_params["lat_buffer_dist"]
        lon_safety = rss_params["lon_buffer_dist"]

        left_ext = d_lat_rss if not right_of_ego else lat_safety
        right_ext = d_lat_rss if right_of_ego else lat_safety
        rear_ext = d_lon_rss
        front_ext = lon_safety

        ego_poly = create_vehicle_polygon(
            x, y, yaw,
            ego_length, ego_width,
            rear_extension=rear_ext,
            front_extension=front_ext,
            left_extension=left_ext,
            right_extension=right_ext
        )



        # Check intersection and compute breach
        if ego_poly.intersects(neighbor_poly):
            intersection = ego_poly.intersection(neighbor_poly)
            breach_area = intersection.area
            ego_total_area = ego_poly.area
            normalized_risk = breach_area / neighbor_poly.area

            print(f"    t={t:.2f}s: RSS Breach Detected!")
            print(f"        ➤ Breach Area: {breach_area:.2f} m²")
            print(f"        ➤ Ego Safe Area: {ego_total_area:.2f} m²")
            print(f"        ➤ Normalized Risk: {normalized_risk:.3f}")
            print(f"        ➤ Dlon_rss: {d_lon_rss:.1f}")
            print(f"        ➤ Dlat_rss: {d_lat_rss:.1f}")

        else:
            normalized_risk = 0.0

        ego_polys.append(ego_poly)
        neighbor_polys.append(neighbor_poly)
        risk_values.append(normalized_risk)

    return ego_polys, neighbor_polys, risk_values




def ext_rss_calc_adjacent_ahead_same_dir(
    rss_params,
    vel_comps_ego,
    vel_comps_neighbor,
    ego_dimen,
    neighbor_dimen,
    ego_pos,
    ego_yaw,
    neighbor_pos,
    neighbor_yaw,
    right_of_ego,
    path_x,
    path_y,
    path_yaw,
    path_vel,
    path_t
):    
    v_lon_front = vel_comps_neighbor[0]
    v_lon_rear = vel_comps_ego[0]

    if right_of_ego:
        v_lat_left = vel_comps_ego[1]
        v_lat_right = vel_comps_neighbor[1]
    else:
        v_lat_right = vel_comps_ego[1]
        v_lat_left = vel_comps_neighbor[1]

    d_lon_rss = rss_longitudinal_distance_same_dir(v_lon_rear, v_lon_front, rss_params)
    d_lat_rss = rss_lateral_distance_left_right(v_lat_left, v_lat_right, rss_params)

    ego_polys = []
    neighbor_polys = []
    risk_values = []

    for i in range(len(path_t)):
        t = path_t[i]
        x = path_x[i]
        y = path_y[i]
        yaw = path_yaw[i]

        ego_length, ego_width = ego_dimen
        neighbor_length, neighbor_width = neighbor_dimen

        # Predict future position of neighbor
        neighbor_x, neighbor_y, neighbor_yaw_pred = predict_future_state(
            neighbor_pos[0], neighbor_pos[1], neighbor_yaw,
            vel_comps_neighbor[0], -rss_params["lon_decel_max"], t
        )

        # Create neighbor polygon
        neighbor_poly = create_vehicle_polygon(
            neighbor_x, neighbor_y, neighbor_yaw_pred,
            neighbor_length, neighbor_width
        )

        # RSS margins
        lat_safety = rss_params["lat_buffer_dist"]
        lon_safety = rss_params["lon_buffer_dist"]

        left_ext = d_lat_rss if not right_of_ego else lat_safety
        right_ext = d_lat_rss if right_of_ego else lat_safety
        front_ext = d_lon_rss
        rear_ext = lon_safety

        ego_poly = create_vehicle_polygon(
            x, y, yaw,
            ego_length, ego_width,
            rear_extension=rear_ext,
            front_extension=front_ext,
            left_extension=left_ext,
            right_extension=right_ext
        )

        # Check intersection and compute breach
        if ego_poly.intersects(neighbor_poly):
            intersection = ego_poly.intersection(neighbor_poly)
            breach_area = intersection.area
            ego_total_area = ego_poly.area
            normalized_risk = breach_area / neighbor_poly.area

            print(f"    t={t:.2f}s: RSS Breach Detected!")
            print(f"        ➤ Breach Area: {breach_area:.2f} m²")
            print(f"        ➤ Ego Safe Area: {ego_total_area:.2f} m²")
            print(f"        ➤ Normalized Risk: {normalized_risk:.3f}")
            print(f"        ➤ Dlon_rss: {d_lon_rss:.1f}")
            print(f"        ➤ Dlat_rss: {d_lat_rss:.1f}")

        else:
            normalized_risk = 0.0


        ego_polys.append(ego_poly)
        neighbor_polys.append(neighbor_poly)
        risk_values.append(normalized_risk)

    return ego_polys, neighbor_polys, risk_values


def ext_rss_calc_adjacent_ahead_opp_dir(
    rss_params,
    vel_comps_ego,
    vel_comps_neighbor,
    ego_dimen,
    neighbor_dimen,
    ego_pos,
    ego_yaw,
    neighbor_pos,
    neighbor_yaw,
    right_of_ego,
    path_x,
    path_y,
    path_yaw,
    path_vel,
    path_t
):    
    v_lon_front = vel_comps_neighbor[0]
    v_lon_rear = vel_comps_ego[0]

    if right_of_ego:
        v_lat_left = vel_comps_ego[1]
        v_lat_right = vel_comps_neighbor[1]
    else:
        v_lat_right = vel_comps_ego[1]
        v_lat_left = vel_comps_neighbor[1]

    d_lon_rss = rss_longitudinal_distance_opp_dir(v_lon_rear, v_lon_front, rss_params)
    d_lat_rss = rss_lateral_distance_left_right(v_lat_left, v_lat_right, rss_params)

    ego_polys = []
    neighbor_polys = []
    risk_values = []

    for i in range(len(path_t)):
        t = path_t[i]
        x = path_x[i]
        y = path_y[i]
        yaw = path_yaw[i]

        ego_length, ego_width = ego_dimen
        neighbor_length, neighbor_width = neighbor_dimen

        # Predict future position of neighbor
        neighbor_x, neighbor_y, neighbor_yaw_pred = predict_future_state(
            neighbor_pos[0], neighbor_pos[1], neighbor_yaw,
            vel_comps_neighbor[0], rss_params["lon_accel_max"], t
        )

        # Create neighbor polygon
        neighbor_poly = create_vehicle_polygon(
            neighbor_x, neighbor_y, neighbor_yaw_pred,
            neighbor_length, neighbor_width
        )

        # RSS margins
        lat_safety = rss_params["lat_buffer_dist"]
        lon_safety = rss_params["lon_buffer_dist"]

        left_ext = d_lat_rss if not right_of_ego else lat_safety
        right_ext = d_lat_rss if right_of_ego else lat_safety
        front_ext = d_lon_rss
        rear_ext = lon_safety

        ego_poly = create_vehicle_polygon(
            x, y, yaw,
            ego_length, ego_width,
            rear_extension=rear_ext,
            front_extension=front_ext,
            left_extension=left_ext,
            right_extension=right_ext
        )

        # Check intersection and compute breach
        if ego_poly.intersects(neighbor_poly):
            intersection = ego_poly.intersection(neighbor_poly)
            breach_area = intersection.area
            ego_total_area = ego_poly.area
            normalized_risk = breach_area / neighbor_poly.area

            print(f"    t={t:.2f}s: RSS Breach Detected!")
            print(f"        ➤ Breach Area: {breach_area:.2f} m²")
            print(f"        ➤ Ego Safe Area: {ego_total_area:.2f} m²")
            print(f"        ➤ Normalized Risk: {normalized_risk:.3f}")
            print(f"        ➤ Dlon_rss: {d_lon_rss:.1f}")
            print(f"        ➤ Dlat_rss: {d_lat_rss:.1f}")

        else:
            normalized_risk = 0.0

        ego_polys.append(ego_poly)
        neighbor_polys.append(neighbor_poly)
        risk_values.append(normalized_risk)

    return ego_polys, neighbor_polys, risk_values



def ext_rss_calc_same_ahead_same_dir(
    rss_params,
    vel_comps_ego,
    vel_comps_neighbor,
    ego_dimen,
    neighbor_dimen,
    ego_pos,
    ego_yaw,
    neighbor_pos,
    neighbor_yaw,
    right_of_ego,
    path_x,
    path_y,
    path_yaw,
    path_vel,
    path_t
):
    v_lon_front = vel_comps_neighbor[0]
    v_lon_rear = vel_comps_ego[0]

    if right_of_ego:
        v_lat_left = vel_comps_ego[1]
        v_lat_right = vel_comps_neighbor[1]
    else:
        v_lat_right = vel_comps_ego[1]
        v_lat_left = vel_comps_neighbor[1]

    d_lon_rss = rss_longitudinal_distance_same_dir(v_lon_rear, v_lon_front, rss_params)
    d_lat_rss = rss_lateral_distance_left_right(v_lat_left, v_lat_right, rss_params)

    ego_polys = []
    neighbor_polys = []
    risk_values = []

    for i in range(len(path_t)):
        t = path_t[i]
        x = path_x[i]
        y = path_y[i]
        yaw = path_yaw[i]

        ego_length, ego_width = ego_dimen
        neighbor_length, neighbor_width = neighbor_dimen

        # Predict future position of neighbor
        neighbor_x, neighbor_y, neighbor_yaw_pred = predict_future_state(
            neighbor_pos[0], neighbor_pos[1], neighbor_yaw,
            vel_comps_neighbor[0], -rss_params["lon_decel_max"], t
        )

        # Create neighbor polygon
        neighbor_poly = create_vehicle_polygon(
            neighbor_x, neighbor_y, neighbor_yaw_pred,
            neighbor_length, neighbor_width
        )

        # RSS margins
        lat_safety = rss_params["lat_buffer_dist"]
        lon_safety = rss_params["lon_buffer_dist"]

        left_ext = d_lat_rss if not right_of_ego else lat_safety
        right_ext = d_lat_rss if right_of_ego else lat_safety
        front_ext = d_lon_rss
        rear_ext = lon_safety

        ego_poly = create_vehicle_polygon(
            x, y, yaw,
            ego_length, ego_width,
            rear_extension=rear_ext,
            front_extension=front_ext,
            left_extension=left_ext,
            right_extension=right_ext
        )        

        if ego_poly.intersects(neighbor_poly):
            intersection = ego_poly.intersection(neighbor_poly)
            breach_area = intersection.area
            normalized_risk = breach_area / neighbor_poly.area

            print(f"    t={t:.2f}s: RSS Breach Detected!")
            print(f"        ➤ Breach Area: {breach_area:.2f} m²")
            print(f"        ➤ Normalized Risk: {normalized_risk:.3f}")
            print(f"        ➤ Dlon_rss: {d_lon_rss:.1f}, Dlat_rss: {d_lat_rss:.1f}")
        else:
            normalized_risk = 0.0

        ego_polys.append(ego_poly)
        neighbor_polys.append(neighbor_poly)
        risk_values.append(normalized_risk)

    return ego_polys, neighbor_polys, risk_values



def ext_rss_calc_same_ahead_opp_dir(
    rss_params,
    vel_comps_ego,
    vel_comps_neighbor,
    ego_dimen,
    neighbor_dimen,
    ego_pos,
    ego_yaw,
    neighbor_pos,
    neighbor_yaw,
    right_of_ego,
    path_x,
    path_y,
    path_yaw,
    path_vel,
    path_t
):    
    v_lon_front = vel_comps_neighbor[0]
    v_lon_rear = vel_comps_ego[0]

    if right_of_ego:
        v_lat_left = vel_comps_ego[1]
        v_lat_right = vel_comps_neighbor[1]
    else:
        v_lat_right = vel_comps_ego[1]
        v_lat_left = vel_comps_neighbor[1]

    d_lon_rss = rss_longitudinal_distance_opp_dir(v_lon_rear, v_lon_front, rss_params)
    d_lat_rss = rss_lateral_distance_left_right(v_lat_left, v_lat_right, rss_params)

    ego_polys = []
    neighbor_polys = []
    risk_values = []

    for i in range(len(path_t)):
        t = path_t[i]
        x = path_x[i]
        y = path_y[i]
        yaw = path_yaw[i]

        ego_length, ego_width = ego_dimen
        neighbor_length, neighbor_width = neighbor_dimen

        # Predict future position of neighbor
        neighbor_x, neighbor_y, neighbor_yaw_pred = predict_future_state(
            neighbor_pos[0], neighbor_pos[1], neighbor_yaw,
            vel_comps_neighbor[0], rss_params["lon_accel_max"], t
        )

        # Create neighbor polygon
        neighbor_poly = create_vehicle_polygon(
            neighbor_x, neighbor_y, neighbor_yaw_pred,
            neighbor_length, neighbor_width
        )

        # RSS margins
        lat_safety = rss_params["lat_buffer_dist"]
        lon_safety = rss_params["lon_buffer_dist"]

        left_ext = d_lat_rss if not right_of_ego else lat_safety
        right_ext = d_lat_rss if right_of_ego else lat_safety
        front_ext = d_lon_rss
        rear_ext = lon_safety

        ego_poly = create_vehicle_polygon(
            x, y, yaw,
            ego_length, ego_width,
            rear_extension=rear_ext,
            front_extension=front_ext,
            left_extension=left_ext,
            right_extension=right_ext
        )

        # Check intersection and compute breach
        if ego_poly.intersects(neighbor_poly):
            intersection = ego_poly.intersection(neighbor_poly)
            breach_area = intersection.area
            ego_total_area = ego_poly.area
            normalized_risk = breach_area / neighbor_poly.area

            print(f"    t={t:.2f}s: RSS Breach Detected!")
            print(f"        ➤ Breach Area: {breach_area:.2f} m²")
            print(f"        ➤ Ego Safe Area: {ego_total_area:.2f} m²")
            print(f"        ➤ Normalized Risk: {normalized_risk:.3f}")
            print(f"        ➤ Dlon_rss: {d_lon_rss:.1f}")
            print(f"        ➤ Dlat_rss: {d_lat_rss:.1f}")
        else:
            normalized_risk = 0.0

        ego_polys.append(ego_poly)
        neighbor_polys.append(neighbor_poly)
        risk_values.append(normalized_risk)

    return ego_polys, neighbor_polys, risk_values

"""

Extended RSS functions

"""

def check_rss_for_frenet_leader_following_trajectory(frenet_path, Xf, Vf, rss_params):
    """
    Check if a Frenet trajectory is RSS safe relative to the lead vehicle.
    
    Args:
        frenet_path (FrenetPath): The ego vehicle's planned trajectory
        Xf (float): Lead vehicle initial longitudinal position
        Vf (float): Lead vehicle velocity
        rss_params (dict): RSS parameters including reaction time and deceleration assumptions
        
    Returns:
        bool: True if the entire trajectory is RSS safe, False otherwise
    """
    if len(frenet_path.x) == 0 or len(frenet_path.s_d) == 0:
        raise ValueError("Frenet path data is incomplete or empty.")

    # Compute average velocity along the trajectory
    avg_velocity = sum(frenet_path.s_d) / len(frenet_path.s_d)
    if avg_velocity <= 0:
        raise ValueError("Average velocity must be positive.")

    # Ego initial position (longitudinal)
    Xe = frenet_path.x[0]

    # Ego average velocity
    Ve = avg_velocity

    # Trajectory completion time
    rho_m = frenet_path.t[-1]

    # Calculate safe following distance using the provided function
    Df_long_rss = rss_longitudinal_distance_same_dir(Ve, Vf, rss_params)

    # Calculate the left side and right side of the inequality
    """CLASS BASED RSS PARAMS CHOICE EG: CAR, BICYCLE, TRUCK"""
    lhs = Xe + (Ve * rho_m)
    rhs = Xf + (Vf * rho_m) - Df_long_rss - (0.5 * rss_params["lon_decel_max"] * (rho_m ** 2))

    print(f"LHS (Ego): {lhs}")
    print(f"RHS (Lead): {rhs}")

    is_safe = (lhs <= rhs)

    return is_safe


def check_rss_for_pedestrian_crossing(frenet_path, crossing_zone_boundary_start, rss_params):
    """
    Check if a Frenet trajectory is RSS safe relative to a pedestrian crossing.

    RSS Equation:
    Xe + Ve * rho_m <= Xstop

    Args:
        frenet_path (FrenetPath): The ego vehicle's planned trajectory
        crossingcrossing_zone_boundary_start (float): Start position of the pedestrian crossing zone (m)
        rss_params (dict): RSS parameters including 'Dsafe_pedestrian' (pedestrian safe distance)

    Returns:
        bool: True if the entire trajectory is RSS safe for pedestrian crossing, False otherwise
    """
    if len(frenet_path.x) < 2 or len(frenet_path.s_d) < 2:
        raise ValueError("Frenet path data is incomplete or empty.")

    # Compute average velocity along the trajectory
    avg_velocity = sum(frenet_path.s_d) / len(frenet_path.s_d)
    if avg_velocity <= 0:
        raise ValueError("Average velocity must be positive.")

    # Ego initial position (longitudinal)
    Xe = frenet_path.x[0]

    # Ego average velocity
    Ve = avg_velocity

    # Planning horizon (total time of trajectory)
    rho_m = frenet_path.t[-1]

    # RSS pedestrian safe stop position considering the buffer
    Xstop = crossing_zone_boundary_start - rss_params["Dsafe_pedestrian"]

    # Predicted position of ego after rho_m time
    future_position = Xe + Ve * rho_m

    # Check RSS condition and comfortable braking limit
    is_safe = (future_position <= Xstop)

    print(f"[Pedestrian RSS Check] FuturePos: {future_position:.2f}, Xstop: {Xstop:.2f}, Safe: {is_safe}")
    return is_safe


def check_rss_for_parking_exit_cut_in(frenet_path, parking_exit_point, rss_params):
    """
    Check if a Frenet trajectory is RSS safe relative to a vehicle cutting in from a parking exit.

    RSS Equation:
    Xe + Ve * rho_m <= Xstop

    Args:
        frenet_path (FrenetPath): The ego vehicle's planned trajectory
        parking_exit_point (float): The longitudinal position where the other vehicle is expected to cut in (m)
        rss_params (dict): RSS parameters including 'Dsafe_parking' (safety distance buffer before parking exit)

    Returns:
        bool: True if the trajectory is RSS safe against parking exit cut-in, False otherwise
    """
    if len(frenet_path.x) == 0 or len(frenet_path.s_d) == 0 or len(frenet_path.t) == 0:
        raise ValueError("Frenet path data is incomplete or empty.")

    # Compute average velocity along the trajectory
    avg_velocity = sum(frenet_path.s_d) / len(frenet_path.s_d)
    if avg_velocity <= 0:
        raise ValueError("Average velocity must be positive.")

    # Ego initial position (longitudinal)
    Xe = frenet_path.x[0]

    # Ego average velocity
    Ve = avg_velocity

    # Planning horizon (total time of trajectory)
    rho_m = frenet_path.t[-1]

    # Define the stop position to safely yield to parking exit vehicle
    Xstop = parking_exit_point - rss_params["Dsafe_parking_exit"]

    # Predict ego's future position after rho_m time
    future_position = Xe + Ve * rho_m

    # Check RSS condition: ensure ego vehicle can yield if needed
    is_safe = (future_position <= Xstop)

    print(f"[Parking Exit RSS Check] FuturePos: {future_position:.2f}, Xstop: {Xstop:.2f}, Safe: {is_safe}")
    return is_safe


def check_rss_for_cut_in_from_adjacent_vehicle(frenet_path, Xo, Vo, rss_params):
    """
    Check if the Frenet trajectory is RSS safe against a detected cut-in from another vehicle.

    RSS Equation:
    Xe + Ve * rho_m <= Xo + Vo * rho_m + 0.5 * assumed_accel_other_vehicle * (rho_m)^2 - Df_long_rss
    Ensuring comfortable braking: required_decel <= a_min_brake after reaction time.

    Args:
        frenet_path (FrenetPath): Ego vehicle's planned trajectory
        other_vehicle (dict): Contains 'position' (Xo), 'velocity' (Vo)
        rss_params (dict): RSS parameters including reaction time, deceleration limits, and braking comfort

    Returns:
        bool: True if the trajectory is RSS safe from cut-in, False otherwise
    """

    if len(frenet_path.x) == 0 or len(frenet_path.s_d) == 0 or len(frenet_path.t) == 0:
        raise ValueError("Frenet path data is incomplete or empty.")

    # Compute average velocity along the trajectory
    avg_velocity = sum(frenet_path.s_d) / len(frenet_path.s_d)
    if avg_velocity <= 0:
        raise ValueError("Average velocity must be positive.")

    # Ego initial position and velocity
    Xe = frenet_path.x[0]
    Ve = avg_velocity
    rho_m = frenet_path.t[-1]  # Planning horizon

    # Compute safe following distance based on RSS rules
    Df_long_rss = rss_longitudinal_distance_same_dir(Ve, Vo, rss_params)

    # Calculate predicted ego position and other vehicle's potential position with acceleration
    ego_future_pos = Xe + Ve * rho_m
    other_future_pos = Xo + Vo * rho_m + 0.5 * rss_params["lon_accel_max"] * (rho_m ** 2)

    # RSS cut-in safety check
    is_safe = ego_future_pos <= other_future_pos - Df_long_rss

    print(f"[Cut-In RSS Check] EgoFuturePos: {ego_future_pos:.2f}, OtherFuturePos: {other_future_pos:.2f}, Safe: {is_safe}")

    return is_safe


def check_rss_for_same_direction_lane_change(frenet_path, case, X_adj_rear, V_adj_rear, X_front, V_front, X_adj_front, V_adj_front, rss_params):
    """
    Check RSS safety during a lane change where both current and adjacent lanes are in the same direction.

    Args:
        frenet_path (FrenetPath): Ego vehicle's planned trajectory.
        case (int): 1 for no vehicle ahead (Case 1), 2 for vehicle present in the ego lane (Case 2).
        adjacent_vehicle (dict): {'X1': float, 'V1': float, 'Acc_V1_accel': float, 'Df_long_rss_1': float}
        front_vehicle (dict): {'X2': float, 'V2': float, 'Acc_V2_brake': float, 'Df_long_rss_2': float}
        rss_params (dict): RSS parameters.

    Returns:
        bool: True if RSS safe for the lane change, False otherwise.
    """
    if len(frenet_path.x) == 0 or len(frenet_path.s_d) == 0 or len(frenet_path.t) == 0:
        raise ValueError("Frenet path data is incomplete or empty.")

    # Compute average velocity along the trajectory
    Ve = sum(frenet_path.s_d) / len(frenet_path.s_d)
    if Ve <= 0:
        raise ValueError("Average velocity must be positive.")

    # Ego initial position
    Xe = frenet_path.x[0]
    # Planning horizon (total time of trajectory)
    rho_m = frenet_path.t[-1]

    # Initialize check
    is_safe = False

    if case == 0:
        # Case 0: Only Vehicle in front in the ego lane, check with adjacent lane vehicle behind ego (X1, V1)
        # Same lane (front) check
        X2 = X_front
        V2 = V_front
        Acc_V2_brake = rss_params["lon_decel_max"]
        Df_long_rss_2 = rss_longitudinal_distance_same_dir(Ve, V2, rss_params)

        ego_predicted_self = Xe + Ve * (rho_m/2.0)
        front_predicted = X2 + V2 * (rho_m/2.0) - (Acc_V2_brake / 8) * (rho_m ** 2) - Df_long_rss_2

        is_safe = (ego_predicted_self <= front_predicted)

        print(f"[Lane Change Case 0] EgoPred Self: {ego_predicted_self:.2f}, FrontPred: {front_predicted:.2f}")

    elif case == 1:
        # Case 1: No vehicle in front in the ego lane, check with adjacent lane vehicle behind ego (X1, V1)
        X1 = X_adj_rear
        V1 = V_adj_rear
        Acc_V1_accel = rss_params["lon_accel_max"]
        Df_long_rss_1 = rss_longitudinal_distance_same_dir(V1, Ve, rss_params)

        ego_predicted = Xe + Ve * (rho_m/2.0)
        adjacent_predicted = X1 + V1 * (rho_m/2.0) + (Acc_V1_accel / 8) * (rho_m ** 2) + Df_long_rss_1

        is_safe = (ego_predicted >= adjacent_predicted)

        print(f"[Lane Change Case 1] EgoPred: {ego_predicted:.2f}, AdjPred: {adjacent_predicted:.2f}, Safe: {is_safe}")

    elif case == 2:
        # Case 2: No vehicle in front in the ego lane, check with adjacent lane vehicle (X1, V1), Vehicle in front in adjacent lane (X2, V2)
        X1 = X_adj_rear
        V1 = V_adj_rear
        Acc_V1_accel = rss_params["lon_accel_max"]
        Df_long_rss_1 = rss_longitudinal_distance_same_dir(V1, Ve, rss_params)

        X2 = X_adj_front
        V2 = V_adj_front
        Acc_V2_brake = rss_params["lon_decel_max"]
        Df_long_rss_2 = rss_longitudinal_distance_same_dir(Ve, V2, rss_params)

        ego_predicted_halftime = Xe + Ve * (rho_m/2.0)
        adjacent_predicted_halftime = X1 + V1 * (rho_m/2.0) + (Acc_V1_accel / 8) * (rho_m ** 2) + Df_long_rss_1

        ego_predicted = Xe + Ve * rho_m - Df_long_rss_2
        adjacent_predicted = X2 + V2 * rho_m - (Acc_V2_brake / 2) * (rho_m ** 2)

        is_safe = (ego_predicted_halftime >= adjacent_predicted_halftime) and (ego_predicted <= adjacent_predicted)

        print(f"[Lane Change Case 2] EgoPred Half: {ego_predicted_halftime:.2f}, AdjPred Half: {adjacent_predicted_halftime:.2f}")
        print(f"[Lane Change Case 2] EgoPred: {ego_predicted:.2f}, AdjPred: {adjacent_predicted:.2f}, Safe: {is_safe}")

    elif case == 3:
        # Case 3: Vehicle present in front and adjacent lane only in rear
        # Adjacent lane check
        X1 = X_adj_rear
        V1 = V_adj_rear
        Acc_V1_accel = rss_params["lon_accel_max"]
        Df_long_rss_1 = rss_longitudinal_distance_same_dir(V1, Ve, rss_params)

        ego_predicted_adj = Xe + Ve * (rho_m/2.0)
        adjacent_predicted = X1 + V1 * (rho_m/2.0) + (Acc_V1_accel / 8) * (rho_m ** 2) + Df_long_rss_1

        # Same lane (front) check
        X2 = X_front
        V2 = V_front
        Acc_V2_brake = rss_params["lon_decel_max"]
        Df_long_rss_2 = rss_longitudinal_distance_same_dir(Ve, V2, rss_params)

        ego_predicted_self = Xe + Ve * (rho_m/2.0)
        front_predicted = X2 + V2 * (rho_m/2.0) - (Acc_V2_brake / 8) * (rho_m ** 2) - Df_long_rss_2

        is_safe = (ego_predicted_adj >= adjacent_predicted) and (ego_predicted_self <= front_predicted)

        print(f"[Lane Change Case 3] EgoPred Adj: {ego_predicted_adj:.2f}, AdjPred: {adjacent_predicted:.2f}")
        print(f"[Lane Change Case 3] EgoPred Self: {ego_predicted_self:.2f}, FrontPred: {front_predicted:.2f}")
        print(f"[Lane Change Case 3] Safe: {is_safe}")

    elif case == 4:
        # Case 2: Vehicle present in front and adjacent lane in front and rear
        # Adjacent lane checkö
        X1 = X_adj_rear
        V1 = V_adj_rear
        Acc_V1_accel = rss_params["lon_accel_max"]
        Df_long_rss_1 = rss_longitudinal_distance_same_dir(V1, Ve, rss_params)

        ego_predicted_adj = Xe + Ve * (rho_m/2.0) - Df_long_rss_1
        adjacent_predicted = X1 + V1 * (rho_m/2.0) + (Acc_V1_accel / 8) * (rho_m ** 2)

        # Same lane (front) check
        X2 = X_front
        V2 = V_front
        Acc_V2_brake = rss_params["lon_decel_max"]
        Df_long_rss_2 = rss_longitudinal_distance_same_dir(Ve, V2, rss_params)

        ego_predicted_self = Xe + Ve * (rho_m/2.0)
        front_predicted = X2 + V2 * (rho_m/2.0) - (Acc_V2_brake / 8) * (rho_m ** 2) - Df_long_rss_2

        X3 = X_adj_front
        V3 = V_adj_front
        Acc_V3_brake = rss_params["lon_decel_max"]
        Df_long_rss_3 = rss_longitudinal_distance_same_dir(Ve, V3, rss_params)

        ego_predicted_end = Xe + Ve * rho_m - Df_long_rss_2
        adjacent_predicted_front = X3 + V3 * rho_m - (Acc_V3_brake / 2) * (rho_m ** 2)

        is_safe = (ego_predicted_adj >= adjacent_predicted) and (ego_predicted_self <= front_predicted) and (ego_predicted_end <= adjacent_predicted_front)

        print(f"[Lane Change Case 3] EgoPred Adj: {ego_predicted_adj:.2f}, AdjPred: {adjacent_predicted:.2f}")
        print(f"[Lane Change Case 3] EgoPred Self: {ego_predicted_self:.2f}, FrontPred: {front_predicted:.2f}")
        print(f"[Lane Change Case 3] EgoPred End: {ego_predicted_end:.2f}, AdjPred Front: {adjacent_predicted_front:.2f}")
        print(f"[Lane Change Case 3] Safe: {is_safe}")

    else:
        raise ValueError("Invalid case. Use 1 (no front vehicle) or 2 (front vehicle present).")

    return is_safe


def check_rss_for_opposite_lane_change(frenet_path, Xopp, Vopp, rss_params):
    """
    Check RSS safety for a lane change into an adjacent lane with opposite direction traffic.

    RSS Equation (Opposite Lane):
    Xe + Ve * ρm <= Xopp - |Vopp| * ρm - |(Acc_V1_accel / 2) * (ρm)^2| - Df_long_rss_opp

    Args:
        frenet_path (FrenetPath): Ego vehicle's planned trajectory.
        opposite_vehicle (dict): {'Xopp': float, 'Vopp': float, 'Acc_V1_accel': float, 'Df_long_rss_opp': float}
        rss_params (dict): RSS-related parameters if needed for future extensions.

    Returns:
        bool: True if RSS safe for the lane change into opposite direction lane, False otherwise.
    """
    if not frenet_path.x or not frenet_path.s_d or not frenet_path.t:
        raise ValueError("Frenet path data is incomplete or empty.")

    # Compute average velocity along the trajectory
    Ve = sum(frenet_path.s_d) / len(frenet_path.s_d)
    if Ve <= 0:
        raise ValueError("Average velocity must be positive.")

    # Ego initial position
    Xe = frenet_path.x[0]
    # Planning horizon (total time of trajectory)
    rho_m = frenet_path.t[-1]

    # Opposite vehicle parameters
    Acc_V1_accel = rss_params["lon_accel_max"]
    Df_long_rss_opp = rss_longitudinal_distance_opp_dir(Ve, Vopp, rss_params)

    # Ego predicted position after rho_m
    ego_predicted = Xe + Ve * rho_m

    # Opposite vehicle predicted position after rho_m (approaching AV)
    opp_predicted = Xopp - abs(Vopp) * rho_m - abs((Acc_V1_accel / 2) * (rho_m ** 2)) - Df_long_rss_opp

    # RSS Safety check
    is_safe = ego_predicted <= opp_predicted

    print(f"[Opposite Lane Change RSS] EgoPred: {ego_predicted:.2f}, OppPred: {opp_predicted:.2f}, Safe: {is_safe}")

    return is_safe


