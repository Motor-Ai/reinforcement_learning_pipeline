import numpy as np
import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import src.envs.actions.act_integration.frenet_polynomials as fp_class
import src.envs.actions.act_integration.spline as cp


def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0, fp_params):
    """
    Calculates several Frenet paths based on the current state and given parameters.

    Parameters:
    c_speed (float): Current speed of the vehicle [m/s].
    c_d (float): Current lateral position [m].
    c_d_d (float): Current lateral speed [m/s].
    c_d_dd (float): Current lateral acceleration [m/s].
    s0 (float): Current course position [m].
    fp_params (list): List of parameters for generating multiple Frenet paths.

    Returns:
    list: A list of generated Frenet paths with associated costs.
    """

    # Initialize an empty list to store the generated Frenet paths
    frenet_paths = []

    # Generate a Frenet path for each set of parameters in fp_params
    for fp_param in fp_params:

        # Create a new Frenet path object
        fp = fp_class.FrenetPath()

        # Generate time intervals based on the given prediction time and time step
        fp.t = np.linspace(0.0, fp_param[2] - fp_param[3], int(fp_param[2] / fp_param[3])).tolist()

        # Lateral motion planning using a Quintic Polynomial
        lat_qp = fp_class.QuinticPolynomial(
            c_d, c_d_d, c_d_dd, fp_param[1], 0.0, 0.0, fp_param[2]
        )
        fp.d, fp.d_d, fp.d_dd, fp.d_ddd = lat_qp.calc_path_derivatives(fp_param[3])

        # Longitudinal motion planning using a Quartic Polynomial
        lon_qp = fp_class.QuarticPolynomial(
            s0, c_speed, 0.0, fp_param[0], 0.0, fp_param[2]
        )
        fp.s, fp.s_d, fp.s_dd, fp.s_ddd = lon_qp.calc_path_derivatives(fp_param[3])

        # Cost calculation of the Frenet path
        J_d = np.sum(np.array(fp.d_ddd) ** 2)  # Sum of squares of lateral jerk
        J_s = np.sum(np.array(fp.s_ddd) ** 2)  # Sum of squares of longitudinal jerk
        J_lat_e = np.sum(np.array(fp.d) ** 2)  # Sum of squares of lateral deviation from reference path

        # Square of the difference from the target speed
        ds = (fp_param[0] - fp.s_d[-1]) ** 2

        # Cost weights
        K_J = 0.1  # Weight for the jerk cost
        K_D = 1.0  # Weight for the terminal lateral deviation cost
        K_LAT_err = 1.0  # Weight for the lateral error cost

        K_LAT = 1.0  # Weight for the lateral cost
        K_LON = 1.0  # Weight for the longitudinal cost

        # Calculate the total cost for the lateral motion
        fp.cd = K_J * J_d + K_D * fp.d[-1] ** 2 + K_LAT_err * J_lat_e

        # Calculate the total cost for the longitudinal motion
        fp.cv = K_J * J_s + K_D * ds

        # Combine the lateral and longitudinal costs into a final cost function
        fp.cf = K_LAT * fp.cd + K_LON * fp.cv

        # Append the generated Frenet path with its associated cost to the list
        frenet_paths.append(fp)

    # Return the list of generated Frenet paths
    return frenet_paths

def calc_global_paths(fplist, tx, ty, tyaw):
    """
    Converts the Frenet Paths into Cartesian coordinates.

    Parameters:
    fplist (list): List of Frenet paths to be converted.
    tx (list): List of x-coordinates of the reference trajectory.
    ty (list): List of y-coordinates of the reference trajectory.
    tyaw (list): List of yaw angles (orientations) along the reference trajectory.
    Returns:
    list: List of Frenet paths with updated global x, y coordinates, and yaw angles.
    """

    
    # Calculate the B-spline representation of the reference trajectory
    x_spline, y_spline, yaw_spline = cp.approximate_b_spline_path_new(tx, ty, tyaw, s=0.5)

    # Iterate over each Frenet path to convert its coordinates to Cartesian form
    for fp in fplist:
        # Find the closest point on the reference trajectory to the current Frenet s-coordinate
        px_i = x_spline(fp.s)
        py_i = y_spline(fp.s)
        pyaw_i  = yaw_spline(fp.s)

        # Calculate the lateral offset (di) and convert to Cartesian coordinates
        di = np.asarray(fp.d)
        fx = px_i + di * np.cos(pyaw_i + np.pi / 2.0)
        fy = py_i + di * np.sin(pyaw_i + np.pi / 2.0)

        # Append the calculated global coordinates to the Frenet path
        fp.x = fx
        fp.y = fy

        # Calculate the yaw (orientation) for the cartesian frenet path x and y coordinates
        fp.yaw = get_heading_arr(fp.x, fp.y)
        # Calculate the curvature for the cartesian frenet path x and y coordinates
        fp.k = cp.get_curvature_arr(fp.x, fp.y)

    # Return the list of Frenet paths with updated global coordinates and yaw angles
    return fplist


def get_car_outline(x, y, yaw):
    """
    Computes and returns the outline of the ego vehicle as a polygon based on its position and orientation.

    Parameters:
    x (float): The x-coordinate of the ego vehicle's position.
    y (float): The y-coordinate of the ego vehicle's position.
    yaw (float): The orientation (yaw angle) of the ego vehicle in radians.

    Returns:
    np.ndarray: A 2D array where each row represents the (x, y) coordinates of a corner of the vehicle outline.
    """

    ## ToDo: Add ego vehicle params to act_utils for a common shared params in Act nodes
    # Dimensions of the vehicle (EQV/EVito)
    LENGTH = 5.140  # Length of the vehicle in meters
    WIDTH = 2.23  # Width of the vehicle in meters
    BACKTOWHEEL = 1.045  # Distance from the rear axle to the back of the vehicle in meters

    # Define the outline of the vehicle as a polygon (before rotation and translation)
    # The polygon is defined in the vehicle's local coordinate system
    outline = np.array(
        [
            [-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],  # x-coordinates
            [WIDTH / 2, WIDTH / 2, -WIDTH / 2, -WIDTH / 2, WIDTH / 2],  # y-coordinates
        ]
    )

    # Create the rotation matrix based on the yaw angle to rotate the vehicle outline
    Rot1 = np.array([
        [math.cos(yaw), math.sin(yaw)],  # First row of rotation matrix
        [-math.sin(yaw), math.cos(yaw)]  # Second row of rotation matrix
    ])

    # Rotate the outline polygon by multiplying with the rotation matrix
    outline = (outline.T.dot(Rot1)).T

    # Translate the rotated outline to the vehicle's current position (x, y)
    outline[0, :] += x  # Translate x-coordinates
    outline[1, :] += y  # Translate y-coordinates

    # Return the translated and rotated vehicle outline as a transposed array
    return outline.T


def check_collision_dynamic(fplist, dynamic_ob, ego_x, ego_y):
    """
    Checks for potential collisions between the planned paths and dynamic obstacles.

    Parameters:
    fplist (list): A list of Frenet paths generated by the path planner.
    dynamic_ob (list): A list of predicted positions (polygons) of dynamic obstacles.
    ego_x (float): The x-coordinate of the ego vehicle's position.
    ego_y (float): The y-coordinate of the ego vehicle's position.

    Returns:
    list: The list of Frenet paths with updated collision flags and intersection areas.
    """

    # Iterate over each path in the list of Frenet paths
    for j, _ in enumerate(fplist):
        # Initialize collision flags and intersection areas for each point in the path
        fplist[j].dynamic_collision = [False] * len(fplist[j].x)
        fplist[j].dynamic_intersection_area = [0.0] * len(fplist[j].x)

        # Check each point on the current path for potential collisions
        for i in range(len(fplist[j].x)):
            # Get the outline (polygon) of the ego vehicle at the current path point
            ego_poly = get_car_outline(fplist[j].x[i], fplist[j].y[i], fplist[j].yaw[i])
            ego_polygon = Polygon(ego_poly)
            
            # Iterate over each dynamic obstacle to check for collisions
            for ob_i in range(len(dynamic_ob)):
                # Only check if the dynamic obstacle has a valid polygon
                if len(dynamic_ob[ob_i][0]):
                    # Create the polygon for the dynamic obstacle
                    dynamic_obj_polygon = Polygon(np.array([dynamic_ob[ob_i][0], dynamic_ob[ob_i][1]]).T)
                    
                    # Check if the ego vehicle's polygon intersects with the obstacle's polygon
                    collision_flag = ego_polygon.intersects(dynamic_obj_polygon)
                    
                    # Update the collision flag for the current path point
                    fplist[j].dynamic_collision[i] = fplist[j].dynamic_collision[i] or collision_flag

                    # Optional: Calculate the intersection area (currently commented out)
                    # collision_poly = ego_polygon.intersection(dynamic_obj_polygon)
                    # fplist[j].dynamic_intersection_area[i] = collision_poly.area

        # Add the sum of intersection areas to the path's cost function
        fplist[j].cf += np.sum(fplist[j].dynamic_intersection_area)

    return fplist


def check_dynamic_collision(self, fp, fp_path_time, fp_path_time_tick):
    """
    Checks if a Frenet path is free from dynamic collisions for a significant duration.

    Parameters:
    fp (FrenetPath): The Frenet path being evaluated.
    fp_path_time (float): Total duration of the Frenet path.
    fp_path_time_tick (float): Time increment for each point in the path.

    Returns:
    bool: True if the path is collision-free for more than 3 seconds, False otherwise.
    """
    time_of_path = 0.0  # Initialize the time a path is collision-free
    for i in range(len(fp.dynamic_collision)):
        if not fp.dynamic_collision[i]:  # If no collision at this point
            time_of_path += fp_path_time_tick  # Increment collision-free time
        else:
            break  # Stop checking if a collision is detected

    # Check if the path is collision-free for more than 3 seconds
    return time_of_path > 3.0


def check_paths(self, fplist, fp_path_time, fp_path_time_tick):
    """
    Filters the Frenet paths based on dynamic collision checks and path length.

    Parameters:
    fplist (list): List of Frenet paths.
    fp_path_time (float): Total duration of each Frenet path.
    fp_path_time_tick (float): Time increment for each point in the path.

    Returns:
    list: List of valid Frenet paths that passed the checks.
    """
    ok_ind = []  # List to store indices of valid paths
    for i, _ in enumerate(fplist):
        # Check if the path has dynamic collisions or is too short
        if not check_dynamic_collision(self, fplist[i], fp_path_time, fp_path_time_tick):
            self.get_logger().debug('Path ID: ' + str(i) + " eliminated due to Dynamic Obstacle")
            continue
        elif fplist[i].s[-1] < 1.5:  # Path length check
            self.get_logger().debug('Path ID: ' + str(i) + " eliminated due to path_length being too short")
            continue

        ok_ind.append(i)  # Add index of valid path

    # Return the list of valid paths
    return [fplist[i] for i in ok_ind]


def plot_fp(fp_list):
    """
    Plots all the calculated Frenet paths.

    Parameters:
    fp_list (list): List of Frenet paths to be plotted.
    """
    for i, _ in enumerate(fp_list):
        plt.plot(fp_list[i].x, fp_list[i].y, "-.b", lw=1.3)  # Plot each path with a dashed blue line


def pi_2_pi(angle):
    """
    Normalize an angle to the range [-pi, pi].

    Parameters:
    angle (float): Angle in radians.

    Returns:
    float: Normalized angle within the range [-pi, pi].
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi  # Subtract 2*pi until the angle is within [-pi, pi]

    while angle < -np.pi:
        angle += 2.0 * np.pi  # Add 2*pi until the angle is within [-pi, pi]

    return angle


def normalize_angle_arr(angle):
    """
    Normalize an array of angles to the range [-pi, pi].

    Parameters:
    angle (array-like): Array of angles in radians.

    Returns:
    numpy.ndarray: Array of normalized angles within the range [-pi, pi].
    """
    angle = np.array(angle)  # Convert input to a numpy array
    sin_val = np.sin(angle)  # Calculate the sine of each angle
    cos_val = np.cos(angle)  # Calculate the cosine of each angle
    normalized_angle = np.arctan2(sin_val, cos_val)  # Calculate the normalized angle using atan2
    return normalized_angle


def get_heading_arr(x_arr, y_arr):
    """
    Calculate the heading (yaw) of the input path.
    
    The heading is the direction in which the path is moving, computed as the angle of the tangent to the path.

    Parameters:
    x_arr: List or array of x coordinates of the path points.
    y_arr: List or array of y coordinates of the path points.
    
    Returns:
    List of headings (angles) in radians corresponding to each segment of the path.
    """
    
    # Check if there are more than one point in the path
    if len(x_arr) > 1:
        # Calculate the differences between consecutive points
        dx = np.diff(x_arr)  # Difference in x coordinates
        dy = np.diff(y_arr)  # Difference in y coordinates
        
        # Calculate the heading (yaw) using the arctangent of the differences
        # atan2 returns the angle between the positive x-axis and the line to the point (dx, dy)
        yaw_arr = np.arctan2(dy, dx)
        
        # Append the last heading value to maintain the same length as the input arrays
        # This ensures that each point has a heading value, even the last one
        yaw_arr = np.append(yaw_arr, yaw_arr[-1])
    else:
        # If there is only one point or no points, return an empty array for headings
        yaw_arr = np.array([])
    
    # Convert the numpy array of headings to a list and return
    return yaw_arr.tolist()