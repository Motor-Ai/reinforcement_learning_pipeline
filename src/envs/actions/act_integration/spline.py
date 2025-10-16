import numpy as np
from scipy.interpolate import splprep, splev
import scipy.interpolate as interpolate


def get_heading_arr(x_arr, y_arr):
    """
    Calculate the heading (yaw angle) of the input path based on x and y coordinates.

    Args:
        x_arr (list or array): X-coordinates of the path.
        y_arr (list or array): Y-coordinates of the path.

    Returns:
        yaw_arr (array): Array of yaw angles corresponding to the input path.
    """
    if len(x_arr) > 1:
        # Calculate the yaw angles using arctan2 and the differences between consecutive points.
        yaw_arr = np.arctan2(np.diff(y_arr), np.diff(x_arr))
        # Append the last yaw value to maintain array length consistency.
        yaw_arr = np.append(yaw_arr, yaw_arr[-1])
    else:
        yaw_arr = np.array([])
    return yaw_arr


def get_curvature_arr(x_arr, y_arr):
    """
    Calculate the curvature of the input path based on x and y coordinates.

    Args:
        x_arr (list or array): X-coordinates of the path.
        y_arr (list or array): Y-coordinates of the path.

    Returns:
        curvature (array): Array of curvatures corresponding to the input path.
    """
    if len(x_arr) > 5:
        # Compute the first derivative (dx/dt, dy/dt) and arc length (ds/dt).
        dx_dt = np.gradient(x_arr)
        dy_dt = np.gradient(y_arr)

        # Compute the second derivative.
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)

        # Calculate curvature using the derivatives.
        curvature = -1.0 * (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
    else:
        curvature = np.array([])

    return curvature


def eliminate_duplicates(arr):
    """
    Eliminates consecutive duplicate rows in a 2D array.

    Args:
        arr (numpy array): A 2D numpy array where each row is compared with the next to identify duplicates.

    Returns:
        numpy array: A 2D array with consecutive duplicate rows removed.
    """

    # Calculate the number of identical elements between consecutive rows.
    repeat_vals = (arr[:-1] == arr[1:]).sum(axis=1)
    
    # Identify the indices of rows that are not completely identical to their consecutive row.
    idx = np.where(repeat_vals != 2)[0] + 1
    
    # Return the array with the rows at the identified indices, effectively removing duplicates.
    return arr[idx]


def filter_input(x_arr, y_arr, dl):
    """
    Filters out repetitive and consecutive points with distances less than the defined resolution 'dl'.

    Args:
        x_arr (list or array): X-coordinates of the path.
        y_arr (list or array): Y-coordinates of the path.
        dl (float): Distance resolution threshold.

    Returns:
        tuple: Filtered x and y coordinates.
    """
    # Combine x and y arrays into a 2D array.
    xy_inp = np.transpose(np.array([x_arr, y_arr]))

    # Remove consecutive duplicate points.
    xy_inp_unique = eliminate_duplicates(xy_inp)

    # Extract unique x and y coordinates.
    x_arr_unique = xy_inp_unique[:, 0]
    y_arr_unique = xy_inp_unique[:, 1]

    if len(x_arr_unique) > 1:
        # Calculate consecutive distances between points.
        x_arr_diff = np.diff(x_arr_unique)
        y_arr_diff = np.diff(y_arr_unique)
        dist_btw = np.round(np.hypot(x_arr_diff, y_arr_diff), 2)

        # Find indices of points that are too close and remove them.
        idx_remove = np.where(dist_btw <= dl)[0] + 1
        x_arr_final = np.delete(x_arr_unique, idx_remove)
        y_arr_final = np.delete(y_arr_unique, idx_remove)

        return x_arr_final, y_arr_final
    else:
        return x_arr_unique, y_arr_unique


def calc_spline_course(x_arr, y_arr, input_res, output_res):
    """
    Generate a cubic spline course from input x and y coordinates.

    Args:
        x_arr (list or array): X-coordinates of the input path.
        y_arr (list or array): Y-coordinates of the input path.
        input_res (float): Original resolution of the input path.
        output_res (float): Desired resolution of the output path.

    Returns:
        tuple: Spline-interpolated x, y, yaw, and curvature arrays.
    """
    # Create a spline function based on the filtered input.
    f, u = splprep([x_arr, y_arr], s=0)

    # Calculate the resolution ratio between input and output.
    res_ratio = int(input_res / output_res)

    # Generate interpolated x and y points using the spline function.
    x_spline, y_spline = splev(np.linspace(0, 1, int(len(x_arr) * res_ratio)), f)

    # Calculate yaw angles along the spline.
    yaw_spline = get_heading_arr(x_spline, y_spline)

    # Calculate curvature along the spline.
    curvature_spline = get_curvature_arr(x_spline, y_spline)

    return x_spline, y_spline, yaw_spline, curvature_spline


def calc_spline_course_2(x_arr, y_arr, path_len, output_res):
    """
    Generate a cubic spline course with specified path length from input x and y coordinates.

    Args:
        x_arr (list or array): X-coordinates of the input path.
        y_arr (list or array): Y-coordinates of the input path.
        path_len (float): Total length of the output path.
        output_res (float): Desired resolution of the output path.

    Returns:
        tuple: Spline-interpolated x, y, yaw, and curvature arrays.
    """
    # Create a spline function based on the filtered input.
    f, u = splprep([x_arr, y_arr])

    # Generate interpolated x and y points using the spline function.
    x_spline, y_spline = splev(np.linspace(0, 1, int(path_len / output_res)), f)

    # Calculate yaw angles along the spline.
    yaw_spline = get_heading_arr(x_spline, y_spline)

    # Calculate curvature along the spline.
    curvature_spline = get_curvature_arr(x_spline, y_spline)

    # Return the results as lists.
    return x_spline.tolist(), y_spline.tolist(), yaw_spline.tolist(), curvature_spline.tolist()

def approximate_b_spline_path_new(x: list,
                              y: list,
                              yaw: list,
                              degree: int = 3,
                              s=None,
                              ) -> tuple:
    """
    Approximate a B-Spline path from given x and y coordinates.

    Args:
        x (list or array): X-coordinates of the path.
        y (list or array): Y-coordinates of the path.
        yaw (list or array): Yaw of the path.
        n_path_points (int): Number of points on the resulting path.
        degree (int): Degree of the B-Spline curve (2 <= k <= 5).
        s (float, optional): Smoothing parameter. Larger values produce smoother paths.

    Returns:
        tuple: x, y and heading arrays for the approximated B-Spline path.
    """

    if len(x) == len(y) and len(x) > 1:
        # Calculate cumulative distances between consecutive points.
        distances_input = _calc_distance_vector(x, y,normalize=False)

        # Adjust degree for insufficient points
        while len(x) <= degree: 
            degree -= 1

        # Fit B-Spline functions to the x and y coordinates.
        spl_i_x = interpolate.UnivariateSpline(distances_input, x, k=degree, s=s)
        spl_i_y = interpolate.UnivariateSpline(distances_input, y, k=degree, s=s)
        spl_i_yaw = interpolate.UnivariateSpline(distances_input, yaw, k=degree, s=s)
        
        # Generate uniformly spaced points along the path.
        return spl_i_x, spl_i_y,spl_i_yaw
    else:
        return [], [], []


def approximate_b_spline_path(x: list,
                              y: list,
                              n_path_points: int,
                              degree: int = 3,
                              s=None,
                              ) -> tuple:
    """
    Approximate a B-Spline path from given x and y coordinates.

    Args:
        x (list or array): X-coordinates of the path.
        y (list or array): Y-coordinates of the path.
        n_path_points (int): Number of points on the resulting path.
        degree (int): Degree of the B-Spline curve (2 <= k <= 5).
        s (float, optional): Smoothing parameter. Larger values produce smoother paths.

    Returns:
        tuple: x, y, heading, and curvature arrays for the approximated B-Spline path.
    """

    if len(x) == len(y) and len(x) > 1:
        # Calculate cumulative distances between consecutive points.
        distances = _calc_distance_vector(x, y)

        # Adjust degree for insufficient points
        while len(x) <= degree: 
            degree -= 1

        # Fit B-Spline functions to the x and y coordinates.
        spl_i_x = interpolate.UnivariateSpline(distances, x, k=degree, s=s)
        spl_i_y = interpolate.UnivariateSpline(distances, y, k=degree, s=s)

        # Generate uniformly spaced points along the path.
        sampled = np.linspace(0.0, distances[-1], n_path_points)
        return _evaluate_spline(sampled, spl_i_x, spl_i_y)
    else:
        return [], []


def _calc_distance_vector(x, y, normalize=True):
    """
    Calculate the cumulative distance vector for a set of points.

    Args:
        x (list or array): X-coordinates of the points.
        y (list or array): Y-coordinates of the points.
        normalize (bool, optional): Normalize the cumulative distance vector.

    Returns:
        array: Normalized cumulative distance vector.
    """
    dx, dy = np.diff(x), np.diff(y)
    distances = np.cumsum([np.hypot(idx, idy) for idx, idy in zip(dx, dy)])
    distances = np.concatenate(([0.0], distances))
    if normalize and distances[-1] > 0:
        distances /= distances[-1]
    return distances


def _evaluate_spline(sampled, spl_i_x, spl_i_y):
    """
    Evaluate the B-spline at sampled points.

    Args:
        sampled (array): Array of parameter values where the spline should be evaluated.
        spl_i_x (UnivariateSpline): Spline function for x-coordinates.
        spl_i_y (UnivariateSpline): Spline function for y-coordinates.

    Returns:
        tuple: Arrays of x and y coordinates evaluated at the sampled points.
    """
    # Evaluate the spline functions at the sampled parameter values
    x = spl_i_x(sampled)
    y = spl_i_y(sampled)
    return np.array(x), np.array(y)


def calc_bspline_course(x_arr, y_arr, input_res, output_res):
    """
    Generate a B-spline course from input x and y coordinates with specified resolution.

    Args:
        x_arr (list or array): X-coordinates of the input path.
        y_arr (list or array): Y-coordinates of the input path.
        input_res (float): Original resolution of the input path.
        output_res (float): Desired resolution of the output path.

    Returns:
        tuple: B-spline interpolated x, y, yaw, and curvature arrays.
    """
    # Calculate the resolution ratio between input and output
    res_ratio = int(input_res / output_res)

    # Generate interpolated x and y points using the B-spline function
    x_spline, y_spline = approximate_b_spline_path(x_arr, y_arr, int(len(x_arr) * res_ratio), s=0.5)

    # Calculate yaw angles along the B-spline
    yaw_spline = get_heading_arr(x_spline, y_spline)

    # Calculate curvature along the B-spline
    curvature_spline = get_curvature_arr(x_spline, y_spline)

    return x_spline, y_spline, yaw_spline, curvature_spline


def calc_bspline_course_2(x_arr, y_arr, path_len, output_res):
    """
    Generate a B-spline course with specified path length and resolution from input x and y coordinates.

    Args:
        x_arr (list or array): X-coordinates of the input path.
        y_arr (list or array): Y-coordinates of the input path.
        path_len (float): Total length of the input path.
        output_res (float): Desired resolution of the output path.

    Returns:
        tuple: B-spline interpolated x, y, yaw, and curvature lists.
    """  
    # Generate interpolated x and y points using the B-spline function
    x_spline, y_spline = approximate_b_spline_path(x_arr, y_arr, int(path_len / output_res), s=0.5)

    # Calculate yaw angles along the B-spline
    yaw_spline = get_heading_arr(x_spline, y_spline)

    # Calculate curvature along the B-spline
    curvature_spline = get_curvature_arr(x_spline, y_spline)

    # Convert the results to lists and return
    return x_spline.tolist(), y_spline.tolist(), yaw_spline.tolist(), curvature_spline.tolist()