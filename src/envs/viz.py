import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import math

from envs.configs.traffic_sign_db import traffic_feat_idx, warning_ts_encoding, warning_ts_encoding
from envs.configs.feature_indices import agent_feat_id


def get_object_rectangle(center_x: float, center_y: float, length: float, width: float):
    """
    Given center and dimensions, return the rectangle describing the object's boundaries
    Args:
        center_x: float
        center_y: float
        length: float
        width: float
    Returns:
        A tuple of two lists of four floats, the first list being the x- and the second list being the y-values of the
        boundaries, each in order bottom-left, bottom-right, top-right, top-left
    """
    rectangle = (
        [
            center_x - length / 2,
            center_x + length / 2,
            center_x + length / 2,
            center_x - length / 2,
        ],
        [
            center_y - width / 2,
            center_y - width / 2,
            center_y + width / 2,
            center_y + width / 2,
        ],
    )
    return rectangle

def plot_dipp_ego(ego, color=None, presentation_mode=False):
    # x, y, heading, vel_x, vel_y, length, width, height
    ego = ego.detach().cpu()

    # Remove batch dimension if still present
    if len(ego.shape) > 2:
        ego = ego[0]

    # Plot settings
    color = "black"
    history_color = "red"
    linewidth = 1.0 if not presentation_mode else 2.0

    # Calculate artists data
    ego_rectangle, ego_traj_x, ego_traj_y = get_ego_plot_data(ego)

    ego_rectangle_plt = Rectangle(
            xy=(ego_rectangle[0][0], ego_rectangle[1][0]),
            width=ego_rectangle[0][1] - ego_rectangle[0][0],
            height=ego_rectangle[1][3] - ego_rectangle[1][0],
            angle=np.degrees(ego[-1, agent_feat_id["yaw"]]),
            #rotation_point="center",
            fc="none",
            ec=color,
            lw=linewidth,
            label="Ego"
        )
    
    # Plot
    ax = plt.gca()
    ax.add_patch(ego_rectangle_plt)
    plt.plot(
        ego_traj_x,
        ego_traj_y,
        c=history_color,
        alpha=0.5,
        zorder=6,
        linewidth=2,
        label="Ego History"
    )

def get_ego_plot_data(ego):
    # Collect trajectory
    ego_traj_x = []
    ego_traj_y = []
    
    for t in range(len(ego)):
        if t == len(ego) - 1:  # Draw Rectangle at current timestep   
            ego_rectangle = get_object_rectangle(
                ego[t, agent_feat_id["x"]-1],
                ego[t, agent_feat_id["y"]-1],
                ego[t, agent_feat_id["length"]-1],
                ego[t, agent_feat_id["width"]-1],  # TODO remove fake car dimensions
            )
        else:  # Collect past trajectory
            if not np.all(ego[t] == 0):
                ego_traj_x.append(ego[t, agent_feat_id["x"]-1])
                ego_traj_y.append(ego[t, agent_feat_id["y"]-1])

    return ego_rectangle, ego_traj_x, ego_traj_y

def plot_dipp_neighbors(neighbors, past_timesteps=40, full_extent=True, color="blue", presentation_mode=False):
    # Batch dimension still present
    if len(neighbors.shape) > 3:
        neighbors = neighbors[0]

    linewidth = 1.0 if not presentation_mode else 2.0
    
    neighbor_rectangles, neighbor_angles, neighbor_x_trajectories, neighbor_y_trajectories = get_neighbor_plot_data(neighbors)

    ax = plt.gca()
    for neighbor_rectangle, neighbor_angle in zip(neighbor_rectangles, neighbor_angles):
        neighbor_rectangle_plt = Rectangle(
            xy=(neighbor_rectangle[0][0], neighbor_rectangle[1][0]),
            width=neighbor_rectangle[0][1] - neighbor_rectangle[0][0],
            height=neighbor_rectangle[1][3] - neighbor_rectangle[1][0],
            angle=np.degrees(neighbor_angle),
            rotation_point="center",
            fc="none",
            ec=color,
            lw=linewidth
        )
    
    for x_traj, y_traj in zip(neighbor_x_trajectories, neighbor_y_trajectories):
        plt.plot(
                x_traj,
                y_traj,
                c=color,
                alpha=0.5,
                zorder=6,
                linewidth=2,
                label="Neighbors" #if n == 0 else None # Prevent multiple entries
            )

def get_neighbor_plot_data(neighbors):
    neighbor_rectangles = []
    neighbor_angles = []
    neighbor_x_trajectories = []
    neighbor_y_trajectories = []

    for neighbor in neighbors:
        if not (neighbor[-1, agent_feat_id["x"]] == 0 and neighbor[-1, agent_feat_id["y"]] == 0):
            neighbor_traj_x = []
            neighbor_traj_y = []

            for t in range(len(neighbor)):
                # if neighbor[t, agent_feat_id["class"]] == 0.0:
                #     continue
                if not (neighbor[t, agent_feat_id["x"]] == 0 and neighbor[t, agent_feat_id["y"]] == 0):  # ignore zero padding
                    if t+1 < len(neighbor):  # Collect trajectory
                        neighbor_traj_x.append(neighbor[t, agent_feat_id["x"]])
                        neighbor_traj_y.append(neighbor[t, agent_feat_id["y"]])
                    else:  # Draw Rectangle at current timestep
                        neighbor_rectangle = get_object_rectangle(
                            neighbor[t, agent_feat_id["x"]],
                            neighbor[t, agent_feat_id["y"]],
                            2,#max(1, neighbor[t, agent_feat_id["length"]]),
                            1,#max(1, neighbor[t, agent_feat_id["width"]]),
                        )
            neighbor_x_trajectories.append(neighbor_traj_x)
            neighbor_y_trajectories.append(neighbor_traj_y)
            neighbor_rectangles.append(neighbor_rectangle)
            neighbor_angles.append(neighbor[t, agent_feat_id["yaw"]])

    return neighbor_rectangles, neighbor_angles, neighbor_x_trajectories, neighbor_y_trajectories
        

def mark_lanes(map_lane, lane_width=3, extend=True, color='lightgrey', alpha=0.5):
    poly_list = get_lane_marks(map_lane, lane_width, extend, color, alpha)
    flattened_poly_list = [
        item 
        for element in poly_list
        for item in (element if isinstance(element, list) else element)
    ]
    # Finally plot
    plt.fill(flattened_poly_list, alpha=alpha)


def get_lane_marks(map_lane, lane_width=3, extend=True, color='lightgrey', alpha=0.5):
     # (l, 50, 49)
    # Distance to offset
    d = lane_width/2

    # Collect polygons
    poly_list = []

    for lane in map_lane:
        lane = lane[np.absolute(lane[...,:2]).sum(-1) > 0]
        if np.size(lane) > 1:
            if np.all(lane[..., traffic_feat_idx["ll_x"]]) and np.all(lane[..., traffic_feat_idx["rl_x"]]):
                # TODO: Test once data is available
                x_poly = np.cat([lane[..., traffic_feat_idx["ll_x"]], lane[..., traffic_feat_idx["rl_x"].flip(0)]])
                y_poly = np.cat([lane[..., traffic_feat_idx["ll_y"]], lane[..., traffic_feat_idx["rl_y"].flip(0)]])
            else:  # Estimate lane width
                x = lane[..., traffic_feat_idx["cl_x"]]
                y = lane[..., traffic_feat_idx["cl_y"]]
                # Compute tangents (dx, dy)
            dx = np.gradient(x)
            dy = np.gradient(y)
            lengths = np.sqrt(dx**2 + dy**2 + 1e-12)  # avoid div by zero

            # Normals (perpendiculars)
            nx = -dy / lengths
            ny = dx / lengths

            # Tangents (normalized)
            tx = dx / lengths
            ty = dy / lengths

            # --- Extend line at start and end ---
            # Start extension
            x_start = x[0] - extend * tx[0]
            y_start = y[0] - extend * ty[0]
            nx_start = nx[0]
            ny_start = ny[0]

            # End extension
            x_end = x[-1] + extend * tx[-1]
            y_end = y[-1] + extend * ty[-1]
            nx_end = nx[-1]
            ny_end = ny[-1]

            # Combine extended line
            x_ext = np.concatenate([[x_start], x, [x_end]])
            y_ext = np.concatenate([[y_start], y, [y_end]])
            nx_ext = np.concatenate([[nx_start], nx, [nx_end]])
            ny_ext = np.concatenate([[ny_start], ny, [ny_end]])

            # Offset line points
            x_upper = x_ext + d * nx_ext
            y_upper = y_ext + d * ny_ext
            x_lower = x_ext - d * nx_ext
            y_lower = y_ext - d * ny_ext

            # Polygon for fill
            x_poly = np.concatenate([x_upper, x_lower[::-1]])
            y_poly = np.concatenate([y_upper, y_lower[::-1]])

            # Add to list
            poly_list.append([x_poly, y_poly, color])
    return poly_list


def remove_duplicate_lanes(map_lanes, check_axes=(2, 3)):
    """
    Remove duplicates from `arr` along the specified `check_axes`.

    Parameters
    ----------
    arr : np.ndarray
        Input array of any shape.
    check_axes : tuple of int
        Axes that should be considered as the "slice" to check for uniqueness.
        The complement axes will be deduplicated.

    Returns
    -------
    unique_arr : np.ndarray
        Unique slices along the specified axes.
    unique_indices : np.ndarray
        Indices of the first occurrence of each unique slice.
    """
    # Ensure tuple
    check_axes = tuple(check_axes)

    # Axes not being checked (these define "positions")
    all_axes = tuple(range(map_lanes.ndim))
    pos_axes = tuple(ax for ax in all_axes if ax not in check_axes)

    # Move axes so that checked ones are last
    arr_perm = np.transpose(map_lanes, pos_axes + check_axes)

    # Collapse check_axes into one vector
    shape_pos = [map_lanes.shape[ax] for ax in pos_axes]
    shape_check = [map_lanes.shape[ax] for ax in check_axes]
    arr_flat = arr_perm.reshape(-1, int(np.prod(shape_check)))

    # Deduplicate
    unique_flat, indices = np.unique(arr_flat, axis=0, return_index=True)

    # Recover unique slices
    unique_arr = arr_perm.reshape(-1, *shape_check)[indices]

    return unique_arr, indices

def plot_centerlines(lanes, fontsize=8, linewidth=1.5, presentation_mode=False):
    # (lanes, points, features)

    clanes = get_centerlines_plot_data(lanes)
    axs = plt.gca()

    for centerline in clanes:
        axs.plot(
            centerline[0],
            centerline[1],
            c="gray",
            alpha=0.5,
            linestyle="--",
            linewidth=linewidth
        )
        # # Plot Lane IDs and direction indicators
        # if centerline[...,:2].abs().sum() > 0:  # check if lane has any points
        #     ids = "R"+str(int(centerline[0, traffic_feat_idx["road_id"]].item()))+" L"+str(int(centerline[0, traffic_feat_idx["lane_id"]].item()))
        #     plt.text(
        #         centerline[0, traffic_feat_idx["cl_x"]],
        #         centerline[0, traffic_feat_idx["cl_y"]],
        #         ids if not presentation_mode else '',
        #         fontsize=fontsize,
        #         c="gray",
        #         clip_on=True
        #     )
        #     for p in range(0, len(centerline), 15):
        #         plt.arrow(
        #             centerline[p, traffic_feat_idx["cl_x"]],
        #             centerline[p, traffic_feat_idx["cl_y"]],
        #             2 * math.cos(centerline[p, traffic_feat_idx["cl_yaw"]]),
        #             2 * math.sin(centerline[p, traffic_feat_idx["cl_yaw"]]),
        #             head_width=1,
        #             head_length=1,
        #             color="gray"
        #         )


def get_centerlines_plot_data(lanes):
    clanes = []
    for centerline in lanes:
        centerline = centerline[np.absolute(centerline[...,:2]).sum(-1) > 0]
        clanes.append([centerline[:, traffic_feat_idx["cl_x"]],
                       centerline[:, traffic_feat_idx["cl_y"]]])
    return clanes