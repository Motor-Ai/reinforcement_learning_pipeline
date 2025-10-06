import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import envs.viz as viz

from src.envs.observation.decision_traffic_rules.feature_indices import agent_feat_id

class MatplotlibAnimationRenderer:
    def __init__(self, save_path='/home/ratul/Workstation/motor-ai/MAI_Bench2Drive/rss_debug/temp_plots/pngs/'):
        """
        Initialize the animation renderer.
        
        Parameters:
            save_path (str): File path where each frame is saved.
        """
        self.save_path = save_path
        self.latest_ego = None
        self.latest_neighbors = None
        self.latest_map = None
        self.FOV = 40

        # Create figure, artists and axis.
        #plt.close("all")
        #self.fig, self.ax = plt.subplots()

        self.fig = plt.gcf()
        self.ax = plt.gca()

        self.ego_hist, = self.ax.plot([], [], c="red", alpha=0.5, lw=2, zorder=6, label="Ego History")

        self.centerlines = []
        self.centerline_polys = []
        for i in range(50):
            l = self.ax.plot([], [], c="gray", alpha=0.5, lw=2, linestyle="--")
            self.centerlines.extend(l)

            p = self.ax.fill([], [], "lightgrey", alpha=0.5)
            self.centerline_polys.extend(p)
        
        self.neighbors = []
        for i in range(20):
            n = Rectangle(
                (0, 0),
                0,
                0,
                fc="none",
                ec="blue",
                lw=2
            )
            self.ax.add_patch(n)
            self.neighbors.append(n)

        self.ego_rect = Rectangle(
            (-0.5, -0.5), 
            1, 
            1,
            #rotation_point="center",
            fc="none",
            ec="black",
            lw=2,
            #label="Ego"
        )
        self.ax.add_patch(self.ego_rect)

        self.ax.set_xlim(-self.FOV, self.FOV)
        self.ax.set_ylim(-self.FOV, self.FOV)
        self.ax.set_title("Simulation Debug Plot")

        # Set to interactive
        plt.ion()
        plt.show(block=False)


    def update_plot(self, step=0):
        """
        Update function for the animation. This function reads the latest data 
        and re-plots the debug visualization. It also saves the current frame.
        
        Parameters:
            frame: Unused but required by FuncAnimation.
        """
        ego_rectangle, ego_traj_x, ego_traj_y = viz.get_ego_plot_data(self.latest_ego[0])
        #print(ego_rectangle)
        # print(ego_traj_x)
        # print(ego_traj_y)
        self.ego_rect.set_xy((ego_rectangle[0][0], ego_rectangle[1][0]))
        self.ego_rect.set_width(ego_rectangle[0][1] - ego_rectangle[0][0])
        self.ego_rect.set_height(ego_rectangle[1][3] - ego_rectangle[1][0])
        #self.ego_rect.set_alpha(np.degrees(self.latest_ego[0, -1, agent_feat_id["yaw"]-1]))

        self.ego_rect.set_visible(True)
        #print(self.ego_rect)

        n_rects, n_angles, n_trajs_x, n_trajs_y = viz.get_neighbor_plot_data(self.latest_neighbors[0])
        for i, n in enumerate(self.neighbors):
            n.set_visible(False)
            try:
                n.set_xy((n_rects[i][0][0], n_rects[i][1][0]))
                n.set_width(n_rects[i][0][1] - n_rects[i][0][0])
                n.set_height(n_rects[i][1][3] - n_rects[i][1][0])
                n.set_angle(n_angles[i])
                n.set_visible(True)
            except IndexError:
                pass

        self.ego_hist.set_xdata(ego_traj_x)
        self.ego_hist.set_ydata(ego_traj_y)

        #print(self.latest_map[0].shape)
        #print(self.latest_map[0][0, :, :2])
        lanes = self.latest_map[0]
        lanes, _ = viz.remove_duplicate_lanes(lanes, check_axes=(1, 2))
        # print(len(lanes))
        # print(len(lanes[0]))
        # print(len(lanes[0][0]))
        
        centerlines = viz.get_centerlines_plot_data(lanes)
        marked_lanes = viz.get_lane_marks(lanes)
        for i, cl in enumerate(self.centerlines):
            cl.set_visible(False)
            try:
                cl.set_xdata(centerlines[i][0])
                cl.set_ydata(centerlines[i][1])
                cl.set_visible(True)  
                
            except IndexError:
                pass

        for i, clp in enumerate(self.centerline_polys):
            clp.visible = False
            try:
                xy = np.stack((marked_lanes[i][0], marked_lanes[i][1]), axis=-1)
                clp.set_xy(xy)
                clp.visible=True
            except IndexError:
                pass
                
        self.ax.legend()
        
        # # Save the current frame
        # plt.savefig(self.save_path)
        # print(f"Plot saved in {self.save_path}{step:04d}.png")
        # return []

    def update_data(self, ego, neighbors, map_data, route=None, route_ef=None):
        """
        Update the data for the animation. Call this function at every simulation step 
        with the latest inputs.
        
        Parameters:
            ego: Updated ego observations.
            neighbors: Updated neighbor observations.
            map_data: Updated map observations.
        """
        self.latest_ego = ego
        self.latest_neighbors = neighbors
        self.latest_map = map_data
        self.route = route
        self.route_ef = route_ef

    def render(self, ego, neighbors, map_data):
        self.update_data(ego, neighbors, map_data)
        self.update_plot()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)
