import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.envs.observation.decision_traffic_rules.feature_indices import agent_feat_id

class MatplotlibAnimationRenderer:
    def __init__(self, save_path=None):
        """
        Initialize the animation renderer.
        
        Parameters:
            save_path (str): File path where each frame is saved.
        """
        self.save_path = os.path.join(os.environ["DATA_DIR"], "temp_plots/pngs/") if save_path is None else save_path
        self.latest_ego = None
        self.latest_neighbors = None
        self.latest_map = None
        self.FOV = 40

        # Create figure and axis.
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-self.FOV, self.FOV)
        self.ax.set_ylim(-self.FOV, self.FOV)
        self.ax.set_title("Simulation Debug Plot")
        
        # Set up the animation. The update_plot function will be called periodically.
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, init_func=self.init_plot, interval=100, blit=False
        )

    def init_plot(self):
        """Initialization function for the animation."""
        self.ax.clear()
        self.ax.set_xlim(-self.FOV, self.FOV)
        self.ax.set_ylim(-self.FOV, self.FOV)
        self.ax.set_title("Simulation Debug Plot")
        return []

    def update_plot(self, step=0):
        """
        Update function for the animation. This function reads the latest data 
        and re-plots the debug visualization. It also saves the current frame.
        
        Parameters:
            frame: Unused but required by FuncAnimation.
        """
        self.ax.clear()
        
        # Plot map lanes (assuming latest_map is structured as a list where the first element is a collection of lanes)
        if self.latest_map is not None:
            for lane in self.latest_map[0]:
                self.ax.plot(lane[:, 0], lane[:, 1], marker='o', 
                             color='black', markersize=0.5, linestyle='None')
        
        # Plot neighbors (using agent_feat_id for keys 'x' and 'y')
        if self.latest_neighbors is not None:
            for neighbor in self.latest_neighbors[0]:
                self.ax.plot(neighbor[:, agent_feat_id['x']], neighbor[:, agent_feat_id['y']], 
                             'o', color='green')
        
        # Plot ego vehicle trajectory
        if self.latest_ego is not None:
            self.ax.plot(self.latest_ego[0, :, agent_feat_id['x']], 
                         self.latest_ego[0, :, agent_feat_id['y']], marker='o', label='ego', color='red')
        
        if self.route is not None:
            self.ax.plot(self.route[:, 0], self.route[:, 1], marker='o', label='Route', color='orange', markersize=5)
        
        if self.route_ef is not None:
            self.ax.plot(self.route_ef[:, 0], self.route_ef[:, 1], marker='o', label='route_ef', color='orange', markersize=5)
        
        self.ax.set_xlim(-self.FOV, self.FOV)
        self.ax.set_ylim(-self.FOV, self.FOV)
        self.ax.legend()
        
        # Save the current frame
        plt.savefig(self.save_path)
        print(f"Plot saved in {self.save_path}{step:04d}.png")
        return []

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

    def show(self):
        """Display the animation window."""
        plt.show()

