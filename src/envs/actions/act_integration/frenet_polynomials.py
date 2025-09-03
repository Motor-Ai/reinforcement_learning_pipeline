import numpy as np


class QuarticPolynomial:
    def __init__(self, start_x, start_vel, start_acc, end_vel, end_acc, traj_time):
        """
        Initialize a QuarticPolynomial object and compute the polynomial coefficients.

        :param start_x: Initial position (x-coordinate)
        :param start_vel: Initial velocity
        :param start_acc: Initial acceleration
        :param end_vel: Final velocity
        :param end_acc: Final acceleration
        :param traj_time: Total trajectory time
        """
        # Coefficients of the quartic polynomial equation
        self.a0 = start_x
        self.a1 = start_vel
        self.a2 = start_acc / 2.0
        self.T = traj_time

        # Define matrices for the equation AX = b
        A = np.array(
            [[3 * (self.T**2), 4 * (self.T**3)], [6 * self.T, 12 * (self.T**2)]]
        )
        b = np.array([end_vel - self.a1 - 2 * self.a2 * self.T, end_acc - 2 * self.a2])

        # Solve for coefficients a3 and a4
        x = np.linalg.solve(A, b)
        self.a3 = x[0]
        self.a4 = x[1]


    def calc_path_derivatives(self, dt):
        """
        Calculate the position, velocity, acceleration, and jerk over time.

        :param dt: Time step for the calculation
        :return: Tuple of lists containing position, velocity, acceleration, and jerk over time
        """
        # Generate time array from 0 to T with step dt
        t_arr = np.linspace(0.0, self.T - dt, int(self.T / dt))

        # Compute position using the quartic polynomial equation
        x = (
            self.a0
            + self.a1 * t_arr
            + self.a2 * (t_arr**2)
            + self.a3 * (t_arr**3)
            + self.a4 * (t_arr**4)
        )
        x = x.tolist()

        # Compute velocity (first derivative of position)
        dx = (
            self.a1
            + 2 * self.a2 * t_arr
            + 3 * self.a3 * (t_arr**2)
            + 4 * self.a4 * (t_arr**3)
        )
        dx = dx.tolist()

        # Compute acceleration (second derivative of position)
        ddx = 2 * self.a2 + 6 * self.a3 * t_arr + 12 * self.a4 * (t_arr**2)
        ddx = ddx.tolist()

        # Compute jerk (third derivative of position)
        dddx = 6 * self.a3 + 24 * self.a4 * t_arr
        dddx = dddx.tolist()

        return x, dx, ddx, dddx


class QuinticPolynomial:
    def __init__(self, start_x, start_vel, start_acc, end_x, end_vel, end_acc, traj_time):
        """
        Initialize a QuinticPolynomial object and compute the polynomial coefficients.

        :param start_x: Initial position (x-coordinate)
        :param start_vel: Initial velocity
        :param start_acc: Initial acceleration
        :param end_x: Final position (x-coordinate)
        :param end_vel: Final velocity
        :param end_acc: Final acceleration
        :param traj_time: Total trajectory time
        """
        # Coefficients of the quintic polynomial equation
        self.a0 = start_x
        self.a1 = start_vel
        self.a2 = start_acc / 2.0
        self.T = traj_time

        # Define matrices for the equation AX = b
        A = np.array(
            [
                [self.T**3, self.T**4, self.T**5],
                [3 * (self.T**2), 4 * (self.T**3), 5 * (self.T**4)],
                [6 * self.T, 12 * (self.T**2), 20 * (self.T**3)],
            ]
        )
        b = np.array(
            [
                end_x - self.a0 - self.a1 * self.T - self.a2 * (self.T**2),
                end_vel - self.a1 - 2 * self.a2 * self.T,
                end_acc - 2 * self.a2,
            ]
        )

        # Solve for coefficients a3, a4, and a5
        x = np.linalg.solve(A, b)
        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]


    def calc_path_derivatives(self, dt):
        """
        Calculate the position, velocity, acceleration, and jerk over time.

        :param dt: Time step for the calculation
        :return: Tuple of lists containing position, velocity, acceleration, and jerk over time
        """
        # Generate time array from 0 to T with step dt
        t_arr = np.linspace(0.0, self.T - dt, int(self.T / dt))

        # Compute position using the quintic polynomial equation
        x = (
            self.a0
            + self.a1 * t_arr
            + self.a2 * (t_arr**2)
            + self.a3 * (t_arr**3)
            + self.a4 * (t_arr**4)
            + self.a5 * (t_arr**5)
        )
        x = x.tolist()

        # Compute velocity (first derivative of position)
        dx = (
            self.a1
            + 2 * self.a2 * t_arr
            + 3 * self.a3 * (t_arr**2)
            + 4 * self.a4 * (t_arr**3)
            + 5 * self.a5 * (t_arr**4)
        )
        dx = dx.tolist()

        # Compute acceleration (second derivative of position)
        ddx = (
            2 * self.a2
            + 6 * self.a3 * t_arr
            + 12 * self.a4 * (t_arr**2)
            + 20 * self.a5 * (t_arr**3)
        )
        ddx = ddx.tolist()

        # Compute jerk (third derivative of position)
        dddx = 6 * self.a3 + 24 * self.a4 * t_arr + 60 * self.a5 * (t_arr**2)
        dddx = dddx.tolist()

        return x, dx, ddx, dddx


class FrenetPath:
    def __init__(self):
        """
        Initialize a FrenetPath object to store trajectory information.
        """
        # Arrays to store the Frenet coordinates and their derivatives
        self.t = []         # Time array
        self.d = []         # Lateral position (d) in Frenet coordinates
        self.d_d = []       # First derivative of lateral position (velocity)
        self.d_dd = []      # Second derivative of lateral position (acceleration)
        self.d_ddd = []     # Third derivative of lateral position (jerk)
        self.s = []         # Longitudinal position (s) in Frenet coordinates
        self.s_d = []       # First derivative of longitudinal position (velocity)
        self.s_dd = []      # Second derivative of longitudinal position (acceleration)
        self.s_ddd = []     # Third derivative of longitudinal position (jerk)
        self.cd = 0.0       # Curvature derivative (dynamic)
        self.cv = 0.0       # Curvature velocity (dynamic)
        self.cf = 0.0       # Curvature factor (dynamic)

        # Arrays to store Cartesian coordinates and related information
        self.x = []         # X coordinates in Cartesian space
        self.y = []         # Y coordinates in Cartesian space
        self.yaw = []       # Yaw angles in Cartesian space
        self.k = []         # Curvature in Cartesian space

        # Arrays to store dynamic obstacle information
        self.dynamic_intersection_area = []  # Intersection areas of dynamic obstacles
        self.dynamic_collision = []          # Collision information with dynamic obstacles