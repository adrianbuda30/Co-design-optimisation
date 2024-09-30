import numpy as np

class QuadcopterTrajectory:
    def __init__(self):
        # Randomize the coefficients a, b, c (same for x, y, z)
        self.a, self.b, self.c = np.random.uniform(-1, 1, 3)
        # For a closed loop, d = 0 and e = 2a + b (for x, y, z)
        self.d = 0
        self.e = 2*self.a + self.b
        # Similarly, for y and z
        self.f, self.g, self.h = np.random.uniform(-1, 1, 3)
        self.i = 0
        self.j = 2*self.f + self.g
        # And for z
        self.k, self.l, self.m = np.random.uniform(-1, 1, 3)
        self.n = 0
        self.o = 2*self.k + self.l

    def get_trajectory_point(self, t):
        x = self.a*t**3 + self.b*t**2 + self.c*t + self.d
        y = self.e*t**3 + self.f*t**2 + self.g*t + self.h
        z = self.i*t**3 + self.j*t**2 + self.k*t + self.l
        return np.array([x, y, z])

    def get_trajectory_velocity(self, t):
        v_x = 3*self.a*t**2 + 2*self.b*t + self.c
        v_y = 3*self.e*t**2 + 2*self.f*t + self.g
        v_z = 3*self.i*t**2 + 2*self.j*t + self.k
        return np.array([v_x, v_y, v_z])
    

class QuadcopterCircularTrajectory:
    def __init__(self, center=np.array([0, 0, 0]), radius=1, height=0):
        # Circle parameters
        self.h, self.k, self.z_const = center  # (h, k) is the center of the circle in xy-plane, z_const is the constant z-height
        self.radius = radius
        self.omega = 2 * np.pi  # Angular velocity for one full circle in 1 time unit
        self.height = height
    def get_trajectory_point(self, t):
        # Compute x and y using parametric equations for a circle
        x = self.h + self.radius * np.cos(self.omega * t)
        y = self.k + self.radius * np.sin(self.omega * t)
        z = self.z_const + self.height  # Assuming constant z for circle in xy-plane
        return np.array([x, y, z])

    def get_trajectory_velocity(self, t):
        # Compute derivatives of x and y w.r.t. t
        v_x = -self.radius * self.omega * np.sin(self.omega * t)
        v_y = self.radius * self.omega * np.cos(self.omega * t)
        v_z = 0  # No change in z direction as the height is constant
        return np.array([v_x, v_y, v_z])
