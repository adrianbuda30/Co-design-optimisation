# import numpy as np
# import random

# min_arm_length = [0.1, 0.1, 0.1, 0.1]
# max_arm_length = [1, 1, 1, 1]
# sample = [2, 2, 2, 2]
# z = 2  # Number of standard deviations to cover
# for _ in range(8):
#     initial_mean = np.array([min_val + (max_val - min_val) * np.random.rand() 
#                     for min_val, max_val in zip(min_arm_length, max_arm_length)])
#     initial_std = np.array([(max_val - min_val) / (2 * z) 
#                     for min_val, max_val, max_val in zip(min_arm_length, max_arm_length, max_arm_length)])
    
#     while True:
#         sample = np.random.normal(initial_mean, initial_std)
#         if np.all(sample >= min_arm_length) and np.all(sample <= max_arm_length):
#                 break  # Exit the loop if the sample is within the range
#     print(initial_mean, initial_std, sample)    
    
# import torch
# from stable_baselines3 import PPO
# import numpy as np

# path = "/home/divij/Documents/quadopter/src/model_dynamics/rl/trained_model/QuadcopterCorrect_Hebo_callback_chop_Tanh_Tsteps_122880000_lr_0.0001_hidden_sizes_256_reward_1.00.250.00.250.250.25_varPoleMass_GaussMix_preOpt"
# trained_model = PPO.load(path)

# obs = np.array([0.1, 0.1, 0.2, 0.3, 0.4, 2, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 0.0])

# # Predict the action using the model
# action, _ = trained_model.predict(obs)

# # Compute the value using the value network
# obs_tensor = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)

# with torch.no_grad():
#     # Pass the observation through the shared layers
#     extract_features = trained_model.policy.features_extractor(obs_tensor)
#     actor_features, critic_features = trained_model.policy.mlp_extractor(extract_features)  # Unpacking the tuple
    
#     # Pass the critic features through the value head
#     value = trained_model.policy.value_net(critic_features).item()

# print(f"Observation: {obs}")
# print(f"Action: {action}")
# print(f"Value: {value}")
# import numpy as np

# def cartesian_to_spherical(pos, origin):
#     # Translate the position so that the origin is at (0, 0, 0)
#     x, y, z = pos - origin

#     # Convert to spherical
#     r = np.sqrt(x**2 + y**2 + z**2)
#     theta = np.arccos(z / r) if r != 0 else 0
#     phi = np.arctan2(y, x)
    
#     return r, theta, phi

# # Example
# pos_world = np.array([0, 0, 0]) # Origin
# point = np.array([0, 2, 0]) # Point in space

# r, theta, phi = cartesian_to_spherical(point, pos_world)
# print(f"r: {r}, theta: {theta}, phi: {phi}")

# import numpy as np
# import matplotlib.pyplot as plt

# def cartesian_to_spherical_with_orientation(point, origin, R):
#     # 1. Translate the point to make origin as (0, 0, 0)
#     translated_point = point - origin

#     # 2. Convert the translated point to spherical coordinates
#     x, y, z = translated_point
#     r = np.sqrt(x**2 + y**2 + z**2)
    
#     # 3. Convert the spherical coordinates' angles with orientation in mind
#     rotated_point = np.dot(R.reshape(3,3), translated_point)
#     x_rot, y_rot, z_rot = rotated_point
#     theta = np.arccos(z_rot / r) if r != 0 else 0
#     phi = np.arctan2(y_rot, x_rot)
    
#     return r, theta, phi

# pos_world = np.array([0, 0, 0])  # Position of the quadcopter
# R = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])  # Rotation matrix of the quadcopter
# point = np.array([0, 0, 1])  # Trajectory point

# r, theta, phi = cartesian_to_spherical_with_orientation(point, pos_world, R)

# # Visualization
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Draw vector from pos_world to point
# ax.quiver(pos_world[0], pos_world[1], pos_world[2], point[0], point[1], point[2], arrow_length_ratio=0.1)

# # Annotate the distance and angles
# ax.text((pos_world[0] + point[0]) / 2,
#         (pos_world[1] + point[1]) / 2,
#         (pos_world[2] + point[2]) / 2, f'r = {r:.2f}', color='red')

# ax.text(pos_world[0] + 0.2, pos_world[1] + 0.2, pos_world[2], f'θ = {np.degrees(theta):.2f}°', color='green')
# ax.text(pos_world[0], pos_world[1] + 0.2, pos_world[2] + 0.2, f'φ = {np.degrees(phi):.2f}°', color='blue')

# ax.set_title('Visualization in Spherical Coordinates with Vector')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_xlim([0, 3])
# ax.set_ylim([0, 3])
# ax.set_zlim([0, 3])
# ax.grid(False)
# plt.show()

# print(f"r: {r}, theta: {theta}, phi: {phi}")

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# class QuadcopterTrajectory:
#     def __init__(self):
#         # Randomize the coefficients a, b, c (same for x, y, z)
#         self.a, self.b, self.c = np.random.uniform(-1, 1, 3)
#         # For a closed loop, d = 0 and e = 2a + b (for x, y, z)
#         self.d = 0
#         self.e = 2*self.a + self.b
#         # Similarly, for y and z
#         self.f, self.g, self.h = np.random.uniform(-1, 1, 3)
#         self.i = 0
#         self.j = 2*self.f + self.g
#         # And for z
#         self.k, self.l, self.m = np.random.uniform(-1, 1, 3)
#         self.n = 0
#         self.o = 2*self.k + self.l

#     def get_trajectory_point(self, t):
#         x = self.a*t**3 + self.b*t**2 + self.c*t + self.d
#         y = self.e*t**3 + self.f*t**2 + self.g*t + self.h
#         z = self.i*t**3 + self.j*t**2 + self.k*t + self.l
#         return np.array([x, y, z])

# def plot_trajectory(ax, trajectory):
#     t_values = np.linspace(0, 1, 2500)
#     points = [trajectory.get_trajectory_point(t) for t in t_values]
#     x = [point[0] for point in points]
#     y = [point[1] for point in points]
#     z = [point[2] for point in points]
#     # Highlight the start point
#     ax.scatter(x[0], y[0], z[0], c='red', s=100, marker='o')  # s is the size, and c is the color

#     ax.plot(x, y, z)

# # Create the figure and axes outside the loop
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot multiple trajectories on the same axes
# for _ in range(10):
#     trajectory = QuadcopterTrajectory()
#     plot_trajectory(ax, trajectory)

# # Setting title and labels once
# ax.set_title('3D Cubic Polynomial Loops')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# class QuadcopterTrajectory:
#     def __init__(self):
#         # Randomize the coefficients a, b, c (same for x, y, z)
#         self.a, self.b, self.c = np.random.uniform(-1, 1, 3)
#         # For a closed loop, d = 0 and e = 2a + b (for x, y, z)
#         self.d = 0
#         self.e = 2*self.a + self.b
#         # Similarly, for y and z
#         self.f, self.g, self.h = np.random.uniform(-1, 1, 3)
#         self.i = 0
#         self.j = 2*self.f + self.g
#         # And for z
#         self.k, self.l, self.m = np.random.uniform(-1, 1, 3)
#         self.n = 0
#         self.o = 2*self.k + self.l

#     def get_trajectory_point(self, t):
#         x = self.a*t**3 + self.b*t**2 + self.c*t + self.d
#         y = self.e*t**3 + self.f*t**2 + self.g*t + self.h
#         z = self.i*t**3 + self.j*t**2 + self.k*t + self.l
#         return np.array([x, y, z])

#     def get_trajectory_velocity(self, t):
#         v_x = 3*self.a*t**2 + 2*self.b*t + self.c
#         v_y = 3*self.e*t**2 + 2*self.f*t + self.g
#         v_z = 3*self.i*t**2 + 2*self.j*t + self.k
#         return np.array([v_x, v_y, v_z])
    
# class QuadcopterCircularTrajectory:

#     def __init__(self, center=np.array([0, 0, 0]), radius=1, height=0, omega=2*np.pi):
#         # Circle parameters
#         self.h, self.k, self.z_const = center  # (h, k) is the center of the circle in xy-plane, z_const is the constant z-height
#         self.radius = radius
#         self.omega = omega  # Angular velocity for one full circle in 1 time unit
#         self.height = height

#     def get_trajectory_point(self, t):
#         # Compute x and y using parametric equations for a circle
#         x = self.h + self.radius * np.cos(self.omega * t)
#         y = self.k + self.radius * np.sin(self.omega * t)
#         z = self.z_const + self.height  # Assuming constant z for circle in xy-plane
#         return np.array([x, y, z])

#     def get_trajectory_velocity(self, t):
#         # Compute derivatives of x and y w.r.t. t
#         v_x = -self.radius * self.omega * np.sin(self.omega * t)
#         v_y = self.radius * self.omega * np.cos(self.omega * t)
#         v_z = 0  # No change in z direction as the height is constant
#         return np.array([v_x, v_y, v_z])

#     def get_approx_velocity(self, t, dt=1e-5):
#         current_point = self.get_trajectory_point(t)
#         previous_point = self.get_trajectory_point(t-dt)
#         velocity = (current_point - previous_point) / dt
#         return velocity

# def plot_trajectory_and_velocity(ax, trajectory):
#     max_steps = 2500
#     max_time = 10
#     ratio = max_time / max_steps
#     print("ratio: ", ratio)
#     t_values = np.linspace(0, max_time, max_steps)
#     points = [trajectory.get_trajectory_point(t) for t in t_values]
#     velocities = [trajectory.get_trajectory_velocity(t) for t in t_values]
#     approx_velocities = [trajectory.get_approx_velocity(t, dt=ratio) for t in t_values]
#     bound = 5
#     points = np.array(points)
#     is_in_bounds = not (np.any(points>bound) or np.any(points<-bound))
#     if not is_in_bounds:
#         print("not in bounds")

#     print(len(points), len(velocities))

#     x = [point[0] for point in points]
#     y = [point[1] for point in points]
#     z = [point[2] for point in points]
    
#     u = [velocity[0] for velocity in velocities]
#     v = [velocity[1] for velocity in velocities]
#     w = [velocity[2] for velocity in velocities]

#     approx_u = [velocity[0] for velocity in approx_velocities]
#     approx_v = [velocity[1] for velocity in approx_velocities]
#     approx_w = [velocity[2] for velocity in approx_velocities]

#     max_approx_velocity = np.max(np.linalg.norm(approx_velocities, axis=1))
#     print("max approx velocity: ", max_approx_velocity)

#     # for velocity in velocities:
#     #     print("velocity: ", velocity)
#     # for point in points:
#     #     print("point: ", point)

#     max_velocity = np.max(np.linalg.norm(velocities, axis=1))
#     print("max velocity: ", max_velocity)
#     obs_velocity = np.array([(abs(v1) - abs(v2)) for i, v1 in enumerate(velocities) for v2 in velocities[i+1:i+11]]).flatten()
#     print(np.max(obs_velocity))
    
#     # Highlight the start point
#     ax.scatter(x[0], y[0], z[0], c='red', s=100, marker='o')
#     ax.plot(x, y, z)

#     # Plotting velocity vectors every 100th step for clarity
#     stride = 100
#     ax.quiver(x[::stride], y[::stride], z[::stride], u[::stride], v[::stride], w[::stride], length=0.1, normalize=True, color='blue')


# def is_in_bounds(trajectory, bound):
#     t_values = np.linspace(0, 1, 2800)
#     points = [trajectory.get_trajectory_point(t) for t in t_values]
#     velocities = [trajectory.get_trajectory_velocity(t) for t in t_values]
#     points = np.array(points)
#     is_in_bounds = not (np.any(points>bound) or np.any(points<-bound))
#     if not is_in_bounds:
#         print("not in bounds")

# # Create the figure and axes outside the loop
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # Plot multiple trajectories on the same axes
# bound = 5
# for _ in range(1):
#     trajectory = QuadcopterCircularTrajectory(center=np.array([0, 0, 0]), radius=1, height=0, omega = 2*np.pi)
#     plot_trajectory_and_velocity(ax, trajectory)
#     # is_in_bounds(trajectory, bound)

# # Setting title and labels once
# ax.set_title('3D Cubic Polynomial Loops with Velocity Vectors')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()


# import numpy as np

# def gradient_direction(x, y):
#     theta = np.arctan2(y, x)
#     tangent_theta = theta + np.pi / 2
    
#     dx = np.cos(tangent_theta)
#     dy = np.sin(tangent_theta)
    
#     return np.array([dx, dy])

# def is_inside_torus(x, y, z, R, r):
#     d = np.sqrt(x**2 + y**2)
#     return (d - R)**2 + z**2 < r**2

# # Example usage:
# R, r = 4, 1  # Sample torus parameters
# x, y, z = 3, 3, 0  # Sample point
# print(is_inside_torus(x, y, z, R, r))  # This will return either True (if inside) or False


# # Example usage:
# x, y = 0, 1  # Point on the circle
# direction = gradient_direction(x, y)
# print(direction)


import numpy as np
import matplotlib.pyplot as plt

def gradient_direction(x, y):
    theta = np.arctan2(y, x)
    tangent_theta = theta + np.pi / 2
    
    dx = np.cos(tangent_theta)
    dy = np.sin(tangent_theta)
    
    return np.array([dx, dy])

# Generate points on a circle
theta_values = np.linspace(0, 2*np.pi, 100)
x_values = np.cos(theta_values)
y_values = np.sin(theta_values)

# Calculate gradient directions
dx_values = []
dy_values = []
for x, y in zip(x_values, y_values):
    dx, dy = gradient_direction(x, y)
    dx_values.append(dx)
    dy_values.append(dy)

# Plot circle and gradient directions
plt.figure()
plt.plot(x_values, y_values, label='Circle', color='blue')
plt.quiver(x_values, y_values, dx_values, dy_values, angles='xy', scale_units='xy', scale=0.01, color='red')
plt.title("Gradient Directions on a Circle")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
