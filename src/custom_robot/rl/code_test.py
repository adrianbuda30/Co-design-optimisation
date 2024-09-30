import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
# Define the FirstOrderDelay class that maintains state
class FirstOrderDelay:
    def __init__(self, alpha, initial_output=0):
        self.alpha = alpha
        self.previous_output = initial_output

    def step(self, current_input):
        current_output = (1 - self.alpha) * self.previous_output + self.alpha * current_input
        self.previous_output = current_output
        return current_output

# Instantiate the FirstOrderDelay class
filter = FirstOrderDelay(alpha=0.1667)

# Simulate a series of input values (for example, a square wave)
input_values = np.zeros(500)
input_values[25:75] = 1

# Prepare a list to store the output values
output_values = []

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, len(input_values))
ax.set_ylim(-0.1, 1.1)
line1, = ax.plot([], [], label='Commanded Torque')
line2, = ax.plot([], [], label='Actual Torque', linewidth=2)
ax.set_xlabel('Time (samples)')
ax.set_ylabel('Torque')
ax.set_title('Live Plot of First-Order Delay Response')
ax.legend()
ax.grid(True)

# This function initializes the plot with empty data
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2,

# The update function to be called for each frame of the animation
def update(frame):
    # Get the current input value
    current_input = input_values[frame]
    # Use the filter to get the current output value
    current_output = filter.step(current_input)
    # Append the current output to the list of output values
    output_values.append(current_output)
    # Update the data for the plot
    line1.set_data(range(frame + 1), input_values[:frame + 1])
    line2.set_data(range(frame + 1), output_values)
    return line1, line2,

# Create an animation object
ani = FuncAnimation(fig, update, frames=range(len(input_values)), init_func=init, blit=True)

plt.show()
