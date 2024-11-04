import mujoco_py
import os

# Define the path to your Walker model XML file
walker_xml_path = "/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/walker/assets/walker2d.xml"

# Ensure the path exists
assert os.path.exists(walker_xml_path), "The specified XML file does not exist."

# Load the model from the XML file
model = mujoco_py.load_model_from_path(walker_xml_path)

# Create a simulation object
sim = mujoco_py.MjSim(model)

# Create a viewer object
viewer = mujoco_py.MjViewer(sim)

# Run the simulation and visualize the model
while True:
    sim.step()
    viewer.render()
