from isaacsim import SimulationApp
config = {"headless": False}
simulation_app = SimulationApp(config)

import numpy as np
import yaml
import py_trees

from omni.isaac.core import World
from tasks.custom_stacking import CustomStacking
from controllers.custom_stacking_controller import CustomStackingController

# -----------------------------------------------------------------------------
# Create the simulation world.
# -----------------------------------------------------------------------------
my_world = World(stage_units_in_meters=1.0)

# -----------------------------------------------------------------------------
# Create a blue stacking task (two blue blocks).
# -----------------------------------------------------------------------------
blue_cubes = np.array([
    [0.3, 0.3, 0.03],   # Block 1 initial position
    [0.3, 0.2, 0.03]    # Block 2 initial position
])
blue_colors = [
    np.array([0, 0, 1]),  # Blue
    np.array([0, 0, 1])
]
blue_target = np.array([0.3, 0.3, 0])  # Stacking target for blue pile

blue_task = CustomStacking(
    name="blue_stacking",
    target_position=blue_target,
    cube_initial_positions=blue_cubes,
    cube_colors=blue_colors
)
my_world.add_task(blue_task)

# -----------------------------------------------------------------------------
# Create a red stacking task (two red blocks) in a separate pile.
# -----------------------------------------------------------------------------
red_cubes = np.array([
    [0.6, 0.3, 0.03],   # Block 1 initial position
    [0.6, 0.2, 0.03]    # Block 2 initial position
])
red_colors = [
    np.array([1, 0, 0]),  # Red
    np.array([1, 0, 0])
]
red_target = np.array([0.55, -0.55, 0])  # Stacking target for red pile

red_task = CustomStacking(
    name="red_stacking",
    target_position=red_target,
    cube_initial_positions=red_cubes,
    cube_colors=red_colors
)

# -----------------------------------------------------------------------------
# Set flag: use_red = True to add red task, False to run blue task only.
# -----------------------------------------------------------------------------
use_red = True
if use_red:
    my_world.add_task(red_task)

# -----------------------------------------------------------------------------
# Reset the world so that the tasks initialize their objects.
# -----------------------------------------------------------------------------
my_world.reset()

# --- Ensure proper initialization by calling post_reset() only on tasks that were added.
blue_task.post_reset()
if use_red:
    red_task.post_reset()

# -----------------------------------------------------------------------------
# Retrieve the robot object(s) and create controller(s).
# -----------------------------------------------------------------------------
# Blue task robot and controller.
blue_robot_name = blue_task.get_params()["robot_name"]["value"]
blue_franka = my_world.scene.get_object(blue_robot_name)
blue_controller = CustomStackingController(
    name="blue_stacking_controller",
    gripper=blue_franka.gripper,
    robot_articulation=blue_franka,
    picking_order_cube_names=blue_task.get_cube_names(),
    robot_observation_name=blue_robot_name,
)
blue_articulation_controller = blue_franka.get_articulation_controller()

if use_red:
    # Red task robot and controller.
    red_robot_name = red_task.get_params()["robot_name"]["value"]
    red_franka = my_world.scene.get_object(red_robot_name)
    red_controller = CustomStackingController(
        name="red_stacking_controller",
        gripper=red_franka.gripper,
        robot_articulation=red_franka,
        picking_order_cube_names=red_task.get_cube_names(),
        robot_observation_name=red_robot_name,
    )
    red_articulation_controller = red_franka.get_articulation_controller()

# -----------------------------------------------------------------------------
# Define a StackBehavior that wraps a stacking controller.
# It accepts parameters for logging (picking and placing) and uses the provided
# controller and its articulation controller.
# -----------------------------------------------------------------------------
class StackBehavior(py_trees.behaviour.Behaviour):
    def __init__(self, name="Stack", picking_param=None, placing_param=None,
                 controller=None, articulation_controller=None):
        super(StackBehavior, self).__init__(name)
        self.initialised = False
        self.picking_param = picking_param
        self.placing_param = placing_param
        self.controller = controller
        self.articulation_controller = articulation_controller

    def initialise(self):
        self.logger.info("[{}] Initialising stacking task with picking: {} and placing: {}"
                         .format(self.name, self.picking_param, self.placing_param))
        self.controller.reset()
        self.initialised = True

    def update(self):
        observations = my_world.get_observations()
        actions = self.controller.forward(observations=observations)
        self.articulation_controller.apply_action(actions)
        if self.controller.is_done():
            self.logger.info("[{}] Stacking task completed".format(self.name))
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

# -----------------------------------------------------------------------------
# Build a behavior tree using a YAML description.
# If only blue is used, the tree contains one node.
# If red is used as well, the tree contains two nodes in sequence.
# -----------------------------------------------------------------------------
if use_red:
    tree_yaml = """
    root:
      type: Sequence
      name: StackSequence
      children:
        - type: StackBehavior
          name: BlueStack
          picking: "cube"
          placing: "stack"
          controller: "blue"
        - type: StackBehavior
          name: RedStack
          picking: "cube"
          placing: "stack"
          controller: "red"
    """
else:
    tree_yaml = """
    root:
      type: Sequence
      name: StackSequence
      children:
        - type: StackBehavior
          name: BlueStack
          picking: "cube"
          placing: "stack"
          controller: "blue"
    """

def build_tree(node_dict):
    node_type = node_dict.get("type")
    node_name = node_dict.get("name")
    if node_type == "Sequence":
        node = py_trees.composites.Sequence(name=node_name, memory=True)
        for child in node_dict.get("children", []):
            child_node = build_tree(child)
            node.add_child(child_node)
        return node
    elif node_type == "StackBehavior":
        picking_param = node_dict.get("picking", None)
        placing_param = node_dict.get("placing", None)
        ctrl_key = node_dict.get("controller")
        if ctrl_key == "blue":
            controller = blue_controller
            articulation_controller = blue_articulation_controller
        elif ctrl_key == "red":
            controller = red_controller
            articulation_controller = red_articulation_controller
        else:
            raise ValueError("Unknown controller key: " + str(ctrl_key))
        return StackBehavior(name=node_name, picking_param=picking_param, placing_param=placing_param,
                             controller=controller, articulation_controller=articulation_controller)
    else:
        raise ValueError("Unknown node type: " + str(node_type))

tree_data = yaml.safe_load(tree_yaml)
root_node = build_tree(tree_data["root"])
tree = py_trees.trees.BehaviourTree(root=root_node)
print(py_trees.display.ascii_tree(root_node))

# -----------------------------------------------------------------------------
# Main simulation loop: tick the behavior tree until the stacking task(s) complete.
# -----------------------------------------------------------------------------
task_completed = False
while simulation_app.is_running():
    my_world.step(render=True)
    if not task_completed:
        tree.tick()
        if tree.root.status == py_trees.common.Status.SUCCESS:
            print("Behavior tree completed. Stacking task(s) finished.")
            task_completed = True

simulation_app.close()
