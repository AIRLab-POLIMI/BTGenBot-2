from isaacsim import SimulationApp
config = {"headless": False}
simulation_app = SimulationApp(config)

import numpy as np
import yaml
import py_trees
from omni.isaac.core import World
from tasks.custom_stacking import CustomStacking
from controllers.custom_stacking_controller import CustomStackingController

cubes = np.array([
    [0.3, 0.3, 0.3], 
    [0.3, -0.3, 0.3]
])

colors = [
    np.array([0, 0, 1]), 
    np.array([1, 0, 0])
]

my_world = World(stage_units_in_meters=1.0)
stacking_task = CustomStacking(cube_initial_positions=cubes, cube_colors=colors)
my_world.add_task(stacking_task)
my_world.reset()

robot_name = stacking_task.get_params()["robot_name"]["value"]
my_franka = my_world.scene.get_object(robot_name)
my_controller = CustomStackingController(
    name="stacking_controller",
    gripper=my_franka.gripper,
    robot_articulation=my_franka,
    picking_order_cube_names=stacking_task.get_cube_names(),
    robot_observation_name=robot_name,
)
articulation_controller = my_franka.get_articulation_controller()

class StackBehavior(py_trees.behaviour.Behaviour):
    """
    Behavior to execute the stacking task.
    Utilizes the stacking controller to pick cubes in order and stack them.
    The behavior takes two parameters:
      - picking: a parameter to indicate details for the picking phase.
      - placing: a parameter to indicate details for the stacking/placing phase.
    """
    def __init__(self, name="Stack", picking_param=None, placing_param=None):
        super(StackBehavior, self).__init__(name)
        self.initialised = False
        self.picking_param = picking_param
        self.placing_param = placing_param

    def initialise(self):
        self.logger.info("[Stack] Initialising stacking task with picking: {} and placing: {}".format(
            self.picking_param, self.placing_param))
        my_controller.reset() 
        self.initialised = True

    def update(self):
        observations = my_world.get_observations()
        actions = my_controller.forward(observations=observations)
        articulation_controller.apply_action(actions)
        # Check if the stacking task is complete.
        if my_controller.is_done():
            self.logger.info("[Stack] Stacking task completed")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

tree_yaml = """
root:
  type: StackBehavior
  name: Stack
  picking: "cube"
  placing: "stack"
"""

tree_data = yaml.safe_load(tree_yaml)

def build_tree(node_dict):
    node_type = node_dict.get("type")
    node_name = node_dict.get("name")
    if node_type == "StackBehavior":
        picking_param = node_dict.get("picking", None)
        placing_param = node_dict.get("placing", None)
        return StackBehavior(name=node_name, picking_param=picking_param, placing_param=placing_param)
    elif node_type == "Sequence":
        node = py_trees.composites.Sequence(name=node_name, memory=True)
        for child in node_dict.get("children", []):
            child_node = build_tree(child)
            node.add_child(child_node)
        return node
    else:
        raise ValueError("Unknown node type: " + str(node_type))

root_node = build_tree(tree_data["root"])
tree = py_trees.trees.BehaviourTree(root=root_node)
print(py_trees.display.ascii_tree(root_node))

task_completed = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not task_completed:
        task_completed = True
    if my_world.is_playing():
        if task_completed:
            my_world.reset()
            my_controller.reset()
            task_completed = False
        tree.tick()
        if tree.root.status == py_trees.common.Status.SUCCESS:
            print("Behavior tree completed. Tasks will no longer restart.")
            task_completed = True

simulation_app.close()
