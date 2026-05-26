from isaacsim import SimulationApp
config = {
    "window_width": "1920",
    "window_height": "1080",
    "headless": False,
}
simulation_app = SimulationApp(config)

import numpy as np
import yaml
import py_trees

from omni.isaac.core import World
from tasks.custom_pick_place import CustomPickPlace
from controllers.pick_controller import PickController
from controllers.place_controller import PlaceController
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import get_stage_units


coordinate_dict = {
    "first_cube": np.array([0.3, 0.3, 0.04]),
    "second_cube": np.array([0.4, -0.1, 0.04]),
    "third_cube": np.array([0.35, -0.2, 0.04]),
    "bin": np.array([-0.3, -0.3, 0.5]),
    "second_target": np.array([0.4, 0.3, 0.05]),
    "third_target": np.array([0.4, 0.3, 0.15]), 
}

colors = [np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 0])]


cube_initial_positions = np.array([
    [0.3, 0.3, 0.3],
    [0.4, -0.1, 0.3],
    [0.35, -0.2, 0.3]
])
my_world = World(stage_units_in_meters=1.0)
pick_place_task = CustomPickPlace(
    cube_initial_positions=cube_initial_positions,
    target_position=np.array([-0.3, -0.3, 0.5]),
    colors=colors
)
my_world.add_task(pick_place_task)
my_world.reset()

task_params = pick_place_task.get_params()

cube_keys = {}
for i, cube in enumerate(task_params["cubes"]):
    key = "cube" if i == 0 else "cube" + str(i+1)
    cube_keys[key] = cube["cube_name"]

# Get the Franka robot object from the scene.
robot_key = task_params["robot_name"]["value"]
my_franka = my_world.scene.get_object(robot_key)

# Optionally, add a bin (for placing) to the stage.
stage_utils.add_reference_to_stage(
    usd_path="/home/*/Documents/bin.usd",
    prim_path="/World/Bin"
)
bt_prim = Articulation(prim_path="/World/Bin", name="bin_prim", position=(-0.3, -0.3, 0.1))

# Create controllers for picking and placing.
pick_controller = PickController(
    name="pick_controller",
    gripper=my_franka.gripper,
    robot_articulation=my_franka,
)
place_controller = PlaceController(
    name="place_controller",
    gripper=my_franka.gripper,
    robot_articulation=my_franka,
)
articulation_controller = my_franka.get_articulation_controller()

# Define PickBehavior and PlaceBehavior behaviors.
class PickBehavior(py_trees.behaviour.Behaviour):
    """
    Behavior to execute the picking phase.
    Uses the provided object parameter (e.g., "first_cube", "second_cube", "third_cube")
    to determine which cube to pick.
    """
    def __init__(self, name="Pick", object_param=None):
        super(PickBehavior, self).__init__(name)
        self.initialised = False
        self.object_param = object_param

    def initialise(self):
        self.logger.info("[Pick] Initialising pick task")
        pick_controller.reset()
        self.initialised = True

    def update(self):
        observations = my_world.get_observations()
        current_joint_positions = observations[robot_key]["joint_positions"]

        cube_key = cube_keys.get(self.object_param, list(cube_keys.values())[0])

        if self.object_param in coordinate_dict:
            pick_pos = coordinate_dict[self.object_param]
        else:
            pick_pos = observations[cube_key]["position"]

        actions = pick_controller.forward(
            picking_position=pick_pos,
            current_joint_positions=current_joint_positions,
            end_effector_offset=np.array([0, 0.005, 0]),
        )
        articulation_controller.apply_action(actions)

        if pick_controller.is_done():
            self.logger.info("[Pick] Pick task completed for " + cube_key)
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

class PlaceBehavior(py_trees.behaviour.Behaviour):
    """
    Behavior to execute the placing phase.
    Uses the location parameter for the target place position and the object parameter to identify the cube.
    """
    def __init__(self, name="Place", location_param=None, object_param=None):
        super(PlaceBehavior, self).__init__(name)
        self.initialised = False
        self.location_param = location_param
        self.object_param = object_param

    def initialise(self):
        self.logger.info("[Place] Initialising place task")
        place_controller.reset()
        self.initialised = True

    def update(self):
        observations = my_world.get_observations()
        current_joint_positions = observations[robot_key]["joint_positions"]

        cube_key = cube_keys.get(self.object_param, list(cube_keys.values())[0])
        cube_current_pos = observations[cube_key]["position"]

        if self.location_param in coordinate_dict:
            target_pos = coordinate_dict[self.location_param]
        else:
            target_pos = observations[cube_key]["target_position"]

        actions = place_controller.forward(
            picking_position=cube_current_pos,
            placing_position=target_pos,
            current_joint_positions=current_joint_positions,
            end_effector_offset=np.array([0, 0.005, 0]),
        )
        articulation_controller.apply_action(actions)

        if place_controller.is_done():
            self.logger.info("[Place] Place task completed for " + cube_key)
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

# Define the behavior tree in YAML
tree_yaml = """
root:
  type: Sequence
  name: MultiPickAndPlaceSequence
  children:
    - type: Sequence
      name: Cube1PickAndPlace
      children:
        - type: PickBehavior
          name: PickCube1
          object: "first_cube"
        - type: PlaceBehavior
          name: PlaceCube1
          location: "bin"
          object: "first_cube"
    - type: Sequence
      name: Cube2PickAndPlace
      children:
        - type: PickBehavior
          name: PickCube2
          object: "second_cube"
        - type: PlaceBehavior
          name: PlaceCube2
          location: "second_target"
          object: "second_cube"
    - type: Sequence
      name: Cube3PickAndPlace
      children:
        - type: PickBehavior
          name: PickCube3
          object: "third_cube"
        - type: PlaceBehavior
          name: PlaceCube3
          location: "third_target"
          object: "third_cube"
"""

tree_data = yaml.safe_load(tree_yaml)

def build_tree(node_dict):
    node_type = node_dict.get("type")
    node_name = node_dict.get("name")
    if node_type == "Sequence":
        node = py_trees.composites.Sequence(name=node_name, memory=True)
        for child in node_dict.get("children", []):
            child_node = build_tree(child)
            node.add_child(child_node)
        return node
    elif node_type == "PickBehavior":
        obj_param = node_dict.get("object", None)
        return PickBehavior(name=node_name, object_param=obj_param)
    elif node_type == "PlaceBehavior":
        loc_param = node_dict.get("location", None)
        obj_param = node_dict.get("object", None)
        return PlaceBehavior(name=node_name, location_param=loc_param, object_param=obj_param)
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
            pick_controller.reset()
            place_controller.reset()
            task_completed = False
        tree.tick()
        if tree.root.status == py_trees.common.Status.SUCCESS:
            print("Behavior tree completed. Tasks will no longer restart.")
            task_completed = True

simulation_app.close()
