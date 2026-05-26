from abc import ABC, abstractmethod
from typing import Optional, List

import numpy as np
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name


class BaseCustomPickPlace(ABC, BaseTask):
    """Custom Pick and Place task that supports initialization with multiple cubes.

    Args:
        name (str): Name of the task.
        cube_initial_positions (np.ndarray): Array of initial positions for cubes (shape: [num_cubes, 3]).
        cube_initial_orientations (Optional[np.ndarray], optional): Array or list of initial orientations for cubes.
            If not provided, defaults to the identity quaternion for each cube.
        target_position (Optional[np.ndarray], optional): Base target position for placing cubes.
            Defaults to np.array([-0.3, -0.3, 0]) / get_stage_units() with the z-coordinate set to half the cube height.
        cube_size (Optional[np.ndarray], optional): Size of each cube.
            Defaults to np.array([0.0515, 0.0515, 0.0515]) / get_stage_units() if not provided.
        offset (Optional[np.ndarray], optional): Offset applied to the target position. Defaults to None.
        colors (Optional[List[np.ndarray]]): List of colors for each cube; if not provided, defaults to blue.
    """
    def __init__(
        self,
        name: str,
        cube_initial_positions: Optional[np.ndarray] = None,
        cube_initial_orientations: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        colors: Optional[List[np.ndarray]] = None,
    ) -> None:
        BaseTask.__init__(self, name=name, offset=offset)
        # Set default cube initial positions if none provided.
        if cube_initial_positions is None:
            cube_initial_positions = np.array([[0.3, 0.3, 0.3], [0.3, 0.0, 0.3]]) / get_stage_units()
        self._cube_initial_positions = cube_initial_positions
        self._num_of_cubes = cube_initial_positions.shape[0]
        
        # Set default cube orientations (identity quaternion) for each cube if none provided.
        if cube_initial_orientations is None:
            self._cube_initial_orientations = [np.array([1, 0, 0, 0]) for _ in range(self._num_of_cubes)]
        else:
            self._cube_initial_orientations = cube_initial_orientations

        if cube_size is None:
            cube_size = np.array([0.0515, 0.0515, 0.0515]) / get_stage_units()
        self._cube_size = cube_size

        # Set default target position if not provided.
        if target_position is None:
            target_position = np.array([-0.3, -0.3, 0]) / get_stage_units()
            target_position[2] = self._cube_size[2] / 2.0
        self._target_position = target_position
        if offset is not None:
            self._target_position = self._target_position + offset

        if colors is None:
            self._colors = [np.array([0, 0, 1]) for _ in range(self._num_of_cubes)]
        else:
            if len(colors) != self._num_of_cubes:
                raise Exception("Length of colors list must match the number of cubes")
            self._colors = [np.array(color) for color in colors]

        self._cubes = []
        self._robot = None
        return

    def set_up_scene(self, scene: Scene) -> None:
        """Set up the scene by adding a ground plane, multiple cubes, and the robot."""
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        for i in range(self._num_of_cubes):
            cube_color = self._colors[i]
            cube_prim_path = find_unique_string_name(
                initial_name="/World/Cube", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
            cube_name = find_unique_string_name(
                initial_name="cube", is_unique_fn=lambda x: not self.scene.object_exists(x)
            )
            cube = scene.add(
                DynamicCuboid(
                    name=cube_name,
                    position=self._cube_initial_positions[i],
                    orientation=self._cube_initial_orientations[i],
                    prim_path=cube_prim_path,
                    scale=self._cube_size,
                    size=1.0,
                    color=cube_color,
                )
            )
            self._cubes.append(cube)
            self._task_objects[cube.name] = cube

        self._robot = self.set_robot()
        scene.add(self._robot)
        self._task_objects[self._robot.name] = self._robot
        self._move_task_objects_to_their_frame()
        return

    @abstractmethod
    def set_robot(self) -> None:
        raise NotImplementedError

    def set_params(
        self,
        cube_positions: Optional[List[np.ndarray]] = None,
        cube_orientations: Optional[List[np.ndarray]] = None,
        target_position: Optional[np.ndarray] = None,
    ) -> None:
        """Set new parameters for cube poses and target position.
        Args:
            cube_positions (Optional[List[np.ndarray]]): List of new positions for cubes.
            cube_orientations (Optional[List[np.ndarray]]): List of new orientations for cubes.
            target_position (Optional[np.ndarray]): New base target position.
        """
        if target_position is not None:
            self._target_position = target_position
        if cube_positions is not None:
            for i, pos in enumerate(cube_positions):
                orient = self._cube_initial_orientations[i] if cube_orientations is None else cube_orientations[i]
                self._cubes[i].set_local_pose(position=pos, orientation=orient)
        return

    def get_params(self) -> dict:
        """Return a dictionary of task parameters, including per-cube settings."""
        params_representation = dict()
        cubes_params = []
        for i, cube in enumerate(self._cubes):
            pos, orient = cube.get_local_pose()
            target_pos = np.array([
                self._target_position[0] + i * (self._cube_size[0] + 0.05),
                self._target_position[1],
                self._target_position[2]
            ])
            cubes_params.append({
                "cube_name": cube.name,
                "cube_position": {"value": pos, "modifiable": True},
                "cube_orientation": {"value": orient, "modifiable": True},
                "target_position": {"value": target_pos, "modifiable": True},
            })
        params_representation["cubes"] = cubes_params
        params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation

    def get_observations(self) -> dict:
        """Return observations for each cube and the robot."""
        observations = {}
        for i, cube in enumerate(self._cubes):
            pos, orient = cube.get_local_pose()
            target_pos = np.array([
                self._target_position[0] + i * (self._cube_size[0] + 0.05),
                self._target_position[1],
                self._target_position[2]
            ])
            observations[cube.name] = {
                "position": pos,
                "orientation": orient,
                "target_position": target_pos,
            }
        joints_state = self._robot.get_joints_state()
        end_effector_position, _ = self._robot.end_effector.get_local_pose()
        observations[self._robot.name] = {
            "joint_positions": joints_state.positions,
            "end_effector_position": end_effector_position,
        }
        return observations

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        return

    def post_reset(self) -> None:
        """Post-reset hook to, for example, reset the robot gripper."""
        from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
        if isinstance(self._robot.gripper, ParallelGripper):
            self._robot.gripper.set_joint_positions(self._robot.gripper.joint_opened_positions)
        return

    def calculate_metrics(self) -> dict:
        raise NotImplementedError

    def is_done(self) -> bool:
        raise NotImplementedError
