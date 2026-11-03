from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name


class BaseCustomStacking(ABC, BaseTask):
    """Base class for custom stacking tasks.

    Args:
        name (str): Task name.
        cube_initial_positions (np.ndarray): Initial positions for the cubes.
        cube_initial_orientations (Optional[np.ndarray], optional): Initial orientations for the cubes.
            Defaults to None.
        stack_target_position (Optional[np.ndarray], optional): The target stacking position. Defaults to None.
        cube_size (Optional[np.ndarray], optional): Size of the cubes. Defaults to None.
        offset (Optional[np.ndarray], optional): Offset to be applied. Defaults to None.
        cube_colors (Optional[List[np.ndarray]], optional): List of colors (RGB) for each cube.
            Defaults to None, which sets all cubes to blue.
    """

    def __init__(
        self,
        name: str,
        cube_initial_positions: np.ndarray,
        cube_initial_orientations: Optional[np.ndarray] = None,
        stack_target_position: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        cube_colors: Optional[List[np.ndarray]] = None,
    ) -> None:
        BaseTask.__init__(self, name=name, offset=offset)
        self._robot = None
        self._num_of_cubes = cube_initial_positions.shape[0]
        self._cube_initial_positions = cube_initial_positions
        self._cube_initial_orientations = cube_initial_orientations
        if self._cube_initial_orientations is None:
            self._cube_initial_orientations = [np.array([1.0, 0.0, 0.0, 0.0]) for _ in range(self._num_of_cubes)]
        self._stack_target_position = stack_target_position
        self._cube_size = cube_size
        if self._cube_size is None:
            self._cube_size = np.array([0.0515, 0.0515, 0.0515]) / get_stage_units()
        if stack_target_position is None:
            self._stack_target_position = np.array([-0.3, -0.3, 0]) / get_stage_units()
        self._stack_target_position = self._stack_target_position + self._offset

        # Set cube colors. If not provided, default to blue for all cubes.
        if cube_colors is None:
            self._cube_colors = [np.array([0.0, 0.0, 1.0]) for _ in range(self._num_of_cubes)]
        else:
            if len(cube_colors) != self._num_of_cubes:
                raise ValueError("Length of cube_colors must match the number of cubes.")
            self._cube_colors = cube_colors

        self._cubes = []
        return

    def set_up_scene(self, scene: Scene) -> None:
        """Sets up the scene for the stacking task.

        Args:
            scene (Scene): The simulation scene.
        """
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        for i in range(self._num_of_cubes):
            color = self._cube_colors[i]
            cube_prim_path = find_unique_string_name(
                initial_name="/World/Cube", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
            cube_name = find_unique_string_name(
                initial_name="cube", is_unique_fn=lambda x: not self.scene.object_exists(x)
            )
            self._cubes.append(
                scene.add(
                    DynamicCuboid(
                        name=cube_name,
                        position=self._cube_initial_positions[i],
                        orientation=self._cube_initial_orientations[i],
                        prim_path=cube_prim_path,
                        scale=self._cube_size,
                        size=1.0,
                        color=color,
                    )
                )
            )
            self._task_objects[self._cubes[-1].name] = self._cubes[-1]
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
        cube_name: Optional[str] = None,
        cube_position: Optional[str] = None,
        cube_orientation: Optional[str] = None,
        stack_target_position: Optional[str] = None,
    ) -> None:
        """Set parameters for the stacking task.

        Args:
            cube_name (Optional[str], optional): Cube name. Defaults to None.
            cube_position (Optional[str], optional): Cube position. Defaults to None.
            cube_orientation (Optional[str], optional): Cube orientation. Defaults to None.
            stack_target_position (Optional[str], optional): Stacking target position. Defaults to None.
        """
        if stack_target_position is not None:
            self._stack_target_position = stack_target_position
        if cube_name is not None:
            self._task_objects[cube_name].set_local_pose(position=cube_position, orientation=cube_orientation)
        return

    def get_params(self) -> dict:
        """Gets the parameters of the stacking task.

        Returns:
            dict: A dictionary with task parameters.
        """
        params_representation = dict()
        params_representation["stack_target_position"] = {"value": self._stack_target_position, "modifiable": True}
        params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation

    def get_observations(self) -> dict:
        """Gets observations from the stacking task.

        Returns:
            dict: Observations dictionary.
        """
        joints_state = self._robot.get_joints_state()
        end_effector_position, _ = self._robot.end_effector.get_local_pose()
        observations = {
            self._robot.name: {
                "joint_positions": joints_state.positions,
                "end_effector_position": end_effector_position,
            }
        }
        for i in range(self._num_of_cubes):
            cube_position, cube_orientation = self._cubes[i].get_local_pose()
            observations[self._cubes[i].name] = {
                "position": cube_position,
                "orientation": cube_orientation,
                "target_position": np.array(
                    [
                        self._stack_target_position[0],
                        self._stack_target_position[1],
                        (self._cube_size[2] * i) + self._cube_size[2] / 2.0,
                    ]
                ),
            }
        return observations

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        return

    def post_reset(self) -> None:
        """Post-reset callback."""
        from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper

        if isinstance(self._robot.gripper, ParallelGripper):
            self._robot.gripper.set_joint_positions(self._robot.gripper.joint_opened_positions)
        return

    def get_cube_names(self) -> List[str]:
        cube_names = []
        for i in range(self._num_of_cubes):
            cube_names.append(self._cubes[i].name)
        return cube_names

    def calculate_metrics(self) -> dict:
        raise NotImplementedError

    def is_done(self) -> bool:
        raise NotImplementedError
