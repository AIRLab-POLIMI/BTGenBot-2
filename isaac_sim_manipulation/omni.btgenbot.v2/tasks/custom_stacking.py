from typing import Optional

import numpy as np
from tasks.base_custom_stacking import BaseCustomStacking
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.franka import Franka


class CustomStacking(BaseCustomStacking):
    """Custom Stacking Task for the Franka robot.

    Args:
        name (str, optional): Task name. Defaults to "franka_stacking".
        target_position (Optional[np.ndarray], optional): The target stacking position. Defaults to None.
        cube_initial_positions (Optional[np.ndarray], optional): Initial positions for the cubes.
            Defaults to None, which falls back to a default 2-cube configuration.
        cube_initial_orientations (Optional[np.ndarray], optional): Initial orientations for the cubes.
            Defaults to None.
        cube_size (Optional[np.ndarray], optional): Cube size. Defaults to None.
        cube_colors (Optional[List[np.ndarray]], optional): List of colors (RGB) for each cube.
            Defaults to None, which sets all cubes to blue.
        offset (Optional[np.ndarray], optional): Offset for the task. Defaults to None.
    """

    def __init__(
        self,
        name: str = "franka_stacking",
        target_position: Optional[np.ndarray] = None,
        cube_initial_positions: Optional[np.ndarray] = None,
        cube_initial_orientations: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = None,
        cube_colors: Optional[list] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        if target_position is None:
            target_position = np.array([0.5, 0.5, 0]) / get_stage_units()
        if cube_initial_positions is None:
            cube_initial_positions = np.array([[0.3, 0.3, 0.3], [0.3, -0.3, 0.3]]) / get_stage_units()
        BaseCustomStacking.__init__(
            self,
            name=name,
            cube_initial_positions=cube_initial_positions,
            cube_initial_orientations=cube_initial_orientations,
            stack_target_position=target_position,
            cube_size=cube_size,
            offset=offset,
            cube_colors=cube_colors,
        )

    def set_robot(self) -> Franka:
        """Sets the robot for the stacking task.

        Returns:
            Franka: The Franka robot instance.
        """
        franka_prim_path = find_unique_string_name(
            initial_name="/World/Franka", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        franka_robot_name = find_unique_string_name(
            initial_name="my_franka", is_unique_fn=lambda x: not self.scene.object_exists(x)
        )
        return Franka(prim_path=franka_prim_path, name=franka_robot_name)
