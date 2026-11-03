from typing import Optional
import numpy as np

from tasks.base_custom_pick_place import BaseCustomPickPlace
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.franka import Franka

class CustomPickPlace(BaseCustomPickPlace):
    """Custom Pick and Place task for the Franka robot supporting multiple cubes initialization.
    
    Args:
        name (str, optional): Name of the task. Defaults to "franka_custom_pick_place".
        cube_initial_positions (Optional[np.ndarray], optional): Array of initial positions for cubes.
        cube_initial_orientations (Optional[np.ndarray], optional): Array of initial orientations for cubes.
        target_position (Optional[np.ndarray], optional): Base target position for cubes.
        cube_size (Optional[np.ndarray], optional): Size of each cube.
        offset (Optional[np.ndarray], optional): Offset applied to the target position.
        colors (Optional[List[np.ndarray]], optional): List of colors for each cube; if not specified, defaults to blue.
    """
    def __init__(
        self,
        name: str = "franka_custom_pick_place",
        cube_initial_positions: Optional[np.ndarray] = None,
        cube_initial_orientations: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        colors: Optional[list] = None,
    ) -> None:
        super().__init__(
            name=name,
            cube_initial_positions=cube_initial_positions,
            cube_initial_orientations=cube_initial_orientations,
            target_position=target_position,
            cube_size=cube_size,
            offset=offset,
            colors=colors,
        )
        return

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
