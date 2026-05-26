# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import typing
import numpy as np
from omni.isaac.core.controllers.base_controller import BaseController
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers.gripper import Gripper

class PlaceController(BaseController):
    """
    A place controller that executes only the placing phase of a pick-and-place task.
    The state machine now runs through these re-indexed phases:
    
      Phase 0: Move end-effector in xy toward the target xy position by interpolating
               from the initial (picking) xy position (height remains constant).
      Phase 1: Move end-effector vertically from the initial (picking) height to the target height
               (with xy fixed at the target xy).
      Phase 2: Open the gripper to release the object.
      
    When phase 2 is complete (i.e. when _event >= 3), the controller reports success.
    
    Args:
        name (str): Name of the controller.
        cspace_controller (BaseController): A cartesian space controller returning an ArticulationAction.
        gripper (Gripper): A gripper controller for open/close actions.
        end_effector_initial_height (typing.Optional[float], optional): Initial height (default is 0.3 m, scaled by stage units).
        events_dt (typing.Optional[typing.List[float]], optional): Dt for each phase (list of exactly 3 values).
    """
    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._h1 = end_effector_initial_height
        if self._h1 is None:
            self._h1 = 0.3 / get_stage_units()
        self._h0 = None  
        if events_dt is None:
            # Default dt for 3 phases.
            self._events_dt = [0.001, 0.0025, 1]
        else:
            if not isinstance(events_dt, (list, np.ndarray)):
                raise Exception("events dt need to be list or numpy array")
            if isinstance(events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 3:
                raise Exception("events dt length must be exactly 3 for PlaceController")
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False

    def is_paused(self) -> bool:
        return self._pause

    def get_current_event(self) -> int:
        return self._event

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        
        # --- Phase-specific logic for placing ---
        if self._event == 2:
            # Phase 2: Open the gripper.
            target_joint_positions = self._gripper.forward(action="open")
        else:
            if self._event == 0:
                # Phase 0: Horizontal movement.
                # Record the initial (picking) xy and height.
                self._current_target_x = picking_position[0]
                self._current_target_y = picking_position[1]
                self._h0 = picking_position[2]
                # Interpolate in xy from the picking position to the target xy.
                alpha = self._mix_sin(self._t)
                xy_target = (1 - alpha) * picking_position[:2] + alpha * placing_position[:2]
                # Maintain the initial height.
                target_height = self._h0
            elif self._event == 1:
                # Phase 1: Vertical movement.
                # Keep xy fixed at the target xy.
                xy_target = placing_position[:2]
                # Interpolate the height from the initial height to the target height.
                alpha = self._mix_sin(self._t)
                target_height = (1 - alpha) * self._h0 + alpha * placing_position[2]
            position_target = np.array([xy_target[0], xy_target[1], target_height]) + end_effector_offset
            if end_effector_orientation is None:
                end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target,
                target_end_effector_orientation=end_effector_orientation
            )
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0
        return target_joint_positions

    def _mix_sin(self, t):
        return 0.5 * (1 - np.cos(t * np.pi))

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if end_effector_initial_height is not None:
            self._h1 = end_effector_initial_height
        self._pause = False
        if events_dt is not None:
            if not isinstance(events_dt, (list, np.ndarray)):
                raise Exception("events dt need to be list or numpy array")
            if isinstance(events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 3:
                raise Exception("events dt length must be exactly 3 for PlaceController")
        return

    def is_done(self) -> bool:
        # The place task is considered done once phase 2 is complete (i.e. _event >= 3).
        return self._event >= 3

    def pause(self) -> None:
        self._pause = True

    def resume(self) -> None:
        self._pause = False
