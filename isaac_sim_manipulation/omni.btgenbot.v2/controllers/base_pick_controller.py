import typing
import numpy as np
from omni.isaac.core.controllers.base_controller import BaseController
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers.gripper import Gripper

class PickController(BaseController):
    """
    A pick controller that only executes the picking phase of a pick-and-place task.
    
    The state machine runs through these phases:
      Phase 0: Move end-effector above the cube center.
      Phase 1: Lower end-effector down to encircle the cube.
      Phase 2: Wait for the robot's inertia to settle.
      Phase 3: Close the gripper.
      Phase 4: Lift the block (move end-effector upward).
      
    When phase 4 is complete (i.e. when _event >= 5), the controller reports success.
    
    Args:
        name (str): Name of the controller.
        cspace_controller (BaseController): A cartesian space controller returning an ArticulationAction.
        gripper (Gripper): A gripper controller for open/close actions.
        end_effector_initial_height (typing.Optional[float], optional): Initial picking height. Defaults to 0.3 m.
        events_dt (typing.Optional[typing.List[float]], optional): Dt for each phase (list of at least 5 values).  
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
        self._events_dt = events_dt
        if self._events_dt is None:
            # Default dt for 10 phases (we only use the first 5 for picking)
            self._events_dt = [0.008, 0.005, 0.1, 0.1, 0.0025, 0.001, 0.0025, 1, 0.008, 0.08]
        else:
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events dt need to be list or numpy array")
            if isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) < 5:
                raise Exception("events dt length must be at least 5 for PickController")
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
        # placing_position: np.ndarray,  # Not used in pick controller.
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])
        # If paused or the pick task is already done, pause and return no action.
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        
        # --- Phase-specific logic for picking ---
        if self._event == 2:
            # Phase 2: Wait for robot inertia to settle; no movement command.
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 3:
            # Phase 3: Close the gripper.
            target_joint_positions = self._gripper.forward(action="close")
        else:
            # For phases 0, 1, and 4:
            # For phases 0 and 1, set the current target from the picking_position.
            if self._event in [0, 1]:
                self._current_target_x = picking_position[0]
                self._current_target_y = picking_position[1]
                self._h0 = picking_position[2]
            # Compute an interpolated XY target.
            interpolated_xy = self._get_interpolated_xy(
                picking_position[0], picking_position[1], self._current_target_x, self._current_target_y
            )
            # Determine the target height.
            target_height = self._get_target_hs(picking_position[2])
            position_target = np.array([
                interpolated_xy[0] + end_effector_offset[0],
                interpolated_xy[1] + end_effector_offset[1],
                target_height + end_effector_offset[2],
            ])
            if end_effector_orientation is None:
                end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target,
                target_end_effector_orientation=end_effector_orientation
            )
        # Increment time within the current phase.
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0
        return target_joint_positions

    def _get_interpolated_xy(self, target_x, target_y, current_x, current_y):
        alpha = self._get_alpha()
        return (1 - alpha) * np.array([current_x, current_y]) + alpha * np.array([target_x, target_y])

    def _get_alpha(self):
        # For picking, we only interpolate during phase 4 (lifting)
        if self._event in [0, 1]:
            return 0
        elif self._event == 4:
            return self._mix_sin(self._t)
        else:
            return 1.0

    def _get_target_hs(self, target_height):
        # For picking:
        # Phase 0: Use initial height (_h1).
        # Phase 1: Interpolate from _h1 to _h0.
        # Phase 3: Maintain _h0.
        # Phase 4: Interpolate from _h0 back up to _h1.
        if self._event == 0:
            return self._h1
        elif self._event == 1:
            a = self._mix_sin(max(0, self._t))
            return self._combine_convex(self._h1, self._h0, a)
        elif self._event == 3:
            return self._h0
        elif self._event == 4:
            a = self._mix_sin(max(0, self._t))
            return self._combine_convex(self._h0, self._h1, a)
        else:
            return self._h1

    def _mix_sin(self, t):
        return 0.5 * (1 - np.cos(t * np.pi))

    def _combine_convex(self, a, b, alpha):
        return (1 - alpha) * a + alpha * b

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
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) < 5:
                raise Exception("events dt length must be at least 5 for PickController")
        return

    def is_done(self) -> bool:
        # The pick task is considered done once phase 4 is complete (_event >= 5).
        return self._event >= 5

    def pause(self) -> None:
        self._pause = True

    def resume(self) -> None:
        self._pause = False
