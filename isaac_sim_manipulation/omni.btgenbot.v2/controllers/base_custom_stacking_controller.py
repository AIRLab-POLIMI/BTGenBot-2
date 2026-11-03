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
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.controllers.pick_place_controller import PickPlaceController


class CustomStackingController(BaseController):
    """[summary]

    Args:
        name (str): [description]
        pick_place_controller (PickPlaceController): [description]
        picking_order_cube_names (typing.List[str]): [description]
        robot_observation_name (str): [description]
    """

    def __init__(
        self,
        name: str,
        pick_place_controller: PickPlaceController,
        picking_order_cube_names: typing.List[str],
        robot_observation_name: str,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._pick_place_controller = pick_place_controller
        self._picking_order_cube_names = picking_order_cube_names
        self._current_cube = 0
        self._robot_observation_name = robot_observation_name
        self.reset()

    def forward(
        self,
        observations: dict,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        end_effector_offset: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        if self._current_cube >= len(self._picking_order_cube_names):
            target_joint_positions = [None] * observations[self._robot_observation_name]["joint_positions"].shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        actions = self._pick_place_controller.forward(
            picking_position=observations[self._picking_order_cube_names[self._current_cube]]["position"],
            placing_position=observations[self._picking_order_cube_names[self._current_cube]]["target_position"],
            current_joint_positions=observations[self._robot_observation_name]["joint_positions"],
            end_effector_orientation=end_effector_orientation,
            end_effector_offset=end_effector_offset,
        )
        if self._pick_place_controller.is_done():
            self._current_cube += 1
            self._pick_place_controller.reset()
        return actions

    def reset(self, picking_order_cube_names: typing.Optional[typing.List[str]] = None) -> None:
        """[summary]

        Args:
            picking_order_cube_names (typing.Optional[typing.List[str]], optional): [description]. Defaults to None.
        """
        self._current_cube = 0
        self._pick_place_controller.reset()
        if picking_order_cube_names is not None:
            self._picking_order_cube_names = picking_order_cube_names
        return

    def is_done(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        if self._current_cube >= len(self._picking_order_cube_names):
            return True
        else:
            return False
