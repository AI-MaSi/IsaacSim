# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .differential_ik import DifferentialIKController


@configclass
class DifferentialIKControllerCfg:
    """Configuration for differential inverse kinematics controller."""

    class_type: type = DifferentialIKController
    """The associated controller class."""

    command_type: Literal["position", "pose"] = MISSING
    """Type of task-space command to control the articulation's body.

    If "position", then the controller only controls the position of the articulation's body.
    Otherwise, the controller controls the pose of the articulation's body.
    """

    use_relative_mode: bool = False
    """Whether to use relative mode for the controller. Defaults to False.

    If True, then the controller treats the input command as a delta change in the position/pose.
    Otherwise, the controller treats the input command as the absolute position/pose.
    """

    ik_method: Literal["pinv", "svd", "trans", "dls"] = MISSING
    """Method for computing inverse of Jacobian."""

    ik_params: dict[str, float | list[float]] | None = None
    """Parameters for the inverse-kinematics method.

    Common parameters for all methods:
    - "k_val": Scaling of computed delta-joint positions (default: 1.0).
    - "position_weight": Weight for position errors (default: 1.0).
    - "rotation_weight": Weight for rotation errors (default: 0.1).
    - "joint_weights": List of per-joint weights (e.g., [1.0, 0.5, 0.1, ..., 1.0]). Works with SVD and DLS methods.

    Method-specific parameters:
    - SVD ("svd"):
        - "min_singular_value": Values below are suppressed to zero (default: 1e-5).
    - Damped least squares ("dls"):
        - "lambda_val": Damping coefficient (default: 0.01).
    """

    def __post_init__(self):
        # Validate command and method
        if self.command_type not in ["position", "pose"]:
            raise ValueError(f"Unsupported inverse-kinematics command: {self.command_type}.")
        if self.ik_method not in ["pinv", "svd", "trans", "dls"]:
            raise ValueError(f"Unsupported inverse-kinematics method: {self.ik_method}.")

        # Default parameters for each method
        default_ik_params = {
            "pinv": {"k_val": 1.0},
            "svd": {"k_val": 1.0, "min_singular_value": 1e-5},
            "trans": {"k_val": 1.0},
            "dls": {"k_val": 1.0, "lambda_val": 0.01},
        }

        # Default weighting parameters (apply to all methods)
        weighting_defaults = {
            "position_weight": 1.0,
            "rotation_weight": 0.1,
        }

        # Start with method-specific defaults
        ik_params = default_ik_params[self.ik_method].copy()

        # Add weighting defaults
        ik_params.update(weighting_defaults)

        # Override with user-provided parameters
        if self.ik_params is not None:
            ik_params.update(self.ik_params)

        self.ik_params = ik_params