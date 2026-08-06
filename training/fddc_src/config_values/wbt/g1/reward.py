# ---------------------------------------------------------------------------------------------------
# Modified from the Holosoma framework (Amazon FAR): https://github.com/amazon-far/holosoma
# Copyright Amazon.com, Inc. or its affiliates. Licensed under the Apache License, Version 2.0.
# This file was CHANGED for FDDC: it carries the deployable support-relative dynamic-CoM observation
# and/or the human-science balance reward library and its config (see training/TRAINING.md).
# The Apache-2.0 license text is in the repository LICENSE; attribution is in training/NOTICE.
# ---------------------------------------------------------------------------------------------------

"""Whole Body Tracking reward presets for the G1 robot."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

g1_29dof_wbt_reward = RewardManagerCfg(
    terms={
        # Motion tracking rewards - global reference frame
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=0.5,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        # Motion tracking rewards - body velocities
        "motion_global_body_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={"sigma": 1.0},
            weight=1.0,
        ),
        "motion_global_body_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={"sigma": 3.14},  # reverted to 072007: loose tracking, more freedom for balance (tightening reduces sway but steals balance freedom; a trade-off -> reverted)
            weight=1.0,
        ),
        # Regularization rewards
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-0.1,
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},
            weight=-10.0,
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": (
                    "^(?!left_foot_contact_point$)(?!right_foot_contact_point$)"
                    "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                    "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"
                ),
            },
            weight=-0.1,
        ),
        # Anti foot-hopping. Penalizes mismatch between the reference support phase (precomputed
        # offline into each motion npz) and the robot's actual foot contact. Logic identical to
        # whole_body_tracking; weight rescaled from WBT's -10 to -2 for holosoma's reward magnitude.
        "reference_support_contact_mismatch_penalty": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:ReferenceSupportContactMismatchPenalty",
            params={
                "force_threshold": 5.0,
                "force_tau": 0.5,
                "stance_miss_weight": 1.0,
                "swing_extra_contact_weight": 1.0,
            },
            weight=-2.0,  # reverted to the 0705 value (re-running the 0705 config, 2026-07-15)
        ),
        # Balance: penalize the extrapolated CoM (xCoM) approaching/leaving the support polygon.
        # LINEAR hinge (penalty_power=1): weight * relu(safety - margin); safety=0.02 ≈ the reference's
        # single-leg xCoM margin, so ~0 for good tracking, rises as xCoM drifts to/over the foot edge.
        # per-step: -0.02 near the foot edge, capped at -0.04 when >=2cm out of bounds. Needs whole-body CoM via simulator.get_body_masses().
        "support_xcom_polygon_margin": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:SupportXcomPolygonMarginPenalty",
            params={
                "threshold": 1.0,
                "single_support_safety_margin": 0.02,
                "double_support_safety_margin": 0.03,
                "gravity": 9.81,
                "com_height_floor": 0.25,
                "max_penalty": 0.04,
                "penalty_power": 1.0,  # linear hinge (squared -> 1): only linear gives an appreciable gradient at meter-scale margins
            },
            weight=-20.0,  # [tuning] xcom_polygon weight -20 (baseline=-50; back to polygon_w20 on 2026-07-24)
        ),
        # Balance: penalize small Time-To-Boundary of the xCoM (time to cross the support polygon).
        "xcom_ttb": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:XcomTtbPenalty",
            params={
                "threshold": 1.0,
                "single_support_ttb_threshold": 0.30,  # reverted to the 072007 value (fine-tune changes only the DR)
                "double_support_ttb_threshold": 0.20,
                "v_min": 0.01,
                "gravity": 9.81,
                "com_height_floor": 0.25,
                "max_penalty": 0.09,
            },
            weight=-15.0,  # xcom_ttb balance reward (baseline value); restored 2026-07-24
        ),
        # Anti-slip: during single-leg support, penalize the support foot's horizontal sliding.
        "single_support_foot_slip_penalty": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:SingleSupportFootSlipPenalty",
            params={"force_threshold": 1.0},
            weight=-1.0,
        ),
        # Stance-foot ankle stability: during single-leg support, penalize ankle (pitch+roll)
        # action-rate of the stance foot (gated by ACTUAL single-support contact, like WBT).
        "stance_ankle_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:StanceAnkleActionRatePenalty",
            params={
                "left_joint_names": ["left_ankle_pitch_joint", "left_ankle_roll_joint"],
                "right_joint_names": ["right_ankle_pitch_joint", "right_ankle_roll_joint"],
                "threshold": 1.0,
                "joint_weights": [1.0, 1.0],
            },
            weight=-0.3,  # stance_ankle_action_rate (polygon_w20/baseline value); restored 2026-07-28
        ),
    }
)

g1_29dof_wbt_fast_sac_reward = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-1.0,  # proven from-scratch value (072007); anti-flail is achieved via convergence/stability, not a heavier action-rate
        ),
        # Action jerk (2nd difference of actions). Surgical anti-chatter: ~0 for smooth motion of any
        # speed, only bites on high-frequency reversals -> targets foot-chatter WITHOUT the over-smoothing
        # a heavier action_rate causes. Measured (072007, 7 motions): jerk raw ~1.28x action_rate, so at
        # -0.1 the per-step contribution ~-0.167 = 13% of action_rate (-1.304) -- a light targeted nudge
        # below the over-smoothing zone (action_rate -1.5 = +50% regressed). Tune by 500-eval.
        # Impl: ActionJerkPenalty in managers/reward/terms/wbt.py (added 2026-07-17).
        "action_jerk": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:ActionJerkPenalty",
            weight=-0.1,  # action_jerk (baseline value); restored 2026-07-24
        ),
        # Stance-leg KNEE action-rate, single-support gated -- mirrors stance_ankle_action_rate on the
        # knee. Reuses the generic stance-joint action-rate impl via a thin StanceKneeActionRatePenalty
        # subclass. KungFu uses ankle > knee, so weight kept below the ankle's -0.3. -0.1 to start; the
        # knee is 1 joint & moves less than the ankle in single-leg, so expect a small contribution.
        "stance_knee_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:StanceKneeActionRatePenalty",
            params={
                "left_joint_names": ["left_knee_joint"],
                "right_joint_names": ["right_knee_joint"],
                "threshold": 1.0,
                "joint_weights": [1.0],
            },
            weight=-0.1,  # stance_knee_action_rate (polygon_w20/baseline value); restored 2026-07-28
        ),
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=2.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
    }
)

g1_29dof_wbt_reward_w_object = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        # Motion tracking rewards - global reference frame
        "object_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "object_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
    }
)

__all__ = ["g1_29dof_wbt_fast_sac_reward", "g1_29dof_wbt_reward", "g1_29dof_wbt_reward_w_object"]
