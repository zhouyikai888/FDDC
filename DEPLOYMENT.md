# Deploying FDDC on a real Unitree G1

FDDC runs **directly** on a physical Unitree G1 (29-DoF) — the *same* actor scored in the benchmark, with
**no teacher–student distillation and no separate student network**. This document is the actual
deployment **recipe and commands** we use, plus the **safety caveats**.

The on-robot runtime is the **Holosoma two-process inference stack** (`run_sim.py` + `run_policy.py`,
from the public [`amazon-far/holosoma`](https://github.com/amazon-far/holosoma) framework the training code
is built on — see [`training/`](training/)). That framework and the Unitree SDK are **not bundled in this
repository**; the commands below document how the policy is deployed on it.

## The deployed artifact

The runtime consumes a **per-motion ONNX** (`model_<step>_<motion>.onnx`) — the actor exported from
`policy/model_0262000.pt` with a specific motion command baked in — run at **50 Hz**. The released
`policy/model_0262000.pt` is that actor; `eval/fast_policy.py` reproduces it in numpy **bit-exact** with
the ONNX export and is the executable reference. Robot constants (`dof_names`, `kp`, `kd`, `action_scale`,
default pose, joint limits) are in `policy/robot_config.json`.

## sim2sim — always validate here before hardware

Two processes (two terminals / two conda envs — a MuJoCo-sim env and an inference env):

```bash
# Terminal A — simulator.  WBT_INIT_MOTION_NPZ must point at the SAME motion as the ONNX,
# otherwise the robot starts from the wrong pose.  WBT_IMU_TYPE=pelvis (see IMU note below).
WBT_IMU_TYPE=pelvis  WBT_INIT_MOTION_NPZ=<motion>.npz \
    python src/holosoma/holosoma/run_sim.py robot:g1-29dof

# Terminal B — controller (run at the same time)
python src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt \
    --task.model-path <model_step_motion>.onnx --task.rl-rate 50
```

Keyboard (in the sim): `]` start the policy · `m` start the motion · `o` stop · `i` return to default stance.

## sim2real — the robot replaces the simulator

Only the controller runs; the real robot supplies `LowState` over the network in place of `run_sim`.
A per-motion ONNX bakes the motion in, so no `WBT_INIT_MOTION_NPZ` is needed.

```bash
python src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt \
    --task.model-path <model_step_motion>.onnx --task.rl-rate 50 \
    --task.interface <robot-DDS-network-interface> --task.use-joystick
```

`--task.use-joystick` is **required** on hardware (without it the remote falls back silently to keyboard).
Joystick: **A** start the policy + play the motion · **B** stop · **Y** default stance ·
**L1+R1** emergency KILL · **select** next policy.

## What the policy uses on-board (paper Appendix A/B)

- **Observation, reconstructed on-board.** Every term of the deployed actor observation is reconstructible
  from the **joint encoders and the torso IMU alone**. Its key term — the support-relative dynamic-CoM
  (the capture point expressed relative to the support foot) — is built from the encoder joint
  positions/velocities, the IMU gyroscope, and projected gravity; the base's absolute linear velocity,
  which no on-board sensor can measure, **cancels identically** in the support-relative difference (full
  derivation in Appendix A), so the robot receives the *same* balance quantity as in simulation. The
  privileged world-frame terms (world xCoM, base linear velocity, body pose) are critic-only in training
  and discarded at deployment.
- **Support-foot selection.** The same gravity-aligned foot-height rule as in training (lower foot
  supports; feet within 3 cm count as double support) — identical in sim and on hardware.
- **Control.** The actor outputs a residual about the default pose; this becomes a joint-position target
  for the robot's on-board PD (gains from `robot_config.json`), with minimal standard processing (numeric
  clipping on observations/actions and a light low-pass on the waist encoders).

## Safety — read before any hardware run

The research deployment runs **without a script-level joint-limit clamp** (`WBT_ENABLE_TARGET_CLAMP=False`),
relying on the physical joint ranges and torque saturation as the backstop; script-level clamping can be
enabled (set it `True`, with per-joint `dof_pos_lower/upper` in the ONNX metadata). Before any hardware run:

- [ ] **Gantry / fall protection** up; start slow and small-amplitude, confirm the stance + swing-foot
      phase, then release.
- [ ] **Emergency stop ready** — joystick **L1+R1** (KILL); and `--task.use-joystick` must be set.
- [ ] **IMU frame** — default `WBT_IMU_TYPE=pelvis`; if your robot's IMU is in the torso, set it to
      `"torso"` on both sides (validate in sim first — a wrong IMU frame corrupts the balance estimate).
- [ ] Confirm `--task.interface` is the robot's DDS network interface.
- [ ] A dry run in **sim2sim** (above) first.

You are responsible for the safeguards appropriate to your hardware. The policy, the deploy-faithful
sim2sim harness (`eval/`), and the deployability proof (Appendix A) substantiate the paper's real-robot
results (§5.6, Appendix I, and the supplementary video).
