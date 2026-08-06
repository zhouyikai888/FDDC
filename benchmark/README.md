# The FDDC single-leg-balance benchmark — method-agnostic scoring

FDDC's benchmark is **method-agnostic**: every policy is scored inside **one shared MuJoCo kernel** at
the robot's command interface, under byte-identical conditions, and success is judged from the robot's
**true physical state** — not from the policy's own reference or reward. This directory documents how
that works and how the paper's eight baselines were scored (Table 1).

## How it works

Three shared pieces (in `../eval/`) plus a thin per-method adapter:

1. **Shared kernel** — `eval/wbt_rollout.py`: the deploy-faithful MuJoCo G1 plant (`G1Sim`), a 50 Hz PD
   controller, the motion set, and the seeded observation noise. Every method runs in *this* plant.
2. **Shared metrics** — `eval/metrics.py`: the outcome tiers (Perfect / Marginal / Failure) and the
   continuous suite, computed from the robot's true state. Method-agnostic by construction.
3. **A thin per-method adapter** that (i) assembles *that method's* observation from the shared sim
   state + the motion reference, (ii) runs its network (ONNX / TorchScript / its own kernel), and
   (iii) decodes the action to joint-position targets for the shared PD.

`robot_meta.onnx` (in `baselines/`) is a tiny metadata-only ONNX carrying the Unitree-G1 constants
(`dof_names`, `kp`, `kd`, `action_scale`, default pose) that the adapters read for the shared PD.

## Reproducing the eight baselines (paper Table 1)

`baselines/` holds the **exact adapters used in the paper** (one per baseline), plus the shared helpers
(`success_metric.py`, `fulllog*.py`). They are provided so the 0/90 baseline result is reproducible and
transparent.

> **These are not turnkey.** Each baseline runs its *own* policy, and several run their *own* sim/obs
> code, so an adapter needs that baseline's **repository and released weights, obtained from the original
> authors** (we do not redistribute third-party code or weights). Per-baseline setup — repo, weights,
> extra dependencies, and the exact command — is in [`baselines/SETUP.md`](baselines/SETUP.md).

The adapters import the shared kernel from `../eval/`, so run them with that on the path, e.g.:

```bash
cd benchmark/baselines
PYTHONPATH=../../eval  GMT_REPO=/path/to/GMT  GMT_WEIGHTS=/path/to/pretrained.pt \
    python gmt_eval.py run  ../../data/data_stratified_900/test  gmt  0 1  ./out
```

## Scoring your own policy

The simplest worked example is the FDDC path itself: `../eval/run_eval.py` + `../eval/fast_policy.py`
score a policy that already speaks the WBT observation. To score a policy with a **different**
observation / action space, write an adapter following the pattern in `baselines/`:

- read the shared sim state and the per-frame motion reference from `wbt_rollout` (`G1Sim`,
  `load_motion_npz`, the reference terms),
- assemble your policy's observation, run your network,
- return joint-position targets for the shared PD (`kp`/`kd`/default pose from `robot_meta.onnx`),
- let `metrics.compute_metrics` judge the outcome from the true state.

`mosaic_eval.py` and `proto_eval.py` are the lightest examples (they drive the shared `wbt_rollout`
plant directly and need only their weights, no external repo).
