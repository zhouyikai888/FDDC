# Reproducing the eight baselines — per-baseline setup

Each adapter here is the **exact script used to score that baseline in the paper** (Table 1). An adapter
runs the *baseline's own* policy — and, for several, the baseline's own sim / observation code — inside
our shared MuJoCo kernel (`../../eval/wbt_rollout.py`). So you must supply that baseline's **repository
and released weights, obtained from the original authors** (we do not redistribute third-party code or
weights). Paths are passed via environment variables; the robot constants come from `robot_meta.onnx`
(bundled here). Run each adapter from this directory with the shared kernel on the path:

```bash
cd benchmark/baselines
export PYTHONPATH=../../eval
# then set the baseline's env vars (below) and run, e.g.:
GMT_REPO=/path/to/GMT GMT_WEIGHTS=/path/to/pretrained.pt \
    python gmt_eval.py run ../../data/data_stratified_900/test gmt 0 1 ./out
```

The exact CLI arguments are in each adapter's top-of-file docstring (English). The common pattern is
`run <motion_dir> <tag> <shard> <nshards> <outdir>` (some also take a `<K>` seed count or a `sanity`
sub-command). Score against `../../data/data_stratified_900/test` (the paper's 90 held-out test clips).

## What each baseline needs

| Baseline | Paper | Env vars to set | Weights (from the authors) | Extra deps |
|----------|-------|-----------------|-----------------------------|------------|
| **ProtoMotions** | Tessler et al. 2025 (github.com/NVlabs/ProtoMotions) | `PROTO_ONNX` | `unified_pipeline.onnx` | onnxruntime |
| **GMT** | Chen et al. 2025, arXiv:2506.14770 | `GMT_REPO`, `GMT_WEIGHTS` | `pretrained.pt` | its repo + `mujoco_viewer` |
| **TWIST** | Ze et al. 2025, arXiv:2505.02833 | `TWIST_REPO`, `TWIST_WEIGHTS` | `twist_general_motion_tracker.pt` (TorchScript) | its repo |
| **SONIC** | Luo et al. 2025, arXiv:2511.07820 | `SONIC_DIR` | `model_encoder.onnx` + `model_decoder.onnx` | onnxruntime |
| **OmniXtreme** | Wang et al. 2026, arXiv:2602.23843 | `OMNI_REPO`, `OMNI_DIR` | `base_policy_trt.onnx`, `fk_trt.onnx`, `residual_policy.onnx` | its repo (runs its `deploy_mujoco`) |
| **MOSAIC** | Sun et al. 2026, arXiv:2602.08594 | `MOSAIC_ONNX` | `gmt.onnx` (770→29 student) | onnxruntime |
| **Humanoid-GPT** | Qi et al. 2026, arXiv:2606.03985 | `HUMANOID_GPT_REPO`, `HGPT_ONNX` | `pns_wo_priv216.onnx` | its repo |
| **HoloMotion** | Chen et al. 2026, arXiv:2605.15336 | `HOLO_REPO`, `HOLO_ONNX` | `model_14000.onnx` | its repo + `warp-lang` |

- **Weights-only (no external repo needed):** ProtoMotions, MOSAIC, SONIC — these drive our shared
  `wbt_rollout` plant directly; you only need their weights.
- **Repo + weights:** GMT, TWIST, OmniXtreme, Humanoid-GPT, HoloMotion — these run the baseline's own
  observation / sim code, so clone that repo and point the `*_REPO` env var at it, plus install its
  requirements.

## Robot metadata

`robot_meta.onnx` is a tiny (~76 KB) metadata-only ONNX carrying the Unitree-G1 constants the adapters
read for the shared PD (`dof_names`, `kp`, `kd`, `action_scale`, default pose). Override with the
`FDDC_ROBOT_META_ONNX` env var if needed. These constants are identical to `../../policy/robot_config.json`.

## Sanity check first

Every method should first pass a tracking sanity check (track a normal motion with low error) before its
single-leg number is trusted — several adapters expose a `sanity` sub-command for this. A method whose
exact specification is unavailable is excluded, never guessed (paper §4.2).
