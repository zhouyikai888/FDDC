# Reproducing the eight baselines (paper Table 1)

Each adapter here is the **exact script used to score that baseline in the paper**. An adapter runs the
baseline's *own* policy — and, for several, the baseline's *own* observation / sim code — inside our
shared MuJoCo kernel (`../../eval/wbt_rollout.py`, `../../eval/metrics.py`). Success is judged from the
robot's true physical state by the same `metrics.py` for every method.

We do **not** redistribute third-party code or weights: you obtain each baseline's **repository and
released weights from its authors** (links below), then point the adapter at them via environment
variables. The Unitree-G1 constants come from the bundled `robot_meta.onnx`.

Expected outcome: **all eight baselines score 0/90 Perfect** (they never hold a clean single-leg stance;
they stay up by hopping / stepping, or fall). Per-baseline Marginal/Failure are in the table below and
match the paper's Table 1.

---

## 1. Common setup (how every run is invoked)

```bash
cd benchmark/baselines
export PYTHONPATH="$(pwd)/../../eval"          # the shared kernel: wbt_rollout.py + metrics.py + fast_policy.py
export WBT_ORT_THREADS=1 OMP_NUM_THREADS=1     # cap threads (recommended)
DATA="$(pwd)/../../data/data_stratified_900/test"   # the 90 held-out test clips (run download_data.py first)
```

- **Python environment.** The lightweight baselines (ProtoMotions, MOSAIC, SONIC) need only
  `mujoco`, `onnxruntime`, `numpy` (and `torch` for TWIST/GMT weights). The repo-based baselines
  (GMT, TWIST, OmniXtreme, Humanoid-GPT, HoloMotion) additionally need **that baseline's own repo
  installed with its requirements** (each was run in its own conda env to avoid dependency clashes) —
  plus `mujoco_viewer` (GMT, OmniXtreme) or `warp-lang` (HoloMotion).
- **Sanity first.** Every adapter has a `sanity` sub-command that tracks a normal motion and reports
  whether the wiring is correct (low tracking error, no fall). Run `python <baseline>_eval.py sanity`
  and confirm it tracks before trusting the single-leg number (paper §4.2 — a method that fails sanity
  is a wiring bug, not a real failure).
- **The metrics run** writes a per-motion JSON (`{tag}__sh{shard}.json` etc.) with `success` / `fell`
  per clip. Aggregate them (below) into Perfect / Marginal / Failure.

## 2. Per-baseline: repo, weights, env vars, and command

Run the full test set with `shard=0 nshards=1`. The **metrics sub-command differs per adapter** (an
artifact of the original per-method scripts): ProtoMotions is positional, MOSAIC uses `run`, GMT / TWIST
/ OmniXtreme / Humanoid-GPT / SONIC use `runs`, HoloMotion uses `runm`. See each adapter's docstring.

| Baseline | Repo / weights (from the authors) | Env vars | Command (from `benchmark/baselines/`) |
|---|---|---|---|
| **ProtoMotions** (weights only) | [NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions) → `unified_pipeline.onnx` | `PROTO_ONNX` | `python proto_eval.py "$DATA" proto 0 1 1 ./out_proto` |
| **MOSAIC** (weights only) | [BAAI-Humanoid/MOSAIC](https://github.com/BAAI-Humanoid/MOSAIC), HF [BAAI-Humanoid/MOSAIC_Model](https://huggingface.co/BAAI-Humanoid/MOSAIC_Model) → `gmt.onnx` | `MOSAIC_ONNX` | `python mosaic_eval.py run "$DATA" mosaic 0 1 ./out_mosaic` |
| **SONIC** (weights only) | [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl), HF [nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC) → the **root** `model_encoder.onnx` (obs **1762**) + `model_decoder.onnx` (obs 994). **NOT** the `low_latency/` variant (1247-dim) — this adapter matches the root model. | `SONIC_DIR` (dir with both root ONNX) | `python sonic_eval.py runs "$DATA" sonic 0 1 ./out_sonic` |
| **GMT** (repo) | [zixuan417/humanoid-general-motion-tracking](https://github.com/zixuan417/humanoid-general-motion-tracking); weights `assets/pretrained_checkpoints/pretrained.pt` in-repo. Needs `mujoco_viewer`. | `GMT_REPO`, `GMT_WEIGHTS` | `python gmt_eval.py runs "$DATA" gmt 0 1 ./out_gmt` |
| **TWIST** (repo) | [YanjieZe/TWIST](https://github.com/YanjieZe/TWIST) → `twist_general_motion_tracker.pt` (TorchScript); adapter reads `$TWIST_REPO/assets/g1/g1_sim2sim_with_wrist_roll.xml`. | `TWIST_REPO`, `TWIST_WEIGHTS` | `python twist_eval.py runs "$DATA" twist 0 1 ./out_twist` |
| **OmniXtreme** (repo) | [Perkins729/OmniXtreme](https://github.com/Perkins729/OmniXtreme) → `policy/{base_policy_trt,residual_policy,fk_trt}.onnx`; runs its `deploy_mujoco.DeployNode`. Needs `mujoco`. | `OMNI_REPO`, `OMNI_DIR` (=`$OMNI_REPO/policy`) | `python omni_eval.py runs "$DATA" omni 0 1 ./out_omni` |
| **Humanoid-GPT** (repo) | [GalaxyGeneralRobotics/Humanoid-GPT](https://github.com/GalaxyGeneralRobotics/Humanoid-GPT) → `pns_wo_priv216.onnx`; runs its `tracking` module (`G1TrackMjSim`). | `HUMANOID_GPT_REPO`, `HGPT_ONNX` | `python hgpt_eval.py runs "$DATA" hgpt 0 1 ./out_hgpt` |
| **HoloMotion** (repo) | [HorizonRobotics/HoloMotion](https://github.com/HorizonRobotics/HoloMotion), HF [HorizonRobotics/HoloMotion_models](https://huggingface.co/HorizonRobotics/HoloMotion_models) → `model_14000.onnx`; uses its warp obs kernel. Needs `warp-lang`. | `HOLO_REPO`, `HOLO_ONNX` | `python holo_eval.py runm "$DATA" holo 0 1 ./out_holo` |

> **SONIC gotcha:** `nvidia/GEAR-SONIC` ships two variants. Use the **root** model (`model_encoder.onnx`
> is 1762-dim), whose SHA256 matches the file used in the paper; the `low_latency/` subfolder is a
> different 1247-dim model that this adapter does **not** match. Point `SONIC_DIR` at the root files.

Example (ProtoMotions, end to end):

```bash
PROTO_ONNX=/path/to/unified_pipeline.onnx \
    python proto_eval.py "$DATA" proto 0 1 1 ./out_proto
```

Example (GMT, needs its repo cloned + `mujoco_viewer` installed in the active env):

```bash
GMT_REPO=/path/to/humanoid-general-motion-tracking \
GMT_WEIGHTS=$GMT_REPO/assets/pretrained_checkpoints/pretrained.pt \
    python gmt_eval.py runs "$DATA" gmt 0 1 ./out_gmt
```

## 3. Aggregate a run into Perfect / Marginal / Failure

```bash
python - <<'PY'
import json, glob
pm = {}
for f in glob.glob("./out_proto/*.json"):       # <- the run's output dir
    pm.update(json.load(open(f)).get("per_motion", {}))
s  = [v.get("success", 0) for v in pm.values()]  # Perfect
fl = [v.get("fell", 0)    for v in pm.values()]  # Failure
n = len(s)
P, F = 100*sum(s)/n, 100*sum(fl)/n
print(f"n={n}  Perfect={P:.1f}%  Marginal={max(0,100-P-F):.1f}%  Failure={F:.1f}%")
PY
```

## 4. Expected results (paper Table 1, clean, n=90)

| Baseline | Perfect | Marginal | Failure |
|----------|:-------:|:--------:|:-------:|
| ProtoMotions | 0.0 | 53.3 | 46.7 |
| OmniXtreme   | 0.0 | 5.6  | 94.4 |
| GMT          | 0.0 | 18.9 | 81.1 |
| MOSAIC       | 0.0 | 64.4 | 35.6 |
| TWIST        | 0.0 | 50.0 | 50.0 |
| Humanoid-GPT | 0.0 | 75.6 | 24.4 |
| HoloMotion   | 0.0 | 75.6 | 24.4 |
| SONIC        | 0.0 | 81.1 | 18.9 |

The weights-only adapters (ProtoMotions, MOSAIC, SONIC) were spot-checked to reproduce **0/90 Perfect**
directly from this released package (ProtoMotions full-90: Perfect 0.0 / Failure 46.7, matching the
table). The repo-based baselines reproduce the same Perfect = 0 once their repo + weights are supplied.

## 5. Robot metadata

`robot_meta.onnx` is a tiny (~76 KB) metadata-only ONNX carrying the Unitree-G1 constants the adapters
read for the shared PD (`dof_names`, `kp`, `kd`, `action_scale`, default pose). Override with
`FDDC_ROBOT_META_ONNX`. These constants equal `../../policy/robot_config.json`.
