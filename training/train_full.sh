#!/usr/bin/env bash
# Modified from the Holosoma framework (Amazon FAR, Apache-2.0): https://github.com/amazon-far/holosoma
# FDDC training launcher — 8-GPU FastSAC WBT single-leg-balance training.
# DATA defaults to the released stratified motion set (data_stratified_900/train); override via env.
# Default matches the paper (Table 6): NUM_ENVS=8192 total = 1024/GPU on 8 GPUs, with the framework's
# default replay buffer. For higher throughput on more memory, raise NUM_ENVS (e.g. 16384 = 2048/GPU);
# if memory then gets tight, cap the replay buffer with BUFFER=<n> (adds --algo.config.buffer_size).
set -e
export OMNI_KIT_ACCEPT_EULA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # anti-fragmentation; required at the memory boundary
# On this box, /sbin/ldconfig -p prints fine but exits 255; triton's check_output for libcuda then throws and torch.compile crashes.
# Set TRITON_LIBCUDA_PATH so triton uses it directly and skips ldconfig (driver.py:libcuda_dirs reads this env first).
export TRITON_LIBCUDA_PATH=${TRITON_LIBCUDA_PATH:-/usr/lib/x86_64-linux-gnu}
# GPU selection: override via CUDA_VISIBLE_DEVICES (on a shared box, avoid GPUs others use, e.g. 0,1,2,3,4,6,7); default all 8
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')   # num processes = num visible GPUs

NUM_ENVS=${NUM_ENVS:-8192}             # global total (paper Table 6); torchrun splits by /8 = 1024/GPU
BUFFER=${BUFFER:-}                      # empty = framework default replay buffer (paper config); set to cap it
DATA=${DATA:-/path/to/data_stratified_900/train}   # motion dir = the released FDDC training set; override via env

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"   # cd to the script's repo root; survives rename/move
MC=--command.setup_terms.motion_command.params.motion_config
TORCHRUN=${TORCHRUN:-torchrun}  # torchrun from the activated training env (hssim)

# Optional fine-tune / resume: CKPT=<model.pt> loads that checkpoint (with global_step) to continue;
# ITERS=<n> overrides num_learning_iterations (when resuming, must exceed the checkpoint step, else it stops immediately).
CKPT=${CKPT:-}
ITERS=${ITERS:-}
PROJECT=${PROJECT:-}       # wandb project name (ablations used balance_ablation); empty = config default WholeBodyTracking
RUN_NAME=${RUN_NAME:-}     # per-run name (wandb run name); empty = auto timestamp name
EXTRA_ARGS=()
[ -n "$CKPT" ]  && EXTRA_ARGS+=("--training.checkpoint=$CKPT")
[ -n "$ITERS" ] && EXTRA_ARGS+=("--algo.config.num_learning_iterations=$ITERS")
[ -n "$PROJECT" ]  && EXTRA_ARGS+=("--training.project=$PROJECT")
[ -n "$RUN_NAME" ] && EXTRA_ARGS+=("--logger.name=$RUN_NAME")
[ -n "$BUFFER" ]   && EXTRA_ARGS+=("--algo.config.buffer_size=$BUFFER")

"$TORCHRUN" --standalone --nproc_per_node=$NPROC src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-fast-sac \
    logger:wandb \
    --logger.video.enabled=False \
    --training.num-envs=$NUM_ENVS \
    "$MC.motion_dir=$DATA" \
    "${EXTRA_ARGS[@]}"
