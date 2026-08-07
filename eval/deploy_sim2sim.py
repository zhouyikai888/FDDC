#!/usr/bin/env python
"""FDDC sim2sim deploy demo — watch the deployed checkpoint hold single-leg balance in MuJoCo.

This is a **self-contained visualization** built on this repo's benchmark harness (`fast_policy.py` +
`wbt_rollout.py`): it rolls out one motion with the deployed checkpoint in the deploy-faithful MuJoCo
plant and either opens an interactive viewer or renders a video. It runs from **this repository alone**
and is **not** the on-robot deploy runtime. The actual deployment pipeline (the Holosoma `run_sim` +
`run_policy` two-process stack with a per-motion ONNX) and the safety caveats are documented in
`../DEPLOYMENT.md`. Sim2sim only — no real robot.

    python deploy_sim2sim.py                        # interactive viewer, a default clip
    python deploy_sim2sim.py --motion left_1077     # a specific clip (ids: data/.../test/)
    python deploy_sim2sim.py --video out.mp4         # headless render to a video (needs imageio)
    python deploy_sim2sim.py --condition noisy       # with deployment-relevant observation noise
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)

COND = {"clean": dict(dof_vel_noise=0.0, dof_vel_delay=0, imu_noise=False),
        "noisy": dict(dof_vel_noise=0.20, dof_vel_delay=1, imu_noise=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=os.path.join(PKG, "policy", "model_0262000.pt"))
    ap.add_argument("--robot-config", default=os.path.join(PKG, "policy", "robot_config.json"))
    ap.add_argument("--robot-xml", default=os.path.join(PKG, "robot", "g1_29dof", "g1_29dof.xml"))
    ap.add_argument("--motion-dir", default=os.path.join(PKG, "data", "data_stratified_900", "test"))
    ap.add_argument("--motion", default="", help="motion id (default: first clip in the dir)")
    ap.add_argument("--condition", choices=list(COND), default="clean")
    ap.add_argument("--video", default="", help="render to this file (.mp4/.gif) instead of a live viewer")
    ap.add_argument("--fps", type=int, default=50)
    args = ap.parse_args()

    os.environ["WBT_ORT_THREADS"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ["WBT_EVAL_ROBOT_XML"] = args.robot_xml
    os.environ["WBT_EVAL_MOTION_DIR"] = args.motion_dir
    if args.video:
        os.environ.setdefault("MUJOCO_GL", "egl")   # headless offscreen render
    sys.path.insert(0, HERE)
    import glob, re, mujoco
    import wbt_rollout as W
    from fast_policy import FastPolicy

    mids = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p))
                  for p in glob.glob(os.path.join(args.motion_dir, "sample_*_mj.npz")))
    if not mids:
        sys.exit(f"no motions in {args.motion_dir} — run download_data.py first")
    mid = args.motion or mids[0]
    if mid not in mids:
        sys.exit(f"motion '{mid}' not found; e.g. {mids[:3]}")

    print(f"policy   : {os.path.basename(args.pt)}")
    print(f"motion   : {mid}   condition: {args.condition}")
    pol = FastPolicy(args.pt, args.robot_config)
    log = W.run_rollout(pol, mid, seed=0, use_npz_ref=True, **COND[args.condition])

    # reconstruct the full MuJoCo qpos per frame = base_pos(3) + base_quat_wxyz(4) + dof_pos(29)
    bp = np.asarray(log["base_pos"]); bq = np.asarray(log["base_quat"])   # base_quat is xyzw
    dq = np.asarray(log["qpos"]); T = len(bp)
    qpos = np.zeros((T, 7 + dq.shape[1]))
    qpos[:, 0:3] = bp
    qpos[:, 3:7] = bq[:, [3, 0, 1, 2]]   # xyzw -> wxyz for the free joint
    qpos[:, 7:] = dq
    fell = np.asarray(log["fell"]); tilt = np.degrees(np.asarray(log["tilt"]))
    print(f"frames   : {T} ({T/args.fps:.1f}s)   fell={bool(fell[-1])}   max_tilt={tilt.max():.0f} deg")

    model = W._load_deploy_model()
    data = mujoco.MjData(model)

    def set_frame(t):
        data.qpos[:qpos.shape[1]] = qpos[t]
        mujoco.mj_forward(model, data)

    if args.video:
        try:
            import imageio.v2 as imageio
        except Exception:
            sys.exit("--video needs imageio:  pip install imageio imageio-ffmpeg   (or render a .gif)")
        renderer = mujoco.Renderer(model, 480, 640)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 3.0, -15.0, 135.0
        frames = []
        for t in range(T):
            set_frame(t)
            cam.lookat[:] = qpos[t, 0:3]
            renderer.update_scene(data, cam)
            frames.append(renderer.render())
        imageio.mimsave(args.video, frames, fps=args.fps)
        print(f"wrote {args.video}  ({T} frames)")
    else:
        import time
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as v:
            print("viewer open — close the window to exit (the trajectory loops)")
            while v.is_running():
                for t in range(T):
                    if not v.is_running():
                        break
                    set_frame(t)
                    v.sync()
                    time.sleep(1.0 / args.fps)


if __name__ == "__main__":
    main()
