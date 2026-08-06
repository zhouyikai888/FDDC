#!/usr/bin/env python
"""Fetch the FDDC stratified single-leg-balance motions (900 clips) from HuggingFace into data/.

The motion .npz are hosted on the HuggingFace Hub (not in git). Run this once before the benchmark:

    pip install huggingface_hub
    python download_data.py

Override the source repo with the FDDC_HF_REPO env var if you mirror the dataset elsewhere.
"""
import os

# TODO(maintainer): set this to the published HuggingFace dataset id once created.
HF_REPO = os.environ.get("FDDC_HF_REPO", "<hf-username>/FDDC-single-leg-balance")
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "data_stratified_900")


def main():
    from huggingface_hub import snapshot_download

    print(f"downloading motions from HuggingFace dataset '{HF_REPO}' -> {DEST}")
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=DEST,
        allow_patterns=["train/*.npz", "val/*.npz", "test/*.npz"],
    )
    for split in ("train", "val", "test"):
        d = os.path.join(DEST, split)
        n = len([f for f in os.listdir(d) if f.endswith(".npz")]) if os.path.isdir(d) else 0
        print(f"  {split}: {n} clips")
    print("done. You can now run:  cd eval && python run_eval.py --condition clean")


if __name__ == "__main__":
    main()
