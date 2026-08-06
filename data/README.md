# Data — FDDC stratified single-leg-balance motions

The **900 motion clips** (`sample_<support>_<id>_mj.npz`; Unitree G1 29-DoF, 50 Hz, 249 frames) are
hosted on the **HuggingFace Hub**, not in this git repo (they are ~645 MB). Only the small metadata
(`data_stratified_900/dataset_info.json`, `manifest.csv`, `*_list.txt`) is tracked here.

**Get the motions:**

```bash
pip install huggingface_hub
python download_data.py        # from the repo root -> fills data/data_stratified_900/{train,val,test}/
```

- **Split:** 720 train / 90 val / 90 test, balanced across the 9 pose classes and both support sides
  (seed 42). Per-clip class / bin / support / split labels are in `data_stratified_900/dataset_info.json`.
- **License: GPL-3.0** (`LICENSE`), because these motions are a derivative of the **AMS** synthetic
  balance dataset (Pan et al., 2025). Attribution and what we changed are in `NOTICE`.
