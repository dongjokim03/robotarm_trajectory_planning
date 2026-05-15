# ADA-L: Adaptive Domain-Aware Learning for Robot Trajectory Optimization

This repository contains the source code accompanying the paper. It includes
the proposed **ADA-L** planner together with baseline planners (Quintic,
Direct transcription, and B-spline), evaluated on a UR5e 6-DoF arm.

## Repository layout

```
code_final/
├── Ex1_Point2Point_new/                          # Experiment 1: Point-to-point
│   ├── 260514_ex1_ada-l_vs_quintic_vs_dircol.py
│   └── ex1_sim_vs_experiment_savgol_wide.py     # sim vs hardware comparison
├── Ex2_Point2Point1Orient/                       # Experiment 2: P2P + one orientation
│   └── 260514_ex2_quintic_vs_ada-l.py
├── Ex3_ContinuousWayPoint/                       # Experiment 3: Multi-waypoint
│   ├── 260514_ex3_bspline_vs_ada-l.py
│   ├── ex3_sim_vs_experiment_savgol_wide.py     # sim vs hardware comparison
│   └── capsule_collision_checker.py
├── Ex4_Payload_Energy_Minimization/              # Experiment 4: Payload energy
│   ├── 260514_ex4_bspline_vs_ada-l.py
│   ├── ex4_sim_vs_experiment_savgol_wide.py     # sim vs hardware comparison
│   └── capsule_collision_checker.py
├── Appendix_B_Point2Point3Orient/                # Appendix B: P2P + full orientation
│   └── 260514_Appendix_B.py
├── Appendix_C/                                   # Appendix C: Cartesian comparison
│   ├── 260514_Appendix_C_acc_x.py
│   └── 260514_Appendix_C_slow.py
├── Hardware_validation/                          # Real-robot CSV rollouts
│   ├── adal_p2p_result.csv
│   ├── adal_multi_result.csv
│   ├── adal_payload_result.csv
│   ├── bspline_multi_result.csv
│   ├── bspline_payload_result.csv
│   └── quintic_p2p_result.csv
├── validate_trajectory.py                        # CSV trajectory validator
├── requirements.txt
└── README.md
```

## Requirements

Python 3.9+ recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

GPU acceleration is optional but recommended for ADA-L training (TensorFlow
will automatically use a CUDA-enabled GPU if available).

## How to run

Each top-level experiment is a single self-contained Python script. Run from
the repository root:

```bash
# Experiment 1 — point-to-point: ADA-L vs Quintic vs Direct transcription
python Ex1_Point2Point_new/260514_ex1_ada-l_vs_quintic_vs_dircol.py

# Experiment 2 — P2P with one orientation constraint
python Ex2_Point2Point1Orient/260514_ex2_quintic_vs_ada-l.py

# Experiment 3 — multi-waypoint: ADA-L vs B-spline
python Ex3_ContinuousWayPoint/260514_ex3_bspline_vs_ada-l.py

# Experiment 4 — payload energy minimization
python Ex4_Payload_Energy_Minimization/260514_ex4_bspline_vs_ada-l.py

# Appendix B — P2P with three-axis orientation
python Appendix_B_Point2Point3Orient/260514_Appendix_B.py

# Appendix C — Cartesian-space comparison (acc_x / slow variants)
python Appendix_C/260514_Appendix_C_acc_x.py
python Appendix_C/260514_Appendix_C_slow.py
```

You can also `cd` into the corresponding directory and run the script
directly — output is written next to the script using
`Path(__file__).resolve().parent`.

### Sim-vs-hardware comparison

After running the main experiment script, each `ex{1,3,4}_sim_vs_experiment_savgol_wide.py`
loads:

- the **simulation** trajectories (`ADAL_trajectory_full.csv`,
  `Quintic_trajectory_full.csv`, `Bspline_trajectory_full.csv`) from the
  most recent `results_*` folder created next to the script;
- the **real-robot** rollouts (`adal_*_result.csv`, etc.) from
  `Hardware_validation/`.

Run them after the corresponding main script:

```bash
python Ex1_Point2Point_new/ex1_sim_vs_experiment_savgol_wide.py
python Ex3_ContinuousWayPoint/ex3_sim_vs_experiment_savgol_wide.py
python Ex4_Payload_Energy_Minimization/ex4_sim_vs_experiment_savgol_wide.py
```

Each comparison script writes its figures into
`compare_sim_vs_exp_savgol_wide/` next to itself.

## Outputs

Each script creates a timestamped folder next to itself, e.g.

```
Ex1_Point2Point_new/results_adal_vs_quintic_vs_dircol_<YYYYMMDD_HHMMSS>/
├── figures/                      # PNG figures referenced in the paper
├── adal_metrics.json             # Scalar metrics for ADA-L
├── quintic_metrics.json          # Scalar metrics for the baseline
├── trajectory_data.npz           # Raw joint trajectories
├── report.md                     # Markdown report
└── report.docx                   # DOCX report (requires python-docx)
```

## Optional: MP4 animation export (FFmpeg)

Animation export is **commented out by default** because it requires a
local FFmpeg binary, which is not a Python package and cannot be
installed via `pip`. The figures and metrics in the paper do **not**
depend on this step.

If you want to regenerate the `*.mp4` animations, do the following.

### 1. Install FFmpeg

| OS      | Command                                                                                     |
|---------|---------------------------------------------------------------------------------------------|
| Windows | Download from <https://www.gyan.dev/ffmpeg/builds/> or `conda install -c conda-forge ffmpeg` |
| macOS   | `brew install ffmpeg`                                                                       |
| Ubuntu  | `sudo apt-get install ffmpeg`                                                               |

Verify the install:

```bash
ffmpeg -version
```

### 2. Point matplotlib to the binary

Each experiment script contains a commented line like:

```python
# matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\Users\user\anaconda3\Library\bin\ffmpeg.exe'
```

Un-comment it and set the path to your local FFmpeg executable, e.g.:

- Windows (conda): `r'C:\Users\<you>\anaconda3\Library\bin\ffmpeg.exe'`
- Windows (standalone): `r'C:\ffmpeg\bin\ffmpeg.exe'`
- macOS (Homebrew): `'/opt/homebrew/bin/ffmpeg'` (Apple Silicon) or `'/usr/local/bin/ffmpeg'` (Intel)
- Ubuntu: `'/usr/bin/ffmpeg'`

You can find the path with `which ffmpeg` (macOS / Linux) or
`where ffmpeg` (Windows).

### 3. Un-comment the animation calls

Each script has a block similar to:

```python
# animate_robot_trajectory(
#     robot=robot, q=adal_res["q"], target_xyz=target_xyz,
#     time_grid=adal_res["t"],
#     save_path=str(Path(out_dir) / "ADAL_animation.mp4"),
#     fps=20, title="ADA-L trajectory",
#     xy_center=xy_center)
```

Remove the leading `# ` from every line in the block (there are usually
2–3 such blocks per file, one per planner).

## Notes

- `validate_trajectory.py` is a standalone CLI utility that validates a
  CSV joint trajectory against UR5e position / velocity / acceleration
  limits. It is independent of the main experiments:
  ```bash
  python validate_trajectory.py <path/to/joint_trajectory.csv> [--plot]
  ```
- `capsule_collision_checker.py` is required only by Experiments 3 and 4
  and is duplicated in both directories so that each script remains
  self-contained.

## License

See `LICENSE`.
