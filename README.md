<h1 align="center">ADA-L</h1>
<p align="center">
  <strong>Manufacturing-Task-Specific Trajectory Planning via Latent Acceleration Panel Representation</strong>
  <br>
  <sub><em>ADA-L = Anti-derivatives Approximator from Legendre polynomials</em></sub>
</p>

This repository contains the source code accompanying the paper.
It includes the proposed **ADA-L** planner together with baseline
planners (**Quintic**, **Direct transcription**, **B-spline**),
all evaluated on a UR5e 6-DoF arm in both **simulation** and on
**real hardware**.

**Requirements:** Python 3.9+, TensorFlow 2.13+. See [`requirements.txt`](requirements.txt).

---

## Contents

- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Running the experiments](#running-the-experiments)
- [Sim-vs-hardware comparison](#sim-vs-hardware-comparison)
- [Output format](#output-format)
- [Optional: MP4 animation export](#optional-mp4-animation-export)
- [Other utilities](#other-utilities)

---

## Quick start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run any experiment (each script is self-contained)
python Ex1_Point2Point_new/260514_ex1_ada-l_vs_quintic_vs_dircol.py

# 3. (Optional) Compare simulation against real-robot data
python Ex1_Point2Point_new/ex1_sim_vs_experiment_savgol_wide.py
```

Each script writes results into a timestamped folder placed next to
itself.

> **GPU is optional.** TensorFlow will automatically use a CUDA-enabled
> GPU if one is visible; CPU-only execution works as well.

---

## Repository layout

<details>
<summary><strong>Click to expand the full tree</strong></summary>

```
code_final/
├── Ex1_Point2Point_new/                     # Experiment 1 — Point-to-point
│   ├── 260514_ex1_ada-l_vs_quintic_vs_dircol.py
│   └── ex1_sim_vs_experiment_savgol_wide.py
│
├── Ex2_Point2Point1Orient/                  # Experiment 2 — P2P + one orientation
│   └── 260514_ex2_quintic_vs_ada-l.py
│
├── Ex3_ContinuousWayPoint/                  # Experiment 3 — Multi-waypoint
│   ├── 260514_ex3_bspline_vs_ada-l.py
│   ├── ex3_sim_vs_experiment_savgol_wide.py
│   └── capsule_collision_checker.py
│
├── Ex4_Payload_Energy_Minimization/         # Experiment 4 — Payload energy
│   ├── 260514_ex4_bspline_vs_ada-l.py
│   ├── ex4_sim_vs_experiment_savgol_wide.py
│   └── capsule_collision_checker.py
│
├── Appendix_B_Point2Point3Orient/           # Appendix B — P2P + full orientation
│   └── 260514_Appendix_B.py
│
├── Appendix_C/                              # Appendix C — Cartesian comparison
│   ├── 260514_Appendix_C_acc_x.py
│   └── 260514_Appendix_C_slow.py
│
├── Hardware_validation/                     # Real-robot CSV rollouts
│   ├── adal_p2p_result.csv
│   ├── adal_multi_result.csv
│   ├── adal_payload_result.csv
│   ├── bspline_multi_result.csv
│   ├── bspline_payload_result.csv
│   └── quintic_p2p_result.csv
│
├── validate_trajectory.py                   # Standalone CSV validator
├── requirements.txt
└── README.md
```

</details>

| Folder | Paper section | Planners compared |
|---|---|---|
| `Ex1_Point2Point_new/`  | Experiment 1 | ADA-L vs Quintic vs Direct transcription |
| `Ex2_Point2Point1Orient/` | Experiment 2 | ADA-L vs Quintic |
| `Ex3_ContinuousWayPoint/` | Experiment 3 | ADA-L vs B-spline |
| `Ex4_Payload_Energy_Minimization/` | Experiment 4 | ADA-L vs B-spline |
| `Appendix_B_Point2Point3Orient/` | Appendix B | ADA-L vs Quintic |
| `Appendix_C/` | Appendix C | ADA-L vs Quintic |

---

## Running the experiments

Run any of the scripts below from the **repository root**:

```bash
# Experiment 1 — point-to-point: ADA-L vs Quintic vs Direct transcription
python Ex1_Point2Point_new/260514_ex1_ada-l_vs_quintic_vs_dircol.py

# Experiment 2 — P2P + one orientation
python Ex2_Point2Point1Orient/260514_ex2_quintic_vs_ada-l.py

# Experiment 3 — multi-waypoint
python Ex3_ContinuousWayPoint/260514_ex3_bspline_vs_ada-l.py

# Experiment 4 — payload energy minimization
python Ex4_Payload_Energy_Minimization/260514_ex4_bspline_vs_ada-l.py

# Appendix B — P2P + three-axis orientation
python Appendix_B_Point2Point3Orient/260514_Appendix_B.py

# Appendix C — Cartesian-space comparison (acc_x / slow variants)
python Appendix_C/260514_Appendix_C_acc_x.py
python Appendix_C/260514_Appendix_C_slow.py
```

> You can also `cd` into the corresponding directory and run the
> script directly — outputs are always written **next to the script**
> via `Path(__file__).resolve().parent`.

---

## Sim-vs-hardware comparison

For Experiments 1, 3 and 4 we additionally compare the simulated
trajectories against rollouts collected on the real UR5e:

```bash
python Ex1_Point2Point_new/ex1_sim_vs_experiment_savgol_wide.py
python Ex3_ContinuousWayPoint/ex3_sim_vs_experiment_savgol_wide.py
python Ex4_Payload_Energy_Minimization/ex4_sim_vs_experiment_savgol_wide.py
```

Each comparison script:

1. **Auto-detects** the most recent `results_*` folder next to itself
   (created by the corresponding main script) and loads the simulation
   trajectories (`ADAL_trajectory_full.csv`, `Quintic_trajectory_full.csv`,
   `Bspline_trajectory_full.csv`).
2. **Loads the real-robot rollouts** from `Hardware_validation/`.
3. **Writes** figures into `compare_sim_vs_exp_savgol_wide/` next to
   the script.

> :warning: Run the main experiment script first; otherwise the
> comparison script will raise a `FileNotFoundError`.

---

## Output format

Every script produces a timestamped results directory next to itself:

```
Ex1_Point2Point_new/results_adal_vs_quintic_vs_dircol_<YYYYMMDD_HHMMSS>/
├── figures/                # PNG figures referenced in the paper
├── adal_metrics.json       # Scalar metrics for ADA-L
├── quintic_metrics.json    # Scalar metrics for the baseline
├── trajectory_data.npz     # Raw joint trajectories
├── report.md               # Markdown report
└── report.docx             # DOCX report (requires python-docx)
```

---

## Optional: MP4 animation export

Animation export is **disabled by default** because it requires an
external FFmpeg binary (not a Python package). The figures and
quantitative metrics in the paper do **not** depend on this step.

<details>
<summary><strong>Enable MP4 export</strong> (click to expand)</summary>

### 1. Install FFmpeg

| OS | Command |
|---|---|
| Windows | Download from <https://www.gyan.dev/ffmpeg/builds/> &nbsp;or&nbsp; `conda install -c conda-forge ffmpeg` |
| macOS   | `brew install ffmpeg` |
| Ubuntu  | `sudo apt-get install ffmpeg` |

Verify with:

```bash
ffmpeg -version
```

### 2. Point matplotlib to the binary

Each experiment script contains a commented line of the form:

```python
# matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\Users\user\anaconda3\Library\bin\ffmpeg.exe'
```

Un-comment it and replace the path with the location of your local
FFmpeg executable:

| OS / install | Typical path |
|---|---|
| Windows (conda)      | `r'C:\Users\<you>\anaconda3\Library\bin\ffmpeg.exe'` |
| Windows (standalone) | `r'C:\ffmpeg\bin\ffmpeg.exe'` |
| macOS (Apple Silicon, Homebrew) | `'/opt/homebrew/bin/ffmpeg'` |
| macOS (Intel, Homebrew)         | `'/usr/local/bin/ffmpeg'` |
| Ubuntu               | `'/usr/bin/ffmpeg'` |

Use `which ffmpeg` (macOS / Linux) or `where ffmpeg` (Windows) to find
the exact path.

### 3. Un-comment the animation calls

Each script has 2 – 3 blocks similar to:

```python
# animate_robot_trajectory(
#     robot=robot, q=adal_res["q"], target_xyz=target_xyz,
#     time_grid=adal_res["t"],
#     save_path=str(Path(out_dir) / "ADAL_animation.mp4"),
#     fps=20, title="ADA-L trajectory",
#     xy_center=xy_center)
```

Remove the leading `# ` from every line in the block.

</details>

---

## Other utilities

- **`validate_trajectory.py`** — standalone CLI that validates a CSV
  joint trajectory against UR5e position / velocity / acceleration
  limits. Independent of the main experiments:

  ```bash
  python validate_trajectory.py <path/to/joint_trajectory.csv> [--plot]
  ```

- **`capsule_collision_checker.py`** — used only by Experiments 3 and
  4. A copy is shipped inside each of those directories so the scripts
  remain self-contained.
