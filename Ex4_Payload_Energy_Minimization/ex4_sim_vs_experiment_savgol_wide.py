"""
Ex4 (payload energy minimization, ADA-L vs B-spline) version of
ex3_sim_vs_experiment_savgol_wide.py. Same end-anchored symmetric
extraction, Savgol params, and plot layout. T_WINDOW = 6.0 s.

Sim trajectories live in `payload_3kg_centered/`; experimental CSVs
sit alongside it (no .csv extension).
"""

from pathlib import Path
from datetime import datetime
import csv
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

_FS = 14
plt.rcParams["font.size"] = _FS
plt.rcParams["axes.titlesize"] = _FS
plt.rcParams["axes.labelsize"] = _FS
plt.rcParams["xtick.labelsize"] = _FS
plt.rcParams["ytick.labelsize"] = _FS
plt.rcParams["legend.fontsize"] = _FS


def _read_csv_to_dict(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = {name: [] for name in header}
        for row in reader:
            for name, v in zip(header, row):
                cols[name].append(v)
    return {name: np.asarray(vals, dtype=np.float64) for name, vals in cols.items()}


# ============================================================
# Paths
# ============================================================
HERE = Path(__file__).resolve().parent
# Hardware experimental data (provided in the repo)
HW_DIR = HERE.parent / "Hardware_validation"


def _find_latest_results(prefix):
    """Return the most recent results_<prefix>_<timestamp> folder next to
    this script. The main script must be run first to generate the sim
    trajectories that this comparison script reads."""
    matches = sorted(HERE.glob(f"{prefix}_*"))
    if not matches:
        raise FileNotFoundError(
            f"No '{prefix}_*' folder found in {HERE}. "
            f"Run the corresponding main experiment script first."
        )
    return matches[-1]


# Simulation results from the main experiment script (Ex4 nests its
# per-payload outputs one level deeper).
SIM_DIR = _find_latest_results("results_payload_energy") / "payload_3kg_centered"
# Experimental CSVs (real robot rollouts).
EXP_ADAL_CSV    = HW_DIR / "adal_payload_result.csv"
EXP_BSPLINE_CSV = HW_DIR / "bspline_payload_result.csv"
# Comparison outputs go next to this script.
OUT_DIR = HERE / "compare_sim_vs_exp_savgol_wide"
OUT_DIR.mkdir(parents=True, exist_ok=True)

T_WINDOW = 6.0            # seconds starting from the motion-onset t=0
SAVGOL_WINDOW = 301       # samples (~500 ms at 500 Hz)
SAVGOL_POLYORDER = 4      # must be > deriv order (=3); 4 keeps jerk non-constant
ONSET_EPS = 1e-9          # ||target_qd|| below this is considered "zero"
MARKER_STEP_S = 0.075       # marker spacing in seconds (markers at 0, 0.1, ..., T_WINDOW)
PAD_S = 0.05              # symmetric padding on both sides of displayed [0, T_WINDOW]
END_OFFSET_S = 0.015      # robot settles ~15ms after target_qd reaches zero


# ============================================================
# Style helpers
# ============================================================
def smart_tick_formatter(x, pos):
    if np.isclose(x, round(x)):
        return f"{int(round(x))}"
    return f"{x:.2f}".rstrip("0").rstrip(".")


def apply_tick_style_2d(ax):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    formatter = FuncFormatter(smart_tick_formatter)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)


def style_2d_axes(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    apply_tick_style_2d(ax)


# ============================================================
# Loaders
# ============================================================
def load_sim_csv(path):
    d = _read_csv_to_dict(path)
    t = d["time"]
    q   = np.stack([d[f"q{j}"]     for j in range(1, 7)], axis=1)
    qd  = np.stack([d[f"qdot{j}"]  for j in range(1, 7)], axis=1)
    qdd = np.stack([d[f"qddot{j}"] for j in range(1, 7)], axis=1)
    qj  = np.stack([d[f"qjerk{j}"] for j in range(1, 7)], axis=1)
    return t, q, qd, qdd, qj


def load_sim_torque_power(path):
    """Sim torque (tau) and mechanical power per joint, time-aligned."""
    d = _read_csv_to_dict(path)
    t = d["time"]
    tau   = np.stack([d[f"tau{j}"]   for j in range(1, 7)], axis=1)
    power = np.stack([d[f"power{j}"] for j in range(1, 7)], axis=1)
    return t, tau, power


def plot_raw_with_detection(name, t_abs_u, q_raw, qd_raw,
                            t_zero, t_end, t_origin, save_path):
    """Plot raw trajectory and velocity for all 6 joints with vertical
    markers at detected motion onset and end. Time is rebased to t_origin
    so motion end lands at displayed t = T_WINDOW."""
    t_rel = t_abs_u - t_origin
    onset_rel = t_zero - t_origin
    end_rel   = t_end  - t_origin
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.4), sharex=True)
    cmap = plt.get_cmap("tab10")
    for j in range(6):
        c = cmap(j)
        axes[0].plot(t_rel, q_raw[:, j],  linewidth=0.8, color=c, label=f"j{j+1}")
        axes[1].plot(t_rel, qd_raw[:, j], linewidth=0.8, color=c)
    for ax in axes:
        ax.axvline(onset_rel, color="k", linestyle=":", linewidth=0.8)
        ax.axvline(end_rel,   color="r", linestyle=":", linewidth=0.8)
        style_2d_axes(ax)
    axes[0].set_ylabel(r"$q$ (rad)")
    axes[1].set_ylabel(r"$\dot q$ (rad/s)")
    axes[1].set_xlabel(r"Time, $t$ (s)")
    axes[0].legend(loc="best", frameon=False, fontsize=12, ncol=6)
    axes[0].set_title(
        f"{name}: raw q / qd  (onset={onset_rel:.3f}s, end={end_rel:.3f}s)",
        fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_experiment_csv_savgol(path, t_window=T_WINDOW,
                                window_length=SAVGOL_WINDOW,
                                polyorder=SAVGOL_POLYORDER):
    """Detect motion onset/end from target_qd, end-anchored symmetric
    extraction, Savitzky-Golay derivatives on a uniform grid."""
    d = _read_csv_to_dict(path)
    t_abs = d["timestamp"]

    # Drop duplicate timestamps up front so indices are consistent across cols.
    _, uniq_idx = np.unique(t_abs, return_index=True)
    uniq_idx.sort()
    t_abs_u = t_abs[uniq_idx]
    target_qd = np.stack([d[f"target_qd_{j}"][uniq_idx] for j in range(6)], axis=1)

    # Onset and end from target_qd (command edges).
    speed = np.linalg.norm(target_qd, axis=1)
    zero = speed < ONSET_EPS
    onset_tr = np.where(zero[:-1] & ~zero[1:])[0]      # 0 -> nonzero
    end_tr   = np.where(~zero[:-1] & zero[1:])[0]      # nonzero -> 0
    if len(onset_tr) > 0:
        i_zero = int(onset_tr[0])
    elif not zero[0]:
        i_zero = 0
    else:
        raise RuntimeError(f"No motion onset found in {path.name}")
    if len(end_tr) > 0:
        i_end = int(end_tr[-1]) + 1
    elif not zero[-1]:
        i_end = len(zero) - 1
    else:
        i_end = len(zero) - 1
    t_zero = float(t_abs_u[i_zero])
    t_end  = float(t_abs_u[i_end])

    # End-anchored symmetric extraction: [end_point - T_WINDOW - PAD, end_point + PAD]
    # where end_point = t_end + END_OFFSET_S maps to displayed t = T_WINDOW.
    end_point_abs = t_end + END_OFFSET_S
    t_origin = end_point_abs - t_window
    t_lo = max(t_origin - PAD_S, float(t_abs_u[0]))
    t_hi = min(end_point_abs + PAD_S, float(t_abs_u[-1]))
    mask = (t_abs_u >= t_lo) & (t_abs_u <= t_hi)
    t = t_abs_u[mask]
    q   = np.stack([d[f"actual_q_{j}"][uniq_idx][mask]      for j in range(6)], axis=1)
    tau_raw = np.stack([d[f"target_moment_{j}"][uniq_idx][mask] for j in range(6)], axis=1)
    pe_raw  = np.stack([d[f"power_elec_{j}"][uniq_idx][mask]    for j in range(6)], axis=1)
    pm_raw  = np.stack([d[f"power_mech_{j}"][uniq_idx][mask]    for j in range(6)], axis=1)
    print(f"  [{path.name}] onset orig ts={t_zero:.3f}s, end orig ts={t_end:.3f}s "
          f"(plot origin={t_origin:.3f}s, end+offset -> t=T_WINDOW), "
          f"raw=[{t_lo:.3f}, {t_hi:.3f}], N_raw={mask.sum()}")

    # Save raw trajectory/velocity plot using the same origin.
    q_raw_all  = np.stack([d[f"actual_q_{j}"][uniq_idx]  for j in range(6)], axis=1)
    qd_raw_all = np.stack([d[f"actual_qd_{j}"][uniq_idx] for j in range(6)], axis=1)
    plot_raw_with_detection(path.stem, t_abs_u, q_raw_all, qd_raw_all,
                            t_zero, t_end, t_origin,
                            save_path=str(OUT_DIR / f"{path.stem}_raw_detection.png"))

    # Rebase to displayed time (motion end + offset will land at t = T_WINDOW).
    t = t - t_origin

    # Resample onto a uniform grid with dt = median of original spacings,
    # so that savgol_filter's `delta` is meaningful.
    dt = float(np.median(np.diff(t)))
    N_uni = int(round((t[-1] - t[0]) / dt)) + 1
    t_u = t[0] + np.arange(N_uni) * dt
    q_u   = np.empty((N_uni, 6), dtype=np.float64)
    tau_u = np.empty((N_uni, 6), dtype=np.float64)
    pe_u  = np.empty((N_uni, 6), dtype=np.float64)
    pm_u  = np.empty((N_uni, 6), dtype=np.float64)
    for j in range(6):
        q_u[:, j]   = np.interp(t_u, t, q[:, j])
        tau_u[:, j] = np.interp(t_u, t, tau_raw[:, j])
        pe_u[:, j]  = np.interp(t_u, t, pe_raw[:, j])
        pm_u[:, j]  = np.interp(t_u, t, pm_raw[:, j])

    wl = min(window_length, (N_uni // 2) * 2 + 1)  # ensure odd and <= N
    if wl < polyorder + 2:
        raise ValueError(f"Too few samples ({N_uni}) for savgol window.")

    # q: raw interpolated actual_q (no smoothing).
    # qdot/qddot/qjerk: SavGol deriv 1/2/3 of q_u.
    qd_s  = savgol_filter(q_u,   wl, polyorder, deriv=1, delta=dt, axis=0)
    qdd_s = savgol_filter(q_u,   wl, polyorder, deriv=2, delta=dt, axis=0)
    qj_s  = savgol_filter(q_u,   wl, polyorder, deriv=3, delta=dt, axis=0)
    tau_s = savgol_filter(tau_u, wl, polyorder, deriv=0, delta=dt, axis=0)
    pe_s  = savgol_filter(pe_u,  wl, polyorder, deriv=0, delta=dt, axis=0)
    pm_s  = savgol_filter(pm_u,  wl, polyorder, deriv=0, delta=dt, axis=0)
    # Return raw torque/power too for noise-visible line plots.
    return t_u, q_u, qd_s, qdd_s, qj_s, tau_s, pe_s, pm_s, tau_u, pe_u, pm_u


# ============================================================
# Plotting
# ============================================================
COLOR_ADAL = "tab:blue"
COLOR_BSPLINE = "tab:orange"


def plot_joint_trajectories_4way(
        t_ls, q_ls, t_le, q_le, t_qs, q_qs, t_qe, q_qe,
        save_path=None, show=False):
    joint_labels = [f"Joint {j+1}" for j in range(6)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 5.0), sharex=True)
    axes = axes.ravel()
    for j, ax in enumerate(axes):
        ax.plot(t_ls, q_ls[:, j], "-", color=COLOR_ADAL,     linewidth=1.2,
                label="ADA-L (sim)"     if j == 0 else None)
        ax.plot(t_le, q_le[:, j], "o", color=COLOR_ADAL,     markersize=5.0,
                markerfacecolor="none", markeredgewidth=0.8,
                linestyle="None",
                label="ADA-L (exp)"     if j == 0 else None)
        ax.plot(t_qs, q_qs[:, j], "-", color=COLOR_BSPLINE, linewidth=1.2,
                label="Bspline (sim)" if j == 0 else None)
        ax.plot(t_qe, q_qe[:, j], "o", color=COLOR_BSPLINE, markersize=5.0,
                markerfacecolor="none", markeredgewidth=0.8,
                linestyle="None",
                label="Bspline (exp)" if j == 0 else None)
        ax.set_xlabel(r"Time, $t$ (s)")
        ax.set_ylabel(r"Angle, $q$ (rad)")
        ax.set_xlim(-0.05, T_WINDOW + 0.05)
        if j == 5:
            span_j6 = max(
                np.max(q_ls[:, j]) - np.min(q_ls[:, j]),
                np.max(q_le[:, j]) - np.min(q_le[:, j]),
                np.max(q_qs[:, j]) - np.min(q_qs[:, j]),
                np.max(q_qe[:, j]) - np.min(q_qe[:, j]),
            )
            if span_j6 < 0.1:
                q0 = float(q_ls[0, j])
                ax.set_ylim(q0 - 0.1, q0 + 0.1)
        style_2d_axes(ax)
        ax.set_title(joint_labels[j], fontsize=14)
    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
    #            bbox_to_anchor=(0.5, 1.00), fontsize=13)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig) if not show else plt.show()


def plot_exp_only_2way(
        t_le, n_le, t_qe, n_qe,
        y_label, save_path=None, show=False, exp_style="marker"):
    """Two-curve overlay of ADA-L-exp and Bspline-exp only (no sim)."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if exp_style == "line":
        ax.plot(t_le, n_le, "--", color=COLOR_ADAL,     linewidth=1.0, label="ADA-L (exp)")
        ax.plot(t_qe, n_qe, "--", color=COLOR_BSPLINE, linewidth=1.0, label="Bspline (exp)")
    else:
        ax.plot(t_le, n_le, "o", color=COLOR_ADAL,     markersize=5.0,
                markerfacecolor="none", markeredgewidth=0.8,
                linestyle="None", label="ADA-L (exp)")
        ax.plot(t_qe, n_qe, "o", color=COLOR_BSPLINE, markersize=5.0,
                markerfacecolor="none", markeredgewidth=0.8,
                linestyle="None", label="Bspline (exp)")
    ax.set_xlabel(r"Time, $t$ (s)")
    ax.set_ylabel(y_label)
    ax.set_xlim(-0.05, T_WINDOW + 0.05)
    style_2d_axes(ax)
    # ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3),
    #           ncol=2, frameon=False, fontsize=13)
    # fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig) if not show else plt.show()


def plot_dual_axis_4way(
        t_ls, n_ls, t_le, n_le, t_qs, n_qs, t_qe, n_qe,
        y_label_left, y_label_right,
        save_path=None, show=False, exp_style="marker"):
    """Sim on left axis, exp on right axis. Each axis fits its own data
    so the exp offset (e.g. idle motor electrical consumption) doesn't
    flatten the sim curve. `exp_style` is 'marker' (sparse circles) or
    'line' (full-resolution solid line)."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(t_ls, n_ls, "-", color=COLOR_ADAL,     linewidth=1.2, label="ADA-L (sim)")
    ax.plot(t_qs, n_qs, "-", color=COLOR_BSPLINE, linewidth=1.2, label="Bspline (sim)")
    ax.set_xlabel(r"Time, $t$ (s)")
    ax.set_ylabel(y_label_left)
    ax.set_xlim(-0.05, T_WINDOW + 0.05)

    ax2 = ax.twinx()
    if exp_style == "line":
        ax2.plot(t_le, n_le, "--", color=COLOR_ADAL,     linewidth=1.0, label="ADA-L (exp)")
        ax2.plot(t_qe, n_qe, "--", color=COLOR_BSPLINE, linewidth=1.0, label="Bspline (exp)")
    else:
        ax2.plot(t_le, n_le, "o", color=COLOR_ADAL,     markersize=5.0,
                 markerfacecolor="none", markeredgewidth=0.8,
                 linestyle="None", label="ADA-L (exp)")
        ax2.plot(t_qe, n_qe, "o", color=COLOR_BSPLINE, markersize=5.0,
                 markerfacecolor="none", markeredgewidth=0.8,
                 linestyle="None", label="Bspline (exp)")
    ax2.set_ylabel(y_label_right)

    # Independent y-ranges: each axis fits its own data with a small margin.
    def _ylim_from(arrs, frac=0.05):
        lo = float(min(np.min(a) for a in arrs))
        hi = float(max(np.max(a) for a in arrs))
        m  = frac * (hi - lo) if hi > lo else 1.0
        return lo - m, hi + m
    ax.set_ylim(*_ylim_from([n_ls, n_qs]))
    ax2.set_ylim(*_ylim_from([n_le, n_qe]))

    style_2d_axes(ax)
    apply_tick_style_2d(ax2)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)

    # handles1, labels1 = ax.get_legend_handles_labels()
    # handles2, labels2 = ax2.get_legend_handles_labels()
    # ax.legend(handles1 + handles2, labels1 + labels2,
    #           loc="upper center", bbox_to_anchor=(0.5, 1.3),
    #           ncol=4, frameon=False, fontsize=13)
    # fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig) if not show else plt.show()


def plot_norm_profile_4way(
        t_ls, n_ls, t_le, n_le, t_qs, n_qs, t_qe, n_qe,
        y_label, save_path=None, show=False, exp_style="marker"):
    """Single-axis overlay. `exp_style` is 'marker' (sparse circles) or
    'line' (full-resolution dashed line)."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(t_ls, n_ls, "-", color=COLOR_ADAL,     linewidth=1.2, label="ADA-L (sim)")
    ax.plot(t_qs, n_qs, "-", color=COLOR_BSPLINE, linewidth=1.2, label="Bspline (sim)")
    if exp_style == "line":
        ax.plot(t_le, n_le, "--", color=COLOR_ADAL,     linewidth=1.0, label="ADA-L (exp)")
        ax.plot(t_qe, n_qe, "--", color=COLOR_BSPLINE, linewidth=1.0, label="Bspline (exp)")
    else:
        ax.plot(t_le, n_le, "o", color=COLOR_ADAL,     markersize=5.0,
                markerfacecolor="none", markeredgewidth=0.8,
                linestyle="None", label="ADA-L (exp)")
        ax.plot(t_qe, n_qe, "o", color=COLOR_BSPLINE, markersize=5.0,
                markerfacecolor="none", markeredgewidth=0.8,
                linestyle="None", label="Bspline (exp)")
    ax.set_xlabel(r"Time, $t$ (s)")
    ax.set_ylabel(y_label)
    ax.set_xlim(-0.05, T_WINDOW + 0.05)
    style_2d_axes(ax)
    # ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3),
    #           ncol=4, frameon=False, fontsize=13)
    # fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig) if not show else plt.show()


# ============================================================
# Experimental metrics report (mirror of sim's report table)
# ============================================================
def compute_exp_metrics(t, q, qdot, qddot, qjerk, tau, pe, pm,
                        t_lo=0.0, t_hi=T_WINDOW):
    """Compute the same scalar metrics as
    `260409_ex4_bspline_v3_coll.compute_motion_metrics_multiwaypoint`,
    plus exp-only `net_joint_elec_energy` and `gross_joint_elec_energy`.
    Integrals use the same `Σ * dt` (left-Riemann) convention as the sim
    so values are directly comparable."""
    m = (t >= t_lo) & (t <= t_hi)
    t = t[m]; q = q[m]; qdot = qdot[m]; qddot = qddot[m]; qjerk = qjerk[m]
    tau = tau[m]; pe = pe[m]; pm = pm[m]
    T = float(t[-1] - t[0])
    n = len(t)
    dt = T / (n - 1) if n > 1 else 0.0

    metrics = {
        "joint_path_length":    float(np.sum(np.linalg.norm(q[1:] - q[:-1], axis=1))),
        "integrated_squared_velocity":     float(np.sum(np.sum(qdot ** 2,  axis=1)) * dt),
        "integrated_squared_acceleration": float(np.sum(np.sum(qddot ** 2, axis=1)) * dt),
        "integrated_squared_jerk":         float(np.sum(np.sum(qjerk ** 2, axis=1)) * dt),
        "mean_squared_velocity":     float(np.mean(np.sum(qdot ** 2,  axis=1))),
        "mean_squared_acceleration": float(np.mean(np.sum(qddot ** 2, axis=1))),
        "mean_squared_jerk":         float(np.mean(np.sum(qjerk ** 2, axis=1))),
        "max_velocity_norm":     float(np.max(np.linalg.norm(qdot,  axis=1))),
        "max_acceleration_norm": float(np.max(np.linalg.norm(qddot, axis=1))),
        "max_jerk_norm":         float(np.max(np.linalg.norm(qjerk, axis=1))),
        "max_abs_velocity_each_joint":     np.max(np.abs(qdot),  axis=0),
        "max_abs_acceleration_each_joint": np.max(np.abs(qddot), axis=0),
        "max_abs_jerk_each_joint":         np.max(np.abs(qjerk), axis=0),
        "final_velocity_norm":     float(np.linalg.norm(qdot[-1])),
        "final_acceleration_norm": float(np.linalg.norm(qddot[-1])),
        "integrated_squared_torque": float(np.sum(np.sum(tau ** 2, axis=1)) * dt),
        "mean_squared_torque":       float(np.mean(np.sum(tau ** 2, axis=1))),
        "max_torque_norm":           float(np.max(np.linalg.norm(tau, axis=1))),
        "max_abs_torque_each_joint": np.max(np.abs(tau), axis=0),
    }

    pm_abs_per_t = np.sum(np.abs(pm), axis=1)
    metrics.update({
        "integrated_joint_abs_work": float(np.sum(pm_abs_per_t) * dt),
        "mean_joint_abs_power":      float(np.mean(pm_abs_per_t)),
        "max_joint_abs_power":       float(np.max(pm_abs_per_t)),
    })

    pe_sum_t = np.sum(pe, axis=1)
    metrics.update({
        "net_joint_elec_energy":   float(np.sum(pe_sum_t) * dt),
        "gross_joint_elec_energy": float(np.sum(np.maximum(0.0, pe_sum_t)) * dt),
    })
    return metrics


def compute_sim_metrics(t, q, qdot, qddot, qjerk, tau, power_joint,
                        t_lo=0.0, t_hi=T_WINDOW):
    """Same shared metrics as `compute_exp_metrics` but for sim arrays.
    No electrical signal -> no net/gross elec keys."""
    m = (t >= t_lo) & (t <= t_hi)
    t = t[m]; q = q[m]; qdot = qdot[m]; qddot = qddot[m]; qjerk = qjerk[m]
    tau = tau[m]; power_joint = power_joint[m]
    T = float(t[-1] - t[0])
    n = len(t)
    dt = T / (n - 1) if n > 1 else 0.0

    metrics = {
        "joint_path_length":    float(np.sum(np.linalg.norm(q[1:] - q[:-1], axis=1))),
        "integrated_squared_velocity":     float(np.sum(np.sum(qdot ** 2,  axis=1)) * dt),
        "integrated_squared_acceleration": float(np.sum(np.sum(qddot ** 2, axis=1)) * dt),
        "integrated_squared_jerk":         float(np.sum(np.sum(qjerk ** 2, axis=1)) * dt),
        "mean_squared_velocity":     float(np.mean(np.sum(qdot ** 2,  axis=1))),
        "mean_squared_acceleration": float(np.mean(np.sum(qddot ** 2, axis=1))),
        "mean_squared_jerk":         float(np.mean(np.sum(qjerk ** 2, axis=1))),
        "max_velocity_norm":     float(np.max(np.linalg.norm(qdot,  axis=1))),
        "max_acceleration_norm": float(np.max(np.linalg.norm(qddot, axis=1))),
        "max_jerk_norm":         float(np.max(np.linalg.norm(qjerk, axis=1))),
        "max_abs_velocity_each_joint":     np.max(np.abs(qdot),  axis=0),
        "max_abs_acceleration_each_joint": np.max(np.abs(qddot), axis=0),
        "max_abs_jerk_each_joint":         np.max(np.abs(qjerk), axis=0),
        "final_velocity_norm":     float(np.linalg.norm(qdot[-1])),
        "final_acceleration_norm": float(np.linalg.norm(qddot[-1])),
        "integrated_squared_torque": float(np.sum(np.sum(tau ** 2, axis=1)) * dt),
        "mean_squared_torque":       float(np.mean(np.sum(tau ** 2, axis=1))),
        "max_torque_norm":           float(np.max(np.linalg.norm(tau, axis=1))),
        "max_abs_torque_each_joint": np.max(np.abs(tau), axis=0),
    }
    pj_abs_per_t = np.sum(np.abs(power_joint), axis=1)
    metrics.update({
        "integrated_joint_abs_work": float(np.sum(pj_abs_per_t) * dt),
        "mean_joint_abs_power":      float(np.mean(pj_abs_per_t)),
        "max_joint_abs_power":       float(np.max(pj_abs_per_t)),
    })
    return metrics


def _reduction_pct(adal, bsp):
    if not np.isfinite(bsp) or bsp == 0.0:
        return float("nan")
    return (adal - bsp) / bsp * 100.0


# Keys present in sim's report table that we can also compute from exp.
SHARED_METRIC_KEYS = [
    "joint_path_length",
    "integrated_squared_velocity", "integrated_squared_acceleration", "integrated_squared_jerk",
    "mean_squared_velocity", "mean_squared_acceleration", "mean_squared_jerk",
    "max_velocity_norm", "max_acceleration_norm", "max_jerk_norm",
    "final_velocity_norm", "final_acceleration_norm",
    "integrated_squared_torque", "mean_squared_torque", "max_torque_norm",
    "integrated_joint_abs_work", "mean_joint_abs_power", "max_joint_abs_power",
]
# Exp-only metrics (no sim counterpart).
EXP_ONLY_METRIC_KEYS = [
    "net_joint_elec_energy", "gross_joint_elec_energy",
]


def write_exp_report_docx(docx_path,
                          adal_sim, bsp_sim, adal_exp, bsp_exp,
                          fig_paths, name_a="ADA-L", name_b="Bspline"):
    """Write a docx report combining sim and exp metrics side-by-side."""
    try:
        from docx import Document
        from docx.shared import Inches
    except Exception as e:
        print(f"[report] python-docx unavailable: {e}")
        return False

    doc = Document()
    doc.add_heading(f"Sim vs Exp metrics: {name_a} vs {name_b}", level=0)
    doc.add_paragraph(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph(
        f"Window: t in [0, {T_WINDOW}] s. "
        f"Savgol window={SAVGOL_WINDOW}, polyorder={SAVGOL_POLYORDER}.")

    def _add_shared_table(keys):
        doc.add_heading("Shared metrics (sim + exp)", level=1)
        table = doc.add_table(rows=1, cols=7)
        try:
            table.style = "Light Grid"
        except KeyError:
            pass
        hdr = table.rows[0].cells
        for i, h in enumerate([
            "Metric",
            f"{name_a} (sim)", f"{name_b} (sim)", f"{name_a} red.% (sim)",
            f"{name_a} (exp)", f"{name_b} (exp)", f"{name_a} red.% (exp)",
        ]):
            hdr[i].text = h
        for k in keys:
            a_s = float(adal_sim.get(k, float("nan")))
            b_s = float(bsp_sim.get(k, float("nan")))
            a_e = float(adal_exp.get(k, float("nan")))
            b_e = float(bsp_exp.get(k, float("nan")))
            row = table.add_row().cells
            row[0].text = k
            row[1].text = f"{a_s:.6e}"
            row[2].text = f"{b_s:.6e}"
            row[3].text = f"{_reduction_pct(a_s, b_s):+.2f}"
            row[4].text = f"{a_e:.6e}"
            row[5].text = f"{b_e:.6e}"
            row[6].text = f"{_reduction_pct(a_e, b_e):+.2f}"

    def _add_exp_only_table(keys):
        doc.add_heading("Experimental-only metrics", level=1)
        table = doc.add_table(rows=1, cols=4)
        try:
            table.style = "Light Grid"
        except KeyError:
            pass
        hdr = table.rows[0].cells
        for i, h in enumerate(
                ["Metric", f"{name_a} (exp)", f"{name_b} (exp)",
                 f"{name_a} red.% (exp)"]):
            hdr[i].text = h
        for k in keys:
            a = float(adal_exp.get(k, float("nan")))
            b = float(bsp_exp.get(k, float("nan")))
            row = table.add_row().cells
            row[0].text = k
            row[1].text = f"{a:.6e}"
            row[2].text = f"{b:.6e}"
            row[3].text = f"{_reduction_pct(a, b):+.2f}"

    _add_shared_table(SHARED_METRIC_KEYS)
    _add_exp_only_table(EXP_ONLY_METRIC_KEYS)

    doc.add_heading("Per-joint extrema (exp)", level=1)
    for lbl, m in [(name_a, adal_exp), (name_b, bsp_exp)]:
        doc.add_paragraph(
            f"{lbl} max abs velocity each joint: "
            f"{np.array2string(m['max_abs_velocity_each_joint'], precision=5)}")
        doc.add_paragraph(
            f"{lbl} max abs acceleration each joint: "
            f"{np.array2string(m['max_abs_acceleration_each_joint'], precision=5)}")
        doc.add_paragraph(
            f"{lbl} max abs jerk each joint: "
            f"{np.array2string(m['max_abs_jerk_each_joint'], precision=5)}")
        doc.add_paragraph(
            f"{lbl} max abs torque each joint: "
            f"{np.array2string(m['max_abs_torque_each_joint'], precision=5)}")

    if fig_paths:
        doc.add_heading("Figures", level=1)
        for title, abs_path in fig_paths:
            if Path(abs_path).exists():
                doc.add_heading(title, level=2)
                doc.add_picture(str(abs_path), width=Inches(6.2))

    doc.save(str(docx_path))
    print(f"[report] DOCX saved to: {docx_path}")
    return True


# ============================================================
# Main
# ============================================================
def main():
    sim_adal_csv     = SIM_DIR / "ADAL_trajectory_full.csv"
    sim_bspline_csv = SIM_DIR / "Bspline_trajectory_full.csv"
    for p in (sim_adal_csv, sim_bspline_csv, EXP_ADAL_CSV, EXP_BSPLINE_CSV):
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    t_ls, q_ls, qd_ls, qdd_ls, qj_ls = load_sim_csv(sim_adal_csv)
    t_qs, q_qs, qd_qs, qdd_qs, qj_qs = load_sim_csv(sim_bspline_csv)
    _, tau_ls, pwr_ls = load_sim_torque_power(sim_adal_csv)
    _, tau_qs, pwr_qs = load_sim_torque_power(sim_bspline_csv)
    (t_le, q_le, qd_le, qdd_le, qj_le,
     tau_le, pe_le, pm_le,
     tau_le_raw, pe_le_raw, pm_le_raw) = load_experiment_csv_savgol(EXP_ADAL_CSV)
    (t_qe, q_qe, qd_qe, qdd_qe, qj_qe,
     tau_qe, pe_qe, pm_qe,
     tau_qe_raw, pe_qe_raw, pm_qe_raw) = load_experiment_csv_savgol(EXP_BSPLINE_CSV)

    print(f"[sim/ADA-L]     N={len(t_ls)}, t in [{t_ls[0]:.3f}, {t_ls[-1]:.3f}] s")
    print(f"[sim/Bspline] N={len(t_qs)}, t in [{t_qs[0]:.3f}, {t_qs[-1]:.3f}] s")
    print(f"[exp/ADA-L]     N={len(t_le)}, t in [{t_le[0]:.3f}, {t_le[-1]:.3f}] s "
          f"(savgol window={SAVGOL_WINDOW}, poly={SAVGOL_POLYORDER})")
    print(f"[exp/Bspline] N={len(t_qe)}, t in [{t_qe[0]:.3f}, {t_qe[-1]:.3f}] s "
          f"(savgol window={SAVGOL_WINDOW}, poly={SAVGOL_POLYORDER})")

    # Pick experiment samples nearest to target marker times 0, MARKER_STEP_S, ..., T_WINDOW.
    target_times = np.arange(0.0, T_WINDOW + 1e-9, MARKER_STEP_S)
    def _pick(t_arr, *arrs):
        idx = np.array([int(np.argmin(np.abs(t_arr - tt))) for tt in target_times])
        return (t_arr[idx],) + tuple(a[idx] for a in arrs)
    t_le_p, q_le_p, qd_le_p, qdd_le_p, qj_le_p, tau_le_p, pe_le_p, pm_le_p = _pick(
        t_le, q_le, qd_le, qdd_le, qj_le, tau_le, pe_le, pm_le)
    t_qe_p, q_qe_p, qd_qe_p, qdd_qe_p, qj_qe_p, tau_qe_p, pe_qe_p, pm_qe_p = _pick(
        t_qe, q_qe, qd_qe, qdd_qe, qj_qe, tau_qe, pe_qe, pm_qe)

    plot_joint_trajectories_4way(
        t_ls, q_ls, t_le_p, q_le_p, t_qs, q_qs, t_qe_p, q_qe_p,
        save_path=str(OUT_DIR / "joint_trajectory_compare.png"))

    v_ls = np.linalg.norm(qd_ls, axis=1)
    v_qs = np.linalg.norm(qd_qs, axis=1)
    a_ls = np.linalg.norm(qdd_ls, axis=1)
    a_qs = np.linalg.norm(qdd_qs, axis=1)
    j_ls = np.linalg.norm(qj_ls, axis=1)
    j_qs = np.linalg.norm(qj_qs, axis=1)
    v_le_p = np.linalg.norm(qd_le_p,  axis=1)
    v_qe_p = np.linalg.norm(qd_qe_p,  axis=1)
    a_le_p = np.linalg.norm(qdd_le_p, axis=1)
    a_qe_p = np.linalg.norm(qdd_qe_p, axis=1)
    j_le_p = np.linalg.norm(qj_le_p,  axis=1)
    j_qe_p = np.linalg.norm(qj_qe_p,  axis=1)

    plot_norm_profile_4way(
        t_ls, v_ls, t_le_p, v_le_p, t_qs, v_qs, t_qe_p, v_qe_p,
        y_label=r"$\|\dot q\|$ (rad/s)",
        save_path=str(OUT_DIR / "velocity_norm_compare.png"))
    plot_norm_profile_4way(
        t_ls, a_ls, t_le_p, a_le_p, t_qs, a_qs, t_qe_p, a_qe_p,
        y_label=r"$\|\ddot q\|$ (rad/s$^2$)",
        save_path=str(OUT_DIR / "acc_norm_compare.png"))
    plot_norm_profile_4way(
        t_ls, j_ls, t_le_p, j_le_p, t_qs, j_qs, t_qe_p, j_qe_p,
        y_label=r"$\|\dddot q\|$ (rad/s$^3$)",
        save_path=str(OUT_DIR / "jerk_norm_compare.png"))

    # Torque norms: sim tau (left) vs exp target_moment (right), dual-axis.
    # Markers use Savgol-filtered exp; lines use raw exp (noise visible).
    tn_ls = np.linalg.norm(tau_ls, axis=1)
    tn_qs = np.linalg.norm(tau_qs, axis=1)
    tn_le_p     = np.linalg.norm(tau_le_p,    axis=1)
    tn_qe_p     = np.linalg.norm(tau_qe_p,    axis=1)
    tn_le_raw   = np.linalg.norm(tau_le_raw,  axis=1)
    tn_qe_raw   = np.linalg.norm(tau_qe_raw,  axis=1)
    plot_norm_profile_4way(
        t_ls, tn_ls, t_le_p, tn_le_p, t_qs, tn_qs, t_qe_p, tn_qe_p,
        y_label=r"$\|\tau\|$ (N$\cdot$m)",
        save_path=str(OUT_DIR / "torque_norm_compare.png"),
        exp_style="marker")
    plot_norm_profile_4way(
        t_ls, tn_ls, t_le, tn_le_raw, t_qs, tn_qs, t_qe, tn_qe_raw,
        y_label=r"$\|\tau\|$ (N$\cdot$m)",
        save_path=str(OUT_DIR / "torque_norm_compare_lines.png"),
        exp_style="line")


    # Power norms: sim P_mech as reference; exp has both P_mech and P_elec.
    pwn_ls = np.linalg.norm(pwr_ls, axis=1)
    pwn_qs = np.linalg.norm(pwr_qs, axis=1)
    pen_le_p   = np.sum(pe_le_p,   axis=1)
    pen_qe_p   = np.sum(pe_qe_p,   axis=1)
    pen_le_raw = np.sum(pe_le_raw, axis=1)
    pen_qe_raw = np.sum(pe_qe_raw, axis=1)
    pmn_le_p   = np.linalg.norm(pm_le_p,   axis=1)
    pmn_qe_p   = np.linalg.norm(pm_qe_p,   axis=1)
    pmn_le_raw = np.linalg.norm(pm_le_raw, axis=1)
    pmn_qe_raw = np.linalg.norm(pm_qe_raw, axis=1)

    # Sim mech vs exp mech (same units): single-axis.
    plot_norm_profile_4way(
        t_ls, pwn_ls, t_le_p, pmn_le_p, t_qs, pwn_qs, t_qe_p, pmn_qe_p,
        y_label=r"$\|P_\mathrm{mech}\|$ (W)",
        save_path=str(OUT_DIR / "power_mech_norm_compare.png"),
        exp_style="marker")
    plot_norm_profile_4way(
        t_ls, pwn_ls, t_le, pmn_le_raw, t_qs, pwn_qs, t_qe, pmn_qe_raw,
        y_label=r"$\|P_\mathrm{mech}\|$ (W)",
        save_path=str(OUT_DIR / "power_mech_norm_compare_lines.png"),
        exp_style="line")

    # Exp electrical power: ADA-L vs Bspline only (no sim — different physics).
    # Use signed joint sum so regen (negative) is visible.
    plot_exp_only_2way(
        t_le_p, pen_le_p, t_qe_p, pen_qe_p,
        y_label=r"$\sum_j P_{\mathrm{elec},j}$ (W)",
        save_path=str(OUT_DIR / "power_elec_exp_only.png"),
        exp_style="marker")
    plot_exp_only_2way(
        t_le, pen_le_raw, t_qe, pen_qe_raw,
        y_label=r"$\sum_j P_{\mathrm{elec},j}$ (W)",
        save_path=str(OUT_DIR / "power_elec_exp_only_lines.png"),
        exp_style="line")

    # ------------------------------------------------------------
    # Experimental energy/effort metrics over the displayed window.
    # ------------------------------------------------------------
    def _integrate_window(t, y, t_lo=0.0, t_hi=T_WINDOW):
        # Σ × dt (left-Riemann), matching sim's convention and compute_*_metrics.
        m = (t >= t_lo) & (t <= t_hi)
        t_m = t[m]; y_m = y[m]
        n = len(t_m)
        if n < 2:
            return 0.0
        dt = (t_m[-1] - t_m[0]) / (n - 1)
        return float(np.sum(y_m) * dt)

    # ∫ ||τ_cmd||² dt   (sum over joints of τ_j²)
    isq_adal  = _integrate_window(t_le, np.sum(tau_le ** 2, axis=1))
    isq_bsp  = _integrate_window(t_qe, np.sum(tau_qe ** 2, axis=1))

    # Net joint electrical energy:  ∫ Σ_j P_elec_j dt   (signed, allows
    # cross-joint regen via shared DC bus — academic-standard metric).
    P_sum_e_adal = np.sum(pe_le, axis=1)
    P_sum_e_bsp = np.sum(pe_qe, axis=1)
    e_net_adal = _integrate_window(t_le, P_sum_e_adal)
    e_net_bsp = _integrate_window(t_qe, P_sum_e_bsp)

    # Gross joint electrical energy:  ∫ max(0, Σ_j P_elec_j) dt   (unidirectional
    # supply + bleeder; regen surplus dumped as heat → mains sees only positive draws).
    e_gross_adal = _integrate_window(t_le, np.maximum(0.0, P_sum_e_adal))
    e_gross_bsp = _integrate_window(t_qe, np.maximum(0.0, P_sum_e_bsp))

    # Integrated absolute mechanical work:  ∫ Σ_j |P_mech_j| dt
    # (matches sim's `integrated_joint_abs_work`).
    iabs_m_adal = _integrate_window(t_le, np.sum(np.abs(pm_le), axis=1))
    iabs_m_bsp = _integrate_window(t_qe, np.sum(np.abs(pm_qe), axis=1))

    print(f"\n[Experimental metrics over t in [0, {T_WINDOW}] s]")
    print(f"  Integrated squared torque  (N^2 m^2 s):")
    print(f"    ADA-L     = {isq_adal:.4f}")
    print(f"    Bspline = {isq_bsp:.4f}")
    print(f"    ADA-L reduction = {_reduction_pct(isq_adal, isq_bsp):+.2f} %")
    print(f"  Net joint electrical energy  ( ∫ Σ_j P_elec_j dt , J ):")
    print(f"    ADA-L     = {e_net_adal:.4f}")
    print(f"    Bspline = {e_net_bsp:.4f}")
    print(f"    ADA-L reduction = {_reduction_pct(e_net_adal, e_net_bsp):+.2f} %")
    print(f"  Gross joint electrical energy  ( ∫ max(0, Σ_j P_elec_j) dt , J ):")
    print(f"    ADA-L     = {e_gross_adal:.4f}")
    print(f"    Bspline = {e_gross_bsp:.4f}")
    print(f"    ADA-L reduction = {_reduction_pct(e_gross_adal, e_gross_bsp):+.2f} %")
    print(f"  Integrated absolute work  ( Σ_j ∫ |P_mech_j| dt , J ):")
    print(f"    ADA-L     = {iabs_m_adal:.4f}")
    print(f"    Bspline = {iabs_m_bsp:.4f}")
    print(f"    ADA-L reduction = {_reduction_pct(iabs_m_adal, iabs_m_bsp):+.2f} %")

    # ------------------------------------------------------------
    # Experimental metrics report (mirror of sim's report table).
    # ------------------------------------------------------------
    metrics_adal_sim = compute_sim_metrics(
        t_ls, q_ls, qd_ls, qdd_ls, qj_ls, tau_ls, pwr_ls)
    metrics_bsp_sim = compute_sim_metrics(
        t_qs, q_qs, qd_qs, qdd_qs, qj_qs, tau_qs, pwr_qs)
    metrics_adal_exp = compute_exp_metrics(
        t_le, q_le, qd_le, qdd_le, qj_le, tau_le, pe_le, pm_le)
    metrics_bsp_exp = compute_exp_metrics(
        t_qe, q_qe, qd_qe, qdd_qe, qj_qe, tau_qe, pe_qe, pm_qe)

    fig_entries = [
        ("Joint trajectory compare",          OUT_DIR / "joint_trajectory_compare.png"),
        ("Velocity norm compare",             OUT_DIR / "velocity_norm_compare.png"),
        ("Acceleration norm compare",         OUT_DIR / "acc_norm_compare.png"),
        ("Jerk norm compare",                 OUT_DIR / "jerk_norm_compare.png"),
        ("Torque norm compare",               OUT_DIR / "torque_norm_compare.png"),
        ("Torque norm compare (lines)",       OUT_DIR / "torque_norm_compare_lines.png"),
        ("Power mech norm compare",           OUT_DIR / "power_mech_norm_compare.png"),
        ("Power mech norm compare (lines)",   OUT_DIR / "power_mech_norm_compare_lines.png"),
        ("Power elec exp only",               OUT_DIR / "power_elec_exp_only.png"),
        ("Power elec exp only (lines)",       OUT_DIR / "power_elec_exp_only_lines.png"),
    ]
    docx_path = OUT_DIR / "report_sim_vs_exp.docx"
    write_exp_report_docx(docx_path,
                          metrics_adal_sim, metrics_bsp_sim,
                          metrics_adal_exp, metrics_bsp_exp,
                          fig_entries, name_a="ADA-L", name_b="Bspline")

    print(f"\nSaved comparison figures under: {OUT_DIR}")


if __name__ == "__main__":
    main()
