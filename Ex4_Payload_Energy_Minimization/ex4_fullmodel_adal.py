#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADA-L payload-energy trajectory optimization with the full UR5e dynamics
========================================================================

The Ex4 ADA-L optimizer is run with the full UR5e rigid-body inverse-dynamics
torque model in the loss (in place of the decoupled diagonal surrogate):

    tau = M(q) qddot + C(q,qdot) qdot + g(q) + D qdot

Full rigid-body dynamics model and parameter sources
----------------------------------------------------
* Equation of motion, inertia matrix M(q), Coriolis/centrifugal term
  C(q,qdot)qdot (Christoffel symbols) and gravity g(q):
  B. Siciliano, L. Sciavicco, L. Villani, G. Oriolo, "Robotics: Modelling,
  Planning and Control", Springer, 2009 (Ch. 3 geometric Jacobian; Ch. 7
  dynamics).
* UR5e DH parameters, link masses and centres of mass:
  Universal Robots, "DH parameters for calculations of kinematics and dynamics".
* UR5e joint-side viscous friction D (recursive Newton-Euler identification):
  E. Clochiatti, L. Scalera, P. Boscariol, A. Gasparetto, "Electro-mechanical
  modeling and identification of the UR5 e-series robot", Robotica 42(7),
  2430-2452, 2024.

Implementation notes
--------------------
* Legendre-ADA-L basis, L-BFGS/Adam optimizer, collision loss, metrics and
  plotting are imported unchanged from the Ex4 script; only the joint-torque
  model (PayloadAwareJointDynamics.torque) is replaced with the full dynamics.
* The full torque is computed in float64 TensorFlow (differentiable). The
  Coriolis/centrifugal term uses Christoffel symbols with dM/dq by central
  finite differences of M(q), so gradients flow for the L-BFGS optimizer.
* On start-up a one-shot numeric check confirms the TensorFlow torque matches
  the inlined reference numpy rigid-body model on the ADA-L trajectory.

Run:  python ex4_fullmodel_adal.py
"""

import os
import sys
import csv
import glob
import json
import importlib.util
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# paper figure style (matches the Ex4 report figures)
plt.rcParams.update({
    "font.family": "Arial", "font.size": 14, "axes.titlesize": 14,
    "axes.labelsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 12, "savefig.dpi": 300,
})

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ============================================================
# Reference numpy UR5e rigid-body dynamics (float64), inlined so this script
# is self-contained. Parameters: UR official DH / link mass / COM; link inertia
# negligible (only link-6 izz); viscous friction D from Clochiatti et al. (2024).
# Used for the start-up numeric check and the post-hoc surrogate-cost comparison.
# ============================================================
DTYPE_NP = np.float64
G_VEC = np.array([0.0, 0.0, -9.81], dtype=DTYPE_NP)
ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0], dtype=DTYPE_NP)
A_DH = np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0], dtype=DTYPE_NP)
D_DH = np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996], dtype=DTYPE_NP)
THETA_OFFSET = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0], dtype=DTYPE_NP)
LINK_MASS = np.array([3.761, 8.058, 2.846, 1.37, 1.3, 0.365], dtype=DTYPE_NP)
LINK_COM = np.array([
    [0.0, -0.02561, 0.00193], [0.2125, 0.0, 0.11336], [0.15, 0.0, 0.0265],
    [0.0, -0.0018, 0.01634], [0.0, 0.0018, 0.01634], [0.0, 0.0, -0.001159],
], dtype=DTYPE_NP)
LINK_INERTIA = np.zeros((6, 3, 3), dtype=DTYPE_NP)
LINK_INERTIA[5, 2, 2] = 0.0002
D_VISC = np.array([4.75, 10.73, 3.82, 2.95, 1.14, 1.88], dtype=DTYPE_NP)
PAYLOAD_MASS = 3.0
PAYLOAD_COM_LOCAL = np.array([0.0, 0.0, 0.0875], dtype=DTYPE_NP)


def load_trajectory(csv_path):
    """time, q(robot conv.), qdot, qddot, tau(surrogate) from a trajectory CSV."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            if r:
                rows.append([float(x) for x in r])
    data = np.array(rows, dtype=DTYPE_NP)
    col = {name: i for i, name in enumerate(header)}
    t = data[:, col["time"]]
    q = np.stack([data[:, col[f"q{j}"]] for j in range(1, 7)], axis=1)
    qdot = np.stack([data[:, col[f"qdot{j}"]] for j in range(1, 7)], axis=1)
    qddot = np.stack([data[:, col[f"qddot{j}"]] for j in range(1, 7)], axis=1)
    tau = np.stack([data[:, col[f"tau{j}"]] for j in range(1, 7)], axis=1)
    return t, q, qdot, qddot, tau


def dh_transform(alpha, a, d, theta):
    N = theta.shape[0]
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    T = np.zeros((N, 4, 4), dtype=DTYPE_NP)
    T[:, 0, 0] = ct; T[:, 0, 1] = -st * ca; T[:, 0, 2] = st * sa; T[:, 0, 3] = a * ct
    T[:, 1, 0] = st; T[:, 1, 1] = ct * ca;  T[:, 1, 2] = -ct * sa; T[:, 1, 3] = a * st
    T[:, 2, 1] = sa; T[:, 2, 2] = ca; T[:, 2, 3] = d; T[:, 3, 3] = 1.0
    return T


def fk_frames(q_robot):
    """q_robot [N,6] robot-convention (theta_i directly). Returns z,o,R (len 7)."""
    N = q_robot.shape[0]
    T = np.tile(np.eye(4, dtype=DTYPE_NP), (N, 1, 1))
    z = [T[:, :3, 2].copy()]; o = [T[:, :3, 3].copy()]; R = [T[:, :3, :3].copy()]
    for i in range(6):
        T = np.matmul(T, dh_transform(ALPHA[i], A_DH[i], D_DH[i], q_robot[:, i]))
        z.append(T[:, :3, 2].copy()); o.append(T[:, :3, 3].copy()); R.append(T[:, :3, :3].copy())
    return z, o, R


def mass_matrix_and_gravity(q_robot):
    """Full pose-dependent mass matrix [N,6,6] and gravity torque [N,6] (3 kg payload)."""
    z, o, R = fk_frames(q_robot)
    N = q_robot.shape[0]
    M = np.zeros((N, 6, 6), dtype=DTYPE_NP)
    g = np.zeros((N, 6), dtype=DTYPE_NP)
    for k in range(6):
        p_com = o[k + 1] + np.einsum("nij,j->ni", R[k + 1], LINK_COM[k])
        Jv = np.zeros((N, 3, 6), dtype=DTYPE_NP)
        Jw = np.zeros((N, 3, 6), dtype=DTYPE_NP)
        for i in range(k + 1):
            Jv[:, :, i] = np.cross(z[i], p_com - o[i]); Jw[:, :, i] = z[i]
        M += LINK_MASS[k] * np.einsum("nai,naj->nij", Jv, Jv)
        Iw = np.einsum("nab,bc,ndc->nad", R[k + 1], LINK_INERTIA[k], R[k + 1])
        M += np.einsum("nai,nab,nbj->nij", Jw, Iw, Jw)
        g += -np.einsum("nai,a->ni", Jv, LINK_MASS[k] * G_VEC)
    p_pl = o[6] + np.einsum("nij,j->ni", R[6], PAYLOAD_COM_LOCAL)
    Jv = np.zeros((N, 3, 6), dtype=DTYPE_NP)
    for i in range(6):
        Jv[:, :, i] = np.cross(z[i], p_pl - o[i])
    M += PAYLOAD_MASS * np.einsum("nai,naj->nij", Jv, Jv)
    g += -np.einsum("nai,a->ni", Jv, PAYLOAD_MASS * G_VEC)
    M = 0.5 * (M + np.transpose(M, (0, 2, 1)))
    return M, g


def christoffel_tensor(q_robot, h=1e-6):
    N = q_robot.shape[0]
    dMdq = np.zeros((N, 6, 6, 6), dtype=DTYPE_NP)
    for k in range(6):
        qp = q_robot.copy(); qp[:, k] += h
        qm = q_robot.copy(); qm[:, k] -= h
        Mp, _ = mass_matrix_and_gravity(qp)
        Mm, _ = mass_matrix_and_gravity(qm)
        dMdq[:, :, :, k] = (Mp - Mm) / (2.0 * h)
    c = 0.5 * (dMdq + np.transpose(dMdq, (0, 1, 3, 2)) - np.transpose(dMdq, (0, 2, 3, 1)))
    return c, dMdq


def coriolis(q_robot, qdot, h=1e-6):
    c, dMdq = christoffel_tensor(q_robot, h)
    Cqd = np.einsum("nijk,nj,nk->ni", c, qdot, qdot)
    Cmat = np.einsum("nijk,nk->nij", c, qdot)
    return Cqd, Cmat, dMdq


# ---- surrogate-optimized ADA-L results (run 260514_ex4_bspline_vs_ada-l.py first) ----
def _find_latest_results(prefix):
    matches = sorted(glob.glob(os.path.join(HERE, f"{prefix}_*")))
    if not matches:
        raise FileNotFoundError(
            f"No '{prefix}_*' results folder in {HERE}. "
            f"Run 260514_ex4_bspline_vs_ada-l.py first to generate it.")
    return matches[-1]


SIM_DIR = os.path.join(_find_latest_results("results_payload_energy"), "payload_3kg_centered")
LPA_CSV = os.path.join(SIM_DIR, "ADAL_trajectory_full.csv")
LPA_METRICS = os.path.join(SIM_DIR, "metrics_adal.json")

# ---- load the Ex4 optimizer module (filename starts with a digit) ----
EX4_PATH = os.path.join(HERE, "260514_ex4_bspline_vs_ada-l.py")
_spec = importlib.util.spec_from_file_location("ex4mod", EX4_PATH)
ex4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex4)
DTYPE = ex4.DTYPE  # tf.float32

OUT_DIR = os.path.join(HERE, "fullmodel_adal_output")
os.makedirs(OUT_DIR, exist_ok=True)

F64 = tf.float64
_ALPHA = [float(x) for x in ALPHA]
_A = [float(x) for x in A_DH]
_D = [float(x) for x in D_DH]
_OFF = [float(x) for x in THETA_OFFSET]
_MASS = tf.constant(LINK_MASS, dtype=F64)
_COM = tf.constant(LINK_COM, dtype=F64)                # [6,3]
_INER = tf.constant(LINK_INERTIA, dtype=F64)           # [6,3,3]
_GVEC = tf.constant([0.0, 0.0, -9.81], dtype=F64)


# ============================================================
# float64 TensorFlow full dynamics (differentiable)
# ============================================================
def _dh_tf(alpha, a, d, theta):
    ca = tf.constant(np.cos(alpha), F64); sa = tf.constant(np.sin(alpha), F64)
    ct = tf.cos(theta); st = tf.sin(theta)
    z = tf.zeros_like(theta); o = tf.ones_like(theta)
    a_ = tf.constant(a, F64); d_ = tf.constant(d, F64)
    r1 = tf.stack([ct, -st * ca, st * sa, a_ * ct], axis=-1)
    r2 = tf.stack([st, ct * ca, -ct * sa, a_ * st], axis=-1)
    r3 = tf.stack([z, sa * o, ca * o, d_ * o], axis=-1)
    r4 = tf.stack([z, z, z, o], axis=-1)
    return tf.stack([r1, r2, r3, r4], axis=1)          # [N,4,4]


def _fk(q_int):
    """q_int [N,6] float64 (internal convention). Returns z,o,R lists (7)."""
    N = tf.shape(q_int)[0]
    T = tf.eye(4, batch_shape=[N], dtype=F64)
    z = [T[:, :3, 2]]; o = [T[:, :3, 3]]; R = [T[:, :3, :3]]
    for i in range(6):
        theta = q_int[:, i] + tf.constant(_OFF[i], F64)
        T = tf.matmul(T, _dh_tf(_ALPHA[i], _A[i], _D[i], theta))
        z.append(T[:, :3, 2]); o.append(T[:, :3, 3]); R.append(T[:, :3, :3])
    return z, o, R


def _full_M_and_g(q_int, pl_mass, pl_com):
    """Full pose-dependent mass matrix [N,6,6] and gravity torque [N,6]."""
    z, o, R = _fk(q_int)
    N = tf.shape(q_int)[0]
    M = tf.zeros([N, 6, 6], F64); g = tf.zeros([N, 6], F64)
    for k in range(6):
        p_com = o[k + 1] + tf.einsum("nij,j->ni", R[k + 1], _COM[k])
        Jv_cols, Jw_cols = [], []
        for i in range(6):
            if i <= k:
                r = p_com - o[i]
                Jv_cols.append(tf.linalg.cross(z[i], r)); Jw_cols.append(z[i])
            else:
                Jv_cols.append(tf.zeros([N, 3], F64)); Jw_cols.append(tf.zeros([N, 3], F64))
        Jv = tf.stack(Jv_cols, axis=2); Jw = tf.stack(Jw_cols, axis=2)   # [N,3,6]
        M += _MASS[k] * tf.einsum("nai,naj->nij", Jv, Jv)
        Iw = tf.einsum("nab,bc,ndc->nad", R[k + 1], _INER[k], R[k + 1])
        M += tf.einsum("nai,nab,nbj->nij", Jw, Iw, Jw)
        g += -tf.einsum("nai,a->ni", Jv, _MASS[k] * _GVEC)
    # payload
    p_pl = o[6] + tf.einsum("nij,j->ni", R[6], pl_com)
    Jv_cols = []
    for i in range(6):
        Jv_cols.append(tf.linalg.cross(z[i], p_pl - o[i]))
    Jv = tf.stack(Jv_cols, axis=2)
    M += pl_mass * tf.einsum("nai,naj->nij", Jv, Jv)
    g += -tf.einsum("nai,a->ni", Jv, pl_mass * _GVEC)
    M = 0.5 * (M + tf.transpose(M, [0, 2, 1]))
    return M, g


def torque_full(self, q, qdot, qddot):
    """Full rigid-body inverse-dynamics torque (drop-in for the surrogate)."""
    q, qdot, qddot = self.align_state(q, qdot, qddot)
    q64 = tf.cast(q, F64); qd = tf.cast(qdot, F64); qdd = tf.cast(qddot, F64)
    pl_m = tf.cast(self.payload_mass, F64)
    pl_c = tf.cast(self.payload_com_local, F64)

    M, g = _full_M_and_g(q64, pl_m, pl_c)
    Mqdd = tf.einsum("nij,nj->ni", M, qdd)

    # dM/dq by central finite differences (differentiable arithmetic)
    h = 1e-6
    dcols = []
    for k in range(6):
        e = tf.one_hot(k, 6, dtype=F64) * h
        Mp, _ = _full_M_and_g(q64 + e, pl_m, pl_c)
        Mm, _ = _full_M_and_g(q64 - e, pl_m, pl_c)
        dcols.append((Mp - Mm) / (2.0 * h))
    dMdq = tf.stack(dcols, axis=-1)                    # [N,6,6,6] (a,b,k)
    c = 0.5 * (dMdq
               + tf.transpose(dMdq, [0, 1, 3, 2])
               - tf.transpose(dMdq, [0, 2, 3, 1]))
    Cqd = tf.einsum("nijk,nj,nk->ni", c, qd, qd)

    D = tf.cast(self.D, F64); K = tf.cast(self.K, F64)
    qref = tf.cast(self.q_ref, F64)
    tau = Mqdd + Cqd + D[None, :] * qd + K[None, :] * (q64 - qref[None, :]) + g
    return tf.cast(tau, DTYPE)


# ============================================================
# one-shot correctness check vs the validated numpy model
# ============================================================
def startup_check():
    t, q_rob, qdot, qddot, _ = load_trajectory(LPA_CSV)
    q_int = q_rob - THETA_OFFSET[None, :]              # internal convention

    dyn = ex4.PayloadAwareJointDynamics(
        robot=ex4.UR5eKinematics(),
        D=[4.75, 10.73, 3.82, 2.95, 1.14, 1.88], K=[0.0] * 6,
        q_ref=[0.0] * 6, tau_limit=[150, 150, 150, 28, 28, 28],
        payload_mass=3.0, payload_com_local=[0.0, 0.0, 0.0875],
        gravity_vec=[0.0, 0.0, -9.81],
    )
    tau_tf = dyn.torque(tf.constant(q_int, DTYPE),
                        tf.constant(qdot, DTYPE),
                        tf.constant(qddot, DTYPE)).numpy()

    # numpy reference (float64) on robot-convention q
    M, g = mass_matrix_and_gravity(q_rob)
    Cqd, _, _ = coriolis(q_rob, qdot)
    Mqdd = np.einsum("nij,nj->ni", M, qddot)
    tau_np = Mqdd + Cqd + g + D_VISC[None, :] * qdot

    err = float(np.max(np.abs(tau_tf - tau_np)))
    rel = err / float(np.max(np.abs(tau_np)))
    print(f"[startup-check] max|tau_TF - tau_numpy| = {err:.3e} N·m "
          f"(rel {rel:.2e})  -> {'OK' if rel < 1e-3 else 'MISMATCH!'}")
    if rel >= 1e-3:
        raise SystemExit("Patched full-dynamics torque does NOT match the "
                         "validated numpy model; aborting before training.")


# ============================================================
# main: patch, check, re-optimize ADA-L with full dynamics
# ============================================================
def main():
    ex4.PayloadAwareJointDynamics.torque = torque_full
    print("Patched PayloadAwareJointDynamics.torque -> full rigid-body dynamics")
    startup_check()

    # same task / hyperparameters as Ex4 __main__ (3 kg centred)
    q0_robot = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, np.pi/2],
                        dtype=np.float32)
    theta_off = np.array(ex4.UR5eKinematics().theta_offset, dtype=np.float32)
    q0 = (q0_robot - theta_off).astype(np.float32)
    waypoints_xyz = np.array([[-0.05, 0.25, 0.3], [0.35, -0.35, 0.7],
                              [0.60, -0.15, 0.55]], dtype=np.float32)
    waypoint_times = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    T_final, Nt = 6.0, 301
    payload_mass, payload_com = 3.0, np.array([0.0, 0.0, 0.0875], dtype=np.float32)

    print("\n=== Re-optimizing ADA-L with FULL dynamics (3 kg) ===")
    adal = ex4.ADALMultiWaypointPlanner(
        q0=q0, waypoints_xyz=waypoints_xyz, waypoint_times=waypoint_times,
        T_final=T_final, Nt=Nt, gamma=1.0, max_order=5, N_p=30, seed=0,
        free_waypoint_times=True, tau_min_gap=0.20, tau_sigma=0.04,
        payload_mass=payload_mass, payload_com_local=payload_com,
    )
    solve_time = adal.train(epochs=500, lr=3e-2, print_every=100,
                            use_lbfgs=True, lbfgs_maxiter=2000)
    res = adal.results()
    metrics = ex4.benchmark_multiwaypoint_metrics(res, solve_time)

    ok, errs, summary = ex4.validate_ur5e_trajectory(res["t_q"], res["q"])
    coll_ok, coll_first, coll_n, coll_total = ex4.validate_collision(res["q"])

    # ---- ADA-L-only figures (full dynamics): every Ex4 figure TYPE, but with
    #      the B-spline curve dropped (ADA-L only). Ex4 paper style, no titles. ----
    figs = os.path.join(OUT_DIR, "figures")
    os.makedirs(figs, exist_ok=True)
    jl = [f"Joint {j+1}" for j in range(6)]
    wp_t = np.asarray(res["waypoint_times_opt"])

    def _norm_fig(t, y, ylabel, fname):
        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        ax.plot(t, y, linewidth=1.0)
        for tk in wp_t:
            ax.axvline(float(tk), color="k", ls=":", alpha=0.3, lw=0.8)
        ax.set_xlabel(r"Time, $t$ (s)"); ax.set_ylabel(ylabel)
        ex4.style_2d_axes(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(figs, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # per-joint bundles (angle / velocity / acceleration / jerk / torque);
    # y-labels copied verbatim from the Ex4 report (two-line format)
    for arr, ylab, fn in (
        (res["q"],     "Angle,\n$q$ (rad)",                          "adal_full_joint_angle.png"),
        (res["qdot"],  "Angular velocity,\n$\\dot{q}$ (rad/s)",       "adal_full_joint_velocity.png"),
        (res["qddot"], "Angular acceleration,\n$\\ddot{q}$ (rad/s$^2$)", "adal_full_joint_acceleration.png"),
        (res["qjerk"], "Angular jerk,\n$\\dddot{q}$ (rad/s$^3$)",     "adal_full_joint_jerk.png"),
        (res["tau"],   "Torque,\n$\\tau$ (N·m)",                      "adal_full_joint_torque.png"),
    ):
        ex4.plot_joint_bundle_single(res["t_q"], arr, "", ylab, jl, wp_t=wp_t,
                                     save_path=os.path.join(figs, fn), show=False)

    # norm profiles (ADA-L only; B-spline curve dropped); y-labels verbatim
    # from the Ex4 report (two-line format)
    _norm_fig(res["t_q"], np.linalg.norm(res["qdot"], axis=1),
              "Angular velocity \nnorm, $\\|\\dot{q}\\|$ (rad/s)",
              "adal_full_velocity_norm.png")
    _norm_fig(res["t_q"], np.linalg.norm(res["qddot"], axis=1),
              "Angular acceleration \nnorm, $\\|\\ddot{q}\\|$ (rad/s$^2$)",
              "adal_full_acceleration_norm.png")
    _norm_fig(res["t_q"], np.linalg.norm(res["qjerk"], axis=1),
              "Angular jerk norm, \n$\\|\\dddot{q}\\|$ (rad/s$^3$)",
              "adal_full_jerk_norm.png")
    _norm_fig(res["t_q"], np.linalg.norm(res["tau"], axis=1),
              "Torque norm, \n$\\|\\tau\\|$ (N·m)", "adal_full_torque_norm.png")
    _norm_fig(res["t_q"], np.sum(np.abs(res["power_joint"]), axis=1),
              r"$\sum_i |\tau_i\dot q_i|$ (W)", "adal_full_joint_abs_power.png")

    # EE path (ADA-L only), matching the Ex4 compare_ee_path 3-D style (no title)
    ee = res["ee"]; wp = np.asarray(res["waypoints_xyz"]); ee_wp = res["ee_wp"]
    wp_colors = ["tab:blue", "tab:orange", "tab:red", "tab:purple", "tab:brown"]
    fig = plt.figure(figsize=(6.6, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], linewidth=2.2, label="ADA-L")
    ax.scatter([ee[0, 0]], [ee[0, 1]], [ee[0, 2]], s=42, color="k", label="start")
    for k in range(len(wp)):
        c = wp_colors[k % len(wp_colors)]
        ax.scatter([wp[k, 0]], [wp[k, 1]], [wp[k, 2]], s=110, marker="*",
                   color=c, label=f"waypoint {k+1}")
        ax.plot([wp[k, 0], ee_wp[k, 0]], [wp[k, 1], ee_wp[k, 1]],
                [wp[k, 2], ee_wp[k, 2]], "-", alpha=0.5, color="C0")
    ex4.style_3d_axes(ax)
    ax.set_xlabel(r"$x$ (m)", labelpad=2)
    ax.set_ylabel(r"$y$ (m)", labelpad=2)
    ax.set_zlabel(r"$z$ (m)", labelpad=2)
    ax.set_xlim(-0.5, 0.6); ax.set_ylim(-0.6, 0.5); ax.set_zlim(0, 1.1)
    ax.legend(loc="upper center", ncol=2, frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "adal_full_ee_path.png"), dpi=300)
    plt.close(fig)

    # waypoint position error (ADA-L only), matching the Ex4 waypoint-error style
    from matplotlib.ticker import ScalarFormatter
    we = np.asarray(metrics.get("waypoint_errors", []))
    if we.size:
        idx = np.arange(1, len(we) + 1)
        fig, ax = plt.subplots(figsize=(5.2, 2.8))
        ax.bar(idx, we, width=0.5)
        ax.set_xlabel("Waypoint index")
        ax.set_ylabel("Position error (m)")
        ex4.style_2d_axes(ax)
        ax.set_xticks(idx)                      # override MaxNLocator for the discrete axis
        sf = ScalarFormatter(useMathText=True)
        sf.set_scientific(True); sf.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(sf)
        fig.tight_layout()
        fig.savefig(os.path.join(figs, "adal_full_waypoint_error.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    # robot snapshots, training losses, panel weights
    try:
        ex4.plot_robot_snapshots_generic(
            robot=adal.robot, q=res["q"], waypoints_xyz=waypoints_xyz,
            waypoint_times=wp_t, time_grid=res["t_q"], n_show=13, title="",
            save_path=os.path.join(figs, "adal_full_robot_snapshots.png"), show=False)
    except Exception as e:
        print(f"  [warn] robot snapshots skipped: {e}")
    ex4.plot_training_losses_generic(
        adal.history, title="",
        save_path=os.path.join(figs, "adal_full_training_losses.png"), show=False)
    try:
        ex4.plot_panel_weights(
            adal, T_final=adal.T_final,
            save_path=os.path.join(figs, "adal_full_panel_weights.png"), show=False)
    except Exception as e:
        print(f"  [warn] panel weights skipped: {e}")
    print(f"[figures] ADA-L-only full-model figure set -> {figs}")

    # ---- surrogate-optimized trajectory, full-model cost (from post-hoc) ----
    # surrogate-optimized ADA-L trajectory, evaluated under the FULL model
    # (inlined post-hoc; replaces the external coriolis_analysis summary).
    ts, qs, qds, qdds, _ = load_trajectory(LPA_CSV)
    Ms, gs = mass_matrix_and_gravity(qs)
    Cs, _, _ = coriolis(qs, qds)
    tau_s = np.einsum("nij,nj->ni", Ms, qdds) + Cs + gs + D_VISC[None, :] * qds
    dts = float((ts[-1] - ts[0]) / (len(ts) - 1))
    sur_full = {
        "integrated_squared_torque": float(np.sum(np.sum(tau_s ** 2, axis=1)) * dts),
        "integrated_joint_abs_work": float(np.sum(np.sum(np.abs(tau_s * qds), axis=1)) * dts),
    }
    with open(LPA_METRICS, "r", encoding="utf-8") as f:
        paper = json.load(f)

    print("\n" + "=" * 68)
    print("RESULT: ADA-L optimized under full dynamics (3 kg)")
    print("=" * 68)
    print(f"  converged solve_time      : {solve_time:.2f} s")
    print(f"  mean waypoint error [m]   : {metrics['mean_waypoint_error']:.3e}")
    print(f"  max  waypoint error [m]   : {metrics['max_waypoint_error']:.3e}")
    print(f"  UR5e limits / collision   : "
          f"{'PASS' if ok else 'FAIL'} / {'PASS' if coll_ok else 'FAIL'}")
    print(f"  waypoint_times_opt        : {np.round(res['waypoint_times_opt'],4)}")
    print("\n  Full-dynamics cost comparison (both evaluated with full model):")
    print(f"                                {'∫||tau||^2 dt':>16} {'work ∫|tau*qd|dt':>18}")
    print(f"    surrogate-optimized traj  : {sur_full['integrated_squared_torque']:16.3f} "
          f"{sur_full['integrated_joint_abs_work']:18.3f}")
    print(f"    full-optimized traj (new) : {metrics['integrated_squared_torque']:16.3f} "
          f"{metrics['integrated_joint_abs_work']:18.3f}")
    d_tau = (metrics['integrated_squared_torque'] /
             sur_full['integrated_squared_torque'] - 1) * 100
    d_wrk = (metrics['integrated_joint_abs_work'] /
             sur_full['integrated_joint_abs_work'] - 1) * 100
    print(f"    change (full-opt vs sur-opt): {d_tau:+.2f}% (torque²), "
          f"{d_wrk:+.2f}% (work)")

    # ---- save ----
    ex4.save_joint_full_csv(
        res["t_q"], res["q"], res["qdot"], res["qddot"], res["qjerk"],
        res["tau"], res["power_joint"],
        os.path.join(OUT_DIR, "ADAL_fullmodel_trajectory.csv"))

    out = {
        "description": "ADA-L re-optimized with full UR5e dynamics (3 kg)",
        "converged": True,
        "solve_time_s": float(solve_time),
        "waypoint_error_mean_m": float(metrics["mean_waypoint_error"]),
        "waypoint_error_max_m": float(metrics["max_waypoint_error"]),
        "ur5e_limits_pass": bool(ok),
        "collision_pass": bool(coll_ok),
        "waypoint_times_opt": [float(x) for x in res["waypoint_times_opt"]],
        "full_optimized_cost": {
            "integrated_squared_torque": float(metrics["integrated_squared_torque"]),
            "integrated_joint_abs_work": float(metrics["integrated_joint_abs_work"]),
            "max_torque_norm": float(metrics["max_torque_norm"]),
        },
        "surrogate_optimized_cost_full_model": {
            "integrated_squared_torque": sur_full["integrated_squared_torque"],
            "integrated_joint_abs_work": sur_full["integrated_joint_abs_work"],
        },
        "surrogate_optimized_cost_surrogate_model_paper": {
            "integrated_squared_torque": paper["integrated_squared_torque"],
            "integrated_joint_abs_work": paper["integrated_joint_abs_work"],
        },
        "pct_change_full_opt_vs_surrogate_opt": {
            "integrated_squared_torque": d_tau,
            "integrated_joint_abs_work": d_wrk,
        },
    }
    with open(os.path.join(OUT_DIR, "fullmodel_adal_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {os.path.join(OUT_DIR, 'fullmodel_adal_summary.json')}")
    print(f"[saved] {os.path.join(OUT_DIR, 'ADAL_fullmodel_trajectory.csv')}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
