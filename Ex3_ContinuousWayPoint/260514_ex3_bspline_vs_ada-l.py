#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark 4: Continuous Multi-Waypoint Passing
- ADA-L multi-waypoint continuous-passing planner
- B-spline multi-waypoint continuous-passing baseline

Metrics, CSV, figures follow ex1_v12_kdj_paperfig_csv style:
  - save_joint_trajectory_csv added
  - compute_loss wrapper removed (compute_loss_train used directly)
  - single-method and training-loss plots removed
  - ex1-style publication-quality axes/fonts

Notes
-----
- T_final is configurable (passed via constructor argument)
- Intermediate waypoints are pass-through constraints only (no stop)
- Final waypoint enforces terminal velocity/acceleration penalties
- ADA-L: analytic derivatives from Legendre anti-derivative basis
- B-spline: finite differences for derivatives
"""

import os
import time
import json
import csv
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, FuncFormatter
from matplotlib.lines import Line2D
import sympy as sp
import scipy.optimize
from pathlib import Path
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

DTYPE = tf.float32
np.random.seed(0)
tf.random.set_seed(0)

fontsize = 14

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = fontsize
plt.rcParams["axes.titlesize"] = fontsize
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize
plt.rcParams["legend.fontsize"] = fontsize


# ============================================================
# 0) IO helpers
# ============================================================
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    def convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.float32, np.float64, np.float16)):
            return float(v)
        if isinstance(v, (np.int32, np.int64, np.int16)):
            return int(v)
        return v
    data2 = {k: convert(v) for k, v in data.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data2, f, indent=2, ensure_ascii=False)


def save_comparison_csv(metrics_a, metrics_b, path, name_a="ADA-L", name_b="B-spline"):
    keys = [
        "mean_waypoint_error",
        "max_waypoint_error",
        "final_waypoint_error",
        "joint_path_length",
        "ee_path_length",
        "integrated_squared_velocity",
        "integrated_squared_acceleration",
        "integrated_squared_jerk",
        "mean_squared_velocity",
        "mean_squared_acceleration",
        "mean_squared_jerk",
        "max_velocity_norm",
        "max_acceleration_norm",
        "max_jerk_norm",
        "final_velocity_norm",
        "final_acceleration_norm",
        "solve_time",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", name_a, name_b, f"{name_a}/{name_b}", f"{name_b}/{name_a}"])
        for k in keys:
            a, b = float(metrics_a[k]), float(metrics_b[k])
            ratio_ab = a / b if abs(b) > 1e-15 else np.nan
            ratio_ba = b / a if abs(a) > 1e-15 else np.nan
            writer.writerow([k, a, b, ratio_ab, ratio_ba])


def save_joint_trajectory_csv(t, q, path):
    """Save joint trajectory CSV with theta_i = q + theta_offset (robot convention)."""
    t = np.asarray(t, dtype=np.float32).reshape(-1)
    q = np.asarray(q, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 6:
        raise ValueError("q must have shape (N, 6)")
    if q.shape[0] != t.shape[0]:
        raise ValueError("len(t) must match q.shape[0]")
    theta_offset = np.array(UR5eKinematics().theta_offset, dtype=np.float32)
    q_robot = q + theta_offset[None, :]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "q1", "q2", "q3", "q4", "q5", "q6"])
        for i in range(len(t)):
            writer.writerow([float(t[i])] + [float(v) for v in q_robot[i]])


def save_joint_full_csv(t, q, qdot, qddot, qjerk, path):
    """Save per-method CSV with time, q (robot convention), qdot, qddot, qjerk for all 6 joints."""
    t = np.asarray(t, dtype=np.float32).reshape(-1)
    q = np.asarray(q, dtype=np.float32)
    qdot = np.asarray(qdot, dtype=np.float32)
    qddot = np.asarray(qddot, dtype=np.float32)
    qjerk = np.asarray(qjerk, dtype=np.float32)
    N = t.shape[0]
    for name, arr in (("q", q), ("qdot", qdot), ("qddot", qddot), ("qjerk", qjerk)):
        if arr.ndim != 2 or arr.shape != (N, 6):
            raise ValueError(f"{name} must have shape ({N}, 6), got {arr.shape}")
    theta_offset = np.array(UR5eKinematics().theta_offset, dtype=np.float32)
    q_robot = q + theta_offset[None, :]
    header = ["time"]
    for base in ("q", "qdot", "qddot", "qjerk"):
        header += [f"{base}{j}" for j in range(1, 7)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(N):
            row = [float(t[i])]
            row += [float(v) for v in q_robot[i]]
            row += [float(v) for v in qdot[i]]
            row += [float(v) for v in qddot[i]]
            row += [float(v) for v in qjerk[i]]
            writer.writerow(row)


# ============================================================
# 1) Symbolic helpers for Legendre anti-derivative basis
# ============================================================
def sympy_poly_to_numpy_coeffs(expr, x_sym, dtype=np.float32):
    poly = sp.Poly(sp.expand(expr), x_sym)
    coeff_dict = poly.as_dict()
    max_deg = poly.degree()
    coeffs = np.zeros((max_deg + 1,), dtype=np.float64)
    for k, v in coeff_dict.items():
        coeffs[k[0]] = float(v)
    return coeffs.astype(dtype)


def build_legendre_symbolic_tables(max_order=6, dtype=np.float32):
    x = sp.symbols("x", real=True)
    P_coeffs, I1_coeffs, I2_coeffs, I3_coeffs = {}, {}, {}, {}
    for n in range(max_order + 1):
        Pn = sp.legendre(n, x)
        I1 = sp.expand(sp.integrate(Pn, x) - sp.integrate(Pn, x).subs(x, -1))
        I2 = sp.expand(sp.integrate(I1, x) - sp.integrate(I1, x).subs(x, -1))
        I3 = sp.expand(sp.integrate(I2, x) - sp.integrate(I2, x).subs(x, -1))
        P_coeffs[n]  = sympy_poly_to_numpy_coeffs(Pn, x, dtype=dtype)
        I1_coeffs[n] = sympy_poly_to_numpy_coeffs(I1, x, dtype=dtype)
        I2_coeffs[n] = sympy_poly_to_numpy_coeffs(I2, x, dtype=dtype)
        I3_coeffs[n] = sympy_poly_to_numpy_coeffs(I3, x, dtype=dtype)
    return P_coeffs, I1_coeffs, I2_coeffs, I3_coeffs


def pad_coeff_dict_to_common_width(coeff_dict, max_order, common_width, dtype=np.float32):
    mat = np.zeros((max_order + 1, common_width), dtype=dtype)
    for n in range(max_order + 1):
        c = coeff_dict[n]
        mat[n, :len(c)] = c
    return mat


def get_legendre_panel_coefs_sympy_on_custom_panels(order, panel_edges, dtype=np.float32):
    x = sp.symbols("x", real=True)
    Pint = sp.integrate(sp.legendre(order, x), x)
    vals = np.array([float(Pint.subs(x, s)) for s in panel_edges], dtype=np.float64)
    coefs = (vals[1:] - vals[:-1]) * (2.0 * order + 1.0) / 2.0
    return coefs.astype(dtype)


def differentiate_poly_matrix(poly_mat_np):
    poly_mat_np = np.asarray(poly_mat_np, dtype=np.float32)
    n_basis, width = poly_mat_np.shape
    out = np.zeros_like(poly_mat_np)
    for k in range(1, width):
        out[:, k - 1] = k * poly_mat_np[:, k]
    return out


# ============================================================
# 2) Legendre anti-derivative approximator
# ============================================================
class LegendreADAFReusable(tf.Module):
    def __init__(self, xgrid_phys_mapped, L=1.0, gamma=1.0, max_order=5,
                 N_p=10, init1=0.0, init2=0.0, init3=0.0,
                 dtype=DTYPE, seed=0, name=None):
        super().__init__(name=name)
        self.dtype = dtype
        self.L = tf.constant(L, dtype)
        self.gamma = float(gamma)
        self.max_order = int(max_order)
        self.N_p = int(N_p)
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError("gamma must satisfy 0 < gamma <= 1")
        self.x_gamma_np = -1.0 + 2.0 * self.gamma
        self.x_gamma = tf.constant(self.x_gamma_np, dtype=dtype)
        xgrid_phys_mapped = np.asarray(xgrid_phys_mapped, dtype=np.float32)
        if xgrid_phys_mapped.ndim != 1:
            raise ValueError("xgrid_phys_mapped must be 1D")
        if np.any(np.diff(xgrid_phys_mapped) < 0):
            raise ValueError("xgrid_phys_mapped must be monotone increasing")
        if xgrid_phys_mapped[0] < -1.0 - 1e-7 or xgrid_phys_mapped[-1] > self.x_gamma_np + 1e-7:
            raise ValueError("xgrid_phys_mapped must lie in [-1, x_gamma]")
        self.x = tf.constant(xgrid_phys_mapped, dtype=dtype)
        self.xm = self.x + tf.constant(1.0, dtype=dtype)
        self.Nt = int(xgrid_phys_mapped.shape[0])
        if self.gamma == 1.0:
            panel_edges_np = np.linspace(-1.0, 1.0, self.N_p + 1).astype(np.float32)
        else:
            panel_edges_np = np.concatenate((
                np.linspace(-1.0, self.x_gamma_np, self.N_p - 1, dtype=np.float32)[:-1],
                np.linspace(self.x_gamma_np, 1.0, 3, dtype=np.float32),
            )).astype(np.float32)
        self.panel_edges = tf.constant(panel_edges_np, dtype=dtype)
        rng = np.random.default_rng(seed)
        self.U = tf.Variable(rng.uniform(-0.5, 0.5, size=(self.N_p - 1,)).astype(np.float32),
                             dtype=dtype, name="U")
        self.init1 = tf.Variable(float(init1), dtype=dtype, trainable=False, name="init1")
        self.init2 = tf.Variable(float(init2), dtype=dtype, trainable=False, name="init2")
        self.init3 = tf.Variable(float(init3), dtype=dtype, trainable=False, name="init3")
        P_dict, I1_dict, I2_dict, I3_dict = build_legendre_symbolic_tables(
            max_order=self.max_order, dtype=np.float32)
        cw = self.max_order + 4
        self.P_poly_mat  = tf.constant(pad_coeff_dict_to_common_width(P_dict,  self.max_order, cw), dtype=dtype)
        self.I1_poly_mat = tf.constant(pad_coeff_dict_to_common_width(I1_dict, self.max_order, cw), dtype=dtype)
        self.I2_poly_mat = tf.constant(pad_coeff_dict_to_common_width(I2_dict, self.max_order, cw), dtype=dtype)
        self.I3_poly_mat = tf.constant(pad_coeff_dict_to_common_width(I3_dict, self.max_order, cw), dtype=dtype)
        coef_rows = [get_legendre_panel_coefs_sympy_on_custom_panels(n, panel_edges_np)
                     for n in range(1, self.max_order + 1)]
        self.coef_mat = tf.constant(np.stack(coef_rows, axis=0).astype(np.float32), dtype=dtype)
        self.mean_vec = tf.constant(np.ones((self.N_p,), dtype=np.float32) / float(self.N_p), dtype=dtype)
        order_idx = tf.constant(np.arange(1, self.max_order + 1), dtype=tf.int32)
        self.P_cache  = self._build_basis_cache(self.P_poly_mat,  order_idx)
        self.I1_cache = self._build_basis_cache(self.I1_poly_mat, order_idx)
        self.I2_cache = self._build_basis_cache(self.I2_poly_mat, order_idx)
        self.I3_cache = self._build_basis_cache(self.I3_poly_mat, order_idx)
        self.P0_cache   = self._eval_poly_matrix(self.P_poly_mat[0:1],  self.x)[0]
        self.I1_0_cache = self._eval_poly_matrix(self.I1_poly_mat[0:1], self.x)[0]
        self.I2_0_cache = self._eval_poly_matrix(self.I2_poly_mat[0:1], self.x)[0]
        self.I3_0_cache = self._eval_poly_matrix(self.I3_poly_mat[0:1], self.x)[0]
        self.Pd_poly_mat = tf.constant(differentiate_poly_matrix(self.P_poly_mat.numpy()), dtype=dtype)
        self.Pd_cache    = self._build_basis_cache(self.Pd_poly_mat, order_idx)
        self.Pd0_cache   = self._eval_poly_matrix(self.Pd_poly_mat[0:1], self.x)[0]

    def get_W(self):
        return tf.concat([self.U, -tf.reduce_sum(self.U, keepdims=True)], axis=0)

    def coeffs(self):
        W = self.get_W()
        return tf.tensordot(self.mean_vec, W, axes=1), tf.linalg.matvec(self.coef_mat, W)

    def _eval_poly_matrix(self, poly_mat, x):
        poly_mat = tf.convert_to_tensor(poly_mat, dtype=self.dtype)
        x = tf.convert_to_tensor(x, dtype=self.dtype)
        y = tf.zeros((tf.shape(poly_mat)[0], tf.shape(x)[0]), dtype=self.dtype)
        for c in tf.unstack(tf.reverse(poly_mat, axis=[1]), axis=1):
            y = y * x[None, :] + c[:, None]
        return y

    def _build_basis_cache(self, poly_mat, order_idx):
        return self._eval_poly_matrix(tf.gather(poly_mat, order_idx, axis=0), self.x)

    def f_from_coeffs(self, a0, A):
        return a0 * self.P0_cache + tf.reduce_sum(A[:, None] * self.P_cache, axis=0)

    def g1_from_coeffs(self, a0, A):
        return a0 * self.I1_0_cache + tf.reduce_sum(A[:, None] * self.I1_cache, axis=0) + self.init1

    def g2_from_coeffs(self, a0, A):
        return (a0 * self.I2_0_cache + tf.reduce_sum(A[:, None] * self.I2_cache, axis=0)
                + self.init1 * self.xm + self.init2)

    def g3_from_coeffs(self, a0, A):
        return (a0 * self.I3_0_cache + tf.reduce_sum(A[:, None] * self.I3_cache, axis=0)
                + 0.5 * self.init1 * tf.square(self.xm) + self.init2 * self.xm + self.init3)

    def dfdx_from_coeffs(self, a0, A):
        return a0 * self.Pd0_cache + tf.reduce_sum(A[:, None] * self.Pd_cache, axis=0)

    def q(self):
        a0, A = self.coeffs()
        return self.g2_from_coeffs(a0, A)

    def q_qdot_qddot_qjerk(self, time_scale):
        a0, A = self.coeffs()
        return (self.g2_from_coeffs(a0, A),
                time_scale * self.g1_from_coeffs(a0, A),
                (time_scale**2) * self.f_from_coeffs(a0, A),
                (time_scale**3) * self.dfdx_from_coeffs(a0, A))


# ============================================================
# 3) UR5e kinematics
# ============================================================
def dh_transform(alpha, a, d, theta):
    theta = tf.cast(theta, DTYPE)
    ca = tf.constant(np.cos(alpha), dtype=DTYPE)
    sa = tf.constant(np.sin(alpha), dtype=DTYPE)
    ct, st = tf.cos(theta), tf.sin(theta)
    row1 = tf.stack([ct, -st*ca,  st*sa, tf.constant(a, dtype=DTYPE)*ct], axis=-1)
    row2 = tf.stack([st,  ct*ca, -ct*sa, tf.constant(a, dtype=DTYPE)*st], axis=-1)
    row3 = tf.stack([tf.zeros_like(theta), sa*tf.ones_like(theta),
                     ca*tf.ones_like(theta), tf.constant(d, dtype=DTYPE)*tf.ones_like(theta)], axis=-1)
    row4 = tf.stack([tf.zeros_like(theta)]*3 + [tf.ones_like(theta)], axis=-1)
    return tf.stack([row1, row2, row3, row4], axis=1)


class UR5eKinematics:
    def __init__(self):
        # UR5e manufacturer DH parameters (Joint 1~6)
        self.alpha = [np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0]
        self.a     = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
        self.d     = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
        self.theta_offset = [0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0]

    def forward_all_points(self, q_batch):
        q_batch = tf.cast(q_batch, DTYPE)
        N = tf.shape(q_batch)[0]
        T = tf.eye(4, batch_shape=[N], dtype=DTYPE)
        points = [T[:, :3, 3]]
        for i in range(6):
            theta_i = q_batch[:, i] + tf.constant(self.theta_offset[i], dtype=DTYPE)
            T = tf.matmul(T, dh_transform(self.alpha[i], self.a[i], self.d[i], theta_i))
            points.append(T[:, :3, 3])
        return points, T[:, :3, 3]


# ============================================================
# 4) B-spline basis helpers
# ============================================================
def make_clamped_uniform_knot_vector(n_ctrl, degree):
    if n_ctrl <= degree:
        raise ValueError("n_ctrl must be > degree")
    n_knots = n_ctrl + degree + 1
    kv = np.zeros(n_knots, dtype=np.float64)
    kv[:degree+1] = 0.0
    kv[-(degree+1):] = 1.0
    n_internal = n_knots - 2*(degree+1)
    if n_internal > 0:
        internal = np.linspace(0.0, 1.0, n_internal+2)[1:-1]
        kv[degree+1:degree+1+n_internal] = internal
    return kv


def bspline_basis_one(i, p, u, knots):
    if p == 0:
        if (knots[i] <= u < knots[i+1]) or (
            np.isclose(u, 1.0) and np.isclose(knots[i+1], 1.0) and knots[i] <= u <= knots[i+1]
        ):
            return 1.0
        return 0.0
    left = 0.0
    denom1 = knots[i+p] - knots[i]
    if denom1 > 1e-14:
        left = (u - knots[i]) / denom1 * bspline_basis_one(i, p-1, u, knots)
    right = 0.0
    denom2 = knots[i+p+1] - knots[i+1]
    if denom2 > 1e-14:
        right = (knots[i+p+1] - u) / denom2 * bspline_basis_one(i+1, p-1, u, knots)
    return left + right


def build_bspline_basis_matrix(u_grid, n_ctrl, degree):
    knots = make_clamped_uniform_knot_vector(n_ctrl, degree)
    B = np.zeros((len(u_grid), n_ctrl), dtype=np.float32)
    for r, u in enumerate(u_grid):
        for i in range(n_ctrl):
            B[r, i] = bspline_basis_one(i, degree, float(u), knots)
    return B, knots


def differencing_matrix(p, n_ctrl, knots):
    """[n_ctrl-1, n_ctrl] matrix Diff such that derivative ctrl pts = Diff @ P."""
    M = np.zeros((n_ctrl - 1, n_ctrl), dtype=np.float32)
    for i in range(n_ctrl - 1):
        denom = knots[i + p + 1] - knots[i + 1]
        if denom > 1e-14:
            c = float(p) / float(denom)
            M[i, i]     = -c
            M[i, i + 1] = +c
    return M


def build_derivative_chain(u_grid, n_ctrl, degree):
    """Build B-spline analytic-derivative chain. See adal_vs_bspline_v2.py."""
    knots_p = make_clamped_uniform_knot_vector(n_ctrl, degree)
    B_q, _   = build_bspline_basis_matrix(u_grid, n_ctrl, degree)
    D1 = differencing_matrix(degree, n_ctrl, knots_p)
    knots_p1 = knots_p[1:-1]
    n1 = n_ctrl - 1
    B_qd, _  = build_bspline_basis_matrix(u_grid, n1, degree - 1)
    D2 = differencing_matrix(degree - 1, n1, knots_p1)
    knots_p2 = knots_p1[1:-1]
    n2 = n1 - 1
    B_qdd, _ = build_bspline_basis_matrix(u_grid, n2, degree - 2)
    D3 = differencing_matrix(degree - 2, n2, knots_p2)
    n3 = n2 - 1
    B_qjerk, _ = build_bspline_basis_matrix(u_grid, n3, degree - 3)
    return B_q, D1, B_qd, D2, B_qdd, D3, B_qjerk


# ============================================================
# Free waypoint-time helpers
# ============================================================
def init_gap_logits_from_waypoint_times(waypoint_times, T_final, tau_min_gap=0.15):
    wp = np.asarray(waypoint_times, dtype=np.float64).reshape(-1).copy()
    M = len(wp)
    if M < 1:
        raise ValueError("Need at least one waypoint time.")
    wp[-1] = float(T_final)
    full = np.concatenate([[0.0], wp], axis=0)
    gaps = np.diff(full)
    min_total = tau_min_gap * M
    if min_total >= T_final:
        raise ValueError("tau_min_gap too large.")
    extra = T_final - min_total
    gaps_clipped = np.maximum(gaps, tau_min_gap + 1e-6)
    frac = np.maximum((gaps_clipped - tau_min_gap) / extra, 1e-8)
    frac /= np.sum(frac)
    return np.log(frac).astype(np.float32)


def gaps_to_ordered_waypoint_times(raw_gap_logits, T_final, tau_min_gap=0.15, dtype=DTYPE):
    raw_gap_logits = tf.convert_to_tensor(raw_gap_logits, dtype=dtype)
    M = tf.shape(raw_gap_logits)[0]
    min_total = tf.cast(M, dtype) * tf.cast(tau_min_gap, dtype)
    extra = tf.cast(T_final, dtype) - min_total
    gaps = tf.cast(tau_min_gap, dtype) + extra * tf.nn.softmax(raw_gap_logits)
    return tf.cumsum(gaps)


def soft_time_sample(values, t_grid, tau_vec, sigma=0.04):
    """
    Differentiable soft sampling:
      values: [Nt, d],  t_grid: [Nt],  tau_vec: [M]
      returns: [M, d]
    """
    values   = tf.convert_to_tensor(values, dtype=DTYPE)
    t_grid   = tf.convert_to_tensor(t_grid, dtype=DTYPE)
    tau_vec  = tf.convert_to_tensor(tau_vec, dtype=DTYPE)
    diff = t_grid[None, :] - tau_vec[:, None]
    w = tf.exp(-0.5 * tf.square(diff / tf.cast(sigma, DTYPE)))
    w = w / (tf.reduce_sum(w, axis=1, keepdims=True) + 1e-12)
    return tf.linalg.matmul(w, values)


# ============================================================
# 4b) Thin proxy for plot helpers that access self.models[j].U / .panel_edges
# ============================================================
class _RowSliceVar:
    def __init__(self, parent, row):
        self._p = parent; self._r = row
    @property
    def shape(self): return self._p.shape[1:]
    def numpy(self): return self._p[self._r].numpy()
    def assign(self, v):
        idx = tf.constant([[self._r]], dtype=tf.int32)
        self._p.assign(tf.tensor_scatter_nd_update(
            self._p, idx, tf.reshape(tf.cast(v, self._p.dtype), (1, -1))))

class _JointProxy:
    def __init__(self, planner, j, panel_edges_tf):
        self._planner = planner; self._j = j
        self.panel_edges = panel_edges_tf
    @property
    def U(self): return _RowSliceVar(self._planner.U_batch, self._j)

def _make_joint_proxies(planner, panel_edges_np):
    pe = tf.constant(panel_edges_np, dtype=DTYPE)
    return [_JointProxy(planner, j, pe) for j in range(6)]


# ============================================================
# 5) ADA-L multi-waypoint planner
# ============================================================
class ADALMultiWaypointPlanner:
    def __init__(
        self, q0, waypoints_xyz, waypoint_times,
        T_final=2.0, Nt=101, gamma=1.0, max_order=5, N_p=10, seed=0,
        free_waypoint_times=True, tau_min_gap=0.15, tau_sigma=0.04,
        z_floor=0.0,
    ):
        self.z_floor = tf.constant(float(z_floor), dtype=DTYPE)
        self.q0 = np.asarray(q0, dtype=np.float32).reshape(6,)
        self.waypoints_xyz = np.asarray(waypoints_xyz, dtype=np.float32).reshape(-1, 3)
        self.waypoint_times = np.asarray(waypoint_times, dtype=np.float32).reshape(-1,)
        if len(self.waypoints_xyz) != len(self.waypoint_times):
            raise ValueError("waypoints_xyz and waypoint_times must have same length")
        if not np.all(np.diff(self.waypoint_times) > 0):
            raise ValueError("waypoint_times must be strictly increasing")
        if self.waypoint_times[-1] > T_final + 1e-8:
            raise ValueError("last waypoint time must be <= T_final")

        self.T_final = float(T_final)
        self.Nt = int(Nt)
        self.dt = self.T_final / (self.Nt - 1)
        self.gamma = float(gamma)
        self.optimizer = None
        self.t_np = np.linspace(0.0, self.T_final, self.Nt).astype(np.float32)
        self.xgrid_np = (-1.0 + 2.0 * gamma * (self.t_np / self.T_final)).astype(np.float32)
        self.waypoint_idx = np.array(
            [int(np.argmin(np.abs(self.t_np - tk))) for tk in self.waypoint_times], dtype=np.int32)
        self.waypoints_xyz_tf = tf.constant(self.waypoints_xyz, dtype=DTYPE)
        self.waypoint_idx_tf  = tf.constant(self.waypoint_idx, dtype=tf.int32)
        self.free_waypoint_times = bool(free_waypoint_times)
        self.tau_min_gap = float(tau_min_gap)
        self.tau_sigma   = float(tau_sigma)
        self.t_tf = tf.constant(self.t_np, dtype=DTYPE)

        if self.free_waypoint_times and len(self.waypoint_times) >= 2:
            init_logits = init_gap_logits_from_waypoint_times(
                self.waypoint_times, T_final=self.T_final, tau_min_gap=self.tau_min_gap)
            self.raw_gap_logits = tf.Variable(init_logits, dtype=DTYPE, trainable=True,
                                              name="adal_raw_gap_logits")
        else:
            self.raw_gap_logits = None

        self.robot = UR5eKinematics()
        self.N_p = int(N_p)
        self.max_order = int(max_order)

        # ---- shared basis matrices (built once, shared across all joints) ----
        P_dict, I1_dict, I2_dict, I3_dict = build_legendre_symbolic_tables(
            max_order=max_order, dtype=np.float32)
        cw = max_order + 4

        def _pad(d):
            return pad_coeff_dict_to_common_width(d, max_order, cw)

        P_np  = _pad(P_dict);  I1_np = _pad(I1_dict)
        I2_np = _pad(I2_dict); I3_np = _pad(I3_dict)

        if gamma == 1.0:
            panel_edges_np = np.linspace(-1.0, 1.0, N_p + 1).astype(np.float32)
        else:
            xg = -1.0 + 2.0 * gamma
            panel_edges_np = np.concatenate((
                np.linspace(-1.0, xg, N_p - 1, dtype=np.float32)[:-1],
                np.linspace(xg, 1.0, 3, dtype=np.float32),
            )).astype(np.float32)

        x_tf = tf.constant(self.xgrid_np, dtype=DTYPE)

        def _eval(mat_np, x):
            mat = tf.constant(mat_np, dtype=DTYPE)
            coeffs_rev = tf.reverse(mat, axis=[1])
            y = tf.zeros((mat_np.shape[0], x.shape[0]), dtype=DTYPE)
            for c in tf.unstack(coeffs_rev, axis=1):
                y = y * x[None, :] + c[:, None]
            return y

        order_idx = np.arange(1, max_order + 1)
        self._P_cache   = _eval(P_np[order_idx],  x_tf)
        self._I1_cache  = _eval(I1_np[order_idx], x_tf)
        self._I2_cache  = _eval(I2_np[order_idx], x_tf)
        self._I3_cache  = _eval(I3_np[order_idx], x_tf)
        self._P0_cache  = _eval(P_np[0:1],  x_tf)[0]
        self._I1_0_cache= _eval(I1_np[0:1], x_tf)[0]
        self._I2_0_cache= _eval(I2_np[0:1], x_tf)[0]
        self._I3_0_cache= _eval(I3_np[0:1], x_tf)[0]
        Pd_np = differentiate_poly_matrix(P_np)
        self._Pd_cache  = _eval(Pd_np[order_idx], x_tf)
        self._Pd0_cache = _eval(Pd_np[0:1], x_tf)[0]
        self._xm = x_tf + tf.constant(1.0, dtype=DTYPE)

        coef_rows = [get_legendre_panel_coefs_sympy_on_custom_panels(n, panel_edges_np)
                     for n in range(1, max_order + 1)]
        self._coef_mat = tf.constant(np.stack(coef_rows).astype(np.float32), dtype=DTYPE)
        self._mean_vec = tf.constant(
            np.ones((N_p,), dtype=np.float32) / float(N_p), dtype=DTYPE)
        self._init2 = tf.constant(self.q0, dtype=DTYPE)   # (6,)

        rng = np.random.default_rng(seed)
        U0 = np.stack([rng.uniform(-0.5, 0.5, size=(N_p - 1,)).astype(np.float32)
                       for _ in range(6)], axis=0)         # (6, N_p-1)
        self.U_batch = tf.Variable(U0, dtype=DTYPE, name="U_batch")

        # thin proxy so external code (plot helpers) using self.models still works
        self.models = _make_joint_proxies(self, panel_edges_np)

        # UR5e joint position limits: elbow (joint 3) is +/-pi, others +/-2*pi
        self.q_min = tf.constant([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi], dtype=DTYPE)
        self.q_max = tf.constant([ 2*np.pi,  2*np.pi,  np.pi,  2*np.pi,  2*np.pi,  2*np.pi], dtype=DTYPE)
        # UR5e velocity limit: 180 deg/s = pi rad/s
        self.qdot_max  = tf.constant(np.pi, dtype=DTYPE)
        # UR5e acceleration limit: 2292 deg/s^2 = 40 rad/s^2
        self.qddot_max = tf.constant(40.0, dtype=DTYPE)
        self.history = {
            "loss": [], "waypoint": [], "path": [],
            "acc": [], "jerk": [], "limit": [],
            "final_vel": [], "final_acc": [], "init_acc": [],
        }

    def get_waypoint_times(self):
        if self.raw_gap_logits is None:
            return tf.constant(self.waypoint_times, dtype=DTYPE)
        return gaps_to_ordered_waypoint_times(
            self.raw_gap_logits, T_final=self.T_final, tau_min_gap=self.tau_min_gap, dtype=DTYPE)

    def evaluate_waypoints_from_ee(self, ee):
        tau_all = self.get_waypoint_times()
        if len(self.waypoints_xyz) == 1:
            ee_wp = ee[-1][None, :]
        else:
            tau_intermediate = tau_all[:-1]
            ee_intermediate = soft_time_sample(ee, self.t_tf, tau_intermediate, sigma=self.tau_sigma)
            ee_wp = tf.concat([ee_intermediate, ee[-1][None, :]], axis=0)
        return tau_all, ee_wp

    def get_trainable_variables(self):
        vars_ = [self.U_batch]
        if self.raw_gap_logits is not None:
            vars_.append(self.raw_gap_logits)
        return vars_

    def _batch_q_qdot_qddot_qjerk(self, time_scale):
        """Batched forward pass for all 6 joints simultaneously. Returns (Nt, 6) tensors."""
        w_last = -tf.reduce_sum(self.U_batch, axis=1, keepdims=True)
        W = tf.concat([self.U_batch, w_last], axis=1)          # (6, N_p)
        a0 = tf.linalg.matvec(W, self._mean_vec)               # (6,)
        A  = tf.matmul(W, tf.transpose(self._coef_mat))        # (6, max_order)

        def _apply(a0_, A_, c0_, cache_):
            return a0_[:, None] * c0_[None, :] + tf.reduce_sum(
                A_[:, :, None] * cache_[None, :, :], axis=1)   # (6, Nt)

        g2 = _apply(a0, A, self._I2_0_cache, self._I2_cache) + self._init2[:, None]
        g1 = _apply(a0, A, self._I1_0_cache, self._I1_cache)
        f  = _apply(a0, A, self._P0_cache,   self._P_cache)
        df = _apply(a0, A, self._Pd0_cache,  self._Pd_cache)

        return (tf.transpose(g2),
                tf.transpose(time_scale * g1),
                tf.transpose((time_scale**2) * f),
                tf.transpose((time_scale**3) * df))

    def q_traj_with_derivatives_and_jerk(self):
        c = tf.constant(2.0 * self.gamma / self.T_final, dtype=DTYPE)
        return self._batch_q_qdot_qddot_qjerk(c)

    def full_ee_from_q(self, q):
        _, ee = self.robot.forward_all_points(q)
        return ee

    def waypoint_ee_from_q(self, q):
        ee = self.full_ee_from_q(q)
        tau_all, ee_wp = self.evaluate_waypoints_from_ee(ee)
        return ee, ee_wp, tau_all

    def compute_loss_train(self):
        q, qdot, qddot, qjerk = self.q_traj_with_derivatives_and_jerk()
        points, ee = self.robot.forward_all_points(q)
        tau_all, ee_wp = self.evaluate_waypoints_from_ee(ee)

        waypoint_loss = tf.reduce_mean(tf.square(ee_wp - self.waypoints_xyz_tf))
        path_loss  = tf.reduce_mean(tf.square(qdot))
        acc_loss   = tf.reduce_mean(tf.square(qddot))
        jerk_loss  = tf.reduce_mean(tf.square(qjerk))
        # L1 penalty with 90% safety margin: pressure builds up before reaching the limit
        margin_frac = tf.constant(0.9, dtype=DTYPE)
        q_mid  = 0.5 * (self.q_min + self.q_max)
        q_half = 0.5 * (self.q_max - self.q_min)
        q_lo_safe = (q_mid - margin_frac * q_half)[None, :]
        q_hi_safe = (q_mid + margin_frac * q_half)[None, :]
        lower_v = tf.nn.relu(q_lo_safe - q)
        upper_v = tf.nn.relu(q - q_hi_safe)
        limit_loss    = tf.reduce_mean(lower_v + upper_v)
        vel_violation = tf.nn.relu(tf.abs(qdot)  - margin_frac * self.qdot_max)
        acc_violation = tf.nn.relu(tf.abs(qddot) - margin_frac * self.qddot_max)
        vel_limit_loss = tf.reduce_mean(vel_violation)
        acc_limit_loss = tf.reduce_mean(acc_violation)
        final_vel_loss = tf.reduce_mean(tf.square(qdot[-1]))
        final_acc_loss = tf.reduce_mean(tf.square(qddot[-1]))
        init_acc_loss  = tf.reduce_mean(tf.square(qddot[0]))

        # Floor constraint: all joint points (except base) must have z >= z_floor
        floor_loss = tf.constant(0.0, dtype=DTYPE)
        for p in points[1:]:
            z = p[:, 2]
            floor_loss = floor_loss + tf.reduce_mean(tf.square(tf.nn.relu(self.z_floor - z)))

        loss = (
            5000000.0 * waypoint_loss
            + 1.0 * path_loss
            # + 1.0 * acc_loss
            + 1.0 * jerk_loss
            + 1000.0 * limit_loss
            + 1000.0 * vel_limit_loss
            + 1000.0 * acc_limit_loss
            # + 3.0 * final_vel_loss
            + 500.0 * final_acc_loss
            + 500.0 * init_acc_loss
            + 1000.0 * floor_loss
        )

        return {
            "loss": loss, "waypoint": waypoint_loss, "path": path_loss,
            "acc": acc_loss, "jerk": jerk_loss, "limit": limit_loss,
            "vel_limit": vel_limit_loss, "acc_limit": acc_limit_loss,
            "final_vel": final_vel_loss, "final_acc": final_acc_loss, "init_acc": init_acc_loss,
            "floor": floor_loss,
            "q": q, "qdot": qdot, "qddot": qddot, "qjerk": qjerk,
            "ee": ee, "ee_wp": ee_wp, "tau_all": tau_all,
        }

    def pack_weights_np(self):
        chunks = [self.U_batch.numpy().reshape(-1)]
        if self.raw_gap_logits is not None:
            chunks.append(self.raw_gap_logits.numpy().reshape(-1))
        return np.concatenate(chunks).astype(np.float64)

    def unpack_weights_np(self, w_flat):
        n_u = 6 * self.U_batch.shape[1]
        self.U_batch.assign(tf.constant(
            np.asarray(w_flat[:n_u], dtype=np.float32).reshape(self.U_batch.shape), dtype=DTYPE))
        if self.raw_gap_logits is not None:
            g = np.asarray(w_flat[n_u:], dtype=np.float32).reshape(self.raw_gap_logits.shape)
            self.raw_gap_logits.assign(tf.constant(g, dtype=DTYPE))

    @tf.function(reduce_retracing=True)
    def _loss_and_grad_tf(self):
        """Compiled loss + gradient for L-BFGS (avoids per-iter eager overhead)."""
        vars_ = self.get_trainable_variables()
        with tf.GradientTape() as tape:
            loss = self.compute_loss_train()["loss"]
        grads = tape.gradient(loss, vars_)
        return loss, grads

    def loss_and_grad_np(self, w_flat):
        self.unpack_weights_np(w_flat)
        loss_tf, grads = self._loss_and_grad_tf()
        grad_flat = np.concatenate([g.numpy().reshape(-1) for g in grads]).astype(np.float64)
        return float(loss_tf.numpy()), grad_flat

    def run_lbfgs(self, maxiter=2000):
        w0 = self.pack_weights_np()
        t0 = time.time()
        res = scipy.optimize.minimize(
            fun=self.loss_and_grad_np, x0=w0, jac=True, method="L-BFGS-B",
            options={"maxiter": int(maxiter), "maxfun": 50000, "maxcor": 50,
                     "maxls": 50, "ftol": 1e-10, "gtol": 1e-10, "iprint": -1},
        )
        elapsed = time.time() - t0
        self.unpack_weights_np(res.x)
        out = self.compute_loss_train()
        print(
            f"[ADA-L-MW-LBFGS] done in {elapsed:.2f} sec | "
            f"loss={out['loss'].numpy():.6e} | wp={out['waypoint'].numpy():.6e} | "
            f"fvel={out['final_vel'].numpy():.6e} | facc={out['final_acc'].numpy():.6e} | "
            f"msg={str(res.message).splitlines()[0]}"
        )
        return res, elapsed

    @tf.function
    def train_step(self):
        vars_ = self.get_trainable_variables()
        with tf.GradientTape() as tape:
            out = self.compute_loss_train()
            loss = out["loss"]
        grads = tape.gradient(loss, vars_)
        grads_and_vars = [(g, v) for g, v in zip(grads, vars_) if g is not None]
        if grads_and_vars:
            self.optimizer.apply_gradients(grads_and_vars)
        else:
            tf.print("WARNING: all gradients are None in ADA-L multi-waypoint train_step")
        return (loss, out["waypoint"], out["path"], out["acc"],
                out["jerk"], out["limit"], out["final_vel"], out["final_acc"], out["init_acc"])

    def train(self, epochs=1200, lr=3e-2, print_every=200, use_lbfgs=True, lbfgs_maxiter=2000):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        t0 = time.time()
        for ep in range(1, epochs + 1):
            loss, wp, path, acc, jerk, limit, final_vel, final_acc, init_acc = self.train_step()
            self.history["loss"].append(float(loss.numpy()))
            self.history["waypoint"].append(float(wp.numpy()))
            self.history["path"].append(float(path.numpy()))
            self.history["acc"].append(float(acc.numpy()))
            self.history["jerk"].append(float(jerk.numpy()))
            self.history["limit"].append(float(limit.numpy()))
            self.history["final_vel"].append(float(final_vel.numpy()))
            self.history["final_acc"].append(float(final_acc.numpy()))
            self.history["init_acc"].append(float(init_acc.numpy()))
            if ep % print_every == 0 or ep == 1:
                tau_now = self.get_waypoint_times().numpy()
                print(
                    f"[ADA-L-MW-Adam {ep:5d}/{epochs}] loss={loss.numpy():.6e} | "
                    f"wp={wp.numpy():.6e} | jerk={jerk.numpy():.6e} | "
                    f"fvel={final_vel.numpy():.6e} | facc={final_acc.numpy():.6e} | "
                    f"tau={np.array2string(tau_now, precision=4)}"
                )
        adam_elapsed = time.time() - t0
        print(f"[ADA-L-MW] Adam finished in {adam_elapsed:.2f} sec")
        lbfgs_elapsed = 0.0
        if use_lbfgs:
            _, lbfgs_elapsed = self.run_lbfgs(maxiter=lbfgs_maxiter)
        return adam_elapsed + lbfgs_elapsed

    def results(self):
        q, qdot, qddot, qjerk = self.q_traj_with_derivatives_and_jerk()
        ee, ee_wp, tau_all = self.waypoint_ee_from_q(q)
        return {
            "q": q.numpy(), "qdot": qdot.numpy(), "qddot": qddot.numpy(), "qjerk": qjerk.numpy(),
            "ee": ee.numpy(), "ee_wp": ee_wp.numpy(),
            "t_q": self.t_np.copy(), "t_qdot": self.t_np.copy(),
            "t_qddot": self.t_np.copy(), "t_qjerk": self.t_np.copy(),
            "waypoints_xyz": self.waypoints_xyz.copy(),
            "waypoint_times_init": self.waypoint_times.copy(),
            "waypoint_times_opt": tau_all.numpy(),
        }


# ============================================================
# 6) B-spline multi-waypoint planner
# ============================================================
class BSplineMultiWaypointPlanner:
    def __init__(
        self, q0, waypoints_xyz, waypoint_times,
        T_final=2.0, Nt=101, n_ctrl=10, degree=3, seed=0,
        free_waypoint_times=True, tau_min_gap=0.15, tau_sigma=0.04,
        z_floor=0.0,
    ):
        self.z_floor = tf.constant(float(z_floor), dtype=DTYPE)
        self.free_waypoint_times = bool(free_waypoint_times)
        self.tau_min_gap = float(tau_min_gap)
        self.tau_sigma   = float(tau_sigma)
        self.q0           = np.asarray(q0, dtype=np.float32).reshape(6,)
        self.waypoints_xyz = np.asarray(waypoints_xyz, dtype=np.float32).reshape(-1, 3)
        self.waypoint_times = np.asarray(waypoint_times, dtype=np.float32).reshape(-1,)
        self.T_final = float(T_final)
        self.Nt      = int(Nt)
        self.dt      = self.T_final / (self.Nt - 1)
        self.n_ctrl  = int(n_ctrl)
        self.degree  = int(degree)
        self.t_np = np.linspace(0.0, self.T_final, self.Nt).astype(np.float32)
        self.t_tf = tf.constant(self.t_np, dtype=DTYPE)
        self.u_np = (self.t_np / self.T_final).astype(np.float32)

        if len(self.waypoints_xyz) != len(self.waypoint_times):
            raise ValueError("waypoints_xyz and waypoint_times must have same length")
        if not np.all(np.diff(self.waypoint_times) > 0):
            raise ValueError("waypoint_times must be strictly increasing")
        if self.waypoint_times[-1] > T_final + 1e-8:
            raise ValueError("last waypoint time must be <= T_final")

        if self.free_waypoint_times and len(self.waypoint_times) >= 2:
            init_logits = init_gap_logits_from_waypoint_times(
                self.waypoint_times, T_final=self.T_final, tau_min_gap=self.tau_min_gap)
            self.raw_gap_logits = tf.Variable(init_logits, dtype=DTYPE, trainable=True,
                                              name="bs_raw_gap_logits")
        else:
            self.raw_gap_logits = None

        # Build analytic-derivative chain (B-spline derivatives via control-point differencing)
        Bq, D1, Bqd, D2, Bqdd, D3, Bqj = build_derivative_chain(self.u_np, self.n_ctrl, self.degree)
        self.B_q     = tf.constant(Bq,  dtype=DTYPE)
        self.D1_tf   = tf.constant(D1,  dtype=DTYPE)
        self.B_qd    = tf.constant(Bqd, dtype=DTYPE)
        self.D2_tf   = tf.constant(D2,  dtype=DTYPE)
        self.B_qdd   = tf.constant(Bqdd, dtype=DTYPE)
        self.D3_tf   = tf.constant(D3,  dtype=DTYPE)
        self.B_qjerk = tf.constant(Bqj, dtype=DTYPE)
        self.B = self.B_q  # backward-compat alias
        self.waypoint_idx    = np.array([int(np.argmin(np.abs(self.t_np - tk))) for tk in self.waypoint_times], dtype=np.int32)
        self.waypoint_idx_tf = tf.constant(self.waypoint_idx, dtype=tf.int32)
        self.waypoints_xyz_tf = tf.constant(self.waypoints_xyz, dtype=DTYPE)
        self.robot = UR5eKinematics()

        # Structural endpoint enforcement:
        #  first 3 ctrl == q0 (fixed) -> q(0)=q0, qd(0)=0, qdd(0)=0
        #  last  3 ctrl == qT_var (single var, tiled 3x) -> qd(T)=0, qdd(T)=0
        #  middle (n_ctrl-6) ctrl pts free
        if self.n_ctrl < 7:
            raise ValueError("n_ctrl must be >= 7 for structural endpoint enforcement (3+1+3)")
        rng = np.random.default_rng(seed)
        n_middle = self.n_ctrl - 6
        ctrl_mid_init = np.tile(self.q0[None, :], (n_middle, 1)).astype(np.float32)
        ctrl_mid_init += 0.01 * rng.standard_normal(size=ctrl_mid_init.shape).astype(np.float32)
        self.ctrl_mid = tf.Variable(ctrl_mid_init, dtype=DTYPE, trainable=True, name="bspline_ctrl_mid")
        self.qT_var   = tf.Variable(self.q0.astype(np.float32), dtype=DTYPE, trainable=True, name="bspline_qT")
        self.q0_tf    = tf.constant(self.q0, dtype=DTYPE)
        # UR5e joint position limits: elbow (joint 3) is +/-pi, others +/-2*pi
        self.q_min = tf.constant([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi], dtype=DTYPE)
        self.q_max = tf.constant([ 2*np.pi,  2*np.pi,  np.pi,  2*np.pi,  2*np.pi,  2*np.pi], dtype=DTYPE)
        # UR5e velocity limit: 180 deg/s = pi rad/s
        self.qdot_max  = tf.constant(np.pi, dtype=DTYPE)
        # UR5e acceleration limit: 2292 deg/s^2 = 40 rad/s^2
        self.qddot_max = tf.constant(40.0, dtype=DTYPE)
        self.optimizer = None
        self.history = {
            "loss": [], "waypoint": [], "start": [], "path": [],
            "acc": [], "jerk": [], "limit": [], "final_vel": [], "final_acc": [], "init_acc": [],
        }

    def get_waypoint_times(self):
        if self.raw_gap_logits is None:
            return tf.constant(self.waypoint_times, dtype=DTYPE)
        return gaps_to_ordered_waypoint_times(
            self.raw_gap_logits, T_final=self.T_final, tau_min_gap=self.tau_min_gap, dtype=DTYPE)

    def evaluate_waypoints_from_ee(self, ee):
        tau_all = self.get_waypoint_times()
        if len(self.waypoints_xyz) == 1:
            ee_wp = ee[-1][None, :]
        else:
            tau_intermediate = tau_all[:-1]
            ee_intermediate = soft_time_sample(ee, self.t_tf, tau_intermediate, sigma=self.tau_sigma)
            ee_wp = tf.concat([ee_intermediate, ee[-1][None, :]], axis=0)
        return tau_all, ee_wp

    def get_trainable_variables(self):
        vars_ = [self.ctrl_mid, self.qT_var]
        if self.raw_gap_logits is not None:
            vars_.append(self.raw_gap_logits)
        return vars_

    def assemble_ctrl(self):
        """Concat first 3 (q0), middle, last 3 (qT_var). shape [n_ctrl, 6]."""
        first3 = tf.tile(self.q0_tf[None, :], (3, 1))
        last3  = tf.tile(self.qT_var[None, :], (3, 1))
        return tf.concat([first3, self.ctrl_mid, last3], axis=0)

    def q_traj(self):
        return tf.linalg.matmul(self.B_q, self.assemble_ctrl())

    def q_traj_with_derivatives(self):
        ctrl = self.assemble_ctrl()
        q       = tf.linalg.matmul(self.B_q, ctrl)
        ctrl_d1 = tf.linalg.matmul(self.D1_tf, ctrl)
        qd_u    = tf.linalg.matmul(self.B_qd, ctrl_d1)
        ctrl_d2 = tf.linalg.matmul(self.D2_tf, ctrl_d1)
        qdd_u   = tf.linalg.matmul(self.B_qdd, ctrl_d2)
        ctrl_d3 = tf.linalg.matmul(self.D3_tf, ctrl_d2)
        qj_u    = tf.linalg.matmul(self.B_qjerk, ctrl_d3)
        T = tf.constant(self.T_final, dtype=DTYPE)
        qdot  = qd_u  / T
        qddot = qdd_u / (T * T)
        qjerk = qj_u  / (T * T * T)
        return q, qdot, qddot, qjerk

    def full_ee_from_q(self, q):
        _, ee = self.robot.forward_all_points(q)
        return ee

    def waypoint_ee_from_q(self, q):
        ee = self.full_ee_from_q(q)
        tau_all, ee_wp = self.evaluate_waypoints_from_ee(ee)
        return ee, ee_wp, tau_all

    def compute_loss_train(self):
        q, qdot, qddot, qjerk = self.q_traj_with_derivatives()
        points, ee = self.robot.forward_all_points(q)
        tau_all, ee_wp = self.evaluate_waypoints_from_ee(ee)

        waypoint_loss = tf.reduce_mean(tf.square(ee_wp - self.waypoints_xyz_tf))
        start_loss    = tf.reduce_mean(tf.square(q[0] - tf.constant(self.q0, dtype=DTYPE)))
        path_loss     = tf.reduce_mean(tf.square(qdot))
        acc_loss      = tf.reduce_mean(tf.square(qddot))
        jerk_loss     = tf.reduce_mean(tf.square(qjerk))
        # L1 penalty with 90% safety margin: pressure builds up before reaching the limit
        margin_frac = tf.constant(0.9, dtype=DTYPE)
        q_mid  = 0.5 * (self.q_min + self.q_max)
        q_half = 0.5 * (self.q_max - self.q_min)
        q_lo_safe = (q_mid - margin_frac * q_half)[None, :]
        q_hi_safe = (q_mid + margin_frac * q_half)[None, :]
        lower_v = tf.nn.relu(q_lo_safe - q)
        upper_v = tf.nn.relu(q - q_hi_safe)
        limit_loss    = tf.reduce_mean(lower_v + upper_v)
        vel_violation = tf.nn.relu(tf.abs(qdot)  - margin_frac * self.qdot_max)
        acc_violation = tf.nn.relu(tf.abs(qddot) - margin_frac * self.qddot_max)
        vel_limit_loss = tf.reduce_mean(vel_violation)
        acc_limit_loss = tf.reduce_mean(acc_violation)
        final_vel_loss = tf.reduce_mean(tf.square(qdot[-1]))
        final_acc_loss = tf.reduce_mean(tf.square(qddot[-1]))
        init_acc_loss  = tf.reduce_mean(tf.square(qddot[0]))

        # Floor constraint: all joint points (except base) must have z >= z_floor
        floor_loss = tf.constant(0.0, dtype=DTYPE)
        for p in points[1:]:
            z = p[:, 2]
            floor_loss = floor_loss + tf.reduce_mean(tf.square(tf.nn.relu(self.z_floor - z)))

        loss = (
            5000000.0 * waypoint_loss
            + 1.0 * path_loss
            # + 1.0 * acc_loss
            + 1.0 * jerk_loss
            + 1000.0 * limit_loss
            + 1000.0 * vel_limit_loss
            + 1000.0 * acc_limit_loss
            # + 100000.0 * final_vel_loss
            # + 5000.0 * final_acc_loss
            # + 5000.0 * init_acc_loss
            + 1000.0 * floor_loss
        )
        return {
            "loss": loss, "waypoint": waypoint_loss, "start": start_loss,
            "path": path_loss, "acc": acc_loss, "jerk": jerk_loss, "limit": limit_loss,
            "vel_limit": vel_limit_loss, "acc_limit": acc_limit_loss,
            "final_vel": final_vel_loss, "final_acc": final_acc_loss, "init_acc": init_acc_loss,
            "floor": floor_loss,
            "q": q, "qdot": qdot, "qddot": qddot, "qjerk": qjerk,
            "ee": ee, "ee_wp": ee_wp, "tau_all": tau_all,
        }

    def pack_weights_np(self):
        chunks = [self.ctrl_mid.numpy().reshape(-1), self.qT_var.numpy().reshape(-1)]
        if self.raw_gap_logits is not None:
            chunks.append(self.raw_gap_logits.numpy().reshape(-1))
        return np.concatenate(chunks).astype(np.float64)

    def unpack_weights_np(self, w_flat):
        cursor = 0
        size_mid = int(np.prod(self.ctrl_mid.shape))
        self.ctrl_mid.assign(
            tf.reshape(tf.convert_to_tensor(w_flat[cursor:cursor+size_mid], dtype=DTYPE), self.ctrl_mid.shape))
        cursor += size_mid
        size_qT = int(np.prod(self.qT_var.shape))
        self.qT_var.assign(
            tf.reshape(tf.convert_to_tensor(w_flat[cursor:cursor+size_qT], dtype=DTYPE), self.qT_var.shape))
        cursor += size_qT
        if self.raw_gap_logits is not None:
            size_tau = int(np.prod(self.raw_gap_logits.shape))
            self.raw_gap_logits.assign(
                tf.reshape(tf.convert_to_tensor(w_flat[cursor:cursor+size_tau], dtype=DTYPE), self.raw_gap_logits.shape))
            cursor += size_tau
        if cursor != len(w_flat):
            raise ValueError(f"Unpack size mismatch: consumed {cursor}, got {len(w_flat)}")

    @tf.function(reduce_retracing=True)
    def _loss_and_grad_tf(self):
        vars_ = self.get_trainable_variables()
        with tf.GradientTape() as tape:
            loss = self.compute_loss_train()["loss"]
        grads = tape.gradient(loss, vars_)
        return loss, grads

    def loss_and_grad_np(self, w_flat):
        self.unpack_weights_np(w_flat)
        loss_tf, grads = self._loss_and_grad_tf()
        grad_flat = np.concatenate([g.numpy().reshape(-1) for g in grads]).astype(np.float64)
        return float(loss_tf.numpy()), grad_flat

    def run_lbfgs(self, maxiter=2000):
        w0 = self.pack_weights_np()
        t0 = time.time()
        res = scipy.optimize.minimize(
            fun=self.loss_and_grad_np, x0=w0, jac=True, method="L-BFGS-B",
            options={"maxiter": int(maxiter), "maxfun": 50000, "maxcor": 50,
                     "maxls": 50, "ftol": 1e-10, "gtol": 1e-10, "iprint": -1},
        )
        elapsed = time.time() - t0
        self.unpack_weights_np(res.x)
        out = self.compute_loss_train()
        print(
            f"[BSpline-MW-LBFGS] done in {elapsed:.2f} sec | "
            f"loss={out['loss'].numpy():.6e} | wp={out['waypoint'].numpy():.6e} | "
            f"fvel={out['final_vel'].numpy():.6e} | facc={out['final_acc'].numpy():.6e} | "
            f"msg={str(res.message).splitlines()[0]}"
        )
        return res, elapsed

    @tf.function
    def train_step(self):
        vars_ = self.get_trainable_variables()
        with tf.GradientTape() as tape:
            out = self.compute_loss_train()
            loss = out["loss"]
        grads = tape.gradient(loss, vars_)
        grads_and_vars = [(g, v) for g, v in zip(grads, vars_) if g is not None]
        if grads_and_vars:
            self.optimizer.apply_gradients(grads_and_vars)
        else:
            tf.print("WARNING: all gradients are None in BSpline train_step")
        return (loss, out["waypoint"], out["start"], out["path"],
                out["acc"], out["jerk"], out["limit"], out["final_vel"], out["final_acc"], out["init_acc"])

    def train(self, epochs=1200, lr=3e-2, print_every=200, use_lbfgs=True, lbfgs_maxiter=2000):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        t0 = time.time()
        for ep in range(1, epochs + 1):
            loss, wp, start, path, acc, jerk, limit, fvel, facc, iacc = self.train_step()
            self.history["loss"].append(float(loss.numpy()))
            self.history["waypoint"].append(float(wp.numpy()))
            self.history["start"].append(float(start.numpy()))
            self.history["path"].append(float(path.numpy()))
            self.history["acc"].append(float(acc.numpy()))
            self.history["jerk"].append(float(jerk.numpy()))
            self.history["limit"].append(float(limit.numpy()))
            self.history["final_vel"].append(float(fvel.numpy()))
            self.history["final_acc"].append(float(facc.numpy()))
            self.history["init_acc"].append(float(iacc.numpy()))
            if ep % print_every == 0 or ep == 1:
                print(
                    f"[BSpline-MW-Adam {ep:5d}/{epochs}] loss={loss.numpy():.6e} | "
                    f"wp={wp.numpy():.6e} | jerk={jerk.numpy():.6e} | "
                    f"fvel={fvel.numpy():.6e} | facc={facc.numpy():.6e}"
                )
        adam_elapsed = time.time() - t0
        print(f"[BSpline-MW] Adam finished in {adam_elapsed:.2f} sec")
        lbfgs_elapsed = 0.0
        if use_lbfgs:
            _, lbfgs_elapsed = self.run_lbfgs(maxiter=lbfgs_maxiter)
        return adam_elapsed + lbfgs_elapsed

    def results(self):
        q, qdot, qddot, qjerk = self.q_traj_with_derivatives()
        ee, ee_wp, tau_all = self.waypoint_ee_from_q(q)
        return {
            "q": q.numpy(), "qdot": qdot.numpy(), "qddot": qddot.numpy(), "qjerk": qjerk.numpy(),
            "ee": ee.numpy(), "ee_wp": ee_wp.numpy(),
            "t_q": self.t_np.copy(), "t_qdot": self.t_np.copy(),
            "t_qddot": self.t_np.copy(), "t_qjerk": self.t_np.copy(),
            "waypoints_xyz": self.waypoints_xyz.copy(),
            "waypoint_times_init": self.waypoint_times.copy(),
            "waypoint_times_opt": tau_all.numpy(),
        }


# ============================================================
# 7) Metrics
# ============================================================
def compute_waypoint_errors(ee_wp, wp_xyz):
    return np.linalg.norm(ee_wp - wp_xyz, axis=1)


def benchmark_multiwaypoint_metrics(res, solve_time):
    q, qdot, qddot, qjerk = res["q"], res["qdot"], res["qddot"], res["qjerk"]
    ee, ee_wp, wp_xyz = res["ee"], res["ee_wp"], res["waypoints_xyz"]

    T_total = float(res["t_q"][-1] - res["t_q"][0])
    dt_qdot  = T_total / (len(qdot)  - 1) if len(qdot)  > 1 else 0.0
    dt_qddot = T_total / (len(qddot) - 1) if len(qddot) > 1 else 0.0
    dt_qjerk = T_total / (len(qjerk) - 1) if len(qjerk) > 1 else 0.0

    wp_err = compute_waypoint_errors(ee_wp, wp_xyz)

    return {
        "mean_waypoint_error":  float(np.mean(wp_err)),
        "max_waypoint_error":   float(np.max(wp_err)),
        "final_waypoint_error": float(wp_err[-1]),
        "joint_path_length":    float(np.sum(np.linalg.norm(q[1:] - q[:-1], axis=1))),
        "ee_path_length":       float(np.sum(np.linalg.norm(ee[1:] - ee[:-1], axis=1))),
        "integrated_squared_velocity":     float(np.sum(np.sum(qdot**2,  axis=1)) * dt_qdot),
        "integrated_squared_acceleration": float(np.sum(np.sum(qddot**2, axis=1)) * dt_qddot),
        "integrated_squared_jerk":         float(np.sum(np.sum(qjerk**2, axis=1)) * dt_qjerk),
        "mean_squared_velocity":     float(np.mean(np.sum(qdot**2,  axis=1))),
        "mean_squared_acceleration": float(np.mean(np.sum(qddot**2, axis=1))),
        "mean_squared_jerk":         float(np.mean(np.sum(qjerk**2, axis=1))),
        "max_velocity_norm":     float(np.max(np.linalg.norm(qdot,  axis=1))),
        "max_acceleration_norm": float(np.max(np.linalg.norm(qddot, axis=1))),
        "max_jerk_norm":         float(np.max(np.linalg.norm(qjerk, axis=1))),
        "max_abs_velocity_each_joint":     np.max(np.abs(qdot),  axis=0),
        "max_abs_acceleration_each_joint": np.max(np.abs(qddot), axis=0),
        "max_abs_jerk_each_joint":         np.max(np.abs(qjerk), axis=0),
        "final_velocity_norm":     float(np.linalg.norm(qdot[-1])),
        "final_acceleration_norm": float(np.linalg.norm(qddot[-1])),
        "solve_time":      float(solve_time),
        "waypoint_errors": wp_err.copy(),
        "n_waypoints":     int(len(wp_xyz)),
        "waypoint_times_opt": res.get("waypoint_times_opt", None),
    }


def print_metrics_table(name, metrics):
    print(f"\n=== {name} metrics ===")
    keys = [
        "mean_waypoint_error", "max_waypoint_error", "final_waypoint_error",
        "joint_path_length", "ee_path_length",
        "integrated_squared_velocity", "integrated_squared_acceleration", "integrated_squared_jerk",
        "mean_squared_velocity", "mean_squared_acceleration", "mean_squared_jerk",
        "max_velocity_norm", "max_acceleration_norm", "max_jerk_norm",
        "final_velocity_norm", "final_acceleration_norm", "solve_time",
    ]
    for k in keys:
        print(f"{k:>36s} : {metrics[k]:.6e}")
    print("waypoint_errors                    :", np.array2string(metrics["waypoint_errors"], precision=6))
    print("max_abs_velocity_each_joint        :", np.array2string(metrics["max_abs_velocity_each_joint"], precision=5))
    print("max_abs_acceleration_each_joint    :", np.array2string(metrics["max_abs_acceleration_each_joint"], precision=5))
    print("max_abs_jerk_each_joint            :", np.array2string(metrics["max_abs_jerk_each_joint"], precision=5))
    if metrics.get("waypoint_times_opt") is not None:
        print("optimized_waypoint_times           :", np.array2string(np.asarray(metrics["waypoint_times_opt"]), precision=6))


def print_comparison_table(metrics_a, metrics_b, name_a="ADA-L", name_b="B-spline"):
    keys = [
        "mean_waypoint_error", "max_waypoint_error", "final_waypoint_error",
        "joint_path_length", "ee_path_length",
        "integrated_squared_velocity", "integrated_squared_acceleration", "integrated_squared_jerk",
        "max_velocity_norm", "max_acceleration_norm", "max_jerk_norm",
        "final_velocity_norm", "final_acceleration_norm", "solve_time",
    ]
    print("\n" + "=" * 105)
    print(f"{'Metric':>36s} | {name_a:>18s} | {name_b:>18s}")
    print("=" * 105)
    for k in keys:
        print(f"{k:>36s} | {metrics_a[k]:18.6e} | {metrics_b[k]:18.6e}")
    print("=" * 105)


# ============================================================
# 7.5) UR5e trajectory validation (joint limits / velocity / acceleration)
#      Ported from validate_trajectory.py for in-memory checking.
# ============================================================
UR5E_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
]
UR5E_POSITION_LIMITS = [
    (-2 * math.pi, 2 * math.pi),
    (-2 * math.pi, 2 * math.pi),
    (-math.pi,      math.pi),
    (-2 * math.pi, 2 * math.pi),
    (-2 * math.pi, 2 * math.pi),
    (-2 * math.pi, 2 * math.pi),
]
UR5E_VELOCITY_LIMIT = math.pi          # 180 deg/s
UR5E_ACCELERATION_LIMIT = 40.0         # rad/s^2


def _finite_diff_vel_acc(times, positions):
    n = len(times)
    vel = np.zeros_like(positions)
    acc = np.zeros_like(positions)
    for i in range(1, n - 1):
        dt_f = times[i + 1] - times[i]
        dt_b = times[i] - times[i - 1]
        vel[i] = (positions[i + 1] - positions[i - 1]) / (dt_f + dt_b)
        acc[i] = (positions[i + 1] - 2 * positions[i] + positions[i - 1]) / (dt_f * dt_b)
    vel[0]  = (positions[1]  - positions[0])  / (times[1]  - times[0])
    vel[-1] = (positions[-1] - positions[-2]) / (times[-1] - times[-2])
    acc[0]  = acc[1]
    acc[-1] = acc[-2]
    return vel, acc


def validate_ur5e_trajectory(times, positions):
    """Check UR5e joint position/velocity/acceleration limits.
    Returns (ok: bool, errors: list[str], summary: dict).
    """
    times = np.asarray(times)
    positions = np.asarray(positions)
    velocities, accelerations = _finite_diff_vel_acc(times, positions)

    errors = []
    for j in range(6):
        lo, hi = UR5E_POSITION_LIMITS[j]
        bad = np.where((positions[:, j] < lo) | (positions[:, j] > hi))[0]
        for idx in bad:
            errors.append(
                f"  [POSITION] J{j+1} ({UR5E_JOINT_NAMES[j]}): "
                f"q={math.degrees(positions[idx, j]):.2f} deg at t={times[idx]:.3f}s "
                f"(limit [{math.degrees(lo):.0f}, {math.degrees(hi):.0f}] deg)"
            )
        bad = np.where(np.abs(velocities[:, j]) > UR5E_VELOCITY_LIMIT)[0]
        for idx in bad:
            errors.append(
                f"  [VELOCITY] J{j+1} ({UR5E_JOINT_NAMES[j]}): "
                f"|v|={math.degrees(abs(velocities[idx, j])):.2f} deg/s at t={times[idx]:.3f}s "
                f"(limit {math.degrees(UR5E_VELOCITY_LIMIT):.0f} deg/s)"
            )
        bad = np.where(np.abs(accelerations[:, j]) > UR5E_ACCELERATION_LIMIT)[0]
        for idx in bad:
            errors.append(
                f"  [ACCEL]    J{j+1} ({UR5E_JOINT_NAMES[j]}): "
                f"|a|={math.degrees(abs(accelerations[idx, j])):.2f} deg/s^2 at t={times[idx]:.3f}s "
                f"(limit {math.degrees(UR5E_ACCELERATION_LIMIT):.0f} deg/s^2)"
            )

    summary = {
        "pos_max_deg": [float(math.degrees(positions[:, j].max())) for j in range(6)],
        "pos_min_deg": [float(math.degrees(positions[:, j].min())) for j in range(6)],
        "vel_max_deg_s":  [float(math.degrees(np.abs(velocities[:, j]).max())) for j in range(6)],
        "acc_max_deg_s2": [float(math.degrees(np.abs(accelerations[:, j]).max())) for j in range(6)],
    }
    return (len(errors) == 0), errors, summary


def print_validation_result(name, ok, errors, summary):
    print(f"\n=== {name} UR5e validation ===")
    print(f"  {'Joint':<14} {'pos min':>10} {'pos max':>10} {'|vel| max':>12} {'|acc| max':>12}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")
    vlim = math.degrees(UR5E_VELOCITY_LIMIT)
    alim = math.degrees(UR5E_ACCELERATION_LIMIT)
    for j in range(6):
        pmin, pmax = summary["pos_min_deg"][j], summary["pos_max_deg"][j]
        vmax, amax = summary["vel_max_deg_s"][j], summary["acc_max_deg_s2"][j]
        vflag = " !" if vmax > vlim else "  "
        aflag = " !" if amax > alim else "  "
        print(f"  {UR5E_JOINT_NAMES[j]:<14} {pmin:>9.2f} {pmax:>9.2f} {vmax:>10.2f}{vflag} {amax:>10.2f}{aflag}")
    if ok:
        print(f"  [OK] {name} trajectory satisfies UR5e limits.")
    else:
        print(f"  [WARN] {name} {len(errors)} violation(s):")
        for e in errors:
            print(e)


def validate_collision(q_traj):
    """
    Check the entire trajectory for UR5e self + floor collision via
    capsule_collision_checker.UR5eCapsuleChecker. No payload (Ex3 is
    payload-free), so capsule-vs-OBB and OBB-vs-floor are skipped. Returns
    (ok, first_idx, n_collide, total). `ok=True` iff no state collides.
    """
    from capsule_collision_checker import UR5eCapsuleChecker
    ck = UR5eCapsuleChecker(include_payload=False)
    q_traj = np.asarray(q_traj)
    first_idx = ck.first_collision_index(q_traj)
    if first_idx is None:
        return True, None, 0, len(q_traj)
    n_coll = sum(1 for i in range(len(q_traj)) if ck.check_state(q_traj[i]))
    return False, int(first_idx), int(n_coll), len(q_traj)


def print_collision_result(name, ok, first_idx, n_coll, n_total, t_traj=None,
                           q_first=None):
    print(f"\n=== {name} collision check (self + floor) ===")
    if ok:
        print(f"  [OK] No collision in {n_total} states.")
        return
    t_str = ""
    if t_traj is not None and first_idx is not None:
        t_str = f", t={float(t_traj[first_idx]):.3f}s"
    print(f"  [FAIL] first collision at idx {first_idx}{t_str}; "
          f"{n_coll}/{n_total} states in collision "
          f"({100.0 * n_coll / n_total:.1f}%)")
    if q_first is not None:
        from capsule_collision_checker import UR5eCapsuleChecker
        ck = UR5eCapsuleChecker(include_payload=False)
        for pa, pb, d, thr in ck.collision_report(q_first):
            print(f"      {pa:>10s} <-> {pb:<10s}  dist={d:.4f}  thr={thr:.4f}  "
                  f"(margin={d-thr:+.4f})")


# ============================================================
# 8) Report helpers
# ============================================================
def metric_better_text(a, b, smaller_is_better=True, name_a="ADA-L", name_b="B-spline"):
    if smaller_is_better:
        if a < b:   return f"{name_a} better ({a:.4e} < {b:.4e})"
        elif a > b: return f"{name_b} better ({b:.4e} < {a:.4e})"
        else:       return "Tie"
    else:
        if a > b:   return f"{name_a} better ({a:.4e} > {b:.4e})"
        elif a < b: return f"{name_b} better ({b:.4e} > {a:.4e})"
        else:       return "Tie"


def generate_metric_summary_text(metrics_a, metrics_b, name_a="ADA-L", name_b="B-spline"):
    lines = ["## Automatic comparison summary\n"]
    keys = [
        "mean_waypoint_error", "max_waypoint_error", "final_waypoint_error",
        "joint_path_length", "ee_path_length",
        "integrated_squared_velocity", "integrated_squared_acceleration", "integrated_squared_jerk",
        "mean_squared_velocity", "mean_squared_acceleration", "mean_squared_jerk",
        "max_velocity_norm", "max_acceleration_norm", "max_jerk_norm",
        "final_velocity_norm", "final_acceleration_norm", "solve_time",
    ]
    for k in keys:
        a, b = float(metrics_a[k]), float(metrics_b[k])
        lines.append(f"- **{k}**: {metric_better_text(a, b, True, name_a, name_b)}")
    return "\n".join(lines)


def write_markdown_report_multiwaypoint(
    report_path, run_config, metrics_a, metrics_b, res_a, res_b,
    fig_paths, name_a="ADA-L", name_b="B-spline",
):
    summary_text = generate_metric_summary_text(metrics_a, metrics_b, name_a, name_b)
    keys = [
        "mean_waypoint_error", "max_waypoint_error", "final_waypoint_error",
        "joint_path_length", "ee_path_length",
        "integrated_squared_velocity", "integrated_squared_acceleration", "integrated_squared_jerk",
        "mean_squared_velocity", "mean_squared_acceleration", "mean_squared_jerk",
        "max_velocity_norm", "max_acceleration_norm", "max_jerk_norm",
        "final_velocity_norm", "final_acceleration_norm", "solve_time",
    ]
    lines = []
    lines.append(f"# {name_a} vs {name_b} Multi-waypoint Comparison Report\n")
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## Problem setup\n")
    lines.append(f"- q0: `{np.array2string(run_config['q0'], precision=5)}`")
    lines.append(f"- waypoints_xyz:\n```text\n{np.array2string(run_config['waypoints_xyz'], precision=5)}\n```")
    lines.append(f"- initial waypoint_times: `{np.array2string(run_config['waypoint_times'], precision=5)}`")
    lines.append(f"- T_final: `{run_config['T_final']}`")
    lines.append(f"- Nt: `{run_config['Nt']}`\n")
    lines.append("## Optimized waypoint times\n")
    lines.append(f"- {name_a}: `{np.array2string(np.asarray(res_a['waypoint_times_opt']), precision=6)}`")
    lines.append(f"- {name_b}: `{np.array2string(np.asarray(res_b['waypoint_times_opt']), precision=6)}`\n")
    lines.append("## Waypoint errors\n")
    lines.append(f"- {name_a}: `{np.array2string(np.asarray(metrics_a['waypoint_errors']), precision=6)}`")
    lines.append(f"- {name_b}: `{np.array2string(np.asarray(metrics_b['waypoint_errors']), precision=6)}`\n")
    lines.append("## Scalar metrics table\n")
    lines.append(f"| Metric | {name_a} | {name_b} |")
    lines.append("|---|---:|---:|")
    for k in keys:
        lines.append(f"| {k} | {float(metrics_a[k]):.6e} | {float(metrics_b[k]):.6e} |")
    lines.append("")
    lines.append(summary_text)
    lines.append("")
    lines.append("## Per-joint extrema\n")
    for lbl, m in [(name_a, metrics_a), (name_b, metrics_b)]:
        lines.append(f"- {lbl} max abs velocity each joint: `{np.array2string(m['max_abs_velocity_each_joint'], precision=5)}`")
        lines.append(f"- {lbl} max abs acceleration each joint: `{np.array2string(m['max_abs_acceleration_each_joint'], precision=5)}`")
        lines.append(f"- {lbl} max abs jerk each joint: `{np.array2string(m['max_abs_jerk_each_joint'], precision=5)}`")
    lines.append("\n## Figures\n")
    for title, rel_path in fig_paths:
        lines.append(f"### {title}\n")
        lines.append(f"![{title}]({rel_path})\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def try_write_docx_report_multiwaypoint(
    docx_path, run_config, metrics_a, metrics_b, res_a, res_b,
    fig_abs_paths, name_a="ADA-L", name_b="B-spline",
):
    try:
        from docx import Document
        from docx.shared import Inches
    except Exception as e:
        print(f"[report] python-docx unavailable: {e}")
        return False
    keys = [
        "mean_waypoint_error", "max_waypoint_error", "final_waypoint_error",
        "joint_path_length", "ee_path_length",
        "integrated_squared_velocity", "integrated_squared_acceleration", "integrated_squared_jerk",
        "mean_squared_velocity", "mean_squared_acceleration", "mean_squared_jerk",
        "max_velocity_norm", "max_acceleration_norm", "max_jerk_norm",
        "final_velocity_norm", "final_acceleration_norm", "solve_time",
    ]
    doc = Document()
    doc.add_heading(f"{name_a} vs {name_b} Multi-waypoint Comparison Report", level=0)
    doc.add_heading("Problem setup", level=1)
    doc.add_paragraph(f"q0: {np.array2string(run_config['q0'], precision=5)}")
    doc.add_paragraph(f"waypoints_xyz: {np.array2string(run_config['waypoints_xyz'], precision=5)}")
    doc.add_paragraph(f"initial waypoint_times: {np.array2string(run_config['waypoint_times'], precision=5)}")
    doc.add_paragraph(f"T_final: {run_config['T_final']}")
    doc.add_paragraph(f"Nt: {run_config['Nt']}")
    doc.add_heading("Optimized waypoint times", level=1)
    doc.add_paragraph(f"{name_a}: {np.array2string(np.asarray(res_a['waypoint_times_opt']), precision=6)}")
    doc.add_paragraph(f"{name_b}: {np.array2string(np.asarray(res_b['waypoint_times_opt']), precision=6)}")
    doc.add_heading("Waypoint errors", level=1)
    doc.add_paragraph(f"{name_a}: {np.array2string(np.asarray(metrics_a['waypoint_errors']), precision=6)}")
    doc.add_paragraph(f"{name_b}: {np.array2string(np.asarray(metrics_b['waypoint_errors']), precision=6)}")
    doc.add_heading("Scalar metrics", level=1)
    table = doc.add_table(rows=1, cols=3)
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text, hdr[2].text = "Metric", name_a, name_b
    for k in keys:
        row = table.add_row().cells
        row[0].text, row[1].text, row[2].text = k, f"{float(metrics_a[k]):.6e}", f"{float(metrics_b[k]):.6e}"
    doc.add_heading("Figures", level=1)
    for title, abs_path in fig_abs_paths:
        doc.add_heading(title, level=2)
        doc.add_picture(str(abs_path), width=Inches(6.2))
    doc.save(docx_path)
    print(f"[report] DOCX saved to: {docx_path}")
    return True


# ============================================================
# 9) Visualization  (ex1-style axes + ex4-specific content)
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


def apply_tick_style_3d(ax):
    # ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    # ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    # ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
    # ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # ax.zaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.set_xticks(np.arange(-1.0, 2.0, 0.25))
    ax.set_yticks(np.arange(-1.0, 2.0, 0.25))
    ax.set_zticks(np.arange(-1.0, 2.0, 0.25))

    formatter = FuncFormatter(smart_tick_formatter)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.zaxis.set_major_formatter(formatter)


def style_2d_axes(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    apply_tick_style_2d(ax)


def style_3d_axes(ax):
    ax.grid(True)
    ax.view_init(elev=25, azim=-120)
    ax.tick_params(axis="x", pad=0.2)
    ax.tick_params(axis="y", pad=0.2)
    ax.tick_params(axis="z", pad=3)
    apply_tick_style_3d(ax)


def plot_training_losses_generic(history, title="Training losses", save_path=None, show=True):
    plt.figure(figsize=(8, 5))
    for k, v in history.items():
        if len(v) == 0:
            continue
        plt.plot(v, label=k)
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss / error")
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_panel_weights(adal, T_final, save_path=None, show=True):
    """Bar chart of learned panel weights W for each joint,
    x-axis positioned at the center of each panel interval in physical time."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.0), sharex=False)
    axes_flat = axes.ravel()

    for j, m in enumerate(adal.models):
        ax = axes_flat[j]
        panel_edges_np = m.panel_edges.numpy()  # shape (N_p+1,)
        W_np = m.U.numpy()                      # shape (N_p-1,): display only these

        # first N_p-1 panels use edges panel_edges_np[:N_p]
        edges_show = panel_edges_np[:len(W_np) + 1]  # shape (N_p,)
        centers_norm = 0.5 * (edges_show[:-1] + edges_show[1:])
        widths_norm  = edges_show[1:] - edges_show[:-1]

        # convert normalized [-1, 1] -> physical time [0, T_final]
        centers_phys = (centers_norm + 1.0) * 0.5 * T_final
        widths_phys  = widths_norm * 0.5 * T_final

        ax.bar(centers_phys, W_np, width=widths_phys,
               color="steelblue", edgecolor="k", linewidth=0.5, align="center")
        # ax.axhline(0, color="k", linewidth=0.6, linestyle="--")
        ax.set_title(f"Joint {j+1}", fontsize=20)
        ax.set_ylabel("Panel weight, $W$", fontsize=20)
        ax.set_xlim(0, (edges_show[-1] + 1.0) * 0.5 * T_final)
        ax.tick_params(axis="both", labelsize=20)

        ## ytick ##
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        formatter = FuncFormatter(smart_tick_formatter)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for edge in edges_show[1:-1]:
            t_edge = (edge + 1.0) * 0.5 * T_final
            ax.axvline(t_edge, color="gray", linewidth=0.5, linestyle=":")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]", fontsize=20)
    fig.suptitle(" ", fontsize=13)
    plt.tight_layout(w_pad=1.5, h_pad=3.0)
    if save_path is not None:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_compare_ee_path(res_a, res_b, name_a="ADA-L", name_b="B-spline",
                         title="EE path comparison", save_path=None, show=False):
    ee_a, ee_b = res_a["ee"], res_b["ee"]
    wp = res_a["waypoints_xyz"]
    ee_wp_a, ee_wp_b = res_a["ee_wp"], res_b["ee_wp"]
    wp_colors = ["tab:blue", "tab:orange", "tab:red", "tab:purple", "tab:brown"]

    fig = plt.figure(figsize=(6.6, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ee_a[:, 0], ee_a[:, 1], ee_a[:, 2], linewidth=2.2, label=name_a)
    ax.plot(ee_b[:, 0], ee_b[:, 1], ee_b[:, 2], "--", linewidth=2.2, label=name_b)
    ax.scatter([ee_a[0, 0]], [ee_a[0, 1]], [ee_a[0, 2]], s=42, color="k", label="start")
    for k in range(len(wp)):
        c = wp_colors[k % len(wp_colors)]
        ax.scatter([wp[k, 0]], [wp[k, 1]], [wp[k, 2]], s=110, marker="*", color=c,
                   label=f"waypoint {k+1}")
        ax.plot([wp[k, 0], ee_wp_a[k, 0]], [wp[k, 1], ee_wp_a[k, 1]], [wp[k, 2], ee_wp_a[k, 2]],
                "-", alpha=0.5, color="C0")
        ax.plot([wp[k, 0], ee_wp_b[k, 0]], [wp[k, 1], ee_wp_b[k, 1]], [wp[k, 2], ee_wp_b[k, 2]],
                "--", alpha=0.5, color="C1")
    style_3d_axes(ax)
    ax.set_xlabel(r"$x$ (m)", labelpad=2)
    ax.set_ylabel(r"$y$ (m)", labelpad=2)
    ax.set_zlabel(r"$z$ (m)", labelpad=2)
    ax.set_title(title)
    ax.set_xlim(-0.5, 0.6)
    ax.set_ylim(-0.6, 0.5)
    ax.set_zlim(0, 1.1)
    ax.legend(loc="upper center", ncol=2, frameon=True, fontsize=9)
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300)
    plt.show() if show else plt.close(fig)


def plot_compare_joint_bundle(res_a, res_b, name_a="ADA-L", name_b="B-spline",
                               title="Joint trajectory comparison", save_path=None, show=False):
    t_a, q_a = res_a["t_q"], res_a["q"]
    t_b, q_b = res_b["t_q"], res_b["q"]
    wp_t = np.asarray(res_a.get("waypoint_times_opt", res_a.get("waypoint_times_init", [])))
    joint_labels = [f"Joint {j+1}" for j in range(6)]

    fig, axes = plt.subplots(3, 2, figsize=(5.6, 6.8), sharex=True)
    for j, ax in enumerate(axes.ravel()):
        ax.plot(t_a, q_a[:, j], linewidth=1.0, label=name_a if j == 0 else None)
        ax.plot(t_b, q_b[:, j], "--", linewidth=1.0, label=name_b if j == 0 else None)
        for tk in wp_t:
            ax.axvline(float(tk), color="k", linestyle=":", alpha=0.3, linewidth=0.8)
        ax.set_xlabel(r"Time, $t$ (s)")
        ax.set_ylabel(r"Angle, $q$ (rad)")
        style_2d_axes(ax)
        if j == 5:
            j6_all = np.concatenate([q_a[:, 5], q_b[:, 5]])
            if float(j6_all.max() - j6_all.min()) < 0.1:
                q6_init = float(q_a[0, 5])
                ax.set_ylim(q6_init - 0.1, q6_init + 0.1)
        dummy = Line2D([], [], linestyle='None', marker=None, linewidth=0, color='none')
        ax.add_artist(ax.legend([dummy], [joint_labels[j]], loc="best", frameon=False,
                                 handlelength=0, handletextpad=0, borderpad=0.2, fontsize=10))
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show() if show else plt.close(fig)


def plot_norm_profiles_compare(t_a, y_a, y_b, title, y_label,
                                label_a="ADA-L", label_b="B-spline",
                                wp_t=None, save_path=None, show=False):
    """Individual figure for one norm comparison (vel, acc, or jerk)."""
    n = min(len(t_a), len(y_a), len(y_b))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(t_a[:n], y_a[:n], linewidth=1.0, label=label_a)
    ax.plot(t_a[:n], y_b[:n], linewidth=1.0, label=label_b)
    if wp_t is not None:
        for tk in wp_t:
            ax.axvline(float(tk), color="k", linestyle=":", alpha=0.3, linewidth=0.8)
    ax.set_xlabel(r"Time, $t$ (s)")
    ax.set_ylabel(y_label)
    style_2d_axes(ax)
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show() if show else plt.close(fig)


def plot_joint_bundle_single(t, y, title, y_label, labels, wp_t=None,
                              save_path=None, show=False):
    """Single-method joint bundle (6 joints in one figure), ex1-style."""
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for j in range(y.shape[1]):
        ax.plot(t, y[:, j], linewidth=1.0, label=labels[j])
    if wp_t is not None:
        for tk in wp_t:
            ax.axvline(float(tk), color="k", linestyle=":", alpha=0.3, linewidth=0.8)
    ax.set_xlabel(r"Time, $t$ (s)")
    ax.set_ylabel(y_label)
    style_2d_axes(ax)
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show() if show else plt.close(fig)


def _draw_robot_arm(ax, robot, q_row, color, linewidth=2.0, alpha=0.95, dot_size=26):
    """Helper: draw one robot arm snapshot onto ax. Returns list of joint positions."""
    L1 = 0.11336
    qk = tf.constant(q_row[None, :], dtype=DTYPE)
    points, _ = robot.forward_all_points(qk)
    pts_np = [p.numpy()[0] for p in points]

    T = tf.eye(4, batch_shape=[1], dtype=DTYPE)
    T01 = None
    for i in range(6):
        theta_i = qk[:, i] + tf.constant(robot.theta_offset[i], dtype=DTYPE)
        T = tf.matmul(T, dh_transform(robot.alpha[i], robot.a[i], robot.d[i], theta_i))
        if i == 0:
            T01 = T
    o1, o2 = pts_np[1], pts_np[2]
    z1_dir = T01[:, :3, 2].numpy()[0]
    p1 = o1 + L1 * z1_dir
    pts_line = np.asarray(pts_np[:2] + [p1, p1 + (o2 - o1)] + pts_np[2:])

    ax.plot(pts_line[:, 0], pts_line[:, 1], pts_line[:, 2],
            "-", color=color, linewidth=linewidth, alpha=alpha)
    ax.scatter(np.asarray(pts_np)[:, 0], np.asarray(pts_np)[:, 1],
               np.asarray(pts_np)[:, 2], color=[color], s=dot_size, depthshade=True)
    return pts_np


def plot_robot_snapshots_generic(robot, q, waypoints_xyz=None, waypoint_times=None,
                                  time_grid=None, n_show=6, title="Robot snapshots",
                                  save_path=None, show=False):
    """
    Robot link snapshots along trajectory.

    Frame selection
    ---------------
    Draws frames at the union of:
      - n_show uniformly-spaced times  (e.g. t = 0, 0.4, 0.8, 1.2, 1.6, 2.0)
      - waypoint_times (tau_opt)       — the optimised passing times

    All frames share the same coolwarm colourmap keyed to time.
    Waypoint frames are drawn with the same style as regular frames —
    the ★ target markers make it clear which postures correspond to waypoints.

    Parameters
    ----------
    waypoint_times : array-like, optional
        Optimised waypoint passing times (tau_opt). If provided, the
        corresponding trajectory frames are added to the uniform grid and
        a ◆ marker is drawn at the EE for each waypoint frame.
    """
    if time_grid is None:
        time_grid = np.linspace(0.0, 1.0, q.shape[0]).astype(np.float32)
    time_grid = np.asarray(time_grid, dtype=np.float32).reshape(-1)

    T_final = float(time_grid[-1])

    # ── Build frame index list ────────────────────────────────
    uniform_times = np.linspace(0.0, T_final, n_show, dtype=np.float32)

    if waypoint_times is not None:
        wpt = np.asarray(waypoint_times, dtype=np.float32).reshape(-1)
        all_times = np.union1d(np.round(uniform_times, 6), np.round(wpt, 6))
    else:
        all_times = uniform_times
        wpt = np.array([], dtype=np.float32)

    frame_idxs = [int(np.argmin(np.abs(time_grid - t))) for t in all_times]
    frame_idxs = sorted(set(frame_idxs))

    wp_colors = ["tab:blue", "tab:orange", "tab:red", "tab:purple", "tab:brown"]

    norm_c = plt.Normalize(vmin=float(time_grid.min()), vmax=float(time_grid.max()))
    cmap = plt.cm.coolwarm

    fig = plt.figure(figsize=(6.6, 5.6))
    ax = fig.add_subplot(111, projection="3d")

    all_start = None

    # ── Draw all frames ───────────────────────────────────────
    for fi, idx in enumerate(frame_idxs):
        color = cmap(norm_c(float(time_grid[idx])))
        pts_np = _draw_robot_arm(ax, robot, q[idx], color,
                                 linewidth=2.0, alpha=0.85, dot_size=22)
        if fi == 0:
            all_start = pts_np[-1]

    # ── Start marker ──────────────────────────────────────────
    if all_start is not None:
        ax.scatter([all_start[0]], [all_start[1]], [all_start[2]],
                   s=60, color="k", marker="o", label="start", zorder=5)

    # ── Target waypoint positions ★ ───────────────────────────
    if waypoints_xyz is not None:
        wp = np.asarray(waypoints_xyz)
        for k in range(len(wp)):
            c = wp_colors[k % len(wp_colors)]
            ax.scatter([wp[k, 0]], [wp[k, 1]], [wp[k, 2]],
                       s=130, marker="*", color=c,
                       label=f"waypoint {k+1}", zorder=6)

    style_3d_axes(ax)
    ax.set_xlabel(r"$x$ (m)", labelpad=2)
    ax.set_ylabel(r"$y$ (m)", labelpad=2)
    ax.set_zlabel(r"$z$ (m)", labelpad=2)
    ax.set_title(title)
    ax.set_xlim(-0.5, 0.6)
    ax.set_ylim(-0.6, 0.5)
    ax.set_zlim(0, 1.1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_c)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.75)
    cbar.set_label(r"Time, $t$ (s)")
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(smart_tick_formatter))
    ax.legend(loc='best', frameon=True, fontsize=9)
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300)
    plt.show() if show else plt.close(fig)


def plot_waypoint_error_comparison(metrics_a, metrics_b, name_a="ADA-L", name_b="B-spline",
                                    save_path=None, show=False):
    err_a = np.asarray(metrics_a["waypoint_errors"])
    err_b = np.asarray(metrics_b["waypoint_errors"])
    idx = np.arange(1, len(err_a) + 1)
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.bar(idx - width/2, err_a, width=width, label=name_a)
    ax.bar(idx + width/2, err_b, width=width, label=name_b)
    ax.set_xlabel("Waypoint index")
    ax.set_ylabel("Position error (m)")
    style_2d_axes(ax)
    ax.set_xticks(idx)
    sf = plt.ScalarFormatter(useMathText=True)
    sf.set_scientific(True)
    sf.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(sf)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.4))
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show() if show else plt.close(fig)


def plot_waypoint_time_comparison(res_a, res_b, name_a="ADA-L", name_b="B-spline",
                                   save_path=None, show=False):
    tau0  = np.asarray(res_a["waypoint_times_init"])
    tau_a = np.asarray(res_a["waypoint_times_opt"])
    tau_b = np.asarray(res_b["waypoint_times_opt"])
    idx = np.arange(1, len(tau0) + 1)

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    # ax.plot(idx, tau0,  "o--", linewidth=1.5, label="Initial")
    ax.plot(idx, tau_a, "o-",  linewidth=1.5, label=name_a)
    ax.plot(idx, tau_b, "s-",  linewidth=1.5, label=name_b)
    ax.set_xlabel("Waypoint index")
    ax.set_ylabel("Waypoint time (s)")
    style_2d_axes(ax)
    ax.set_xticks(idx)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.legend(frameon=False)
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show() if show else plt.close(fig)


# ============================================================
# animate_robot_trajectory (multi-waypoint version)
# ============================================================
def animate_robot_trajectory(robot, q, waypoints_xyz, time_grid,
                             save_path="robot_animation.mp4", fps=20,
                             title="Robot trajectory"):
    """
    Render the full trajectory frame-by-frame and save as mp4.
    All waypoints in waypoints_xyz are displayed as scatter markers.

    Parameters
    ----------
    robot        : UR5eKinematics (must have forward_all_points method)
    q            : np.ndarray, shape (N, 6)
    waypoints_xyz: array-like, shape (M, 3) — all waypoints
    time_grid    : array-like, shape (N,)
    save_path    : str  output mp4 path
    fps          : int  frames per second
    title        : str  axes title
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    L1 = 0.11336
    waypoints_xyz = np.asarray(waypoints_xyz)
    time_grid = np.asarray(time_grid, dtype=np.float32).reshape(-1)
    N = q.shape[0]
    if len(time_grid) != N:
        raise ValueError("time_grid length must match q.shape[0]")

    norm = plt.Normalize(vmin=float(time_grid.min()), vmax=float(time_grid.max()))
    cmap = plt.cm.coolwarm

    # Pre-compute all poses
    all_pts_line = []
    all_pts_scatter = []
    for k in range(N):
        qk = tf.constant(q[k:k+1], dtype=DTYPE)
        points, _ = robot.forward_all_points(qk)
        pts_np = [p.numpy()[0] for p in points]

        T = tf.eye(4, batch_shape=[1], dtype=DTYPE)
        T01 = None
        for i in range(6):
            theta_i = qk[:, i] + tf.constant(robot.theta_offset[i], dtype=DTYPE)
            A_i = dh_transform(robot.alpha[i], robot.a[i], robot.d[i], theta_i)
            T = tf.matmul(T, A_i)
            if i == 0:
                T01 = T

        o1 = pts_np[1]
        o2 = pts_np[2]
        z1_dir = T01[:, :3, 2].numpy()[0]
        p1_base = o1 + L1 * z1_dir
        p2_base = p1_base + (o2 - o1)

        all_pts_line.append(np.asarray(pts_np[:2] + [p1_base, p2_base] + pts_np[2:]))
        all_pts_scatter.append(np.asarray(pts_np))

    fig = plt.figure(figsize=(6.6, 5.6))
    ax = fig.add_subplot(111, projection="3d")

    # Fixed elements: start first, then all waypoints
    ax.scatter([all_pts_line[0][-1, 0]], [all_pts_line[0][-1, 1]], [all_pts_line[0][-1, 2]],
               s=100, color="k", label="start", zorder=5)
    wp_colors = ["tab:blue", "tab:orange", "tab:red", "tab:purple", "tab:brown"]
    for wi in range(len(waypoints_xyz)):
        c = wp_colors[wi % len(wp_colors)]
        ax.scatter([waypoints_xyz[wi, 0]], [waypoints_xyz[wi, 1]], [waypoints_xyz[wi, 2]],
                   s=120, marker="*", color=c, label=f"Waypoint {wi+1}", zorder=5)

    style_3d_axes(ax)
    ax.set_xlabel(r"$x$ (m)", labelpad=2)
    ax.set_ylabel(r"$y$ (m)", labelpad=2)
    ax.set_zlabel(r"$z$ (m)", labelpad=2)
    ax.set_title(title)
    ax.set_xlim(-0.5, 0.6)
    ax.set_ylim(-0.6, 0.5)
    ax.set_zlim(0, 1.1)
    ax.legend(frameon=True, fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.75)
    cbar.set_label(r"Time, $t$ (s)")
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    fig.tight_layout()

    # Animation state
    robot_line, = ax.plot([], [], [], "-", linewidth=2.0, alpha=0.95)
    robot_scatter = ax.scatter([], [], [], s=26, depthshade=True)
    trail_line, = ax.plot([], [], [], "-", linewidth=0.8, alpha=0.35, color="gray")

    ee_trail_x, ee_trail_y, ee_trail_z = [], [], []

    def init():
        robot_line.set_data([], [])
        robot_line.set_3d_properties([])
        trail_line.set_data([], [])
        trail_line.set_3d_properties([])
        return robot_line, robot_scatter, trail_line

    n_freeze = 5  # freeze last frame for 5 extra frames

    def update(k):
        # Clamp to last real frame for freeze frames
        idx = min(k, N - 1)
        color = cmap(norm(float(time_grid[idx])))
        pts_l = all_pts_line[idx]
        pts_s = all_pts_scatter[idx]

        # Current robot pose
        robot_line.set_data(pts_l[:, 0], pts_l[:, 1])
        robot_line.set_3d_properties(pts_l[:, 2])
        robot_line.set_color(color)

        robot_scatter._offsets3d = (pts_s[:, 0], pts_s[:, 1], pts_s[:, 2])
        robot_scatter.set_color([color] * len(pts_s))

        # End-effector trail (only append for real frames)
        if k < N:
            ee_trail_x.append(pts_l[-1, 0])
            ee_trail_y.append(pts_l[-1, 1])
            ee_trail_z.append(pts_l[-1, 2])
        trail_line.set_data(ee_trail_x, ee_trail_y)
        trail_line.set_3d_properties(ee_trail_z)

        return robot_line, robot_scatter, trail_line

    anim = FuncAnimation(fig, update, frames=N + n_freeze, init_func=init,
                         interval=1000 / fps, blit=False)

    writer = FFMpegWriter(fps=fps, metadata={"title": title}, bitrate=1800)
    anim.save(save_path, writer=writer, dpi=150)
    plt.close(fig)
    print(f"[animate] Saved to: {save_path}")


# ============================================================
# 10) save_full_report
# ============================================================
def save_full_report_multiwaypoint(
    out_dir, run_config, adal, adal_res, bs, bs_res,
    adal_metrics, bs_metrics, show_figures=False,
):
    ensure_dir(out_dir)
    fig_dir = Path(out_dir) / "figures"
    ensure_dir(fig_dir)
    fig_entries_md, fig_entries_docx = [], []

    def add_fig(title, filename):
        relp = f"figures/{filename}"
        absp = fig_dir / filename
        fig_entries_md.append((title, relp))
        fig_entries_docx.append((title, absp))
        return str(absp)

    joint_labels = [f"joint{j+1}" for j in range(6)]
    robot = adal.robot
    wp_t_adal = np.asarray(adal_res["waypoint_times_opt"])
    wp_t_bs  = np.asarray(bs_res["waypoint_times_opt"])
    waypoints_xyz = run_config["waypoints_xyz"]

    # ---- ADA-L joint derivative bundles ----
    plot_joint_bundle_single(
        adal_res["t_q"], adal_res["q"],
        title="ADA-L joint angle",
        y_label="Angle,\n$q$ (rad)",
        labels=joint_labels, wp_t=wp_t_adal,
        save_path=add_fig("ADA-L joint angle", "adal_joint_angle.png"),
        show=show_figures,
    )
    plot_joint_bundle_single(
        adal_res["t_qdot"], adal_res["qdot"],
        title="ADA-L joint angular velocity",
        y_label="Angular velocity,\n$\\dot{q}$ (rad/s)",
        labels=joint_labels, wp_t=wp_t_adal,
        save_path=add_fig("ADA-L joint angular velocity", "adal_joint_angular_velocity.png"),
        show=show_figures,
    )
    plot_joint_bundle_single(
        adal_res["t_qddot"], adal_res["qddot"],
        title="ADA-L joint angular acceleration",
        y_label="Angular acceleration,\n$\\ddot{q}$ (rad/s$^2$)",
        labels=joint_labels, wp_t=wp_t_adal,
        save_path=add_fig("ADA-L joint angular acceleration", "adal_joint_angular_acceleration.png"),
        show=show_figures,
    )
    plot_joint_bundle_single(
        adal_res["t_qjerk"], adal_res["qjerk"],
        title="ADA-L joint angular jerk",
        y_label="Angular jerk,\n$\\dddot{q}$ (rad/s$^3$)",
        labels=joint_labels, wp_t=wp_t_adal,
        save_path=add_fig("ADA-L joint angular jerk", "adal_joint_angular_jerk.png"),
        show=show_figures,
    )

    # ---- B-spline joint derivative bundles (commented out — not needed) ----
    # plot_joint_bundle_single(
    #     bs_res["t_qdot"], bs_res["qdot"],
    #     title="B-spline joint angular velocity",
    #     y_label="Angular velocity,\n$\\dot{q}$ (rad/s)",
    #     labels=joint_labels, wp_t=wp_t_bs,
    #     save_path=add_fig("B-spline joint angular velocity", "bs_joint_angular_velocity.png"),
    #     show=show_figures,
    # )
    # plot_joint_bundle_single(
    #     bs_res["t_qddot"], bs_res["qddot"],
    #     title="B-spline joint angular acceleration",
    #     y_label="Angular acceleration,\n$\\ddot{q}$ (rad/s$^2$)",
    #     labels=joint_labels, wp_t=wp_t_bs,
    #     save_path=add_fig("B-spline joint angular acceleration", "bs_joint_angular_acceleration.png"),
    #     show=show_figures,
    # )
    # plot_joint_bundle_single(
    #     bs_res["t_qjerk"], bs_res["qjerk"],
    #     title="B-spline joint angular jerk",
    #     y_label="Angular jerk,\n$\\dddot{q}$ (rad/s$^3$)",
    #     labels=joint_labels, wp_t=wp_t_bs,
    #     save_path=add_fig("B-spline joint angular jerk", "bs_joint_angular_jerk.png"),
    #     show=show_figures,
    # )

    # ---- Comparison figures ----
    plot_compare_ee_path(
        adal_res, bs_res, name_a="ADA-L", name_b="B-spline",
        title="End-effector path comparison",
        save_path=add_fig("EE path comparison", "compare_ee_path.png"),
        show=show_figures,
    )
    plot_compare_joint_bundle(
        adal_res, bs_res, name_a="ADA-L", name_b="B-spline",
        title="Joint trajectory comparison",
        save_path=add_fig("Joint trajectory comparison", "compare_joint_bundle.png"),
        show=show_figures,
    )

    adal_vel_norm  = np.linalg.norm(adal_res["qdot"],  axis=1)
    bs_vel_norm   = np.linalg.norm(bs_res["qdot"],   axis=1)
    adal_acc_norm  = np.linalg.norm(adal_res["qddot"], axis=1)
    bs_acc_norm   = np.linalg.norm(bs_res["qddot"],  axis=1)
    adal_jerk_norm = np.linalg.norm(adal_res["qjerk"], axis=1)
    bs_jerk_norm  = np.linalg.norm(bs_res["qjerk"],  axis=1)

    plot_norm_profiles_compare(
        adal_res["t_qdot"], adal_vel_norm, bs_vel_norm,
        title="Angular velocity norm: ADA-L vs B-spline",
        y_label="Angular velocity \nnorm, $\\|\\dot{q}\\|$ (rad/s)",
        label_a="ADA-L", label_b="B-spline", wp_t=wp_t_adal,
        save_path=add_fig("Angular velocity norm comparison", "compare_angular_velocity_norm.png"),
        show=show_figures,
    )
    plot_norm_profiles_compare(
        adal_res["t_qddot"], adal_acc_norm, bs_acc_norm,
        title="Angular acceleration norm: ADA-L vs B-spline",
        y_label="Angular acceleration \nnorm, $\\|\\ddot{q}\\|$ (rad/s$^2$)",
        label_a="ADA-L", label_b="B-spline", wp_t=wp_t_adal,
        save_path=add_fig("Angular acceleration norm comparison", "compare_angular_acceleration_norm.png"),
        show=show_figures,
    )
    plot_norm_profiles_compare(
        adal_res["t_qjerk"], adal_jerk_norm, bs_jerk_norm,
        title="Angular jerk norm: ADA-L vs B-spline",
        y_label="Angular jerk norm, \n$\\|\\dddot{q}\\|$ (rad/s$^3$)",
        label_a="ADA-L", label_b="B-spline", wp_t=wp_t_adal,
        save_path=add_fig("Angular jerk norm comparison", "compare_angular_jerk_norm.png"),
        show=show_figures,
    )

    # ---- Waypoint-specific comparison ----
    plot_waypoint_error_comparison(
        adal_metrics, bs_metrics, name_a="ADA-L", name_b="B-spline",
        save_path=add_fig("Waypoint error comparison", "compare_waypoint_errors.png"),
        show=show_figures,
    )
    plot_waypoint_time_comparison(
        adal_res, bs_res, name_a="ADA-L", name_b="B-spline",
        save_path=add_fig("Waypoint time comparison", "compare_waypoint_times.png"),
        show=show_figures,
    )

    # ---- Robot snapshots ----
    plot_robot_snapshots_generic(
        robot=robot, q=adal_res["q"], waypoints_xyz=waypoints_xyz,
        waypoint_times=np.asarray(adal_res["waypoint_times_opt"]),
        time_grid=adal_res["t_q"], n_show=13,
        title="ADA-L trajectory snapshots",
        save_path=add_fig("ADA-L robot snapshots", "adal_robot_snapshots.png"),
        show=show_figures,
    )
    plot_robot_snapshots_generic(
        robot=robot, q=bs_res["q"], waypoints_xyz=waypoints_xyz,
        waypoint_times=np.asarray(bs_res["waypoint_times_opt"]),
        time_grid=bs_res["t_q"], n_show=13,
        title="B-spline trajectory snapshots",
        save_path=add_fig("B-spline robot snapshots", "bs_robot_snapshots.png"),
        show=show_figures,
    )

    plot_training_losses_generic(
        adal.history,
        title="ADA-L training losses",
        save_path=add_fig("ADA-L training losses", "adal_training_losses.png"),
        show=show_figures,
    )

    plot_training_losses_generic(
        bs.history,
        title="B-spline training losses",
        save_path=add_fig("B-spline training losses", "bs_training_losses.png"),
        show=show_figures,
    )

    plot_panel_weights(
        adal,
        T_final=adal.T_final,
        save_path=add_fig("ADA-L panel weights", "adal_panel_weights.png"),
        show=show_figures,
    )

    # ---- Save raw files ----
    save_json(adal_metrics, Path(out_dir) / "metrics_adal.json")
    save_json(bs_metrics,  Path(out_dir) / "metrics_bspline.json")
    save_comparison_csv(adal_metrics, bs_metrics,
                        Path(out_dir) / "metrics_comparison.csv",
                        name_a="ADA-L", name_b="B-spline")

    # joint_trajectory_*.csv saved earlier in main (before validation step).

    # Full trajectory CSVs (q, qdot, qddot, qjerk)
    save_joint_full_csv(
        adal_res["t_q"], adal_res["q"], adal_res["qdot"],
        adal_res["qddot"], adal_res["qjerk"],
        Path(out_dir) / "ADAL_trajectory_full.csv")
    save_joint_full_csv(
        bs_res["t_q"], bs_res["q"], bs_res["qdot"],
        bs_res["qddot"], bs_res["qjerk"],
        Path(out_dir) / "Bspline_trajectory_full.csv")

    np.savez(
        Path(out_dir) / "trajectory_data.npz",
        q0=run_config["q0"],
        waypoints_xyz=run_config["waypoints_xyz"],
        waypoint_times=run_config["waypoint_times"],
        q_adal=adal_res["q"],    qdot_adal=adal_res["qdot"],
        qddot_adal=adal_res["qddot"], qjerk_adal=adal_res["qjerk"],
        ee_adal=adal_res["ee"],  ee_wp_adal=adal_res["ee_wp"],
        tau_adal=adal_res["waypoint_times_opt"],
        q_bs=bs_res["q"],      qdot_bs=bs_res["qdot"],
        qddot_bs=bs_res["qddot"],   qjerk_bs=bs_res["qjerk"],
        ee_bs=bs_res["ee"],    ee_wp_bs=bs_res["ee_wp"],
        tau_bs=bs_res["waypoint_times_opt"],
    )

    write_markdown_report_multiwaypoint(
        report_path=Path(out_dir) / "report.md",
        run_config=run_config, metrics_a=adal_metrics, metrics_b=bs_metrics,
        res_a=adal_res, res_b=bs_res, fig_paths=fig_entries_md,
        name_a="ADA-L", name_b="B-spline",
    )
    print(f"[report] Markdown saved to: {Path(out_dir) / 'report.md'}")

    try_write_docx_report_multiwaypoint(
        docx_path=Path(out_dir) / "report.docx",
        run_config=run_config, metrics_a=adal_metrics, metrics_b=bs_metrics,
        res_a=adal_res, res_b=bs_res, fig_abs_paths=fig_entries_docx,
        name_a="ADA-L", name_b="B-spline",
    )


# ============================================================
# 11) Main
# ============================================================
if __name__ == "__main__":
    # Initial pose in URDF/robot convention (same as Ex1/Ex2 unified setting)
    q0_UR5e = np.array(
        [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, np.pi/2],
        dtype=np.float32,
    )
    # Convert to solver input: theta_i = q + theta_offset
    _theta_offset = np.array(UR5eKinematics().theta_offset, dtype=np.float32)
    q0 = (q0_UR5e - _theta_offset).astype(np.float32)


    waypoints_xyz = np.array([
        [-0.05, 0.25, 0.3],
        [0.35, -0.35, 0.7],
        [0.60, -0.15, 0.55],
        # [-0.05, 0.25, 0.3],
        # [0.3, -0.3, 0.8],
        # [0.60, -0.15, 0.55],
        # [0.15, 0.25, 0.7],
        # [0.1, -0.5, 0.7],
        # [0.60, -0.35, 0.55],
    ], dtype=np.float32)


    waypoint_times = np.array([2.0, 4.0, 6.0], dtype=np.float32)

    T_final = 6.0
    Nt = 301

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = str(Path(__file__).resolve().parent / f"results_multiwaypoint_adal_vs_bspline_L1_{timestamp}")
    ensure_dir(out_dir)

    # ---- ADA-L ----
    print("\nRunning ADA-L multi-waypoint continuous-passing planner...")
    adal = ADALMultiWaypointPlanner(
        q0=q0, waypoints_xyz=waypoints_xyz, waypoint_times=waypoint_times,
        T_final=T_final, Nt=Nt, gamma=1.0, max_order=5, N_p=30, seed=0,
        free_waypoint_times=True, tau_min_gap=0.20, tau_sigma=0.04,
        z_floor=0.0,
    )
    adal_solve_time = adal.train(epochs=500, lr=3e-2, print_every=100,
                                use_lbfgs=True, lbfgs_maxiter=2000)
    adal_res     = adal.results()
    adal_metrics = benchmark_multiwaypoint_metrics(adal_res, adal_solve_time)

    # ---- B-spline ----
    print("\nRunning B-spline multi-waypoint continuous-passing planner...")
    bs = BSplineMultiWaypointPlanner(
        q0=q0, waypoints_xyz=waypoints_xyz, waypoint_times=waypoint_times,
        T_final=T_final, Nt=Nt, n_ctrl=20, degree=5, seed=0,
        free_waypoint_times=True, tau_min_gap=0.20, tau_sigma=0.04,
        z_floor=0.0,
    )
    bs_solve_time = bs.train(epochs=500, lr=3e-2, print_every=100,
                              use_lbfgs=True, lbfgs_maxiter=2000)
    bs_res     = bs.results()
    bs_metrics = benchmark_multiwaypoint_metrics(bs_res, bs_solve_time)

    # ---- Print ----
    print_metrics_table("ADA-L", adal_metrics)
    print_metrics_table("B-spline", bs_metrics)
    print_comparison_table(adal_metrics, bs_metrics, name_a="ADA-L", name_b="B-spline")

    # ---- Joint trajectory CSVs (saved early for validation) ----
    adal_csv_path = Path(out_dir) / "joint_trajectory_adal.csv"
    bs_csv_path  = Path(out_dir) / "joint_trajectory_bspline.csv"
    save_joint_trajectory_csv(adal_res["t_q"], adal_res["q"], adal_csv_path)
    save_joint_trajectory_csv(bs_res["t_q"],  bs_res["q"],  bs_csv_path)

    # ---- UR5e trajectory validation ----
    adal_ok, _, _ = validate_ur5e_trajectory(adal_res["t_q"], adal_res["q"])
    bs_ok,  _, _ = validate_ur5e_trajectory(bs_res["t_q"],  bs_res["q"])

    print("\n=== UR5e Validation Verdict ===")
    print(f"  ADA-L      : {'PASS' if adal_ok else 'FAIL'}  ({adal_csv_path.name})")
    print(f"  B-spline : {'PASS' if bs_ok  else 'FAIL'}  ({bs_csv_path.name})")

    # ---- collision check (capsule-based UR5e self + floor; no payload) ----
    adal_coll_ok, adal_coll_first, adal_coll_n, adal_coll_total = (
        validate_collision(adal_res["q"])
    )
    bs_coll_ok, bs_coll_first, bs_coll_n, bs_coll_total = (
        validate_collision(bs_res["q"])
    )
    print_collision_result(
        "ADA-L", adal_coll_ok, adal_coll_first, adal_coll_n, adal_coll_total,
        t_traj=adal_res["t_q"],
        q_first=(adal_res["q"][adal_coll_first] if adal_coll_first is not None else None),
    )
    print_collision_result(
        "B-spline", bs_coll_ok, bs_coll_first, bs_coll_n, bs_coll_total,
        t_traj=bs_res["t_q"],
        q_first=(bs_res["q"][bs_coll_first] if bs_coll_first is not None else None),
    )
    print("\n=== Collision Verdict ===")
    print(f"  ADA-L      : {'PASS' if adal_coll_ok else 'FAIL'}")
    print(f"  B-spline : {'PASS' if bs_coll_ok  else 'FAIL'}")

    # ---- Worst penetration across ALL states (capsule margins) ----
    from capsule_collision_checker import UR5eCapsuleChecker as _UR5eCC
    _ck = _UR5eCC(include_payload=False)
    print("\n=== Worst penetration (capsule margins, across all states) ===")
    for _tag, _q_arr, _t_arr in [("ADA-L",      adal_res["q"], adal_res["t_q"]),
                                  ("B-spline", bs_res["q"],  bs_res["t_q"])]:
        _worst = None
        for _i in range(len(_q_arr)):
            for _pa, _pb, _d, _thr in _ck.collision_report(_q_arr[_i]):
                _m = _d - _thr
                if _worst is None or _m < _worst[3]:
                    _worst = (_i, _pa, _pb, _m)
        if _worst is None:
            print(f"  {_tag:<10}: no penetration in {len(_q_arr)} states.")
        else:
            _i, _pa, _pb, _m = _worst
            print(f"  {_tag:<10}: worst {_m*1000:+.3f} mm at "
                  f"idx={_i} t={float(_t_arr[_i]):.3f}s ({_pa} <-> {_pb})")

    # ---- Save ----
    run_config = {
        "q0": q0.copy(),
        "waypoints_xyz": waypoints_xyz.copy(),
        "waypoint_times": waypoint_times.copy(),
        "T_final": T_final,
        "Nt": Nt,
    }

    save_full_report_multiwaypoint(
        out_dir=out_dir, run_config=run_config,
        adal=adal, adal_res=adal_res, bs=bs, bs_res=bs_res,
        adal_metrics=adal_metrics, bs_metrics=bs_metrics,
        show_figures=False,
    )

    import matplotlib
    # matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\Users\user\anaconda3\Library\bin\ffmpeg.exe'

    # ---- Robot animations ----
    robot = adal.robot
    # animate_robot_trajectory(
        # robot=robot, q=adal_res["q"], waypoints_xyz=waypoints_xyz,
        # time_grid=adal_res["t_q"],
        # save_path=str(Path(out_dir) / "ADAL_animation.mp4"),
        # fps=30, title="ADA-L trajectory",
    # )
    # animate_robot_trajectory(
        # robot=robot, q=bs_res["q"], waypoints_xyz=waypoints_xyz,
        # time_grid=bs_res["t_q"],
        # save_path=str(Path(out_dir) / "Bspline_animation.mp4"),
        # fps=30, title="B-spline trajectory",
    # )

    print(f"\nAll outputs saved under: {out_dir}")
