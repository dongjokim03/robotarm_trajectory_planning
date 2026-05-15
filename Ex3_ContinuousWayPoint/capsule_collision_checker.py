#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UR5e capsule-based self/payload collision checker (Windows-friendly, ROS-free).

Each robot link is approximated as a capsule (line segment + radius).
The end-effector payload is treated as an oriented bounding box (OBB)
attached to the tool0 frame. Collision is decided by:
  - segment-segment distance for capsule-capsule pairs
  - sample-based capsule-OBB distance for link-payload pairs

Adjacent links and link-payload pairs that always touch are excluded
from the check (Allowed Collision Matrix, ACM).

Conservative manual capsule parameters are used (slightly larger than
the actual link meshes) so the checker errs on the side of declaring
collision. Joint angles `q` follow the DH convention (i.e. the value
optimized in the planner, BEFORE adding theta_offset).

Public API
----------
    UR5eCapsuleChecker(payload_size=(0.15, 0.15, 0.0175),
                       payload_offset_z=None)
        .check_state(q)                -> bool   (True = collision)
        .check_trajectory(q_traj)      -> bool   (True if any state collides)
        .first_collision_index(q_traj) -> Optional[int]
        .collision_report(q)           -> List[(name_a, name_b, distance)]
"""

from __future__ import annotations

import numpy as np

# UR5e DH parameters
_ALPHA = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]
_A     = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
_D     = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
_THETA_OFFSET = [0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0]


def _dh_matrix(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ], dtype=np.float64)


def _fk_chain(q: np.ndarray) -> list[np.ndarray]:
    """Return list of 7 SE(3) transforms [T0=I, T1, ..., T6=tool0]."""
    Ts = [np.eye(4, dtype=np.float64)]
    T = np.eye(4, dtype=np.float64)
    for i in range(6):
        theta = float(q[i]) + _THETA_OFFSET[i]
        T = T @ _dh_matrix(_ALPHA[i], _A[i], _D[i], theta)
        Ts.append(T.copy())
    return Ts


def _segment_segment_distance(p1: np.ndarray, q1: np.ndarray,
                              p2: np.ndarray, q2: np.ndarray) -> float:
    """Closest distance between segment p1-q1 and segment p2-q2 (Eberly)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    EPS = 1e-12

    if a <= EPS and e <= EPS:
        return float(np.linalg.norm(r))
    if a <= EPS:
        s = 0.0
        t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, r))
        if e <= EPS:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            if denom != 0.0:
                s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0))
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))

    closest1 = p1 + d1 * s
    closest2 = p2 + d2 * t
    return float(np.linalg.norm(closest1 - closest2))


def _point_obb_distance(p: np.ndarray, box_center: np.ndarray,
                        box_axes: np.ndarray, half_extents: np.ndarray) -> float:
    """Distance from point p to OBB; returns 0 if p is inside."""
    d = p - box_center
    delta_sq = 0.0
    for i in range(3):
        e = float(np.dot(d, box_axes[:, i]))
        h = float(half_extents[i])
        if e > h:
            delta_sq += (e - h) ** 2
        elif e < -h:
            delta_sq += (e + h) ** 2
    return float(np.sqrt(delta_sq))


def _capsule_obb_distance(c0: np.ndarray, c1: np.ndarray,
                          box_center: np.ndarray, box_axes: np.ndarray,
                          half_extents: np.ndarray, n_samples: int = 12) -> float:
    """
    Approximate the closest distance between the central segment c0-c1 of
    a capsule and an OBB by sampling the segment.

    n_samples = 12 gives an interval of ~3.3 cm on the longest UR5e link
    (forearm, 0.392 m). With link radii >= 6 cm and payload thickness
    1.75 cm, this is dense enough to catch any real intersection.
    """
    dmin = np.inf
    for k in range(n_samples + 1):
        t = k / n_samples
        p = c0 * (1.0 - t) + c1 * t
        d = _point_obb_distance(p, box_center, box_axes, half_extents)
        if d < dmin:
            dmin = d
            if dmin == 0.0:
                return 0.0
    return float(dmin)


class UR5eCapsuleChecker:
    """
    Capsule-based collision checker for UR5e + box payload.

    Parameters
    ----------
    payload_size : tuple of 3 floats, default (0.15, 0.15, 0.0175)
        Payload dimensions in meters (x, y, z) in the tool0 local frame.
    payload_offset_z : float or None
        Distance from the flange (tool0 origin) to the payload box center
        along tool0 +Z. If None, defaults to payload_size[2] / 2 so the box
        sits flush against the flange (mirrors moveit_collision_checker.py).
    """

    # Manually defined capsule radii for UR5e links (slightly conservative).
    R_BASE      = 0.080
    R_SHOULDER  = 0.075
    R_UPPER_ARM = 0.075
    R_FOREARM   = 0.062
    R_WRIST_1   = 0.060
    R_WRIST_2   = 0.060
    R_WRIST_3   = 0.055

    LINK_NAMES = (
        "base", "shoulder", "upper_arm", "forearm",
        "wrist_1", "wrist_2", "wrist_3",
    )

    def __init__(self, payload_size=(0.15, 0.15, 0.0175),
                 payload_offset_z=None):
        size = np.asarray(payload_size, dtype=np.float64)
        if size.shape != (3,):
            raise ValueError("payload_size must have 3 components")
        self.payload_size = size
        self.payload_half_extents = size / 2.0
        if payload_offset_z is None:
            payload_offset_z = float(size[2]) / 2.0
        self.payload_offset_z = float(payload_offset_z)

        # Static base capsule (in world frame). Approximates the cylindrical
        # pedestal between the floor and the shoulder joint.
        self._base_p0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._base_p1 = np.array([0.0, 0.0, 0.10], dtype=np.float64)

        # Allowed Collision Matrix: pairs always considered non-colliding.
        # Adjacent UR5e links + payload-to-end-of-arm.
        self._acm = self._build_acm()

    @staticmethod
    def _build_acm() -> set:
        # Mirrors the standard UR5e MoveIt SRDF "disable_collisions" list:
        #   - "Adjacent" pairs (parent/child in URDF)
        #   - "Never" pairs that geometrically can't be separated in any
        #     reachable joint configuration of the standard arm.
        # Plus payload-near-flange pairs.
        adj = [
            # Adjacent
            ("base", "shoulder"),
            ("shoulder", "upper_arm"),
            ("upper_arm", "forearm"),
            ("forearm", "wrist_1"),
            ("wrist_1", "wrist_2"),
            ("wrist_2", "wrist_3"),
            # Never collide in standard arm geometry
            ("base", "upper_arm"),
            ("wrist_1", "wrist_3"),
            # Payload rigidly attached at the flange
            ("wrist_3", "payload"),
            ("wrist_2", "payload"),
        ]
        acm = set()
        for a, b in adj:
            acm.add((a, b))
            acm.add((b, a))
        return acm

    def _capsules_for_state(self, q: np.ndarray):
        """
        Compute link capsules (p0, p1, radius) and payload OBB for a state.

        Returns
        -------
        capsules : dict[str, (np.ndarray, np.ndarray, float)]
        payload  : dict with keys "center", "axes" (3x3), "half_extents"
        """
        Ts = _fk_chain(q)
        P = [T[:3, 3] for T in Ts]  # P[0]..P[6]

        # Shoulder: short capsule centered on P[1] along joint-1 z-axis.
        z1 = Ts[1][:3, 2]
        shoulder_p0 = P[1] - 0.04 * z1
        shoulder_p1 = P[1] + 0.04 * z1

        capsules = {
            "base":      (self._base_p0,        self._base_p1,        self.R_BASE),
            "shoulder":  (shoulder_p0,          shoulder_p1,          self.R_SHOULDER),
            "upper_arm": (P[1].copy(),          P[2].copy(),          self.R_UPPER_ARM),
            "forearm":   (P[2].copy(),          P[3].copy(),          self.R_FOREARM),
            "wrist_1":   (P[3].copy(),          P[4].copy(),          self.R_WRIST_1),
            "wrist_2":   (P[4].copy(),          P[5].copy(),          self.R_WRIST_2),
            "wrist_3":   (P[5].copy(),          P[6].copy(),          self.R_WRIST_3),
        }

        T_tool0 = Ts[6]
        R = T_tool0[:3, :3]
        payload_center = P[6] + R[:, 2] * self.payload_offset_z
        payload = {
            "center":       payload_center,
            "axes":         R.copy(),
            "half_extents": self.payload_half_extents.copy(),
        }
        return capsules, payload

    def _is_skipped(self, name_a: str, name_b: str) -> bool:
        return (name_a, name_b) in self._acm

    def check_state(self, q) -> bool:
        """Return True if the joint state q (DH convention, length 6) collides."""
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape != (6,):
            raise ValueError(f"q must have shape (6,), got {q.shape}")
        capsules, payload = self._capsules_for_state(q)
        names = list(capsules.keys())

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ni, nj = names[i], names[j]
                if self._is_skipped(ni, nj):
                    continue
                c0i, c1i, ri = capsules[ni]
                c0j, c1j, rj = capsules[nj]
                d = _segment_segment_distance(c0i, c1i, c0j, c1j)
                if d < ri + rj:
                    return True

        for ni, (c0, c1, r) in capsules.items():
            if self._is_skipped(ni, "payload"):
                continue
            d = _capsule_obb_distance(
                c0, c1,
                payload["center"], payload["axes"], payload["half_extents"],
            )
            if d < r:
                return True

        # Floor (z=0) check: capsule lowest point = min(c0.z, c1.z) - r.
        # Skip "base" (the pedestal capsule sits at the floor by design).
        for ni, (c0, c1, r) in capsules.items():
            if ni == "base":
                continue
            if min(float(c0[2]), float(c1[2])) - float(r) < 0.0:
                return True
        # Payload OBB lowest point: center.z - sum_j half_extent_j * |axis_j.z|.
        abs_axis_z = np.abs(payload["axes"][2, :])
        min_z_payload = float(payload["center"][2]) - float(
            np.sum(payload["half_extents"] * abs_axis_z))
        if min_z_payload < 0.0:
            return True

        return False

    def check_trajectory(self, q_traj) -> bool:
        """True if any state along the (N, 6) trajectory is in collision."""
        q_traj = np.asarray(q_traj, dtype=np.float64)
        if q_traj.ndim != 2 or q_traj.shape[1] != 6:
            raise ValueError(f"q_traj must have shape (N, 6), got {q_traj.shape}")
        for i in range(q_traj.shape[0]):
            if self.check_state(q_traj[i]):
                return True
        return False

    def first_collision_index(self, q_traj):
        """Return the first index i where q_traj[i] is in collision, or None."""
        q_traj = np.asarray(q_traj, dtype=np.float64)
        if q_traj.ndim != 2 or q_traj.shape[1] != 6:
            raise ValueError(f"q_traj must have shape (N, 6), got {q_traj.shape}")
        for i in range(q_traj.shape[0]):
            if self.check_state(q_traj[i]):
                return i
        return None

    def collision_report(self, q):
        """
        Return a list of (name_a, name_b, distance, threshold) for every pair
        in collision (distance < radius_a + radius_b for capsules, or
        distance < radius for capsule-OBB). Useful for debugging.
        """
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        capsules, payload = self._capsules_for_state(q)
        names = list(capsules.keys())
        violations = []

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ni, nj = names[i], names[j]
                if self._is_skipped(ni, nj):
                    continue
                c0i, c1i, ri = capsules[ni]
                c0j, c1j, rj = capsules[nj]
                d = _segment_segment_distance(c0i, c1i, c0j, c1j)
                if d < ri + rj:
                    violations.append((ni, nj, d, ri + rj))

        for ni, (c0, c1, r) in capsules.items():
            if self._is_skipped(ni, "payload"):
                continue
            d = _capsule_obb_distance(
                c0, c1,
                payload["center"], payload["axes"], payload["half_extents"],
            )
            if d < r:
                violations.append((ni, "payload", d, r))

        for ni, (c0, c1, r) in capsules.items():
            if ni == "base":
                continue
            d = min(float(c0[2]), float(c1[2])) - float(r)
            if d < 0.0:
                violations.append((ni, "floor", d, 0.0))
        abs_axis_z = np.abs(payload["axes"][2, :])
        min_z_payload = float(payload["center"][2]) - float(
            np.sum(payload["half_extents"] * abs_axis_z))
        if min_z_payload < 0.0:
            violations.append(("payload", "floor", min_z_payload, 0.0))

        return violations


__all__ = ["UR5eCapsuleChecker"]
