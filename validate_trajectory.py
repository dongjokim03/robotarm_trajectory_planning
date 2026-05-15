"""
Validate CSV trajectory against UR5e joint limits and dynamics constraints.

Usage:
    python3 src/script/validate_trajectory.py src/trajectory/joint_trajectory_quintic.csv
    python3 src/script/validate_trajectory.py src/trajectory/joint_trajectory_adal.csv --plot
"""

import argparse
import csv
import math
import sys

import numpy as np

# UR5e joint limits (Universal Robots e-Series User Manual, UR5e)
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

# rad
POSITION_LIMITS = [
    (-2 * math.pi, 2 * math.pi),  # shoulder_pan
    (-2 * math.pi, 2 * math.pi),  # shoulder_lift
    (-math.pi,      math.pi),      # elbow
    (-2 * math.pi, 2 * math.pi),  # wrist_1
    (-2 * math.pi, 2 * math.pi),  # wrist_2
    (-2 * math.pi, 2 * math.pi),  # wrist_3
]

# rad/s — UR5e User Manual v5.8: all joints 180 deg/s
VELOCITY_LIMITS = [
    3.14159,  # 180 deg/s
    3.14159,
    3.14159,
    3.14159,
    3.14159,
    3.14159,
]

# rad/s^2 - UR5e official spec is not published; using a practical safe value
ACCELERATION_LIMITS = [
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
]

# Continuity check: maximum allowed position change between adjacent waypoints (rad)
MAX_POSITION_JUMP = 0.5


def load_csv(filepath):
    times, positions = [], []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time"]))
            positions.append([float(row[f"q{i}"]) for i in range(1, 7)])
    return np.array(times), np.array(positions)


def finite_diff(times, values):
    n = len(times)
    vel = np.zeros_like(values)
    acc = np.zeros_like(values)

    for i in range(1, n - 1):
        dt_f = times[i + 1] - times[i]
        dt_b = times[i] - times[i - 1]
        vel[i] = (values[i + 1] - values[i - 1]) / (dt_f + dt_b)
        acc[i] = (values[i + 1] - 2 * values[i] + values[i - 1]) / (dt_f * dt_b)

    vel[0]  = (values[1]  - values[0])  / (times[1]  - times[0])
    vel[-1] = (values[-1] - values[-2]) / (times[-1] - times[-2])
    acc[0]  = acc[1]
    acc[-1] = acc[-2]

    return vel, acc


def check_position_limits(times, positions):
    errors = []
    for j in range(6):
        lo, hi = POSITION_LIMITS[j]
        violations = np.where((positions[:, j] < lo) | (positions[:, j] > hi))[0]
        for idx in violations:
            errors.append(
                f"  [POSITION] Joint {j+1} ({JOINT_NAMES[j]}): "
                f"q={math.degrees(positions[idx, j]):.2f} deg "
                f"at t={times[idx]:.3f}s  (limit: [{math.degrees(lo):.0f}, {math.degrees(hi):.0f}] deg)"
            )
    return errors


def check_velocity_limits(times, positions):
    velocities, _ = finite_diff(times, positions)
    errors = []
    for j in range(6):
        vmax = VELOCITY_LIMITS[j]
        violations = np.where(np.abs(velocities[:, j]) > vmax)[0]
        for idx in violations:
            errors.append(
                f"  [VELOCITY] Joint {j+1} ({JOINT_NAMES[j]}): "
                f"|v|={math.degrees(abs(velocities[idx, j])):.2f} deg/s "
                f"at t={times[idx]:.3f}s  (limit: {math.degrees(vmax):.0f} deg/s)"
            )
    return errors, velocities


def check_acceleration_limits(times, positions):
    _, accelerations = finite_diff(times, positions)
    errors = []
    for j in range(6):
        amax = ACCELERATION_LIMITS[j]
        violations = np.where(np.abs(accelerations[:, j]) > amax)[0]
        for idx in violations:
            errors.append(
                f"  [ACCEL]    Joint {j+1} ({JOINT_NAMES[j]}): "
                f"|a|={math.degrees(abs(accelerations[idx, j])):.2f} deg/s² "
                f"at t={times[idx]:.3f}s  (limit: {math.degrees(amax):.0f} deg/s²)"
            )
    return errors, accelerations


def check_continuity(times, positions):
    errors = []
    diffs = np.abs(np.diff(positions, axis=0))
    for i in range(len(times) - 1):
        for j in range(6):
            if diffs[i, j] > MAX_POSITION_JUMP:
                errors.append(
                    f"  [JUMP]     Joint {j+1} ({JOINT_NAMES[j]}): "
                    f"Δq={math.degrees(diffs[i, j]):.2f} deg "
                    f"between t={times[i]:.3f}s and t={times[i+1]:.3f}s"
                )
    return errors


def check_timestep(times):
    dts = np.diff(times)
    errors = []
    if np.any(dts <= 0):
        bad = np.where(dts <= 0)[0]
        for idx in bad:
            errors.append(f"  [TIMESTEP] Non-monotonic time at index {idx}: dt={dts[idx]:.6f}s")
    if np.any(dts > 0.5):
        bad = np.where(dts > 0.5)[0]
        for idx in bad:
            errors.append(f"  [TIMESTEP] Large gap at t={times[idx]:.3f}s: dt={dts[idx]:.3f}s")
    return errors


def print_summary(times, positions, velocities, accelerations):
    print("\n========== Trajectory Summary ==========")
    print(f"  Waypoints  : {len(times)}")
    print(f"  Duration   : {times[-1]:.3f} s")
    dt = np.diff(times)
    print(f"  Timestep   : mean={dt.mean()*1000:.2f}ms  min={dt.min()*1000:.2f}ms  max={dt.max()*1000:.2f}ms")
    print()
    print(f"  {'Joint':<18} {'pos min':>10} {'pos max':>10} {'|vel| max':>12} {'|acc| max':>12}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")
    for j in range(6):
        pmin = math.degrees(positions[:, j].min())
        pmax = math.degrees(positions[:, j].max())
        vmax = math.degrees(np.abs(velocities[:, j]).max())
        amax = math.degrees(np.abs(accelerations[:, j]).max())
        vlim = math.degrees(VELOCITY_LIMITS[j])
        alim = math.degrees(ACCELERATION_LIMITS[j])
        vflag = " !" if vmax > vlim else "  "
        aflag = " !" if amax > alim else "  "
        print(f"  {JOINT_NAMES[j]:<18} {pmin:>9.2f}° {pmax:>9.2f}° {vmax:>10.2f}°/s{vflag} {amax:>10.2f}°/s²{aflag}")


def plot_trajectory(times, positions, velocities, accelerations):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[INFO] matplotlib not installed, skipping plot. Install with: pip3 install matplotlib")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    for j in range(6):
        axes[0].plot(times, np.degrees(positions[:, j]), color=colors[j], label=JOINT_NAMES[j])
    axes[0].set_ylabel("Position (deg)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True)

    for j in range(6):
        axes[1].plot(times, np.degrees(velocities[:, j]), color=colors[j], label=JOINT_NAMES[j])
        axes[1].axhline(math.degrees(VELOCITY_LIMITS[j]),  color=colors[j], linestyle="--", linewidth=0.8, alpha=0.5)
        axes[1].axhline(-math.degrees(VELOCITY_LIMITS[j]), color=colors[j], linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1].set_ylabel("Velocity (deg/s)")
    axes[1].grid(True)

    for j in range(6):
        axes[2].plot(times, np.degrees(accelerations[:, j]), color=colors[j], label=JOINT_NAMES[j])
        axes[2].axhline(math.degrees(ACCELERATION_LIMITS[j]),  color=colors[j], linestyle="--", linewidth=0.8, alpha=0.5)
        axes[2].axhline(-math.degrees(ACCELERATION_LIMITS[j]), color=colors[j], linestyle="--", linewidth=0.8, alpha=0.5)
    axes[2].set_ylabel("Acceleration (deg/s²)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True)

    fig.suptitle("UR5e Trajectory Validation", fontsize=13)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Validate CSV trajectory for UR5e")
    parser.add_argument("csv_file", help="Path to joint trajectory CSV file")
    parser.add_argument("--plot", action="store_true", help="Plot position/velocity/acceleration")
    args = parser.parse_args()

    print(f"\nValidating: {args.csv_file}")
    times, positions = load_csv(args.csv_file)

    velocities, accelerations = finite_diff(times, positions)[0], finite_diff(times, positions)[1]

    all_errors = []
    all_errors += check_timestep(times)
    all_errors += check_continuity(times, positions)
    all_errors += check_position_limits(times, positions)
    vel_errors, velocities = check_velocity_limits(times, positions)
    all_errors += vel_errors
    acc_errors, accelerations = check_acceleration_limits(times, positions)
    all_errors += acc_errors

    print_summary(times, positions, velocities, accelerations)

    print("\n========== Validation Results ==========")
    if not all_errors:
        print("  [OK] No violations found. Trajectory looks safe for UR5e.")
    else:
        print(f"  [WARN] {len(all_errors)} violation(s) found:\n")
        for e in all_errors:
            print(e)
        print(f"\n  -> Fix the above before running on real hardware.")

    if args.plot:
        plot_trajectory(times, positions, velocities, accelerations)

    return 0 if not all_errors else 1


if __name__ == "__main__":
    sys.exit(main())
