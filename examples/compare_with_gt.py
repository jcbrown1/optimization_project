#!/usr/bin/env python3
"""
Script to compare CORA solution with ground truth using evo
"""

import os
import sys
import argparse
import math
import matplotlib.pyplot as plt
from py_factor_graph.io.pyfg_text import read_from_pyfg_text

def export_ground_truth_trajectory(pyfg_path, output_dir):
    """Export ground truth trajectory from .pyfg file to .tum format"""
    print(f"Loading {pyfg_path}...")
    fg = read_from_pyfg_text(pyfg_path)

    os.makedirs(output_dir, exist_ok=True)
    # Create output directory if it doesn't exist

    # Export ground truth trajectories with pose indices as timestamps (matching CORA)
    gt_files = []
    for i, pose_chain in enumerate(fg.pose_variables):
        filename = f"gt_traj_{chr(ord('A') + i)}.tum"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as fw:
            for pose_idx, pose in enumerate(pose_chain):
                # Use pose index as timestamp to match CORA's timestamp scheme
                timestamp = pose_idx
                qx, qy, qz, qw = pose.true_quat
                fw.write(
                    f"{timestamp} "
                    f"{pose.true_x} {pose.true_y} {pose.true_z} "
                    f"{qx} {qy} {qz} {qw}\n"
                )
        gt_files.append(filepath)
        print(f"Exported ground truth trajectory: {filepath}")

    return gt_files


def _load_tum_xy(tum_path):
    xs = []
    ys = []
    with open(tum_path, "r", encoding="utf-8") as fr:
        for line in fr:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            xs.append(float(parts[1]))
            ys.append(float(parts[2]))
    return xs, ys


def _load_tum_poses(tum_path):
    """Load (x,y,z,qx,qy,qz,qw) tuples from a TUM trajectory file."""
    poses = []
    with open(tum_path, "r", encoding="utf-8") as fr:
        for line in fr:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = (
                float(parts[4]),
                float(parts[5]),
                float(parts[6]),
                float(parts[7]),
            )
            poses.append((x, y, z, qx, qy, qz, qw))
    return poses


def _align_xy_similarity(src_x, src_y, ref_x, ref_y):
    """Align src trajectory to ref via 2D similarity transform."""
    n = min(len(src_x), len(src_y), len(ref_x), len(ref_y))
    if n == 0:
        return src_x, src_y, 0.0, 0.0, 0.0, 1.0

    sx = src_x[:n]
    sy = src_y[:n]
    rx = ref_x[:n]
    ry = ref_y[:n]

    c_sx = sum(sx) / n
    c_sy = sum(sy) / n
    c_rx = sum(rx) / n
    c_ry = sum(ry) / n

    a = 0.0
    b = 0.0
    src_var = 0.0
    for i in range(n):
        qsx = sx[i] - c_sx
        qsy = sy[i] - c_sy
        qrx = rx[i] - c_rx
        qry = ry[i] - c_ry
        a += qsx * qrx + qsy * qry
        b += qsx * qry - qsy * qrx
        src_var += qsx * qsx + qsy * qsy

    theta = math.atan2(b, a)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    if src_var > 1e-12:
        scale = math.sqrt(a * a + b * b) / src_var
    else:
        scale = 1.0

    tx = c_rx - scale * (cos_t * c_sx - sin_t * c_sy)
    ty = c_ry - scale * (sin_t * c_sx + cos_t * c_sy)

    aligned_x = []
    aligned_y = []
    for x, y in zip(src_x, src_y):
        aligned_x.append(scale * (cos_t * x - sin_t * y) + tx)
        aligned_y.append(scale * (sin_t * x + cos_t * y) + ty)
    return aligned_x, aligned_y, theta, tx, ty, scale


def _quat_angle_error_deg(q1, q2):
    """Smallest angular distance between quaternions in degrees."""
    dot = abs(
        q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]
    )
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def _quat_multiply(q1, q2):
    """Quaternion multiply q = q1 * q2 for (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _yaw_quaternion(theta):
    """Quaternion for rotation about +Z by theta radians."""
    half = 0.5 * theta
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _compute_robot_errors(gt_tum, sol_tum):
    """
    Compute per-robot errors after 2D global alignment.

    Returns:
      mean_translation_error_m: mean Euclidean position error [m]
      mean_rotation_error_deg: mean quaternion angular error [deg]
      global_alignment_rotation_deg: global alignment yaw [deg]
      global_alignment_translation_m: global alignment translation magnitude [m]
    """
    gt_xy_x, gt_xy_y = _load_tum_xy(gt_tum)
    sol_xy_x, sol_xy_y = _load_tum_xy(sol_tum)
    aligned_x, aligned_y, theta, tx, ty, _ = _align_xy_similarity(
        sol_xy_x, sol_xy_y, gt_xy_x, gt_xy_y
    )

    n_xy = min(len(aligned_x), len(aligned_y), len(gt_xy_x), len(gt_xy_y))
    if n_xy == 0:
        return 0.0, 0.0, 0.0, 0.0

    trans_errors = []
    for i in range(n_xy):
        dx = aligned_x[i] - gt_xy_x[i]
        dy = aligned_y[i] - gt_xy_y[i]
        trans_errors.append(math.sqrt(dx * dx + dy * dy))
    mean_translation_error_m = sum(trans_errors) / len(trans_errors)

    gt_poses = _load_tum_poses(gt_tum)
    sol_poses = _load_tum_poses(sol_tum)
    n_rot = min(len(gt_poses), len(sol_poses))
    rot_errors = []
    q_align = _yaw_quaternion(theta)
    for i in range(n_rot):
        gt_q = gt_poses[i][3:]
        sol_q = sol_poses[i][3:]
        # Apply the same global XY alignment yaw to solver orientation before
        # comparing orientation error, so frame offset is not counted as error.
        sol_q_aligned = _quat_multiply(q_align, sol_q)
        rot_errors.append(_quat_angle_error_deg(sol_q_aligned, gt_q))
    mean_rotation_error_deg = (
        sum(rot_errors) / len(rot_errors) if len(rot_errors) > 0 else 0.0
    )

    global_alignment_rotation_deg = abs(math.degrees(theta))
    global_alignment_translation_m = math.sqrt(tx * tx + ty * ty)
    return (
        mean_translation_error_m,
        mean_rotation_error_deg,
        global_alignment_rotation_deg,
        global_alignment_translation_m,
    )


def _compute_optimal_solution_loss_placeholder(
    avg_rotation_error_deg, avg_translation_error_m, optimal_solution=None
):
    """
    Placeholder loss function based on an "optimal solution".

    Replace this with your true objective once available:
      - If you have an optimal factor-graph objective value, compare this run's
        objective to that optimum.
      - If you have per-robot optimal trajectories, compute trajectory-space
        discrepancy to those trajectories.
    """
    # Placeholder proxy loss:
    # This combines average translation and average rotation with hand-tuned
    # weights. Update these weights (or the entire formula) for your task.
    w_trans = 1.0
    w_rot = 0.01
    proxy_loss = w_trans * avg_translation_error_m + w_rot * avg_rotation_error_deg

    # TODO: Replace with a real optimality-gap loss, e.g.
    # loss = (current_objective - optimal_objective) / max(optimal_objective, eps)
    _ = optimal_solution  # placeholder for future use
    return proxy_loss


def _report_evaluation_metrics(gt_files, cora_solution_files):
    """Report per-robot and averaged translation/rotation metrics."""
    n = min(len(gt_files), len(cora_solution_files))
    if n == 0:
        print("No trajectories provided for evaluation metrics.")
        return

    translation_errors = []
    rotation_errors = []
    align_rot_errors = []
    align_trans_errors = []
    for i in range(n):
        (
            mean_t_err,
            mean_r_err,
            align_r_err,
            align_t_err,
        ) = _compute_robot_errors(gt_files[i], cora_solution_files[i])
        translation_errors.append(mean_t_err)
        rotation_errors.append(mean_r_err)
        align_rot_errors.append(align_r_err)
        align_trans_errors.append(align_t_err)
        print(
            f"Robot {i}: mean translation error={mean_t_err:.4f} m, "
            f"mean rotation error={mean_r_err:.4f} deg, "
            f"global align rot={align_r_err:.4f} deg, "
            f"global align trans={align_t_err:.4f} m"
        )

    avg_translation_error_m = sum(translation_errors) / len(translation_errors)
    avg_rotation_error_deg = sum(rotation_errors) / len(rotation_errors)
    avg_global_align_rot_deg = sum(align_rot_errors) / len(align_rot_errors)
    avg_global_align_trans_m = sum(align_trans_errors) / len(align_trans_errors)

    print(
        "Average errors across all robots: "
        f"translation={avg_translation_error_m:.6f} m, "
        f"rotation={avg_rotation_error_deg:.6f} deg"
    )
    print(
        "Average global alignment transform across all robots: "
        f"rotation={avg_global_align_rot_deg:.6f} deg, "
        f"translation={avg_global_align_trans_m:.6f} m"
    )

    optimal_loss = _compute_optimal_solution_loss_placeholder(
        avg_rotation_error_deg=avg_rotation_error_deg,
        avg_translation_error_m=avg_translation_error_m,
        optimal_solution=None,
    )
    print(f"Optimal-solution loss (placeholder) = {optimal_loss:.6f}")


def _run_evo_translation_and_rotation_ape(gt_tum, sol_tum):
    """Run evo_ape for translation and angular error (deg)."""
    import subprocess

    trans_cmd = ['evo_ape', 'tum', gt_tum, sol_tum, '-a']
    print(f"Computing APE translation: {' '.join(trans_cmd)}")
    subprocess.run(trans_cmd)

    rot_cmd = [
        'evo_ape',
        'tum',
        gt_tum,
        sol_tum,
        '-a',
        '--pose_relation',
        'angle_deg',
    ]
    print(f"Computing APE rotation (deg): {' '.join(rot_cmd)}")
    subprocess.run(rot_cmd)


def _plot_square_trajectory_overlays(gt_files, cora_solution_files, output_dir):
    """Plot each robot in its own subplot with global alignment."""
    n = min(len(gt_files), len(cora_solution_files))
    if n == 0:
        print("No trajectories provided for square overlay plot.")
        return

    grid_cols = math.ceil(math.sqrt(n))
    grid_rows = math.ceil(n / grid_cols)
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(6 * grid_cols, 5 * grid_rows))
    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i in range(n):
        ax = axes[i]
        gt_x, gt_y = _load_tum_xy(gt_files[i])
        sol_x, sol_y = _load_tum_xy(cora_solution_files[i])
        aligned_sol_x, aligned_sol_y, _, _, _, _ = _align_xy_similarity(
            sol_x, sol_y, gt_x, gt_y
        )
        ax.plot(gt_x, gt_y, label=f"GT robot {i}", linewidth=2)
        ax.plot(aligned_sol_x, aligned_sol_y, label=f"Solver robot {i} (aligned)", linewidth=1.5)
        ax.set_title(f"Robot {i}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend(loc="best", fontsize=8)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    square_plot_file = os.path.join(output_dir, "trajectory_overlay_square.png")
    fig.savefig(square_plot_file, dpi=200)
    print(f"Saved square trajectory overlay plot: {square_plot_file}")
    _report_evaluation_metrics(gt_files[:n], cora_solution_files[:n])
    for i in range(n):
        _run_evo_translation_and_rotation_ape(gt_files[i], cora_solution_files[i])

def main():
    parser = argparse.ArgumentParser(description='Compare CORA solution with ground truth')
    parser.add_argument('pyfg_file', help='Path to the .pyfg data file')
    parser.add_argument(
        '--cora_solution',
        nargs='+',
        help='Path(s) to CORA solution .tum file(s). For multi-robot, pass one .tum per robot.',
    )
    parser.add_argument('--output_dir', default='outputs', help='Output directory for ground truth files')

    args = parser.parse_args()

    if not os.path.exists(args.pyfg_file):
        print(f"Error: {args.pyfg_file} does not exist")
        sys.exit(1)

    # Export ground truth
    gt_files = export_ground_truth_trajectory(args.pyfg_file, args.output_dir)

    if not gt_files:
        print("No ground truth trajectories found in the .pyfg file")
        sys.exit(1)

    # If CORA solution is provided, compare using evo
    if args.cora_solution:
        for sol_path in args.cora_solution:
            if not os.path.exists(sol_path):
                print(f"Error: {sol_path} does not exist")
                sys.exit(1)

        # Multi-file mode: plot each robot separately in a square layout.
        if len(args.cora_solution) > 1:
            if len(args.cora_solution) != len(gt_files):
                print(
                    "Error: number of --cora_solution files must match number of robots "
                    f"({len(gt_files)} GT trajectories, {len(args.cora_solution)} provided)."
                )
                sys.exit(1)
            print(
                f"Comparing {len(args.cora_solution)} solution trajectories with "
                f"{len(gt_files)} GT trajectories..."
            )
            _plot_square_trajectory_overlays(gt_files, args.cora_solution, args.output_dir)
            return

        # Backward-compatible single-file mode.
        cora_solution = args.cora_solution[0]
        print(f"Comparing {cora_solution} with ground truth...")

        # Use evo_traj to save plots with alignment
        import subprocess
        plot_file = os.path.join(args.output_dir, 'trajectory_overlay.png')
        # Align trajectories using Umeyama method so they visually overlap
        cmd = ['evo_traj', 'tum', cora_solution] + gt_files + ['-a', '--ref', gt_files[0], '--save_plot', plot_file, '--plot_mode', 'xy']
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        
        # Also compute APE metrics (translation and rotation)
        _run_evo_translation_and_rotation_ape(gt_files[0], cora_solution)
        _report_evaluation_metrics([gt_files[0]], [cora_solution])
    else:
        print("Ground truth trajectories exported. To compare with CORA solution, run:")
        print(f"evo_traj tum /path/to/cora_solution.tum {' '.join([f'{args.output_dir}/' + os.path.basename(f) for f in gt_files])} --save_plot comparison.png")

if __name__ == "__main__":
    main()