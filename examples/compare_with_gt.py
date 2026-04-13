#!/usr/bin/env python3
"""
Script to compare CORA solution with ground truth using evo
"""

import os
import sys
import argparse
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

def main():
    parser = argparse.ArgumentParser(description='Compare CORA solution with ground truth')
    parser.add_argument('pyfg_file', help='Path to the .pyfg data file')
    parser.add_argument('--cora_solution', help='Path to CORA solution .tum or .g2o file')
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
        if not os.path.exists(args.cora_solution):
            print(f"Error: {args.cora_solution} does not exist")
            sys.exit(1)

        print(f"Comparing {args.cora_solution} with ground truth...")

        # Use evo_traj to save plots with alignment
        import subprocess
        plot_file = os.path.join(args.output_dir, 'trajectory_overlay.png')
        # Align trajectories using Umeyama method so they visually overlap
        cmd = ['evo_traj', 'tum', args.cora_solution] + gt_files + ['-a', '--ref', gt_files[0], '--save_plot', plot_file, '--plot_mode', 'xy']
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        
        # Also compute APE
        ape_cmd = ['evo_ape', 'tum', gt_files[0], args.cora_solution, '-a']
        print(f"Computing APE: {' '.join(ape_cmd)}")
        subprocess.run(ape_cmd)
    else:
        print("Ground truth trajectories exported. To compare with CORA solution, run:")
        print(f"evo_traj tum /path/to/cora_solution.tum {' '.join([f'{args.output_dir}/' + os.path.basename(f) for f in gt_files])} --save_plot comparison.png")

if __name__ == "__main__":
    main()