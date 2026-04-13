"""
A file to visualize the different experimental data sets in the examples folder.

NOTE: the odometry visualization/calibration is unrefined and may not work well
for all data sets. It is likely a 80% solution that may need some more work if
you want to really inspect your odometry data/calibration.
"""

from py_factor_graph.io.pyfg_text import read_from_pyfg_text
from py_factor_graph.calibrations.odom_measurement_calibration import (
    calibrate_odom_measures,
)
import matplotlib.pyplot as plt
import os
import math


def _get_pyfg_files_in_dir(dir_path):
    # search in all subdirectories as well
    pyfg_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".pyfg"):
                pyfg_files.append(os.path.join(root, file))
    return pyfg_files


def _select_from_list_based_on_requested_input(list_of_items, message):
    print(message)
    for i, item in enumerate(list_of_items):
        print(f"{i+1}: {item}")
    selected_index = int(input("Enter the index of the item you want to select: "))
    return list_of_items[selected_index - 1]


def _visualize_dataset(
    fg, show_gt, show_range_lines, show_range_circles, num_timesteps_keep_ranges
):
    dim = fg.dimension
    if dim == 2:
        fg.animate_odometry(
            show_gt=show_gt,
            draw_range_lines=show_range_lines,
            draw_range_circles=show_range_circles,
            num_timesteps_keep_ranges=num_timesteps_keep_ranges,
        )
    elif dim == 3:
        fg.animate_odometry_3d(
            show_gt=show_gt,
            draw_range_lines=show_range_lines,
            num_timesteps_keep_ranges=num_timesteps_keep_ranges,
        )
    else:
        raise ValueError(
            f"Only 2D and 3D data sets are supported. Received {dim}D data set."
        )


def _visualize_range_errors(fg):
    variable_positions = fg.variable_true_positions_dict
    assoc_to_true = {}
    assoc_to_measured = {}

    for measure in fg.range_measurements:
        key_a, key_b = measure.association
        if key_a not in variable_positions or key_b not in variable_positions:
            continue

        pos_a = variable_positions[key_a]
        pos_b = variable_positions[key_b]
        true_dist = math.dist(pos_a, pos_b)
        measured_dist = float(measure.dist)

        assoc = (key_a, key_b)
        if assoc not in assoc_to_true:
            assoc_to_true[assoc] = []
            assoc_to_measured[assoc] = []
        assoc_to_true[assoc].append(true_dist)
        assoc_to_measured[assoc].append(measured_dist)

    if len(assoc_to_true) == 0:
        print("No valid range measurements found to plot.")
        return

    all_true = [d for values in assoc_to_true.values() for d in values]
    all_measured = [d for values in assoc_to_measured.values() for d in values]
    all_residuals = [m - t for m, t in zip(all_measured, all_true)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    scatter_ax, residual_ax = axes

    for assoc, true_vals in assoc_to_true.items():
        measured_vals = assoc_to_measured[assoc]
        label = f"{assoc[0]}-{assoc[1]}"
        scatter_ax.scatter(true_vals, measured_vals, s=16, alpha=0.7, label=label)

    min_val = min(min(all_true), min(all_measured))
    max_val = max(max(all_true), max(all_measured))
    scatter_ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)
    scatter_ax.set_title("Range Measurements vs Ground Truth")
    scatter_ax.set_xlabel("Ground truth range")
    scatter_ax.set_ylabel("Measured range")
    if len(assoc_to_true) <= 12:
        scatter_ax.legend(loc="best", fontsize=8)
    scatter_ax.grid(True, alpha=0.25)

    residual_ax.scatter(all_true, all_residuals, s=12, alpha=0.6)
    residual_ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    residual_ax.set_title("Range Residuals")
    residual_ax.set_xlabel("Ground truth range")
    residual_ax.set_ylabel("Measured - ground truth")
    residual_ax.grid(True, alpha=0.25)

    fig.tight_layout()
    plt.show(block=True)


def _visualize_relative_pose_errors(fg):
    calibrate_odom_measures(fg)

if __name__ == "__main__":
    SHOW_GT = True
    SHOW_RANGE_LINES = True
    SHOW_RANGE_CIRCLES = False
    NUM_TIMESTEPS_KEEP_RANGES = 10

    # get the directory of the current file
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # get all the pyfg files inside the directory
    pyfg_files = _get_pyfg_files_in_dir(dir_path)
    pyfg_fnames = [os.path.basename(f) for f in pyfg_files]

    keep_looping = True
    loop_options = ["Visualize a data set", "Visualize measurement errors", "Exit"]

    # read the pyfg files
    while keep_looping:
        loop_choice = _select_from_list_based_on_requested_input(
            loop_options, "What would you like to do?"
        )
        if loop_choice == "Exit":
            keep_looping = False
            continue
        elif loop_choice == "Visualize a data set":
            pyfg_fname = _select_from_list_based_on_requested_input(
                pyfg_fnames, "Select a data set to visualize:"
            )
            if "mrclam" in pyfg_fname:
                print(
                    "\nWARNING:\n\nthe visualization isn't configured to handle the time-sync for the MRCLAM data set.\n"
                    "For example, the range measurements may not be in sync with the odometry.\n"
                    "This is only a visualization issue, the factor graph is still correct.\n\n"
                )
            pyfg_file = pyfg_files[pyfg_fnames.index(pyfg_fname)]
            fg = read_from_pyfg_text(pyfg_file)
            _visualize_dataset(
                fg,
                SHOW_GT,
                SHOW_RANGE_LINES,
                SHOW_RANGE_CIRCLES,
                NUM_TIMESTEPS_KEEP_RANGES,
            )
        elif loop_choice == "Visualize measurement errors":
            pyfg_fname = _select_from_list_based_on_requested_input(
                pyfg_fnames, "Select a data set to visualize:"
            )
            pyfg_file = pyfg_files[pyfg_fnames.index(pyfg_fname)]
            fg = read_from_pyfg_text(pyfg_file)
            _visualize_range_errors(fg)
            # _visualize_relative_pose_errors(fg)
        else:
            raise ValueError(f"Invalid loop choice. Received {loop_choice}.")
