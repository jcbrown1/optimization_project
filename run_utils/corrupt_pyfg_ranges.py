#!/usr/bin/env python3
"""
Corrupt or replace EDGE_RANGE measurements in a .pyfg file.

This script preserves all non-range lines and edits EDGE_RANGE values with:
  - uniform: corrupt selected measurements with U(lower, upper)
  - gamma: corrupt selected measurements with Gamma(k, theta)
  - student_t_replace: replace all measurements with r_true + Student-t noise

Ground-truth support:
  - Optional zero-error export writes EDGE_RANGE = geometric range from input vertices.
  - Student-t replacement uses input vertices as ground truth.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


Point = Tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrupt or replace EDGE_RANGE measurements in a .pyfg file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input .pyfg file path.",
    )
    parser.add_argument(
        "--method",
        choices=("uniform", "gamma", "student_t_replace"),
        required=True,
        help=(
            "uniform: U(lower,upper) for selected measurements; "
            "gamma: Gamma(k,theta) for selected measurements; "
            "student_t_replace: replace all ranges using GT + Student-t noise."
        ),
    )
    parser.add_argument(
        "--probability",
        type=float,
        default=1.0,
        help=(
            "Probability (0..1) that an EDGE_RANGE measurement is edited. "
            "Used for uniform/gamma; ignored by student_t_replace."
        ),
    )
    parser.add_argument(
        "--k",
        type=float,
        default=None,
        help="Gamma shape parameter k (required when --method gamma).",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=None,
        help="Gamma scale parameter theta (required when --method gamma).",
    )
    parser.add_argument(
        "--student-dof",
        type=float,
        default=None,
        help="Student-t degrees of freedom (required for student_t_replace).",
    )
    parser.add_argument(
        "--student-scale",
        type=float,
        default=None,
        help="Student-t scale (required for student_t_replace).",
    )
    parser.add_argument(
        "--student-loc",
        type=float,
        default=0.0,
        help="Student-t location (default 0.0).",
    )
    parser.add_argument(
        "--write-zero-error",
        action="store_true",
        help=(
            "Also write a *_zero_error.pyfg file where each EDGE_RANGE equals "
            "the geometric range from input vertices."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional random seed for reproducibility.",
    )
    return parser.parse_args()


def _parse_vertex_point(tokens: Sequence[str]) -> Optional[Tuple[str, Point]]:
    rec_type = tokens[0]

    if rec_type == "VERTEX_SE2" and len(tokens) >= 6:
        return tokens[2], (float(tokens[3]), float(tokens[4]))
    if rec_type == "VERTEX_SE3:QUAT" and len(tokens) >= 7:
        return tokens[2], (float(tokens[3]), float(tokens[4]), float(tokens[5]))
    if rec_type == "VERTEX_XY" and len(tokens) >= 4:
        return tokens[1], (float(tokens[2]), float(tokens[3]))
    if rec_type == "VERTEX_XYZ" and len(tokens) >= 5:
        return tokens[1], (float(tokens[2]), float(tokens[3]), float(tokens[4]))
    return None


def collect_vertex_points(lines: Sequence[str]) -> Dict[str, Point]:
    points: Dict[str, Point] = {}
    for line in lines:
        tokens = line.strip().split()
        if not tokens:
            continue
        parsed = _parse_vertex_point(tokens)
        if parsed is None:
            continue
        sym, point = parsed
        points[sym] = point
    return points


def compute_uniform_bounds_from_true_ranges(
    true_ranges_by_line: Optional[Dict[int, float]], measured_ranges: Sequence[float]
) -> Tuple[float, float]:
    if true_ranges_by_line and len(true_ranges_by_line) > 0:
        true_vals = list(true_ranges_by_line.values())
        lower = min(true_vals)
        upper = max(true_vals)
    elif measured_ranges:
        lower = min(measured_ranges)
        upper = max(measured_ranges)
    else:
        lower, upper = 0.0, 1.0

    if upper <= lower:
        upper = lower + 1e-6
    return lower, upper


def _euclidean_distance(a: Point, b: Point) -> float:
    dim = min(len(a), len(b))
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(dim)))


def sample_gamma_outlier(k: float, theta: float) -> float:
    g = random.gammavariate(k, theta)
    return g - k * theta  # center to zero mean


def sample_student_t(dof: float, loc: float, scale: float) -> float:
    z = random.gauss(0.0, 1.0)
    chi_sq = random.gammavariate(dof / 2.0, 2.0)
    t = z / math.sqrt(chi_sq / dof)
    return loc + scale * t


def compute_true_ranges_by_line(
    lines: Sequence[str], gt_points: Dict[str, Point]
) -> Dict[int, float]:
    true_ranges: Dict[int, float] = {}
    for idx, line in enumerate(lines):
        tokens = line.strip().split()
        if not tokens or tokens[0] != "EDGE_RANGE" or len(tokens) < 6:
            continue
        sym_a, sym_b = tokens[2], tokens[3]
        if sym_a not in gt_points or sym_b not in gt_points:
            continue
        true_ranges[idx] = _euclidean_distance(gt_points[sym_a], gt_points[sym_b])
    return true_ranges


def rewrite_with_true_ranges(
    lines: Sequence[str], true_ranges: Dict[int, float]
) -> Tuple[List[str], int, int]:
    output_lines: List[str] = []
    total_ranges = 0
    rewritten = 0

    for idx, line in enumerate(lines):
        raw = line.rstrip("\n")
        tokens = raw.split()
        if not tokens or tokens[0] != "EDGE_RANGE" or len(tokens) < 6:
            output_lines.append(raw)
            continue

        total_ranges += 1
        if idx not in true_ranges:
            output_lines.append(raw)
            continue

        timestamp, sym_a, sym_b = tokens[1], tokens[2], tokens[3]
        covariance = tokens[5]
        new_range = max(0.0, true_ranges[idx])
        output_lines.append(
            f"EDGE_RANGE {timestamp} {sym_a} {sym_b} {new_range:.9f} {covariance}"
        )
        rewritten += 1

    return output_lines, total_ranges, rewritten


def transform_ranges(
    lines: Sequence[str],
    probability: float,
    method: str,
    k: Optional[float],
    theta: Optional[float],
    true_ranges_by_line: Optional[Dict[int, float]],
    student_dof: Optional[float],
    student_scale: Optional[float],
    student_loc: float,
) -> Tuple[List[str], int, int]:
    range_values: List[float] = []
    for line in lines:
        tokens = line.strip().split()
        if tokens and tokens[0] == "EDGE_RANGE" and len(tokens) >= 6:
            range_values.append(float(tokens[4]))

    uniform_lower: Optional[float] = None
    uniform_upper: Optional[float] = None
    if method == "uniform":
        uniform_lower, uniform_upper = compute_uniform_bounds_from_true_ranges(
            true_ranges_by_line, range_values
        )

    replace_all = method == "student_t_replace"
    output_lines: List[str] = []
    total_ranges = 0
    changed_ranges = 0

    for idx, line in enumerate(lines):
        raw = line.rstrip("\n")
        tokens = raw.split()
        if not tokens or tokens[0] != "EDGE_RANGE" or len(tokens) < 6:
            output_lines.append(raw)
            continue

        total_ranges += 1
        if (not replace_all) and random.random() >= probability:
            output_lines.append(raw)
            continue

        timestamp, sym_a, sym_b = tokens[1], tokens[2], tokens[3]
        original_range = float(tokens[4])
        covariance = tokens[5]

        if method == "uniform":
            assert uniform_lower is not None and uniform_upper is not None
            new_range = random.uniform(uniform_lower, uniform_upper)
        elif method == "gamma":
            assert k is not None and theta is not None
            # Gamma samples the corruption error, not the full measurement.
            gamma_error = sample_gamma_outlier(k, theta)
            if true_ranges_by_line is not None and idx in true_ranges_by_line:
                baseline = true_ranges_by_line[idx]
            else:
                baseline = original_range
            new_range = max(0.0, baseline + gamma_error)
        else:
            assert method == "student_t_replace"
            assert true_ranges_by_line is not None
            assert student_dof is not None and student_scale is not None
            if idx not in true_ranges_by_line:
                output_lines.append(raw)
                continue
            true_range = true_ranges_by_line[idx]
            noise = sample_student_t(student_dof, student_loc, student_scale)
            new_range = max(0.0, true_range + noise)

        output_lines.append(
            f"EDGE_RANGE {timestamp} {sym_a} {sym_b} {new_range:.9f} {covariance}"
        )
        changed_ranges += 1

    return output_lines, total_ranges, changed_ranges


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.probability <= 1.0:
        raise ValueError("--probability must be in [0, 1].")

    if args.method == "gamma":
        if args.k is None or args.theta is None:
            raise ValueError("--k and --theta are required when --method gamma.")
        if args.k <= 0.0:
            raise ValueError("--k must be > 0 for gamma distribution.")
        if args.theta <= 0.0:
            raise ValueError("--theta must be > 0 for gamma distribution.")

    if args.method == "student_t_replace":
        if args.student_dof is None or args.student_scale is None:
            raise ValueError(
                "--student-dof and --student-scale are required for --method student_t_replace."
            )
        if args.student_dof <= 0.0:
            raise ValueError("--student-dof must be > 0.")
        if args.student_scale <= 0.0:
            raise ValueError("--student-scale must be > 0.")


def _method_suffix(args: argparse.Namespace) -> str:
    if args.method == "gamma":
        return f"gamma_k{args.k:g}_theta{args.theta:g}"
    if args.method == "student_t_replace":
        return (
            f"student_t_replace_dof{args.student_dof:g}"
            f"_scale{args.student_scale:g}_loc{args.student_loc:g}"
        )
    return args.method


def main() -> None:
    args = parse_args()
    _validate_args(args)

    if args.seed is not None:
        random.seed(args.seed)

    in_path = Path(args.input)
    if in_path.suffix != ".pyfg":
        raise ValueError("Input file must have .pyfg extension.")
    if not in_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {in_path}")

    out_path_root = in_path.parent
    out_path = out_path_root / in_path.name.replace(
        ".pyfg", f"_corrupted_{args.probability:.4f}_{_method_suffix(args)}.pyfg"
    )

    lines = in_path.read_text(encoding="utf-8").splitlines()

    gt_points = collect_vertex_points(lines)
    true_ranges_by_line: Optional[Dict[int, float]] = compute_true_ranges_by_line(
        lines, gt_points
    )

    if args.write_zero_error:
        zero_lines, zero_total, zero_written = rewrite_with_true_ranges(
            lines, true_ranges_by_line
        )
        zero_path = out_path_root / in_path.name.replace(".pyfg", "_zero_error.pyfg")
        zero_path.write_text("\n".join(zero_lines) + "\n", encoding="utf-8")
        print(
            f"Wrote {zero_path} with {zero_written}/{zero_total} EDGE_RANGE "
            "measurements replaced by geometric ranges from input vertices."
        )

    transformed_lines, total, changed = transform_ranges(
        lines=lines,
        probability=args.probability,
        method=args.method,
        k=args.k,
        theta=args.theta,
        true_ranges_by_line=true_ranges_by_line,
        student_dof=args.student_dof,
        student_scale=args.student_scale,
        student_loc=args.student_loc,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(transformed_lines) + "\n", encoding="utf-8")

    operation = "replaced" if args.method == "student_t_replace" else "corrupted"
    print(
        f"Wrote {out_path} with {changed}/{total} EDGE_RANGE measurements {operation} "
        f"(method={args.method}, probability={args.probability})."
    )


if __name__ == "__main__":
    main()
