#!/usr/bin/env python3
"""
Corrupt EDGE_RANGE measurements in a .pyfg file.

This script preserves all non-range lines and only edits range values in
EDGE_RANGE entries according to:
  1) A per-measurement corruption probability
  2) A selected outlier model:
     - uniform: sample a new range inside problem-scale bounds only (from vertex layout + range stats)
     - realistic: heavy-tailed bias vs robust scale sigma, loose sigma-based clip (not the uniform box)
     - exponential: occlusion — false range z < r_true from a truncated exponential on [0, r_true]
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple


Point = Tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrupt EDGE_RANGE measurements in a .pyfg file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input .pyfg file path.",
    )
    parser.add_argument(
        "--probability",
        type=float,
        required=True,
        help="Probability (0..1) that an EDGE_RANGE measurement is corrupted.",
    )
    parser.add_argument(
        "--method",
        choices=("uniform", "realistic", "exponential"),
        required=True,
        help=(
            "uniform: U(lower,upper) from problem-scale bounds only; "
            "realistic: heavy-tailed NLOS bias with sigma-based clip (not the uniform box); "
            "exponential: occlusion — truncated Exp on [0,r_true] (measured range < true range)."
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


def _robust_scale(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    med = median(values)
    deviations = [abs(v - med) for v in values]
    mad = median(deviations)
    sigma = 1.4826 * mad
    if sigma <= 1e-12:
        sigma = max(med * 0.05, 0.5)
    return sigma


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


def compute_uniform_bounds(
    lines: Sequence[str], ranges: Sequence[float]
) -> Tuple[float, float]:
    max_range = max(ranges) if ranges else 1.0
    min_range = min(ranges) if ranges else 0.1

    points = collect_vertex_points(lines)

    if points:
        dims = max(len(p) for p in points.values())
        mins = [
            min(p[i] for p in points.values() if i < len(p))
            for i in range(dims)
        ]
        maxs = [
            max(p[i] for p in points.values() if i < len(p))
            for i in range(dims)
        ]
        diag = math.sqrt(sum((maxs[i] - mins[i]) ** 2 for i in range(len(mins))))
        problem_scale = max(diag, max_range, 1.0)
    else:
        problem_scale = max(max_range, 1.0)

    lower = max(0.0, min(min_range * 0.25, problem_scale * 0.05))
    upper = max(max_range * 1.25, problem_scale * 1.25, lower + 1e-6)
    return lower, upper


def _euclidean_distance(a: Point, b: Point) -> float:
    dim = min(len(a), len(b))
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(dim)))


def sample_truncated_exponential_below_true(r_true: float, lam: float) -> float:
    """Sample z from the truncated distribution p(z) ∝ λ exp(−λ z) on z ∈ [0, r_true].

    Models an early return from an unexpected obstacle: the reported range z is
    always at most the geometric true range r_true, with smaller z more likely
    when λ is larger (closer occlusions more probable).
    """
    r_true = max(float(r_true), 1e-12)
    if lam <= 0.0:
        lam = 1.0 / r_true
    u = random.random()
    u = min(max(u, 1e-15), 1.0 - 1e-15)
    exp_m = math.exp(-lam * r_true)
    z = -math.log(1.0 - u * (1.0 - exp_m)) / lam
    return min(z, r_true)


def generate_uniform_outlier(lower: float, upper: float) -> float:
    return random.uniform(lower, upper)


def generate_realistic_outlier(original: float, sigma: float) -> float:
    """Heavy-tailed NLOS-style bias; clipped only by a loose sigma-based band."""
    floor = max(1e-9, original * 0.01)
    upper = original + 30.0 * max(sigma, original * 0.02, 0.1)
    tail = abs(random.gauss(0.0, 1.0) / math.sqrt(random.uniform(1e-6, 1.0)))
    bias = sigma * (2.0 + tail)
    if random.random() < 0.8:
        corrupted = original + bias
    else:
        corrupted = original - 0.5 * bias
    return min(max(corrupted, floor), upper)


def generate_exponential_occlusion(r_true: float, sigma: float) -> float:
    """Occlusion: measured range z < r_true with p(z) ∝ λ exp(−λ z) on [0, r_true].

    Rate λ is set from the robust range scale σ so closer false returns are more
    likely than far ones (λ larger → typical z smaller).
    """
    r_true = max(float(r_true), 1e-12)
    mean_free = max(sigma, 0.05 * r_true, 0.5)
    lam = 1.0 / mean_free
    return sample_truncated_exponential_below_true(r_true, lam)


def corrupt_ranges(
    lines: Sequence[str],
    probability: float,
    method: str,
) -> Tuple[List[str], int, int]:
    range_values: List[float] = []
    for line in lines:
        tokens = line.strip().split()
        if tokens and tokens[0] == "EDGE_RANGE" and len(tokens) >= 6:
            range_values.append(float(tokens[4]))

    sigma = _robust_scale(range_values)
    uniform_lower: Optional[float] = None
    uniform_upper: Optional[float] = None
    if method == "uniform":
        uniform_lower, uniform_upper = compute_uniform_bounds(lines, range_values)

    vertex_points: Dict[str, Point] = {}
    if method == "exponential":
        vertex_points = collect_vertex_points(lines)

    output_lines: List[str] = []
    total_ranges = 0
    changed_ranges = 0

    for line in lines:
        raw = line.rstrip("\n")
        tokens = raw.split()
        if not tokens or tokens[0] != "EDGE_RANGE" or len(tokens) < 6:
            output_lines.append(raw)
            continue

        total_ranges += 1
        if random.random() >= probability:
            output_lines.append(raw)
            continue

        timestamp, sym_a, sym_b = tokens[1], tokens[2], tokens[3]
        original_range = float(tokens[4])
        covariance = tokens[5]

        if method == "uniform":
            assert uniform_lower is not None and uniform_upper is not None
            new_range = generate_uniform_outlier(uniform_lower, uniform_upper)
        elif method == "realistic":
            new_range = generate_realistic_outlier(original_range, sigma)
        else:
            assert method == "exponential"
            if sym_a in vertex_points and sym_b in vertex_points:
                r_true = _euclidean_distance(
                    vertex_points[sym_a], vertex_points[sym_b]
                )
            else:
                r_true = original_range
            new_range = generate_exponential_occlusion(r_true, sigma)

        new_line = (
            f"EDGE_RANGE {timestamp} {sym_a} {sym_b} "
            f"{new_range:.9f} {covariance}"
        )
        output_lines.append(new_line)
        changed_ranges += 1

    return output_lines, total_ranges, changed_ranges


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.probability <= 1.0:
        raise ValueError("--probability must be in [0, 1].")

    if args.seed is not None:
        random.seed(args.seed)

    in_path = Path(args.input)
    out_path_root = in_path.parent
    out_path = out_path_root / in_path.name.replace(".pyfg", f"_corrupted_{args.probability:.4f}_{args.method}.pyfg")

    if in_path.suffix != ".pyfg":
        raise ValueError("Input file must have .pyfg extension.")
    if not in_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {in_path}")

    lines = in_path.read_text(encoding="utf-8").splitlines()
    corrupted_lines, total, changed = corrupt_ranges(
        lines=lines, probability=args.probability, method=args.method
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(corrupted_lines) + "\n", encoding="utf-8")

    print(
        f"Wrote {out_path} with {changed}/{total} EDGE_RANGE measurements corrupted "
        f"(method={args.method}, probability={args.probability})."
    )


if __name__ == "__main__":
    main()
