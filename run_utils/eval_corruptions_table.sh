#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_utils/eval_corruptions_table.sh \
    --pyfg <ground_truth.pyfg> \
    --baseline <baseline_solution.tum> \
    --uniform <uniform_solution.tum> \
    --gamma <gamma_solution.tum> \
    [--compare-script <path/to/compare_with_gt.py>]

Description:
  Evaluates three solution sets (baseline, uniform, gamma) against one
  ground-truth .pyfg file by calling examples/compare_with_gt.py, then prints
  a metrics table with:
    - RMSE rotation (deg)
    - RMSE translation (m)
    - optimization loss (placeholder, averaged per robot when multi-robot)

Notes:
  - Each solution path can be either:
      * a single .tum file (single robot), or
      * a directory containing robot .tum files (e.g. tiers with 4 robots).
  - The compare script is run with a unique --output_dir per row to avoid
    interactive overwrite prompts from evo plot files.
EOF
}

PYFG=""
BASELINE_SOL=""
UNIFORM_SOL=""
GAMMA_SOL=""
COMPARE_SCRIPT="examples/compare_with_gt.py"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pyfg)
      PYFG="$2"
      shift 2
      ;;
    --baseline)
      BASELINE_SOL="$2"
      shift 2
      ;;
    --uniform)
      UNIFORM_SOL="$2"
      shift 2
      ;;
    --gamma)
      GAMMA_SOL="$2"
      shift 2
      ;;
    --compare-script)
      COMPARE_SCRIPT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$PYFG" || -z "$BASELINE_SOL" || -z "$UNIFORM_SOL" || -z "$GAMMA_SOL" ]]; then
  echo "Error: missing required arguments." >&2
  usage
  exit 1
fi

for p in "$PYFG" "$BASELINE_SOL" "$UNIFORM_SOL" "$GAMMA_SOL" "$COMPARE_SCRIPT"; do
  if [[ ! -e "$p" ]]; then
    echo "Error: path does not exist: $p" >&2
    exit 1
  fi
done

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

collect_solution_files() {
  local sol_path="$1"
  local -n out_arr_ref=$2
  out_arr_ref=()

  if [[ -f "$sol_path" ]]; then
    out_arr_ref=("$sol_path")
    return
  fi

  if [[ ! -d "$sol_path" ]]; then
    echo "Error: solution path must be a file or directory: $sol_path" >&2
    exit 1
  fi

  local -a files=()
  shopt -s nullglob globstar
  files=("$sol_path"/**/*_cora_*.tum)
  if [[ ${#files[@]} -eq 0 ]]; then
    files=("$sol_path"/**/*.tum)
  fi
  shopt -u nullglob globstar

  if [[ ${#files[@]} -eq 0 ]]; then
    echo "Error: no .tum files found under $sol_path" >&2
    exit 1
  fi

  # Sort paths naturally so robot_0, robot_1, ... are in order.
  mapfile -t out_arr_ref < <(printf '%s\n' "${files[@]}" | sort -V)
}

run_eval() {
  local label="$1"
  local sol="$2"
  local out_dir="$TMP_ROOT/out_${label}"
  mkdir -p "$out_dir"

  local log_file="$TMP_ROOT/${label}.log"
  local -a sol_files=()
  collect_solution_files "$sol" sol_files

  python "$COMPARE_SCRIPT" "$PYFG" --cora_solution "${sol_files[@]}" --output_dir "$out_dir" > "$log_file" 2>&1

  # evo_ape prints rmse lines in sequence:
  #   translation, rotation, translation, rotation, ...
  # (for multi-robot there are multiple pairs). We average each category.
  local rmse_trans rmse_rot loss
  rmse_trans="$(awk '
    /^[[:space:]]*rmse[[:space:]]+/ {
      c += 1
      if (c % 2 == 1) {
        s += $2
        n += 1
      }
    }
    END {
      if (n > 0) printf "%.6f", s / n
    }
  ' "$log_file")"
  rmse_rot="$(awk '
    /^[[:space:]]*rmse[[:space:]]+/ {
      c += 1
      if (c % 2 == 0) {
        s += $2
        n += 1
      }
    }
    END {
      if (n > 0) printf "%.6f", s / n
    }
  ' "$log_file")"

  # Average placeholder per-robot loss from per-robot metrics line:
  # loss_i = mean_translation_error_i + 0.01 * mean_rotation_error_i
  loss="$(awk '
    /Robot [0-9]+: mean translation error=/ {
      line = $0
      trans_part = line
      rot_part = line

      sub(/^.*mean translation error=/, "", trans_part)
      sub(/ m,.*$/, "", trans_part)

      sub(/^.*mean rotation error=/, "", rot_part)
      sub(/ deg,.*$/, "", rot_part)

      if (trans_part != "" && rot_part != "") {
        s += (trans_part + 0.0) + 0.01 * (rot_part + 0.0)
        n += 1
      }
    }
    END {
      if (n > 0) printf "%.6f", s / n
    }
  ' "$log_file")"

  if [[ -z "${loss:-}" ]]; then
    loss="$(awk -F'= ' '/Optimal-solution loss \(placeholder\)/ {print $2; exit}' "$log_file")"
  fi

  if [[ -z "${rmse_trans:-}" ]]; then rmse_trans="NA"; fi
  if [[ -z "${rmse_rot:-}" ]]; then rmse_rot="NA"; fi
  if [[ -z "${loss:-}" ]]; then loss="NA"; fi

  printf "| %-8s | %-14s | %-16s | %-16s |\n" "$label" "$rmse_rot" "$rmse_trans" "$loss"
}

echo "| Corruption | RMSE rot (deg) | RMSE trans (m) | Optimization loss |"
echo "|------------|----------------|----------------|-------------------|"
run_eval "baseline" "$BASELINE_SOL"
run_eval "uniform" "$UNIFORM_SOL"
run_eval "gamma" "$GAMMA_SOL"

