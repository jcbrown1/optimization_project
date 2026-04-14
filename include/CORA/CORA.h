/** @file
    @brief The main header file for the CORA library.

    This file provides the primary interface to the CORA library. All
    usage of the library should be done through this file.
*/

#pragma once

#include <CORA/CORA_problem.h>
#include <CORA/CORA_types.h>
#include <CORA/pyfg_text_parser.h>
#include <string>
#include <utility>
#include <vector>

namespace CORA {

using CoraTntResult = Optimization::Riemannian::TNTResult<Matrix, Scalar>;
using CoraResult = std::pair<CoraTntResult, std::vector<Matrix>>;

struct GncParams {
  Scalar c_bar = 1.5;
  Scalar mu_init = 100.0;
  Scalar mu_factor = 0.8;
  Scalar mu_min = 1.0;
  int max_outer_iters = 12;
  int max_inner_iters = 10;
  Scalar weight_tol = 1e-3;
  Scalar min_weight = 1e-3;
  int max_relaxation_rank = 20;
  bool verbose = false;
  bool log_iterates = false;
  bool show_iterates = false;
};

// struct GncParams {
//   // Conservative defaults: preserve most range information unless residuals are
//   // clearly inconsistent. These can still be overridden via CORA_GNC_* env vars.
//   Scalar c_bar = 120.0;
//   Scalar mu_init = 100.0;
//   Scalar mu_factor = 0.9;
//   Scalar mu_min = 1.0;
//   int max_outer_iters = 8;
//   int max_inner_iters = 4;
//   Scalar weight_tol = 1e-3;
//   Scalar min_weight = 5e-2;
//   int max_relaxation_rank = 20;
//   bool verbose = false;
//   bool log_iterates = false;
//   bool show_iterates = false;
// };

struct GncStageStats {
  Scalar mu = 1.0;
  int inner_iterations = 0;
  Scalar min_weight = 1.0;
  Scalar mean_weight = 1.0;
  Scalar max_weight = 1.0;
  Scalar max_weight_delta = 0.0;
};

struct RobustCoraResult {
  CoraResult result;
  std::vector<Scalar> final_range_weights;
  std::vector<GncStageStats> stage_stats;
  int total_weighted_solves = 0;
};

CoraResult solveCORA(Problem &problem, const Matrix &x0,
                     int max_relaxation_rank = 20, bool verbose = false,
                     bool log_iterates = false, bool show_iterates = false);

RobustCoraResult solveCORAWithGNC(Problem &problem, const Matrix &x0,
                                  const GncParams &params = GncParams());

inline CoraResult solveCORA(std::string filepath) {
  Problem problem = parsePyfgTextToProblem(filepath);
  Matrix x0 = Matrix();
  throw std::runtime_error(
      "Not implemented -- need to decide how to get initialization");
  return solveCORA(problem, x0);
}

Matrix saddleEscape(const Problem &problem, const Matrix &Y, Scalar theta,
                    const Vector &v, Scalar gradient_tolerance,
                    Scalar preconditioned_gradient_tolerance);

Matrix projectSolution(const Problem &problem, const Matrix &Y,
                       bool verbose = false);

} // namespace CORA
