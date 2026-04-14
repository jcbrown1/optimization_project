#include <CORA/CORA.h>
#include <CORA/Measurements.h>
#include <test_utils.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace CORA {

TEST_CASE("range weights scale only range blocks",
          "[robust::range-weights::assembly]") {
  Problem problem = getProblem("small_ra_slam_problem");
  problem.updateProblemData();

  const CoraDataSubmatrices base_submatrices = problem.getDataSubmatrices();
  const SparseMatrix base_data_matrix = problem.getDataMatrix();
  const auto range_measurements = problem.getRangeMeasurements();
  REQUIRE_FALSE(range_measurements.empty());

  std::vector<Scalar> weights(problem.numRangeMeasurements(), 1.0);
  weights[0] = 0.25;
  if (weights.size() > 1) {
    weights[1] = 0.75;
  }
  problem.setRangeWeights(weights);
  problem.updateProblemData();

  const CoraDataSubmatrices weighted_submatrices = problem.getDataSubmatrices();
  const SparseMatrix weighted_data_matrix = problem.getDataMatrix();

  CHECK_THAT(weighted_submatrices.rel_pose_translation_precision_matrix,
             IsApproximatelyEqual(
                 base_submatrices.rel_pose_translation_precision_matrix, 1e-12));
  CHECK_THAT(weighted_submatrices.rotation_conn_laplacian,
             IsApproximatelyEqual(base_submatrices.rotation_conn_laplacian,
                                  1e-12));

  for (size_t k = 0; k < range_measurements.size(); ++k) {
    const Scalar expected_precision =
        weights[k] * range_measurements[k].getPrecision();
    CHECK_THAT(weighted_submatrices.range_precision_matrix.coeff(k, k),
               Catch::Matchers::WithinAbs(expected_precision, 1e-12));
  }

  const int rot_sz = problem.numPosesDim();
  Matrix base_rot_block = base_data_matrix.block(0, 0, rot_sz, rot_sz).toDense();
  Matrix weighted_rot_block =
      weighted_data_matrix.block(0, 0, rot_sz, rot_sz).toDense();
  CHECK_THAT(weighted_rot_block, IsApproximatelyEqual(base_rot_block, 1e-10));
}

TEST_CASE("GM closed-form update decreases with larger squared residual",
          "[robust::gm::weights]") {
  const Scalar mu = 10.0;
  const Scalar c_bar = 2.0;
  const Scalar c_sq = c_bar * c_bar;
  const Scalar sigma_sq = 4.0;
  const Scalar precision = 1.0 / sigma_sq;

  Vector t_i(2);
  t_i << 0.0, 0.0;
  Vector t_j(2);
  t_j << 3.0, 4.0;
  Vector u(2);
  u << 1.0, 0.0;
  const Scalar r_tilde = 5.0;

  Vector residual = t_j - t_i - r_tilde * u;
  const Scalar s_small = precision * residual.squaredNorm();
  CHECK_THAT(s_small, Catch::Matchers::WithinAbs(5.0, 1e-12));

  const Scalar s_large = 50.0;
  const Scalar w_small = std::pow((mu * c_sq) / (s_small + mu * c_sq), 2);
  const Scalar w_large = std::pow((mu * c_sq) / (s_large + mu * c_sq), 2);

  CHECK(w_large < w_small);
  CHECK_THAT(w_small, Catch::Matchers::WithinAbs(std::pow(40.0 / 45.0, 2), 1e-12));
}

void checkNominalAndOneStageGncMatch(Formulation formulation) {
  Problem nominal_problem = getProblem("small_ra_slam_problem");
  nominal_problem.setFormulation(formulation);
  nominal_problem.updateProblemData();

  Problem robust_problem = getProblem("small_ra_slam_problem");
  robust_problem.setFormulation(formulation);
  robust_problem.updateProblemData();

  const std::string x0_path =
      getTestDataFpath("small_ra_slam_problem", "X_odom.mm");
  Matrix x0_full = readMatrixMarketFile(x0_path).toDense();
  Matrix x0 = x0_full;
  if (formulation == Formulation::Implicit) {
    x0 = x0_full.topRows(nominal_problem.rotAndRangeMatrixSize());
  }
  const int max_rank = 8;
  CoraResult nominal_result =
      solveCORA(nominal_problem, x0, max_rank, false, false, false);

  GncParams params;
  params.mu_init = 1.0;
  params.mu_min = 1.0;
  params.mu_factor = 0.5;
  params.max_outer_iters = 1;
  params.max_inner_iters = 1;
  params.max_relaxation_rank = max_rank;
  params.verbose = false;
  params.log_iterates = false;
  params.show_iterates = false;
  RobustCoraResult robust_result = solveCORAWithGNC(robust_problem, x0, params);

  CHECK(robust_result.total_weighted_solves == 1);
  CHECK_THAT(robust_result.result.first.f,
             Catch::Matchers::WithinAbs(nominal_result.first.f, 1e-6));
}

TEST_CASE("one-stage GNC equals nominal solve in explicit mode",
          "[robust::integration::explicit]") {
  checkNominalAndOneStageGncMatch(Formulation::Explicit);
}

TEST_CASE("one-stage GNC equals nominal solve in implicit mode",
          "[robust::integration::implicit]") {
  checkNominalAndOneStageGncMatch(Formulation::Implicit);
}

Problem makeConflictingRangeProblem() {
  const int dim = 2;
  const int rank = 3;
  Problem problem(dim, rank, Formulation::Explicit);

  Symbol x1("x1");
  Symbol x2("x2");
  Symbol x3("x3");
  problem.addPoseVariable(x1);
  problem.addPoseVariable(x2);
  problem.addPoseVariable(x3);

  Matrix rot = Matrix::Identity(dim, dim);
  Vector t = Vector::Zero(dim);
  t(0) = 1.0;
  Matrix cov = Matrix::Identity(dim + 1, dim + 1);
  cov(0, 0) = 1e-4;
  cov(1, 1) = 1e-4;
  cov(2, 2) = 1e-4;
  problem.addRelativePoseMeasurement(RelativePoseMeasurement(x1, x2, rot, t, cov));
  problem.addRelativePoseMeasurement(RelativePoseMeasurement(x2, x3, rot, t, cov));

  problem.addRangeMeasurement(RangeMeasurement(x1, x3, 2.0, 0.01));
  problem.addRangeMeasurement(RangeMeasurement(x1, x2, 10.0, 0.01));

  problem.updateProblemData();
  return problem;
}

TEST_CASE("GNC downweights at least one inconsistent range edge",
          "[robust::integration::downweight]") {
  Problem problem = makeConflictingRangeProblem();
  Matrix x0 = problem.getRandomInitialGuess();

  GncParams params;
  params.c_bar = 1.0;
  params.mu_init = 100.0;
  params.mu_factor = 0.5;
  params.mu_min = 1.0;
  params.max_outer_iters = 8;
  params.max_inner_iters = 5;
  params.max_relaxation_rank = 8;
  params.weight_tol = 1e-3;
  params.min_weight = 1e-6;

  RobustCoraResult robust_result = solveCORAWithGNC(problem, x0, params);
  REQUIRE(robust_result.final_range_weights.size() == 2);

  const bool has_downweighted_edge = std::any_of(
      robust_result.final_range_weights.begin(),
      robust_result.final_range_weights.end(),
      [](Scalar w) { return w < 0.95; });
  CHECK(has_downweighted_edge);
}

} // namespace CORA
