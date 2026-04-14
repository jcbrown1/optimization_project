// robust_cora — baseline CORA solve; customize optimization (e.g. TNT / loss)
// in this file and in src/CORA.cpp as needed.

#include <CORA/CORA.h>
#include <CORA/CORA_problem.h>
#include <CORA/CORA_types.h>
#include <CORA/CORA_utils.h>
#include <CORA/pyfg_text_parser.h>
#include <CORA/Symbol.h>

#ifdef GPERFTOOLS
#include <gperftools/profiler.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <vector>

using PoseChain = std::vector<CORA::Symbol>;
using PoseChains = std::vector<PoseChain>;

namespace {

PoseChains getRobotPoseChains(const CORA::Problem &problem) {
  std::set<unsigned char> seen_pose_chars;
  for (auto const &all_pose_symbols : problem.getPoseSymbolMap()) {
    CORA::Symbol pose_symbol = all_pose_symbols.first;
    seen_pose_chars.insert(pose_symbol.chr());
  }

  std::vector<unsigned char> unique_pose_chars = {seen_pose_chars.begin(),
                                                  seen_pose_chars.end()};
  std::sort(unique_pose_chars.begin(), unique_pose_chars.end());

  PoseChains robot_pose_chains;
  for (auto const &pose_char : unique_pose_chars) {
    PoseChain robot_pose_chain = problem.getPoseSymbols(pose_char);
    std::sort(robot_pose_chain.begin(), robot_pose_chain.end());
    robot_pose_chains.push_back(robot_pose_chain);
  }

  return robot_pose_chains;
}

std::filesystem::path getOutputRoot() {
  if (const char *env = std::getenv("CORA_OUTPUT_ROOT")) {
    return std::filesystem::path(env);
  }
  return std::filesystem::path(CORA_REPO_ROOT) / "outputs";
}

std::filesystem::path output_dir_for_robust_dataset(
    const std::filesystem::path &pyfg_path) {
  const std::string dataset_name = pyfg_path.stem().string();
  return getOutputRoot() / dataset_name / "robust";
}

void save_solution(const CORA::Problem &problem,
                   const CORA::Matrix &aligned_soln,
                   const std::filesystem::path &output_dir,
                   const std::filesystem::path &pyfg_path) {
  std::filesystem::create_directories(output_dir);
  PoseChains robot_pose_chains = getRobotPoseChains(problem);
  const std::string stem = pyfg_path.stem().string();

  for (size_t robot_index = 0; robot_index < robot_pose_chains.size();
       ++robot_index) {
    const PoseChain &robot_pose_chain = robot_pose_chains[robot_index];
    const std::filesystem::path tum_path =
        output_dir / (stem + "_robust_cora_" + std::to_string(robot_index) + ".tum");
    const std::filesystem::path g2o_path =
        output_dir / (stem + "_robust_cora_" + std::to_string(robot_index) + ".g2o");
    CORA::saveSolnToTum(robot_pose_chain, problem, aligned_soln,
                        tum_path.string());
    CORA::saveSolnToG20(robot_pose_chain, problem, aligned_soln,
                        g2o_path.string());
  }
  std::cout << "Saved solution to " << output_dir.string() << std::endl;
}

double envToDouble(const char *name, double default_value) {
  if (const char *env = std::getenv(name)) {
    return std::stod(env);
  }
  return default_value;
}

int envToInt(const char *name, int default_value) {
  if (const char *env = std::getenv(name)) {
    return std::stoi(env);
  }
  return default_value;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " [input .pyfg file]" << std::endl;
    exit(1);
  }

  const std::filesystem::path pyfg_path = argv[1];

  CORA::Problem problem = CORA::parsePyfgTextToProblem(pyfg_path.string());
  problem.updateProblemData();

#ifdef GPERFTOOLS
  ProfilerStart("robust_cora.prof");
#endif

  CORA::Matrix x0 = problem.getRandomInitialGuess();
  int max_rank = 10;

  CORA::GncParams gnc_params;
  gnc_params.c_bar = envToDouble("CORA_GNC_C_BAR", gnc_params.c_bar);
  gnc_params.mu_init = envToDouble("CORA_GNC_MU_INIT", gnc_params.mu_init);
  gnc_params.mu_factor = envToDouble("CORA_GNC_MU_FACTOR", gnc_params.mu_factor);
  gnc_params.mu_min = envToDouble("CORA_GNC_MU_MIN", gnc_params.mu_min);
  gnc_params.max_outer_iters =
      envToInt("CORA_GNC_MAX_OUTER_ITERS", gnc_params.max_outer_iters);
  gnc_params.max_inner_iters =
      envToInt("CORA_GNC_MAX_INNER_ITERS", gnc_params.max_inner_iters);
  gnc_params.weight_tol =
      envToDouble("CORA_GNC_WEIGHT_TOL", gnc_params.weight_tol);
  gnc_params.min_weight =
      envToDouble("CORA_GNC_MIN_WEIGHT", gnc_params.min_weight);
  gnc_params.max_relaxation_rank = max_rank;

  CORA::RobustCoraResult robust_soln =
      CORA::solveCORAWithGNC(problem, x0, gnc_params);
  CORA::Matrix projected_soln =
      CORA::projectSolution(problem, robust_soln.result.first.x, false);
  CORA::Matrix explicit_soln = projected_soln;
  if (problem.getFormulation() == CORA::Formulation::Implicit) {
    explicit_soln = problem.getTranslationExplicitSolution(projected_soln);
  }
  CORA::Matrix aligned_soln = problem.alignEstimateToOrigin(explicit_soln);

  const std::filesystem::path out_dir = output_dir_for_robust_dataset(pyfg_path);
  save_solution(problem, aligned_soln, out_dir, pyfg_path);

  if (!robust_soln.stage_stats.empty()) {
    const CORA::GncStageStats &last_stage = robust_soln.stage_stats.back();
    std::cout << "GNC stages: " << robust_soln.stage_stats.size()
              << ", weighted solves: " << robust_soln.total_weighted_solves
              << ", final mu: " << last_stage.mu
              << ", final mean weight: " << last_stage.mean_weight << std::endl;
  }

  // std::cout << "robust_cora solution: " << std::endl;
  // std::cout << aligned_soln << std::endl;

#ifdef GPERFTOOLS
  ProfilerStop();
#endif
}
