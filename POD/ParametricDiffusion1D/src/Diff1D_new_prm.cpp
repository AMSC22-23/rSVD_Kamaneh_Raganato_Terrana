#include <deal.II/base/convergence_table.h>
#include <deal.II/base/conditional_ostream.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <map>

#include <chrono>
using namespace std::chrono;

#include "POD.hpp"
#include "AdvDiff1D.hpp"
#include "AdvDiff1D_POD.hpp"

#include <unsupported/Eigen/SparseExtra>

// Main function.
int
main(int argc, char * argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1); // PETSc does not support multi-threaded access, set the thread limit
  // to 1 in MPI_InitFinalize().
  const unsigned int               mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  dealii::ConditionalOStream pcout(std::cout, mpi_rank == 0);

  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " [pod_parameter_file] [advdiff_parameter_file]" << std::endl;
    return 1;
  }

  std::string pod_parameter_file, advdiff_parameter_file;
  if (argc >= 3)
  {
    pod_parameter_file = argv[1];
    advdiff_parameter_file = argv[2];
  }

  // Read the parameters from the pod_parameter_file.
  unsigned int dim  = 0;   // Dimension of the problem
  unsigned int n    = 0;   // Number of parameters
  double mu_min     = 0.0; // Minimum diffusion coefficient
  double mu_max     = 0.0; // Maximum diffusion coefficient
  double mu_new     = 0.0; // New parameter
  unsigned int rank = 0;   // Rank --- in POD riguarda per cosa
  double tol        = 0.0; // sempre per POD...
  std::vector<Eigen::Index> rom_sizes; // Sizes for the reduced order models

  std::string key;
  std::ifstream pod_prm(pod_parameter_file);
  if (!pod_prm) {
      std::cerr << "Error opening file " << pod_parameter_file << std::endl;
      return 1;
  }
  while (pod_prm >> key)
  {
    if (key == "dim")
      pod_prm >> dim;
    else if (key == "n")
      pod_prm >> n;
    else if (key == "mu_min")
      pod_prm >> mu_min;
    else if (key == "mu_max")
      pod_prm >> mu_max;
    else if (key == "mu_new")
      pod_prm >> mu_new;
    else if (key == "rank")
      pod_prm >> rank;
    else if (key == "tol")
      pod_prm >> tol;
    else if (key == "rom_sizes")
    {
      Eigen::Index aux;
      std::string line;
      std::getline(pod_prm, line);
      std::istringstream iss(line);
      while (iss >> aux)
        rom_sizes.push_back(aux);
    }
  }
  pod_prm.close();

  if (dim != 1) {
    std::cerr << "The problem dimension should be 1. Check 'dim' in the parameter file." << std::endl;
    return 1;
  }
  constexpr unsigned int d = 1;

  // Solve the advection diffusion problem on the full order model and collect snapshots in the snapshot matrix.
  pcout << "===================================================================" << std::endl;
  pcout << "Run FOM and collect snapshots" << std::endl;

  std::vector<double> prm_diffusion_coefficient;
  prm_diffusion_coefficient.resize(n); 

  Eigen::Index snapshot_length = 0;
  Eigen::Index time_steps = 0;
  Mat_m snapshots;
  Mat_m solutions; // The solutions matrix stores the full order model solutions for all parameters. It is then exported.

  for (unsigned int i=0; i<n; i++)
  {
    if (n == 1) // The snapshot matrix is composed by solving the problem only once.
      prm_diffusion_coefficient[i] = mu_min;
    else
      prm_diffusion_coefficient[i] = (mu_min+i*(mu_max-mu_min)/(n-1));
    pcout << "  Check prm_diffusion_coefficient = " << prm_diffusion_coefficient[i] << std::endl;
  }
    
  auto start_snapshot = high_resolution_clock::now();
  for (unsigned int i=0; i<n; i++)
  {
    pcout << "  Computing snapshot matrix stripe " << i+1 << " out of " << n << std::endl;

    AdvDiff<d> problem(prm_diffusion_coefficient[i], advdiff_parameter_file);    

    problem.setup();
    problem.solve();

    // Now the snapshot_matrix, defined with standard library, is required to fit in snapshots, defined in Eigen, since the SVD
    // method is implemented in Eigen.
    if (i == 0) { // At the first iteration it is useful to resize snapshots.
      snapshot_length = problem.snapshot_matrix.size();
      time_steps = problem.snapshot_matrix[0].size();
      snapshots.resize(snapshot_length, n*time_steps);
      solutions.resize(snapshot_length, n);
    }
    for (Eigen::Index j=0; j<snapshots.rows(); j++)
      for (Eigen::Index k=0; k<time_steps; k++)
        snapshots(j, (i*time_steps)+k) = problem.snapshot_matrix[j][k];

    solutions.col(i) = snapshots.col((i*time_steps)+time_steps-1);

    pcout << "\n  Check snapshots size:\t\t" << snapshots.rows() << " * " << snapshots.cols() << std::endl << std::endl;
    // This print is commented to save time and space in the output.
    pcout << "  Check snapshots and problem solution values:" << std::endl;
    pcout << "    snapshots(0, time_steps-1)  = " << snapshots(0, (i*time_steps)+time_steps-1) << std::endl;
    pcout << "    problem.solution(0)         = " << problem.solution(0) << std::endl;
    pcout << "    snapshots(1, time_steps-1)  = " << snapshots(1, (i*time_steps)+time_steps-1) << std::endl;
    pcout << "    problem.solution(1)         = " << problem.solution(1) << std::endl;
    pcout << "    snapshots(17, time_steps-1) = " << snapshots(17, (i*time_steps)+time_steps-1) << std::endl;
    pcout << "    problem.solution(17)        = " << problem.solution(17) << std::endl; 
  }
  auto stop_snapshot = high_resolution_clock::now();
  auto duration_snapshot = duration_cast<milliseconds>(stop_snapshot - start_snapshot);
  pcout << "\n  Time for building the snapshot matrix sequentially: " << duration_snapshot.count() << " ms" << std::endl;

  // Compute U by applying SVD or one POD algorithm to snapshots.
  pcout << "===================================================================" << std::endl;
  pcout << "Compute POD modes" << std::endl;
  pcout << "  Check rank = " << rank << std::endl;

  // VERSIONE CON SVD //////
  // Vec_v sigma = Vec_v::Zero(rank);

  // Initialize the other inputs required by the SVD method, Note that the SVD method returns sigma as a vector, then it has
  // to be converted into a diagonal matrix
  // Mat_m U = Mat_m::Zero(snapshots.rows(), snapshots.rows());
  // Mat_m V = Mat_m::Zero(snapshots.cols(), snapshots.cols());
  /////////

  // const double tol = 1e-12;
  // const double tol = 1e-9;
  // POD compute_modes(snapshots, rank, tol); ///////versione


  // PROVA CON ENERGY ma devi capire significato e costruzione Xh///////////



  Mat_m Xh = Mat_m::Zero(snapshot_length, snapshot_length);
  for (Eigen::Index i=0; i<snapshot_length; i++) {
      Xh.coeffRef(i, i) = 2.0;
    if(i>0) Xh.coeffRef(i, i-1) = -1.0;
      if(i<snapshot_length-1) Xh.coeffRef(i, i+1) = -1.0;	
  }
  POD compute_modes(snapshots, Xh, rank, tol);


  // PROVA CON WEIGHT ///// versione
  // Mat_m D = Mat_m::Zero(snapshots.cols(), snapshots.cols());
  // for (int i=0; i<snapshots.cols(); i++) {
  //     D.coeffRef(i, i) = 1.0;
  // }
  // POD compute_modes(snapshots, Xh, D, rank, tol);

  // SVD(snapshots, sigma, U, V, rank);
  // pcout << "\nCheck dimensions of U:     " << U.rows() << " * " << U.cols() << endl;
  // pcout << "Check dimensions of sigma: " << sigma.size() << endl;
  // pcout << "Check dimensions of V:     " << V.rows() << " * " << V.cols() << endl;


  // pcout << " Check U" << U << std::endl;

  /// SISTEMARE





  // Create the modes matrix containing the first rom_sizes columns of U.
  pcout << "===================================================================" << std::endl;
  pcout << "Construct and run ROM" << std::endl;

  std::vector<std::vector<double>> modes;

  // The approximations matrix stores the final fom_state for each rom size. 
  Mat_m approximations = Mat_m::Zero(snapshot_length, rom_sizes.size()*n);

  // The vector errors stores the relative errors in L2 norm between the full and reconstructed solution for each rom size and each parameter.
  Mat_m errors = Mat_m::Zero(n, rom_sizes.size());

  /**
  // Now we want to use the class AdvDiffPOD to solve a reduced problem in which the diffusion parameter wasn't used for the
  // construction of the snapshot matrix. The aim is to save time and space thanks to this prediction.
  // At first, we solve the full order problem with the new parameter through the AdvDiff class, its solution will be used for 
  // computing the relative error. Then we solve the reduced order problem changing the number of reduced order size.
  */

  AdvDiff<d> problem_new_parameter(mu_new, advdiff_parameter_file);    

  auto start_full = high_resolution_clock::now();

  problem_new_parameter.setup();
  problem_new_parameter.solve();

  auto stop_full = high_resolution_clock::now();
  auto duration_full = duration_cast<milliseconds>(stop_full - start_full);
  auto duration_full_step = problem_new_parameter.duration_full_avg.count();

  Vec_v solution_new_parameter = Vec_v::Zero(problem_new_parameter.solution.size());
  for (size_t i=0; i<problem_new_parameter.solution.size(); i++)
    solution_new_parameter(i) = problem_new_parameter.solution(i);
  
  for (size_t h=0; h<rom_sizes.size(); h++)
  {
    pcout << "-------------------------------------------------------------------" << std::endl;
    pcout << "Creating ROM for " << rom_sizes[h] << " modes\n" << std::endl;

    //VERSIONE PER SVD MI SA
    // modes.resize(U.rows());
    // for(auto &row : modes)
    //   row.resize(rom_sizes[i], 0.0);
    // for (size_t j=0; j<U.rows(); j++)
    //   for (size_t k=0; k<rom_sizes[i]; k++)
    //     modes[j][k] = U(j, k);

    modes.resize(compute_modes.W.rows());
    for(auto &row : modes)
      row.resize(rom_sizes[h], 0.0);
    for (Eigen::Index j=0; j<compute_modes.W.rows(); j++)
      for (Eigen::Index k=0; k<rom_sizes[h]; k++)
        modes[j][k] = compute_modes.W(j, k);

    AdvDiffPOD<d> problemPOD(modes, mu_new, advdiff_parameter_file);

    auto start_reduced = high_resolution_clock::now();
    
    problemPOD.setup();
    problemPOD.solve_reduced();

    auto stop_reduced = high_resolution_clock::now();
    auto duration_reduced = duration_cast<milliseconds>(stop_reduced - start_reduced);

    // The final fom_solution is copied in an Eigen vector fom_state in order to easily compare it with snapshots and compute 
    // the relative error. For the same reason, we can initialize the fom_state with the snapshot_length.
    Vec_v fom_state = Vec_v::Zero(snapshot_length);
    for (Eigen::Index j=0; j<snapshot_length; j++)
      fom_state(j) = problemPOD.fom_solution[j];
    
    // The current fom_state is stored in approximations matrix.
    approximations.col(h) = fom_state;

    // Compute the relative l2 error.
    double fom_solution_norm = solution_new_parameter.norm();
    double err = (solution_new_parameter - fom_state).norm();

    pcout << "  Check fom_state values:" << std::endl;
    pcout << "    fom solution(0)              = " << solution_new_parameter(0) << std::endl;
    pcout << "    fom approximated solution(0) = " << fom_state(0) << std::endl;
    pcout << "    fom solution(1)              = " << solution_new_parameter(1) << std::endl;
    pcout << "    fom approximated solution(1) = " << fom_state(1) << std::endl;
    pcout << "    fom solution(2)              = " << solution_new_parameter(2) << std::endl;
    pcout << "    fom approximated solution(2) = " << fom_state(2) << std::endl;

    pcout << "\n  With " << rom_sizes[h] << " modes, final relative l2 error: " << err/fom_solution_norm << std::endl;
    pcout << "  Time for solving the full order problem:          " << duration_full.count() << " ms" << std::endl;
    pcout << "  Time for solving the reduced order problem:       " << duration_reduced.count() << " ms" << std::endl;
    pcout << "  Average time for solving one step of fom problem: " << duration_full_step << " ms" << std::endl;
    pcout << "  Average time for solving one step of rom problem: " << problemPOD.duration_reduced_avg.count() << " ms" << std::endl;

    // Clear the modes matrix for the next iteration.
    modes.resize(0);
    modes[0].resize(0, 0.0);
  }

  // Export the vector containing the full order model solutions for the new parameter.
  std::string matrixFileOut1("../output/full.mtx");
  Eigen::saveMarket(solution_new_parameter, matrixFileOut1);  

  // Export the matrix containing the approximated solutions for all rom_sizes and the new parameter.
  std::string matrixFileOut2("../output/reconstruction.mtx");
  Eigen::saveMarket(approximations, matrixFileOut2); 

  // Export the vector containing the relative errors for all rom_sizes and the new parameter.
  std::string matrixFileOut3("../output/errors.mtx");
  Eigen::saveMarket(errors, matrixFileOut3); 

  return 0;
}
