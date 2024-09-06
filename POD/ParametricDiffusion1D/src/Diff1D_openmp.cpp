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
  unsigned int dim      = 0;   // Dimension of the problem
  unsigned int n        = 0;   // Number of parameters
  double mu_min         = 0.0; // Minimum diffusion coefficient
  double mu_max         = 0.0; // Maximum diffusion coefficient
  double mu_new         = 0.0; // New parameter
  unsigned int rank     = 0;   // Rank --- in POD riguarda per cosa
  double tol            = 0.0; // sempre per POD...
  unsigned int pod_type = 0;   // POD types: naive, standard, energy, weight, ...
  unsigned int svd_type = 0;   // SVD types: Power, Jacobi, Dynamic Jacobi, Parallel Jacobi
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
    else if (key == "pod_type")
      pod_prm >> pod_type;
    else if (key == "svd_type")
      pod_prm >> svd_type;
    else if (key == "rom_sizes")
    {
      Eigen::Index aux;
      std::string line;
      std::getline(pod_prm, line);
      std::istringstream iss(line);
      while (iss >> aux)
      {
        if (dim != 1)
        {
          std::cerr << "The rom size can't be 1. Check 'rom_sizes' in the parameter file." << std::endl;
          return 1;
        }
        rom_sizes.push_back(aux);
      }
    }
  }
  pod_prm.close();

  if (dim != 1)
  {
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

  omp_set_num_threads(n); // Number of parameters

  #pragma omp parallel for ordered
  for (unsigned int i=0; i<n; i++)
  {
    if (n == 1) // The snapshot matrix is composed by solving the problem only once.
      prm_diffusion_coefficient[i] = mu_min;
    else
      prm_diffusion_coefficient[i] = (mu_min+i*(mu_max-mu_min)/(n-1));
    
    #pragma omp ordered
    {
      pcout << "  Check prm_diffusion_coefficient = " << prm_diffusion_coefficient[i] << std::endl;
    }
  }

  auto start_snapshot = high_resolution_clock::now();
  #pragma omp parallel for ordered
  for (unsigned int i=0; i<n; i++)
  {
    #pragma omp ordered
    {
      pcout << "  Computing snapshot matrix stripe " << i+1 << " out of " << n << std::endl;
    }

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

    #pragma omp ordered
    {
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
  }
  auto stop_snapshot = high_resolution_clock::now();
  auto duration_snapshot = duration_cast<milliseconds>(stop_snapshot - start_snapshot);
  pcout << "\n  Time for building the snapshot matrix with 5 threads: " << duration_snapshot.count() << " ms" << std::endl;

  // Compute U by applying SVD or one POD algorithm to snapshots.
  pcout << "===================================================================" << std::endl;
  pcout << "Compute POD modes" << std::endl;
  pcout << "  Check rank = " << rank << std::endl;

  // Commentare
  std::unique_ptr<POD> naive_pod;
  std::unique_ptr<POD> standard_pod;
  std::unique_ptr<POD> energy_pod;
  std::unique_ptr<POD> weight_pod;
  POD compute_modes;

  switch (pod_type)
  {
    case 0:
    {
      naive_pod = std::make_unique<POD>(snapshots, rank, svd_type);
      compute_modes = *naive_pod;
      break;
    }
    case 1:
    {
      standard_pod = std::make_unique<POD>(snapshots, rank, tol, svd_type);
      compute_modes = *standard_pod;
      break;
    }
    case 2:
    {
      Mat_m Xh = Mat_m::Zero(snapshot_length, snapshot_length);
      for (Eigen::Index i=0; i<snapshot_length; i++)
      {
        Xh.coeffRef(i, i) = 2.0;
        if(i>0) Xh.coeffRef(i, i-1) = -1.0;
          if(i<snapshot_length-1) Xh.coeffRef(i, i+1) = -1.0;	
      }
      energy_pod = std::make_unique<POD>(snapshots, Xh, rank, tol, svd_type);
      compute_modes = *energy_pod;
      break;
    }
    case 3:
    {
      Mat_m Xh = Mat_m::Zero(snapshot_length, snapshot_length);
      for (Eigen::Index i=0; i<snapshot_length; i++)
      {
        Xh.coeffRef(i, i) = 2.0;
        if(i>0) Xh.coeffRef(i, i-1) = -1.0;
          if(i<snapshot_length-1) Xh.coeffRef(i, i+1) = -1.0;	
      }
      Mat_m D = Mat_m::Zero(snapshot_length, snapshot_length); // CAMBIARE
      weight_pod = std::make_unique<POD>(snapshots, Xh, D, rank, tol, svd_type);
      compute_modes = *weight_pod;
      break;
    }
    default:
    {
      std::cerr << "The pod_type should be among 0, 1, 2, 3. Check 'pod_type' in the parameter file." << std::endl;
      return 1;
    }
  }

  // Store the singular values
  Vec_v sigma = compute_modes.sigma; // commento su esportare per fare plot

  // Create the modes matrix containing the first rom_sizes columns of U.
  pcout << "===================================================================" << std::endl;
  pcout << "Construct and run ROM" << std::endl;

  std::vector<std::vector<double>> modes;

  // The approximations matrix stores the final fom_state for each rom size. 
  Mat_m approximations = Mat_m::Zero(snapshot_length, rom_sizes.size()*n);

  // The vector errors stores the relative errors in L2 norm between the full and reconstructed solution for each rom size and each parameter.
  Mat_m errors = Mat_m::Zero(n, rom_sizes.size());

  // for (size_t i=0; i<rom_sizes.size(); i++) {
  // for (size_t i=0; i<1; i++) {
  #pragma omp parallel for ordered
  for (size_t h=0; h<rom_sizes.size(); h++)
  {
    for (unsigned int i=0; i<n; i++)
    {
      
      #pragma omp ordered
      {
        pcout << "-------------------------------------------------------------------" << std::endl;
        pcout << "Creating ROM for " << rom_sizes[h] << " modes\n" << std::endl;
      }
      modes.resize(compute_modes.W.rows());
      for(auto &row : modes)
        row.resize(rom_sizes[h], 0.0);
      for (Eigen::Index j=0; j<compute_modes.W.rows(); j++)
        for (Eigen::Index k=0; k<rom_sizes[h]; k++)
          modes[j][k] = compute_modes.W(j, k);

      AdvDiffPOD<d> problemPOD(modes, prm_diffusion_coefficient[i], advdiff_parameter_file);
      
      problemPOD.setup();
      problemPOD.solve_reduced();

      // The final fom_solution is copied in an Eigen vector fom_state in order to easily compare it with snapshots and compute 
      // the relative error. For the same reason, we can initialize the fom_state with the snapshot_length.
      Vec_v fom_state = Vec_v::Zero(snapshot_length);
      for (Eigen::Index j=0; j<snapshot_length; j++)
        fom_state(j) = problemPOD.fom_solution[j];
      
      // The current fom_state is stored in approximations matrix.
      approximations.col(h*n+i) = fom_state;

      // Compute the relative l2 error.
      double fom_solution_norm = (snapshots.col((i*time_steps)+time_steps-1)).norm();
      double err = (snapshots.col((i*time_steps)+time_steps-1) - fom_state).norm();
      errors(i, h) = err/fom_solution_norm;

      // POI EVENTUALMENTE COMMENTARE
      #pragma omp ordered
      {
        pcout << "  Check fom_state values:" << std::endl;
        pcout << "    fom solution(0)              = " << snapshots(0, (i*time_steps)+time_steps-1) << std::endl;
        pcout << "    fom approximated solution(0) = " << fom_state(0) << std::endl;
        pcout << "    fom solution(1)              = " << snapshots(1, (i*time_steps)+time_steps-1) << std::endl;
        pcout << "    fom approximated solution(1) = " << fom_state(1) << std::endl;
        pcout << "    fom solution(2)              = " << snapshots(2, (i*time_steps)+time_steps-1) << std::endl;
        pcout << "    fom approximated solution(2) = " << fom_state(2) << std::endl;

        pcout << "\n  With " << rom_sizes[h] << " modes, final relative l2 error: " << errors(i, h) << std::endl;
      }

      // Clear the modes matrix for the next iteration.
      modes.resize(0);
      modes[0].resize(0, 0.0);
    }
  }

  // Export the matrix containing the full order model solutions for all parameters.
  std::string matrixFileOut1("../output/full.mtx");
  Eigen::saveMarket(solutions, matrixFileOut1);  

  // Export the matrix containing the approximated solutions for all rom_sizes and all parameters.
  std::string matrixFileOut2("../output/reconstruction.mtx");
  Eigen::saveMarket(approximations, matrixFileOut2); 

  // Export the vector containing the relative errors for all rom_sizes and all parameters.
  std::string matrixFileOut3("../output/errors.mtx");
  Eigen::saveMarket(errors, matrixFileOut3); 

  // Export the vector containing the singular values from the SVD performed in the POD algorithm.
  std::string matrixFileOut4("../output/sigma.txt");
  Eigen::saveMarketVector(sigma, matrixFileOut4); 

  return 0;
}
