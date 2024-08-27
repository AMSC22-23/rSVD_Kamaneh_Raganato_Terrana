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

// Main function.
int
main(int argc, char * argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1); // PETSc does not support multi-threaded access, set the thread limit
  // to 1 in MPI_InitFinalize().
  const unsigned int               mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  dealii::ConditionalOStream pcout(std::cout, mpi_rank == 0);

  const unsigned int N = 119;
  // const unsigned int N = 49;
  const unsigned int r = 1;

  const double T         = 0.05;
  // const double deltat    = 1e-5; // 5000 steps
  // const double deltat    = 1e-4; // 500 steps
  // const double deltat    = 5e-4; // 100 steps
  const double deltat    = 1e-3; // 50 steps

  // const double theta  = 1.0; // Implicit Euler
  const double theta  = 0.0; // Explicit Euler

  // This parameter establishes how much frequently the snapshots are collected in the snapshot matrix. The default value 1 means
  // that all the snapshots are collected.
  const unsigned int sample_every = 1;
  // const unsigned int sample_every = 5;

  // Solve the advection diffusion problem on the full order model and collect snapshots in the snapshot matrix.
  pcout << "===================================================================" << std::endl;
  pcout << "Run FOM and collect snapshots" << std::endl;

  // Only one parameter. The snapshot matrix is composed by solving the problem only once.
  double prm_diffusion_coefficient = 0.0001;

  auto start_snapshot = high_resolution_clock::now();

  AdvDiff problem(N, r, T, deltat, theta, sample_every, prm_diffusion_coefficient);    

  problem.setup();
  problem.solve();

  // Now the snapshot_matrix, defined with standard library, is required to fit in snapshots, defined in Eigen, since the SVD
  // method is implemented in Eigen.
  Eigen::Index snapshot_length = problem.snapshot_matrix.size();
  size_t time_steps = problem.snapshot_matrix[0].size();
  Mat_m snapshots = Mat_m::Zero(snapshot_length, time_steps);
  for (Eigen::Index i=0; i<snapshots.rows(); i++) // 'Eigen::Index' {aka 'long int'}
    for (Eigen::Index j=0; j<snapshots.cols(); j++)
      snapshots(i, j) = problem.snapshot_matrix[i][j];
  
  auto stop_snapshot = high_resolution_clock::now();
  auto duration_snapshot = duration_cast<milliseconds>(stop_snapshot - start_snapshot);
  pcout << "\n  Time for building the snapshot matrix: " << duration_snapshot.count() << " ms" << std::endl;

  pcout << "\n  Check snapshots size:\t\t" << snapshots.rows() << " * " << snapshots.cols() << std::endl << std::endl;

  pcout << "  Check snapshots and problem solution values:" << std::endl;
  pcout << "    snapshots(0, time_steps-1)  = " << snapshots(0, time_steps-1) << std::endl;
  pcout << "    problem.solution(0)         = " << problem.solution(0) << std::endl;
  pcout << "    snapshots(1, time_steps-1)  = " << snapshots(1, time_steps-1) << std::endl;
  pcout << "    problem.solution(1)         = " << problem.solution(1) << std::endl;
  pcout << "    snapshots(17, time_steps-1) = " << snapshots(17, time_steps-1) << std::endl;
  pcout << "    problem.solution(17)        = " << problem.solution(17) << std::endl;

  // Compute U by applying SVD or one POD algorithm to snapshots.
  pcout << "===================================================================" << std::endl;
  pcout << "Compute POD modes" << std::endl;

  const int rank = std::min(snapshots.rows(), snapshots.cols()); // Maximum rank
  // const int rank = 15;
  pcout << "  Check rank = " << rank << std::endl;

  // VERSIONE CON SVD //////
  // Vec_v sigma = Vec_v::Zero(rank);

  // Initialize the other inputs required by the SVD method, Note that the SVD method returns sigma as a vector, then it has
  // to be converted into a diagonal matrix
  // Mat_m U = Mat_m::Zero(snapshots.rows(), snapshots.rows());
  // Mat_m V = Mat_m::Zero(snapshots.cols(), snapshots.cols());
  /////////

  const double tol = 1e-12;
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

  // const double deltat_rom = 1e-3; // CAMBIA
  // const double deltat_rom = 3.01204e-4; // CAMBIA
  // ORA PROVA A TENERE STESSA deltat



  std::vector<Eigen::Index> rom_sizes = {2, 4, 6}; // CAMBIA
  // std::vector<size_t> rom_sizes = {5, 10, 25, 50, 75, 100}; // comportamento strano con 10 modes CAMBIA

  std::vector<std::vector<double>> modes;

  // The approximations matrix stores the final fom_state for each rom size. 
  Mat_m approximations = Mat_m::Zero(snapshot_length, rom_sizes.size());

  for (size_t i=0; i<rom_sizes.size(); i++) {
  // for (size_t i=0; i<1; i++) {
    pcout << "-------------------------------------------------------------------" << std::endl;
    pcout << "Creating ROM for " << rom_sizes[i] << " modes\n" << std::endl;

    //VERSIONE PER SVD MI SA
    // modes.resize(U.rows());
    // for(auto &row : modes)
    //   row.resize(rom_sizes[i], 0.0);
    // for (size_t j=0; j<U.rows(); j++)
    //   for (size_t k=0; k<rom_sizes[i]; k++)
    //     modes[j][k] = U(j, k);

    modes.resize(compute_modes.W.rows());
    for(auto &row : modes)
      row.resize(rom_sizes[i], 0.0);
    for (Eigen::Index j=0; j<compute_modes.W.rows(); j++)
      for (Eigen::Index k=0; k<rom_sizes[i]; k++)
        modes[j][k] = compute_modes.W(j, k);

    AdvDiffPOD problemPOD(N, r, T, deltat, theta, modes, prm_diffusion_coefficient);
    
    problemPOD.setup();
    problemPOD.solve_reduced();

    // The final fom_solution is copied in an Eigen vector fom_state in order to easily compare it with snapshots and compute 
    // the relative error. For the same reason, we can initialize the fom_state with the snapshot_length.
    Vec_v fom_state = Vec_v::Zero(snapshot_length);
    for (Eigen::Index j=0; j<snapshot_length; j++)
      fom_state(j) = problemPOD.fom_solution[j];
    
    // The current fom_state is stored in approximations matrix.
    approximations.col(i) = fom_state;

    // Compute the relative l2 error.
    double fom_solution_norm = (snapshots.col(time_steps-1)).norm();
    double err = (snapshots.col(time_steps-1) - fom_state).norm();

    // POI EVENTUALMENTE COMMENTARE
    pcout << "  Check fom_state values:" << std::endl;
    pcout << "    fom solution(0)              = " << snapshots(0, time_steps-1) << std::endl;
    pcout << "    fom approximated solution(0) = " << fom_state(0) << std::endl;
    pcout << "    fom solution(1)              = " << snapshots(1, time_steps-1) << std::endl;
    pcout << "    fom approximated solution(1) = " << fom_state(1) << std::endl;
    pcout << "    fom solution(2)              = " << snapshots(2, time_steps-1) << std::endl;
    pcout << "    fom approximated solution(2) = " << fom_state(2) << std::endl;

    pcout << "\n  With " << rom_sizes[i] << " modes, final relative l2 error: " << err/fom_solution_norm << std::endl;

    // Clear the modes matrix for the next iteration.
    modes.resize(0);
    modes[0].resize(0, 0.0);
  }

  return 0;
}
