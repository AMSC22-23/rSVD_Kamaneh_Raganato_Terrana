#include <deal.II/base/convergence_table.h>
#include <deal.II/base/conditional_ostream.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <map>

#include "SVD.hpp"
#include "AdvDiff1D.hpp"
#include "AdvDiff1D_POD.hpp"

// Main function.
int
main(int argc, char * argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const unsigned int               mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  dealii::ConditionalOStream pcout(std::cout, mpi_rank == 0);

  const unsigned int N = 119;
  // const unsigned int N = 60;
  const unsigned int r = 1;

  const double T         = 0.05;
  // const double deltat    = 1e-5; // 5000 steps
  // const double deltat    = 1e-4; // 500 steps
  const double deltat    = 5e-4; // 50 steps
  // const double deltat    = 2.5e-3;

  // const double theta  = 1.0; // Implicit Euler
  const double theta  = 0.0; // Explicit Euler

  const unsigned int sample_every = 1;


  // RISOLUZIONE DEL PROBLEMA DI DIFFUSIONE E TRASPORTO
  pcout << "===============================================" << std::endl;
  pcout << "Run FOM and collect snapshots" << std::endl;

  AdvDiff problem(N, r, T, deltat, theta, sample_every);    

  problem.setup();
  problem.solve();

  // COSTRUZIONE SNAPSHOT MATRIX IN EIGEN IN MODO DA POTER APPLICARE SVD
  // Now the snapshot_matrix, defined with standard library, is required to fit in snapshots, defined in Eigen, since the SVD
  // method is implemented in Eigen.
  size_t snapshot_length = problem.snapshot_matrix.size();
  size_t time_steps = problem.snapshot_matrix[0].size();
  Mat_m snapshots = Mat_m::Zero(snapshot_length, time_steps);
  for (size_t i=0; i<snapshots.rows(); i++)
    for (size_t j=0; j<snapshots.cols(); j++)
      snapshots(i, j) = problem.snapshot_matrix[i][j];

  pcout << "\nCheck dimensions of snapshots: "
            << snapshots.rows() << " * " << snapshots.cols() << std::endl << std::endl;

  pcout << "  Check snapshots + sol coincidente:" << std::endl;
  pcout << snapshots(0, time_steps-1) << std::endl;
  pcout << problem.solution(0) << std::endl;
  pcout << snapshots(1, time_steps-1) << std::endl;
  pcout << problem.solution(1) << std::endl;
  pcout << snapshots(17, time_steps-1) << std::endl;
  pcout << problem.solution(17) << std::endl;

  // CALCOLO DI U APPLICANDO SVD A SNAPSHOT MATRIX
  pcout << "===============================================" << std::endl;
  pcout << "Compute POD modes" << std::endl;
  // Initialize the vector sigma to store the singular values
  int rank = std::min(snapshots.rows(), snapshots.cols()); // rank così cioè su tutto
  Vec_v sigma = Vec_v::Zero(rank);

  // Initialize the other inputs required by the SVD method, Note that the SVD method returns sigma as a vector, then it has
  // to be converted into a diagonal matrix
  Mat_m U = Mat_m::Zero(snapshots.rows(), snapshots.rows());
  Mat_m V = Mat_m::Zero(snapshots.cols(), snapshots.cols());

  SVD(snapshots, sigma, U, V, rank);
  pcout << "\nCheck dimensions of U:     " << U.rows() << " * " << U.cols() << endl;
  pcout << "Check dimensions of sigma: " << sigma.size() << endl;
  pcout << "Check dimensions of V:     " << V.rows() << " * " << V.cols() << endl;


  // pcout << " Check U" << U << std::endl;

  // COSTRUZIONE DELLA MATRICE DI TRASFORMAZIONE A PARTIRE DA U E ROM_SIZES
  // POI EVENTUALMENTE CHIAMA V -> Z e modes -> V
  pcout << "===============================================" << std::endl;
  pcout << "Construct and run ROM" << std::endl;

  // const double deltat_rom = 1e-3; // CAMBIA
  // const double deltat_rom = 3.01204e-4; // CAMBIA
  // ORA PROVA A TENERE STESSA deltat
  // std::vector<size_t> rom_sizes = {2, 4, 6}; // CAMBIA
  std::vector<size_t> rom_sizes = {5, 10, 100};
  std::vector<std::vector<double>> modes;
  // Mat_m modes;
  Mat_m approximations = Mat_m::Zero(snapshots.rows(), rom_sizes.size());

  for (size_t i=0; i<rom_sizes.size(); i++) {
  // for (size_t i=0; i<1; i++) {
    pcout << "  Creating ROM for " << rom_sizes[i] << " modes" << std::endl;

    // POD hpp vuole versione modes std, non Eigen
    modes.resize(U.rows());
    for(auto &row : modes)
      row.resize(rom_sizes[i], 0.0);
    for (size_t j=0; j<U.rows(); j++)
      for (size_t k=0; k<rom_sizes[i]; k++)
        modes[j][k] = U(j, k);
    // modes.resize(U.rows(), rom_sizes[i]);
    // for (unsigned int j=0; j<rom_sizes[i]; j++)
    //   modes.col(j) = U.col(j);
    // modes = U.leftCols(rom_sizes[i]);

    // pcout << "  Check modes" << modes << std::endl;

    // PROIEZIONE DELLO STATO FOM INIZIALE SULLA BASE ROM
    // Projection of the initial FOM state on the ROM basis
    // Vec_v rom_state = modes.transpose() * snapshots.col(0);

    // pcout << "  Check rom_state" << rom_state(0) << std::endl;
    // if (rom_state.size() == rom_sizes[i])
    //   pcout << "  Check dimensions of rom_state: " << rom_state.size() << std::endl;
    // else
    //   std::cerr << "  Error in computing rom_state" << std::endl;
    
  
    // DEFINIZIONE DELLO STESSO PROBLEMA CON deltat_rom IN MODO DA OTTENERE system_rhs –> PEZZO SICURAMENTE DA SISTEMARE,
    // PER ORA HO RIUTILIZZATO LA STESSA CLASSE AdvDiff IN MODO IMPROPRIO PERCHÉ IL PROBLEMA COSÌ VIENE NUOVAMENTE RISOLTO
    //PROVA a farti la classe per farti dare rhs e prenderlo e usarlo di qui
    // const unsigned int sample_every_pod = 166;



    AdvDiffPOD problemPOD(N, r, T, deltat, theta, modes); // qui sample every non serve
    
    problemPOD.setup();
    problemPOD.solve_reduced();

    // PASSAGGIO NON ANCORA CAPITO
    // INITIAL CONDITION rom_initial_state
    // ho time steps e delta_t_rom
    // unsigned int rom_time_steps = floor(T/deltat_rom);
    // double       time = 0.0;
    // for(unsigned int time_step=1; time_step<=rom_time_steps; time_step++) // FAI PARTIRE DA 1 ? perchè zero già fuori ??? O FARE <= rom_time_steps???
    // {
    //   // time += deltat_rom;
    //   // pcout << "  Check time" << time << std::endl;
    //   // pcout << "  Check romstate initial" << rom_state(0) << std::endl;
    //   // pcout << "  Check rhs" << problemPOD.system_rhs_matrix[time_step][2] << std::endl;

    //   if (problemPOD.system_rhs_matrix.size() == 0)
    //   {
    //     pcout << "Error: system_rhs_matrix is empty." << std::endl;
    //     MPI_Abort(MPI_COMM_WORLD, 1);
    //   }
    //   Vec_v fom_rhs = Vec_v::Zero(problemPOD.system_rhs_matrix.size()); // SISTEMA DIMENSIONI
    //   for(unsigned int j=0; j<problemPOD.system_rhs_matrix.size(); j++)
    //     fom_rhs(j) = problemPOD.system_rhs_matrix[time_step][j];
    //   Vec_v rom_rhs = Vec_v::Zero(rom_sizes[i]);
    //   rom_rhs = modes.transpose() * fom_rhs;
    //   // rom_rhs = -rom_state;
    //   rom_state += deltat_rom * rom_rhs;
    //   pcout << "  Check romstate final" << rom_state(0) << std::endl;
    // }


    // The final ROM state corresponds to the solution of the ROM problem.
    // Vec_v rom_final_state = Vec_v::Zero(rom_sizes[i]);
    // for(size_t j=0; j<rom_sizes[i]; j++)
    //   rom_final_state(j) = 0.0;

    // RICOSTRUZIONE FOM E CALCOLO DELL'ERRORE
    // Reconstruction of the FOM state from the final ROM state
    // Vec_v fom_state = modes * rom_state; // (U.rows() * rom_sizes[i]) * (rom_sizes[i] * 1)

    // IDEM QUI POD hpp VUOLE VERSIONE SENZA EIGEN
    // Vec_v rom_state = Vec_v::Zero(rom_sizes[i]);
    // for(size_t j=0; j<rom_sizes[i]; j++)
    //   rom_state(j) = problemPOD.solution[j]; // CONTROLLA
    // Vec_v fom_state = modes * rom_state; // (U.rows() * rom_sizes[i]) * (rom_sizes[i] * 1)
    std::vector<double> rom_state;
    rom_state.resize(rom_sizes[i]);
      for(size_t j=0; j<rom_sizes[i]; j++)
        rom_state[j] = problemPOD.reduced_solution[j]; // CONTROLLA
    Vec_v fom_state = Vec_v::Zero(U.rows()); // SISTEMA DIMENSIONI
    for (size_t j=0; j<U.rows(); j++)
      for (size_t k=0; k<rom_sizes[i]; k++)
        fom_state(j) += modes[j][k] * rom_state[k]; // PRODOTTO MATRICE VETTORE MAGARI ESISTE

    // The current fom_state is stored in approximations matrix.
    approximations.col(i) = fom_state;

    // Compute the relative l2 error
    double fom_solution_norm = (snapshots.col(time_steps-1)).norm(); // HAI TOLTO time_steps -1
    double err = (snapshots.col(time_steps-1) - fom_state).norm();
    pcout << " fom_solution" << snapshots(0, time_steps-1) << std::endl;
    pcout << " rom_solution riproiettata" << fom_state(0) << std::endl;
    pcout << " fom_solution" << snapshots(1, time_steps-1) << std::endl;
    pcout << " rom_solution riproiettata" << fom_state(1) << std::endl;
    pcout << " fom_solution" << snapshots(2, time_steps-1) << std::endl;
    pcout << " rom_solution riproiettata" << fom_state(2) << std::endl;
    pcout << "  With " << rom_sizes[i] << " modes, final relative l2 error: " << err/fom_solution_norm << std::endl;

    // Clear the modes matrix for the next iteration
    modes.resize(0);
    modes[0].resize(0, 0.0);
  }

  return 0;
}
