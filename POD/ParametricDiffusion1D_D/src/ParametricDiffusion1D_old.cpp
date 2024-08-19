#include <deal.II/base/convergence_table.h>
#include <deal.II/base/conditional_ostream.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <map>
// #include <Eigen/Dense>

#include "SVD.hpp"
#include "AdvDiff1D.hpp"
// #include "AdvDiff1D_POD.hpp"

// Main function.
int
main(int argc, char * argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const unsigned int               mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  dealii::ConditionalOStream pcout(std::cout, mpi_rank == 0);

  const unsigned int N = 120;
  // const unsigned int N = 60;
  const unsigned int r = 1;

  const double T         = 0.05;
  // const double deltat    = 1e-5; // 5000 steps
  const double deltat    = 1e-3; // 50 steps
  // const double deltat    = 2.5e-3; // 50 steps

  // const double theta  = 1.0; // Implicit Euler
  const double theta  = 0.0; // Explicit Euler

  // std::vector<double> errors_L2;
  // std::vector<double> errors_H1;

  // ----------------------------------------------------------------------------------------------------------------------------
  // MatLab code from build_PODbased_ROM.m
  // FOM is struct containing the full order model

  // SAMPLE_GRID is a matrix of dimension number of snaphots x FOM.P
  // containing the coordinates of the parameter points where to
  // compute the snapshots

  // Ns = size(sample_grid,1);
  // S  = zeros(length(FOM.MESH.internal_dof), Ns);

  // %% Compute snapshots
  // fprintf('\n Compute snapshots ...')
  // parfor i = 1 : Ns

  //     fprintf('\n   Computing snapshot %d out of %d', i, Ns);
  //     uh      = solve_HFsystem(FOM, sample_grid(i,:));
  //     S(:, i) = uh(FOM.MESH.internal_dof);

  // end
  // ----------------------------------------------------------------------------------------------------------------------------

  // INTERMEDIATE SNAPSHOT MATRIX
  // NOTE: boundary_dofs_idx_new will store the indices of the boundary DOFs computed in Diffusion::setup()
  // but that is of type std::vector<types::global_dof_index>, here we use std::vector<unsigned int>
  // std::vector<unsigned int> boundary_dofs_idx_int;
  // std::vector<unsigned int> boundary_dofs_idx_old;
  // Vec_v boundary_dofs_idx_v;

  // NOTE: snapshot_array will store the solution computed in Diffusion::solve() but that solution is of type
  // Vector<double>, here we use std::vector<double>
  // std::vector<double> snapshot_array;

  // Compute only one snapshot to understand better what entries in snapshot_array correspond to internal DOFs
  // const unsigned int ns = 2; 
  // const unsigned int ns = 50;
  // NOTE: the intermediate number of rows of the snapshot matrix will be n_dofs,
  // the final number of rows will be n_dofs - n_boundary_dofs
  // unsigned int rows_default = 1; // default value that will be set to n_dofs at the first iteration of the for loop
  // Mat_m snapshot_matrix = Mat_m::Zero(rows_default, ns);

  // double prm_diffusion_coefficient_min = 0.0;
  // double prm_diffusion_coefficient_max = 1.0;
  // std::vector<double> prm_diffusion_coefficient;
  // prm_diffusion_coefficient.resize(ns); 

  pcout << "===============================================" << std::endl;
  pcout << "Run FOM and collect snapshots" << std::endl;

  // for (unsigned int i=0; i<ns; i++) {

    // pcout << "  Computing snapshot " << i+1 << " out of " << ns << std::endl;
    // prm_diffusion_coefficient[i] = (prm_diffusion_coefficient_min +
    //                                 i*(prm_diffusion_coefficient_max - prm_diffusion_coefficient_min)/(ns-1));
    // pcout << "  Check prm_diffusion_coefficient = " << prm_diffusion_coefficient[i] << std::endl;                                   

    // NOTE: boundary_dofs_idx_int and snapshot_array are public members, the other arguments are protected members 
    AdvDiff problem(N, r, T, deltat, theta);    

    problem.setup();
    problem.solve();

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

      // pcout << "  Check snapshots:" << std::endl << snapshots << std::endl << std::endl;

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

      pcout << "===============================================" << std::endl;
      pcout << "Construct and run ROM" << std::endl;

      const double deltat_rom = 1e-3; // CAMBIA
      std::vector<size_t> rom_sizes = {2, 4, 6};
      // std::vector<std::vector<double>> modes;
      Mat_m modes;
      Mat_m approximations = Mat_m::Zero(snapshots.rows(), rom_sizes.size());

      // for (size_t i=0; i<rom_sizes.size(); i++) {
      for (size_t i=0; i<1; i++) {
        pcout << "  Creating ROM for " << rom_sizes[i] << " modes" << std::endl;
        // modes.resize(U.rows());
        // for(auto &row : modes)
        //   row.resize(rom_sizes[i], 0.0);
        // for (size_t j=0; j<U.rows(); j++)
        //   for (size_t k=0; k<rom_sizes[i]; k++)
        //     modes[j][k] = U(j, k);
        modes.resize(U.rows(), rom_sizes[i]);
        for (unsigned int j=0; j<rom_sizes[i]; j++)
          modes.col(j) = U.col(j);
        // modes = U.leftCols(rom_sizes[i]);

        // pcout << "  Check modes" << modes << std::endl;

        // Projection of the initial FOM state on the ROM basis
        Vec_v rom_state = modes.transpose() * snapshots.col(0);

        pcout << "  Check rom_state" << rom_state(0) << std::endl;
        if (rom_state.size() == rom_sizes[i])
          pcout << "  Check dimensions of rom_state: " << rom_state.size() << std::endl;
        else
          std::cerr << "  Error in computing rom_state" << std::endl;
        
        // std::vector<double> initial_state(rom_sizes[i]);
        // for(size_t j=0; j<rom_sizes[i]; j++)
        //   initial_state[i] = rom_initial_state(j);

        // USARE rom_state COME CONDIZIONE INIZIALE PER IL ROM? vedi nota ?????????
        // USARE romsize come N???????

        // MODES SAREBBE V SUL LIBRO
        // AdvDiffPOD problemPOD(N, r, T, deltat_rom, theta, modes);    

        // problemPOD.setup();
        // problemPOD.solve();

        // RISOLUZIONE
        // ? fom reference state?
  //         # create problem
  // scheme = ode.stepscheme.ForwardEuler
  // problem = rom.galerkin.DefaultExplicitProblem(scheme, fomObj, linearDecoder, romState, fomReferenceState)

  // # create object to monitor the romState at every iteration
  // myObs = RomStateObserver()
  // # solve problem
  // ode.advance_n_steps_and_observe(problem, romState, 0., dt, nsteps, myObs)


      // Time-stepping parameters
    // double dt = 0.01;
    // int nsteps = 100;

    // // Time-stepping loop
    // for (int step = 0; step < nsteps; ++step) {
    //     // Compute the derivative in ROM space (this is problem-specific)
    //     // Here, we assume a simple linear model for illustration
    //     Eigen::VectorXd romDerivative = -romState; // Replace with your actual derivative computation

    //     // Advance the ROM state
    //     romState = advanceOneStep(romState, romDerivative, dt);
    // }

        // Set the reduced order problem with Forward Euler

      
        //PROVA a farti la classe per farti dare rhs e prenderlo e usarlo di qui
        AdvDiff problemPOD(N, r, T, deltat_rom, theta);
        
        problemPOD.setup();
        problemPOD.solve();


        // INITIAL CONDITION rom_initial_state
        // ho time steps e delta_t_rom
        unsigned int rom_time_steps = T/deltat_rom;
        double       time = 0.0;
        for(unsigned int time_step=1; time_step<=rom_time_steps; time_step++) // FAI PARTIRE DA 1 ? perchè zero già fuori
        {
          time += deltat_rom;
          // pcout << "  Check time" << time << std::endl;
          // pcout << "  Check romstate initial" << rom_state(0) << std::endl;
          // pcout << "  Check rhs" << problemPOD.system_rhs_matrix[time_step][2] << std::endl;

          if (problemPOD.system_rhs_matrix.size() == 0)
          {
            pcout << "Error: system_rhs_matrix is empty." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
          }
          Vec_v fom_rhs = Vec_v::Zero(problemPOD.system_rhs_matrix.size()); // SISTEMA DIMENSIONI
          for(unsigned int j=0; j<problemPOD.system_rhs_matrix.size(); j++)
            fom_rhs(j) = problemPOD.system_rhs_matrix[time_step][j];
          Vec_v rom_rhs = Vec_v::Zero(rom_sizes[i]);
          rom_rhs = modes.transpose() * fom_rhs;
          // rom_rhs = -rom_state;
          rom_state += deltat_rom * rom_rhs;
          pcout << "  Check romstate final" << rom_state(0) << std::endl;
        }


        // The final ROM state corresponds to the solution of the ROM problem.
        // Vec_v rom_final_state = Vec_v::Zero(rom_sizes[i]);
        // for(size_t j=0; j<rom_sizes[i]; j++)
        //   rom_final_state(j) = 0.0;

        // Reconstruction of the FOM state from the final ROM state
        Vec_v fom_state = modes * rom_state; // (U.rows() * rom_sizes[i]) * (rom_sizes[i] * 1)

        // The current fom_state is stored in approximations matrix.
        approximations.col(i) = fom_state;

        // Compute the relative l2 error
        double fom_solution_norm = (snapshots.col(time_steps-1)).norm();
        double err = (snapshots.col(time_steps-1) - fom_state).norm();
        pcout << "  With " << rom_sizes[i] << " modes, final relative l2 error: " << err/fom_solution_norm << std::endl;

        // Clear the modes matrix for the next iteration
        modes.resize(0, 0);
      }

      // fomReferenceState per te è ?
      // fomInitialState per te è snapshots[:, 0)


    // errors_L2.push_back(problem.compute_error(VectorTools::L2_norm));
    // errors_H1.push_back(problem.compute_error(VectorTools::H1_norm));

    // At the first iteration set the intermediate number of iterations as n_dofs = snapshot_array.size()
    // if (i == 0)
    //   snapshot_matrix.conservativeResize(problem.snapshot_array.size(), Eigen::NoChange);
    // snapshot_matrix.col(i) = Eigen::Map<Vec_v>(problem.snapshot_array.data(), problem.snapshot_array.size());

    // Check if the boundary DOFs are always the same
    // Otherwise the snapshot matrix can't be modified by erasing rows corresponding to boundary DOFs indices
    // if (problem.boundary_dofs_idx_int == boundary_dofs_idx_old)
    //   pcout << "  Check on boundary DOFs indices passed." << std::endl;
    // else
    //   pcout << "  Check on boundary DOFs indices failed." << std::endl;
    // pcout << "===============================================" << std::endl;
    // pcout << "===============================================" << std::endl << std::endl;

    // problem.snapshot_array.clear();
    // boundary_dofs_idx_old.assign(problem.boundary_dofs_idx_int.begin(), problem.boundary_dofs_idx_int.end());
    // problem.boundary_dofs_idx_int.clear();
  // }

  // for (int i=0; i<2; i++)
  //   pcout << "Check the snapshot matrix:" << std::endl << snapshot_matrix.col(i) << std::endl << std::endl;

  // ----------------------------------------------------------------------------------------------------------------------------

  // FROM THE INTERMEDIATE TO THE FINAL SNAPSHOT MATRIX
  // Since we want snapshot to contain only the solution of the internal DOFs, we want to erase the rows of the snapshot matrix
  // corresponding to the boundary DOFs indices 
  // pcout << "  Check intermediate dimensions of the snapshot_matrix: "
  //           << snapshot_matrix.rows() << " * " << snapshot_matrix.cols() << std::endl << std::endl;

  // const unsigned int Nh = snapshot_matrix.rows() - boundary_dofs_idx_old.size(); // n_dofs - n_boundary_dofs

  // Create a map to distinguish between internal and boundary DOFs
  // std::map<unsigned int, bool> boundary_dofs_idx_map;
  // bool b = false;
  // unsigned int j = 0; // auxiliary index for traversing boundary_dofs_idx_old
  // for (unsigned int key=0; key<snapshot_matrix.rows(); key++) {
  //   if (boundary_dofs_idx_old[j] == key) {
  //     b = true;
  //     j++;
  //   }
  //   boundary_dofs_idx_map.insert(std::make_pair(key, b));
  //   b = false;
  // }

  // Check the creation of boundary_dofs_idx_map by traversing it with an iterator and printing "key -> b"
  // std::map<unsigned int, bool>::iterator it;
  // pcout << "  Check boundary_dofs_idx_map:" << std::endl;
  // for (it=boundary_dofs_idx_map.begin(); it!=boundary_dofs_idx_map.end(); it++)
  //   pcout << "    " << it->first << " -> " << it->second << std::endl;
  // pcout << std::endl;

  // Erase the rows corresponding to the boundary DOFs indices: starting from boundary_dofs_idx_map keep "false"
  // https://stackoverflow.com/questions/13290395/how-to-remove-a-certain-row-or-column-while-using-eigen-library-c
  // https://eigen.tuxfamily.org/dox/group__TutorialSlicingIndexing.html
  // std::map<unsigned int, bool>::iterator it;
  // Vec_v internal_dofs_idx; // indices to keep
  // j = 0; // auxiliary index for traversing internal_dofs_idx
  // for (it=boundary_dofs_idx_map.begin(); it!=boundary_dofs_idx_map.end(); it++) {
  //   if (it->second == false) {
  //     internal_dofs_idx(j) = it->first;
  //     j++;
  //   }
  // }
  // pcout << "  Check internal_dofs_idx:" << std::endl << internal_dofs_idx << std::endl << std::endl;

  // snapshot_matrix = snapshot_matrix(internal_dofs_idx, Eigen::placeholders::all);

  // PROVVISORIO
  // Mat_m snapshot_matrix_reduced = Mat_m::Zero(internal_dofs_idx.size(), ns);
  // j = 0; // auxiliary index for traversing internal_dofs_idx
  // for (unsigned int i=0; i<snapshot_matrix.rows(); i++) {
  //   if (internal_dofs_idx(j) == i) {
  //     snapshot_matrix_reduced.row(j) = snapshot_matrix.row(i);
  //     j++;
  //   }
  // }

  // pcout << "  Expected dimensions of the snapshot_matrix:           "
  //           << Nh << " * " << ns << std::endl << std::endl;
  // pcout << "  Check final dimensions of the snapshot_matrix:        "
  //           << snapshots.rows() << " * " << snapshots.cols() << std::endl << std::endl;

  // A QUESTO PUNTO SVD?

  // const int r = ns-10;
  // const double tol = 1e-4;
  // POD pod(snapshot_matrix, r, tol);
  // POD pod(snapshot_matrix_reduced, r, tol);

  // E POI?
  // Ottieni W, ma cosa puoi farcene?

  // Print the errors and estimate the convergence order.
  // if (mpi_rank == 0)
  // {
  //   pcout << "==============================================="
  //             << std::endl;
  //   ConvergenceTable table;
  //   std::ofstream convergence_file("convergence.csv");
  //   convergence_file << "h,eL2,eH1" << std::endl;

  //   for (unsigned int i = 0; i < h_vals.size(); ++i)
  //     {
  //       table.add_value("h", h_vals[i]);
  //       table.add_value("L2", errors_L2[i]);
  //       table.add_value("H1", errors_H1[i]);
        
  //       convergence_file << h_vals[i] << "," << errors_L2[i] << ","
  //                         << errors_H1[i] << std::endl;

  //       pcout << std::scientific << "h = " << std::setw(4)
  //                 << std::setprecision(2) << h_vals[i];

  //       pcout << std::scientific << " | eL2 = " << errors_L2[i];

  //       // Estimate the convergence order.
  //       if (i > 0)
  //         {
  //           const double p =
  //             std::log(errors_L2[i] / errors_L2[i - 1]) /
  //             std::log(h_vals[i] / h_vals[i - 1]);

  //           pcout << " (" << std::fixed << std::setprecision(2)
  //                     << std::setw(4) << p << ")";
  //         }
  //       else
  //         pcout << " (  - )";

  //       pcout << std::scientific << " | eH1 = " << errors_H1[i];

  //       // Estimate the convergence order.
  //       if (i > 0)
  //         {
  //           const double p =
  //             std::log(errors_H1[i] / errors_H1[i - 1]) /
  //             std::log(h_vals[i] / h_vals[i - 1]);

  //           pcout << " (" << std::fixed << std::setprecision(2)
  //                     << std::setw(4) << p << ")";
  //         }
  //       else
  //         pcout << " (  - )";

  //       pcout << "\n";
  //     }

  //   table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
  //   table.set_scientific("L2", true);
  //   table.set_scientific("H1", true);
  //   table.write_text(pcout);
  // }

  return 0;
}
