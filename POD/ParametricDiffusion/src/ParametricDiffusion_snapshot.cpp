#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <map>
// #include <Eigen/Dense>

#include "POD.hpp"
#include "Poisson2D_snapshot.hpp"

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  const std::string mesh_file_name = "../mesh/gmsh/mesh-square-h0.100000.msh";
  const unsigned int p = 1;

  // -----------------------------------------------------------------------------------------------
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
  // -----------------------------------------------------------------------------------------------

  // NOTE: boundary_dofs_idx_new will store the indices of the boundary DOFs computed in Poisson2D::setup()
  // but that is of type std::vector<types::global_dof_index>, here we use std::vector<unsigned int>
  std::vector<unsigned int> boundary_dofs_idx_int;
  std::vector<unsigned int> boundary_dofs_idx_old;
  Vec_v boundary_dofs_idx_v;

  // NOTE: snapshot_array will store the solution computed in Poisson2D::solve() but that solution is of type
  // Vector<double>, here we use std::vector<double>
  std::vector<double> snapshot_array;

  // Compute only one snapshot to understand better what entries in snapshot_array correspond to internal DOFs
  // const unsigned int ns = 2; 
  const unsigned int ns = 50;
  // NOTE: the intermediate number of rows of the snapshot matrix will be n_dofs,
  // the final number of rows will be n_dofs - n_boundary_dofs
  unsigned int rows_default = 1; // default value that will be set to n_dofs at the first iteration of the for loop
  Mat_m snapshot_matrix = Mat_m::Zero(rows_default, ns);

  double prm_diffusion_coefficient_min = 0.1;
  double prm_diffusion_coefficient_max = 10.0;
  std::vector<double> prm_diffusion_coefficient;
  prm_diffusion_coefficient.resize(ns); 

  std::cout << "===============================================" << std::endl;
  std::cout << "Compute snapshots ..." << std::endl;

  for (unsigned int i=0; i<ns; i++) {

    std::cout << "  Computing snapshot " << i+1 << " out of " << ns << std::endl;
    prm_diffusion_coefficient[i] = (prm_diffusion_coefficient_min +
                                    i*(prm_diffusion_coefficient_max - prm_diffusion_coefficient_min)/(ns-1));
    std::cout << "  Check prm_diffusion_coefficient = " << prm_diffusion_coefficient[i] << std::endl;                                   

    Poisson2D problem(mesh_file_name, p, boundary_dofs_idx_int, snapshot_array, prm_diffusion_coefficient[i]);

    problem.setup();
    problem.assemble();
    problem.solve();
    // problem.output(); COMMENTO PER NON APPESANTIRE

    // At the first iteration set the intermediate number of iterations as n_dofs = snapshot_array.size()
    if (i == 0)
      snapshot_matrix.conservativeResize(snapshot_array.size(), Eigen::NoChange);
    snapshot_matrix.col(i) = Eigen::Map<Vec_v>(snapshot_array.data(), snapshot_array.size());

    // Check if the boundary DOFs are always the same
    // Otherwise the snapshot matrix can't be modified by erasing rows corresponding to boundary DOFs indices
    if (boundary_dofs_idx_int == boundary_dofs_idx_old)
      std::cout << "  Check on boundary DOFs indices passed." << std::endl;
    else
      std::cout << "  Check on boundary DOFs indices failed." << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << "===============================================" << std::endl << std::endl;

    snapshot_array.clear();

    boundary_dofs_idx_old = boundary_dofs_idx_int;
    boundary_dofs_idx_int.clear();
  }

  // FAI MAPPA

  // for (int i=0; i<2; i++)
  //   std::cout << "Check the snapshot matrix:" << std::endl << snapshot_matrix.col(i) << std::endl << std::endl;

  // CAPIRE Nh inizia provando ad farlo ritornare
  // RISOLVERE IL PROBLEMA DEI DOF sul bordo
  // Mi sa che non sono solo gli zeri ma ce ne sono altri
  // far ritornare al problem Nh e indici dei dof interni e/o esterni

  // Export the snapshot matrix
  // PER ORA INIZIA A USARLA DENTRO
  // POI CAMBIA ns COME SENCONDO ARGOMENTO
  
  // FAI MAPPA
  // STA COSA SI POTREBBE FARE DIRETTAMENTE DENTRO A solve




  // Since we want snapshot to contain only the solution of the internal DOFs, we want to erase the rows of the snapshot matrix
  // corresponding to the boundary DOFs indices 
  // std::cout << "  Check intermediate dimensions of the snapshot_matrix: "
  //           << snapshot_matrix.rows() << " * " << snapshot_matrix.cols() << std::endl << std::endl;

  // const unsigned int Nh = snapshot_array.size() - boundary_dofs_idx_old.size(); // n_dofs - n_boundary_dofs

  // std::map<unsigned int, bool> boundary_dofs_idx_map;
  // std::map<unsigned int, bool>::iterator it;
  // bool b = false;
  // for (it=snapshot_array.begin(); it!=snapshot_array.end(); it++) {
  //   if (boundary_dofs_idx_old[it->first] == it->first)
  //     b = true;
  //   boundary_dofs_idx_map.insert(std::make_pair(it->first, b));
  //   b = false;
  // }

  // std::cout << "  Check boundary_dofs_idx_map:" << std::endl;
  // for (it=snapshot_array.begin(); it!=snapshot_array.end(); it++)
  //   std::cout << "  " << it->first << " -> " << it->second << std::endl;
  // std::cout << std::endl;





  std::cout << "  Expected dimensions of the snapshot_matrix:           "
            << Nh << " * " << ns << std::endl << std::endl;
  std::cout << "  Check final dimensions of the snapshot_matrix:        "
            << snapshot_matrix.rows() << " * " << snapshot_matrix.cols() << std::endl << std::endl;

  const int r = ns-10;
  const double tol = 1e-4;
  POD pod(snapshot_matrix, r, tol);

  return 0;
}
