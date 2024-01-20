#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "Poisson2D_snapshot.hpp"


// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  const std::string mesh_file_name = "../mesh/gmsh/mesh-square-h0.100000.msh";
  const unsigned int r = 1;

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

  double prm_diffusion_coefficient_min = 0.1;
  double prm_diffusion_coefficient_max = 10.0;
  const unsigned int ns = 50;
  const unsigned int Nh = 121;

  std::vector<double> prm_diffusion_coefficient;
  prm_diffusion_coefficient.resize(ns); 

  std::vector<double> snapshot_array;
  snapshot_array.resize(Nh);
  
  Eigen::MatrixXd snapshot_matrix = Eigen::MatrixXd::Zero(Nh, ns);

  std::cout << "===============================================" << std::endl;
  std::cout << "Compute snapshots ..." << std::endl;

  for (unsigned int i=0; i<ns; i++) {

    std::cout << "  Computing snapshot " << i+1 << " out of " << ns << std::endl;
    prm_diffusion_coefficient[i] = (prm_diffusion_coefficient_min +
                                    i*(prm_diffusion_coefficient_max - prm_diffusion_coefficient_min)/ns);
    std::cout << "Check prm_diffusion_coefficient = " << prm_diffusion_coefficient[i] << std::endl;                                   

    Poisson2D problem(mesh_file_name, r, snapshot_array, prm_diffusion_coefficient[i]);

    problem.setup();
    problem.assemble();
    problem.solve();
    problem.output();

    for (unsigned int j=0; j<Nh; j++)
      snapshot_matrix(j, i) = snapshot_array[j];
    // snapshot_matrix.col(i) = Eigen::Map<Eigen::VectorXd>(snapshot_array.data(), Nh);
  }

  for (int i=0; i<10; i++)
    std::cout << "Check the snapshot matrix:" << std::endl << snapshot_matrix.col(i) << std::endl << std::endl;

  // Export the snapshot matrix

  return 0;
}
