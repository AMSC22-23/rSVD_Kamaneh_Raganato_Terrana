#include "Heat.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string  mesh_file_name = "../mesh/mesh-cube-20.msh";
  const unsigned int degree         = 1;

  const double T      = 1.0;
  const double deltat = 0.05;

  // PUNTO 1.4
  const double theta  = 1.0; // Implicit Euler

  // PUNTO 1.5
  // const double theta  = 0.0; // Explicit Euler

  Heat problem(mesh_file_name, degree, T, deltat, theta);

  problem.setup();
  problem.solve();

  return 0;
}