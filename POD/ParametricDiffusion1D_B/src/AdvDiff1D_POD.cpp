#include "AdvDiff1D_POD.hpp"

void
AdvDiffPOD::setup()
{
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;
    GridGenerator::subdivided_hyper_cube(mesh_serial, N + 1, 0.0, 1.0, true);
    pcout << "  Number of elements = " << mesh.n_active_cells()
              << std::endl;

    // Write the mesh to file.
    const std::string mesh_file_name = "mesh-" + std::to_string(N + 1) + ".vtk";
    GridOut           grid_out;
    std::ofstream     grid_out_file(mesh_file_name);
    grid_out.write_vtk(mesh_serial, grid_out_file);
    pcout << "  Mesh saved to " << mesh_file_name << std::endl;

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-------------------------------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_Q<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGauss<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  }

  pcout << "-------------------------------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-------------------------------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    pcout << "  Initializing the matrices" << std::endl;
    mass_matrix.reinit(sparsity);
    stiffness_matrix.reinit(sparsity);
    lhs_matrix.reinit(sparsity);
    rhs_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  }
}

void
AdvDiffPOD::setup_reduced()
{
  // Create the mesh.
  {
    pcout << "Initializing the reduced mesh" << std::endl;

    Triangulation<dim> mesh_serial_r;
    GridGenerator::subdivided_hyper_cube(mesh_serial_r, modes[0].size(), 0.0, 1.0, true);
    // CAPIRE
        // QUI HAI CAMBIATO IN MODO CHE LA MATRICE NON ABBIA UN ELEMENTO IN PIU
    pcout << "  Number of elements = " << mesh_r.n_active_cells()
              << std::endl;

    // Write the mesh to file.
    const std::string mesh_file_name_r = "mesh_r-" + std::to_string(modes[0].size()) + ".vtk";
    GridOut           grid_out_r;
    std::ofstream     grid_out_file_r(mesh_file_name_r);
    grid_out_r.write_vtk(mesh_serial_r, grid_out_file_r);
    pcout << "  Mesh saved to " << mesh_file_name_r << std::endl;

    GridTools::partition_triangulation(mpi_size, mesh_serial_r);
    const auto construction_data_r = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial_r, MPI_COMM_WORLD);
    mesh_r.create_triangulation(construction_data_r);

    pcout << "  Number of elements = " << mesh_r.n_global_active_cells()
          << std::endl;
  }

  pcout << "-------------------------------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe_r = std::make_unique<FE_Q<dim>>(r);

    pcout << "  Degree                     = " << fe_r->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe_r->dofs_per_cell
          << std::endl;

    quadrature_r = std::make_unique<QGauss<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature_r->size()
          << std::endl;
  }

  pcout << "-------------------------------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler_r.reinit(mesh_r);
    dof_handler_r.distribute_dofs(*fe_r);

    locally_owned_dofs_r = dof_handler_r.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler_r, locally_relevant_dofs_r);

    pcout << "  Number of DoFs = " << dof_handler_r.n_dofs() << std::endl;
  }

  pcout << "-------------------------------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    TrilinosWrappers::SparsityPattern sparsity_r(locally_owned_dofs_r,
                                                 MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler_r, sparsity_r);
    sparsity_r.compress();

    // Note that in the reduced order system there's no need to assemble the mass matrix, stiffness matrix and right-hand side
    // matrix. Since the reduced left-hand side matrix and the reduced right-hand side vector are obtained by projection.
    pcout << "  Initializing the matrices" << std::endl;
    reduced_system_lhs.reinit(sparsity_r);

    pcout << "  Initializing the system right-hand side" << std::endl;
    reduced_system_rhs.reinit(locally_owned_dofs_r, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    reduced_solution_owned.reinit(locally_owned_dofs_r, MPI_COMM_WORLD);
    reduced_solution.reinit(locally_owned_dofs_r, locally_relevant_dofs_r, MPI_COMM_WORLD);
  }
}

void
AdvDiffPOD::assemble_matrices()
{
  pcout << "===================================================================" << std::endl;
  pcout << "Assembling the system matrices" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  mass_matrix      = 0.0;
  stiffness_matrix = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_mass_matrix      = 0.0;
      cell_stiffness_matrix = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Evaluate coefficients on this quadrature node.
          const double mu_loc   = mu.value(fe_values.quadrature_point(q));

          // Evaluate the transport term on this quadrature node.
          Vector<double> beta_vector(dim);
          beta.vector_value(fe_values.quadrature_point(q), beta_vector);

          // Convert the transport term to a tensor to use it with scalar_product.
          Tensor<1, dim> beta_tensor;
          for (unsigned int i = 0; i < dim; ++i)
            beta_tensor[i] = beta_vector[i];
          
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_mass_matrix(i, j) += fe_values.shape_value(i, q) *
                                            fe_values.shape_value(j, q) /
                                            deltat * fe_values.JxW(q);

                  // Diffusion term.
                  cell_stiffness_matrix(i, j) +=
                    mu_loc * fe_values.shape_grad(i, q) *
                    fe_values.shape_grad(j, q) * fe_values.JxW(q);

                  // Transport term.
                  cell_stiffness_matrix(i, j) += 
                    scalar_product(beta_tensor, fe_values.shape_grad(j, q)) *
                    fe_values.shape_value(i, q) * fe_values.JxW(q);
                    
                }
            }
        }

      cell->get_dof_indices(dof_indices);

      mass_matrix.add(dof_indices, cell_mass_matrix);
      stiffness_matrix.add(dof_indices, cell_stiffness_matrix);
    }

  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);

  // Matrix on the left-hand side of the algebraic problem.
  lhs_matrix.copy_from(mass_matrix);
  lhs_matrix.add(theta, stiffness_matrix);

  // Matrix on the right-hand side.
  rhs_matrix.copy_from(mass_matrix);
  rhs_matrix.add(-(1.0 - theta), stiffness_matrix);
}

void
AdvDiffPOD::assemble_rhs(const double &time)
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_quadrature_points |
                            update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_rhs = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_rhs = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Compute f(tn+1)
          forcing_term.set_time(time);
          const double f_new_loc =
            forcing_term.value(fe_values.quadrature_point(q));

          // Compute f(tn)
          forcing_term.set_time(time - deltat);
          const double f_old_loc =
            forcing_term.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              cell_rhs(i) += (theta * f_new_loc + (1.0 - theta) * f_old_loc) *
                             fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);
      system_rhs.add(dof_indices, cell_rhs);
    }

  system_rhs.compress(VectorOperation::add);

  // Add the term that comes from the old solution: rhs_matrix.vmult_add(system_rhs, solution_owned);
  // Note that we don't have solution_owned at each time step but we can use fom_solution instead. When we assemble the full order
  // right-hand side, we use the approximated full order solution at the previous time step, that is fom_solution. Moreover
  // fom_solution is used some lines below to apply the boundary values.
  rhs_matrix.vmult_add(system_rhs, fom_solution);
  
  // Boundary conditions.
  {
    std::map<types::global_dof_index, double> boundary_values;

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    boundary_functions[0] = &function_g;
    boundary_functions[1] = &function_g;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
      boundary_values, lhs_matrix, fom_solution, system_rhs, false);
  }
}

// Copy of the standard matrix of modes in a TrilinosWrappers::SparseMatrix. The transformation matrix is the used for projecting
// the full order space into the reduced order space.
void
AdvDiffPOD::convert_modes(TrilinosWrappers::SparseMatrix &transformation_matrix)
{
  for (unsigned int i = 0; i < modes.size(); ++i)
    for (unsigned int j = 0; j < modes[0].size(); ++j)
      transformation_matrix.set(i, j, modes[i][j]);
  transformation_matrix.compress(VectorOperation::insert);

  // This print is commented to save time and space in the output.
  // pcout << "  Check transformation_matrix values:" << std::endl;
  // pcout << "    modes[0][0] = " << modes[0][0] << std::endl;
  // pcout << "    transformation_matrix(0, 0) = " << transformation_matrix(0, 0) << std::endl;
  // pcout << "    modes[1][0] = " << modes[1][0] << std::endl;
  // pcout << "    transformation_matrix(1, 0) = " << transformation_matrix(1, 0) << std::endl;
  // pcout << "    modes[1][1] = " << modes[1][1] << std::endl;
  // pcout << "    transformation_matrix(1, 1) = " << transformation_matrix(1, 1) << std::endl;
  // pcout << "    modes[40][1] = " << modes[40][1] << std::endl;
  // pcout << "    transformation_matrix(40, 1) = " << transformation_matrix(40, 1) << std::endl;
  // pcout << "    modes[44][1] = " << modes[44][1] << std::endl;
  // pcout << "    transformation_matrix(44, 1) = " << transformation_matrix(44, 1) << std::endl;
  // pcout << "    modes[89][1] = " << modes[89][1] << std::endl;
  // pcout << "    transformation_matrix(89, 1) = " << transformation_matrix(89, 1) << std::endl;
  // pcout << "    modes[108][1] = " << modes[108][1] << std::endl;
  // pcout << "    transformation_matrix(108, 1) = " << transformation_matrix(108, 1) << std::endl;
  // pcout << "    modes[27][0] = " << modes[27][0] << std::endl;
  // pcout << "    transformation_matrix(27, 0) = " << transformation_matrix(27, 0) << std::endl;

  pcout << "\n  Check transformation_matrix size:" << std::endl;
  pcout << "    modes.size() = " << modes.size() << "\t\ttransformation_matrix.m() = " << transformation_matrix.m() << std::endl;
  pcout << "    modes[0].size() = " << modes[0].size() << "\t\ttransformation_matrix.n() = " << transformation_matrix.n() << std::endl;
}

// Projection of the initial condition.
void
AdvDiffPOD::project_u0(TrilinosWrappers::SparseMatrix &transformation_matrix)
{
  // Note that at time 0 solution_owned is defined and contains the initial condition.
  reduced_solution_owned = 0.0;
  // Projection: reduced_solution_owned = T^T * solution_owned
  transformation_matrix.Tvmult(reduced_solution_owned, solution_owned);
  reduced_solution_owned.compress(VectorOperation::add);

  // This print is commented to save time and space in the output.
  // pcout << "  Check reduced_solution_owned values:" << std::endl;
  // pcout << "    reduced_solution_owned(0) = "  << reduced_solution_owned(0) << std::endl;
  // pcout << "    reduced_solution_owned(1) = "  << reduced_solution_owned(1) << std::endl;
}

// Projection of the left-hand side matrix.
void
AdvDiffPOD::project_lhs(TrilinosWrappers::SparseMatrix &transformation_matrix)
{
  TrilinosWrappers::SparseMatrix aux;

  reduced_system_lhs = 0.0;
  assert(transformation_matrix.m() == lhs_matrix.m()); // Check on sizes
  pcout << "  Check lhs_matrix size:\t\t" << lhs_matrix.m() << " * " << lhs_matrix.n() << std::endl;

  pcout << "lhs_matrix " << lhs_matrix(2, 0) << std::endl;

  // Intermediate step of projection: aux = T^T * lhs_matrix
  transformation_matrix.Tmmult(aux, lhs_matrix);
  pcout << "  Check auxiliary matrix size:\t\t" << aux.m() << " * " << aux.n() << std::endl;

  assert(aux.n() == transformation_matrix.m()); // Check on sizes
  // Projection: reduced_system_lhs = aux * T = T^T * lhs_matrix * T
  aux.mmult(reduced_system_lhs, transformation_matrix);
  pcout << "  Check reduced_system_lhs size:\t" << reduced_system_lhs.m() << " * " << reduced_system_lhs.n() << std::endl;

  reduced_system_lhs.compress(VectorOperation::insert);

  // This print is commented to save time and space in the output.
  // pcout << "  Check reduced_system_lhs values:" << std::endl;
  // pcout << "    reduced_system_lhs(0, 0) = " << reduced_system_lhs(0, 0) << std::endl;
  // pcout << "    reduced_system_lhs(1, 0) = " << reduced_system_lhs(1, 0) << std::endl;
}

// Projection of the right-hand side vector.
void
AdvDiffPOD::project_rhs(TrilinosWrappers::SparseMatrix &transformation_matrix)
{
  reduced_system_rhs = 0.0;
  // Projection: reduced_system_rhs = T^T * system_rhs
  transformation_matrix.Tvmult(reduced_system_rhs, system_rhs);
  reduced_system_rhs.compress(VectorOperation::insert);

  // This print is commented to save time and space in the output.
  // pcout << "  Check reduced_system_rhs values:" << std::endl;
  // pcout << "    reduced_system_rhs(0) = "  << reduced_system_rhs(0) << std::endl;
  // pcout << "    reduced_system_rhs(1) = "  << reduced_system_rhs(1) << std::endl;
}

// Expansion of the reduced order solution to the full order solution.
void
AdvDiffPOD::expand_solution(TrilinosWrappers::SparseMatrix &transformation_matrix)
{
  fom_solution = 0.0;
  // Expansion: fom_solution = T * reduced_solution
  transformation_matrix.vmult(fom_solution, reduced_solution);
  fom_solution.compress(VectorOperation::insert);

  // pcout << "  Check fom_solution size: " << fom_solution.size() << std::endl;
  // pcout << "  Check fom_solution values: " << std::endl;
  // pcout << "    fom_solution(17) = " << fom_solution(17) << std::endl;
  // pcout << "    fom_solution(80) = " << fom_solution(80) << std::endl;
}

// Solve the reduced order system.
void
AdvDiffPOD::solve_time_step_reduced()
{
  SolverControl solver_control(1000, 1e-6 * reduced_system_rhs.l2_norm());

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionSSOR      preconditioner;
  preconditioner.initialize(
    reduced_system_lhs, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  solver.solve(reduced_system_lhs, reduced_solution_owned, reduced_system_rhs, preconditioner);
  // This print is commented to save time and space in the output.
  // pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;

  reduced_solution = reduced_solution_owned;
}

void
AdvDiffPOD::output(const unsigned int &time_step) const
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler_r, reduced_solution, "u");

  std::vector<unsigned int> partition_int(mesh_r.n_active_cells());
  GridTools::get_subdomain_association(mesh_r, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(
    "./", "output", time_step, MPI_COMM_WORLD, 3);
}

void
AdvDiffPOD::solve_reduced()
{
  assemble_matrices();
  setup_reduced();

  TrilinosWrappers::SparseMatrix transformation_matrix(modes.size(), modes[0].size(), modes[0].size());
  convert_modes(transformation_matrix);
  try
  {
    project_lhs(transformation_matrix);
  }
  catch (const std::runtime_error &e)
  {
    std::cerr << "Runtime error: " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  pcout << "===================================================================" << std::endl;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;

    VectorTools::interpolate(dof_handler, u_0, solution_owned);
    fom_solution = solution_owned;

    // This print is commented to save time and space in the output.
    // pcout << "  Check fom_solution values:" << std::endl;
    // pcout << "    solution_owned(0)   = " << solution_owned(0) << std::endl;
    // pcout << "    fom_solution(0)     = " << fom_solution(0) << std::endl;
    // pcout << "    solution_owned(17)  = " << solution_owned(17) << std::endl;
    // pcout << "    fom_solution(17)    = " << fom_solution(17) << std::endl;
    // pcout << "    solution_owned(100) = " << solution_owned(100) << std::endl;
    // pcout << "    fom_solution(100)   = " << fom_solution(100) << std::endl;

    project_u0(transformation_matrix);
    reduced_solution = reduced_solution_owned;

    // Output the initial solution.
    output(0);

  pcout << "-------------------------------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;
  double       time      = 0;

  while (time < T)
    {
      time += deltat;
      ++time_step;

      // This print is commented to save time and space in the output.
      // pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
      //       << time << ":" << std::flush;

      assemble_rhs(time);
      project_rhs(transformation_matrix);

      solve_time_step_reduced();
      expand_solution(transformation_matrix);
      output(time_step);
    }
}

// double
// AdvDiffPOD::compute_error(const VectorTools::NormType &norm_type)
// {
//   FE_Q<dim> fe_linear(1);
//   MappingFE mapping(fe_linear);

//   const QGauss<dim> quadrature_error = QGauss<dim>(r + 2);

//   exact_solution.set_time(time);

//   Vector<double> error_per_cell;
//   VectorTools::integrate_difference(mapping,
//                                     dof_handler,
//                                     solution,
//                                     exact_solution,
//                                     error_per_cell,
//                                     quadrature_error,
//                                     norm_type);

//   const double error =
//     VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

//   return error;
// }