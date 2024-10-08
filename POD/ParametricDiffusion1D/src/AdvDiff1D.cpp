#include "AdvDiff1D.hpp"

template class AdvDiff<1>;

template <int dim>
void AdvDiff<dim>::setup()
{
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    const unsigned int N = parameters.get_integer("N");
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

    const unsigned int r = parameters.get_integer("degree");
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
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }
}

template <int dim>
void AdvDiff<dim>::assemble_matrices()
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

  double deltat = 0.0;
  if (convergence_deltat == 0.0) // If convergence_deltat assumes its default value, then the time step is set by deltat.
    deltat = parameters.get_double("deltat");
  else
    deltat = convergence_deltat;
  const double theta = parameters.get_double("theta");

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

template <int dim>
void AdvDiff<dim>::assemble_rhs(const double &time)
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

  const double theta = parameters.get_double("theta");
  double deltat = 0.0;
  if (convergence_deltat == 0.0) // If convergence_deltat assumes its default value, then the time step is set by deltat.
    deltat = parameters.get_double("deltat");
  else
    deltat = convergence_deltat;

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

  // Add the term that comes from the old solution.
  rhs_matrix.vmult_add(system_rhs, solution_owned);
  
  // Boundary conditions.
  {
    std::map<types::global_dof_index, double> boundary_values;

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    boundary_functions[0] = &function_g;
    boundary_functions[1] = &function_g;
    // boundary_functions[0] = &exact_solution;
    // boundary_functions[1] = &exact_solution;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
      boundary_values, lhs_matrix, solution_owned, system_rhs, false);
  }
}

template <int dim>
void AdvDiff<dim>::solve_time_step()
{
  SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionSSOR      preconditioner;
  preconditioner.initialize(
    lhs_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  solver.solve(lhs_matrix, solution_owned, system_rhs, preconditioner);

  // This print is commented to save time and space in the output.
  // pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;

  solution = solution_owned;
}

template <int dim>
void AdvDiff<dim>::assemble_snapshot_matrix(const unsigned int &time_step)
{
  const double T = parameters.get_double("T");
  double deltat = 0.0;
  if (convergence_deltat == 0.0) // If convergence_deltat assumes its default value, then the time step is set by deltat.
    deltat = parameters.get_double("deltat");
  else
    deltat = convergence_deltat;
  const unsigned int sample_every = parameters.get_integer("sample_every");

  /**
  // At the first call, it is useful to resize the snapshot matrix so that it can be easily filled. It has as many rows as the
  // solution size and as many columns as the number of time steps.
  */
  if(time_step == 0) {
    snapshot_matrix.resize(solution.size());
    for(auto &row : snapshot_matrix)
      row.resize(static_cast<std::vector<double>::size_type>(T/(deltat*sample_every)+1), 0.0); 
  }

  /**
  // It is not necessary to build a snapshot_array, since snapshot_matrix can be directly filled with solution.
  // The idea of a snapshot_array helps in understanding that the snapshot_matrix will be filled with column vectors that
  // represent the solution at each time step.
  // std::vector<double> snapshot_array(solution.size());
  // for (unsigned int i=0; i<solution.size(); i++)
  //   snapshot_array[i] = solution[i];
  // pcout << "  Check solution.size()       = " << solution.size() << std::endl;
  // pcout << "  Check snapshot_array.size() = " << snapshot_array.size() << std::endl;
  */

  std::pair<unsigned int, unsigned int> solution_local_range = solution.local_range();
  
  // for (unsigned int i=0; i<solution.size(); i++)
  for (unsigned int i=solution_local_range.first; i<solution_local_range.second; i++)
    snapshot_matrix[i][time_step/sample_every] = solution[i];
}

// template <int dim>
// void AdvDiff<dim>::output(const unsigned int &time_step) const
// {
//   DataOut<dim> data_out;
//   data_out.add_data_vector(dof_handler, solution, "u");

//   std::vector<unsigned int> partition_int(mesh.n_active_cells());
//   GridTools::get_subdomain_association(mesh, partition_int);
//   const Vector<double> partitioning(partition_int.begin(), partition_int.end());
//   data_out.add_data_vector(partitioning, "partitioning");

//   data_out.build_patches();

//   data_out.write_vtu_with_pvtu_record(
//     "./", "output", time_step, MPI_COMM_WORLD, 3);
// }

template <int dim>
void AdvDiff<dim>::solve()
{
  assemble_matrices();

  pcout << "===================================================================" << std::endl;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;

    VectorTools::interpolate(dof_handler, u_0, solution_owned);
    solution = solution_owned;

    // Output the initial solution.
    // output(0);

    // Assemble the snapshot matrix.
    assemble_snapshot_matrix(0);
  }

  unsigned int time_step = 0;
  double       time      = 0;

  const double T = parameters.get_double("T");
  double deltat = 0.0;
  if (convergence_deltat == 0.0) // If convergence_deltat assumes its default value, then the time step is set by deltat.
    deltat = parameters.get_double("deltat");
  else
    deltat = convergence_deltat;

  pcout << "-------------------------------------------------------------------" << std::endl;
  pcout << "Solving the full order system" << std::endl;

  /**
  // This parameter establishes how much frequently the snapshots are collected in the snapshot matrix. The default value 1 means
  // that all the snapshots are collected.
  */
  const unsigned int sample_every = parameters.get_integer("sample_every");

  while (time < T && time_step < floor(T/deltat))
    {
      time += deltat;
      ++time_step;

      // This print is commented to save time and space in the output.
      // pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
      //       << time << ":" << std::flush;

      assemble_rhs(time);

      auto start_full = high_resolution_clock::now();
      solve_time_step();
      auto stop_full = high_resolution_clock::now();
      auto duration_full = duration_cast<microseconds>(stop_full - start_full);
      duration_full_vec.push_back(duration_full);

      if (time_step % sample_every == 0)
        assemble_snapshot_matrix(time_step);

      // output(time_step);
    }

  // Compute the average duration of solving a single time step.
  auto aux = std::reduce(duration_full_vec.begin(), duration_full_vec.end())/static_cast<double>(duration_full_vec.size());
  duration_full_avg = std::chrono::duration_cast<std::chrono::microseconds>(aux);
}

template <int dim>
double AdvDiff<dim>::compute_error(const VectorTools::NormType &norm_type)
{
  FE_Q<dim> fe_linear(1);
  MappingFE mapping(fe_linear);

  const unsigned int r = parameters.get_integer("degree");
  const QGauss<dim> quadrature_error = QGauss<dim>(r + 2);

  exact_solution.set_time(time);

  Vector<double> error_per_cell;
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  const double error =
    VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}