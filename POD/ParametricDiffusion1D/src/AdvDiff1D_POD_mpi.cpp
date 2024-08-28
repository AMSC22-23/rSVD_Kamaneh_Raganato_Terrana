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
    GridGenerator::subdivided_hyper_cube(mesh_r, modes[0].size()-1, 0.0, 1.0, true);
    pcout << "  Number of elements = " << mesh_r.n_active_cells()
              << std::endl;

    // Write the mesh to file.
    const std::string mesh_file_name_r = "mesh_r-" + std::to_string(modes[0].size()-1) + ".vtk";
    GridOut           grid_out_r;
    std::ofstream     grid_out_file_r(mesh_file_name_r);
    grid_out_r.write_vtk(mesh_r, grid_out_file_r);
    pcout << "  Mesh saved to " << mesh_file_name_r << std::endl;
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

    // locally_owned_dofs_r = dof_handler_r.locally_owned_dofs();
    // DoFTools::extract_locally_relevant_dofs(dof_handler_r, locally_relevant_dofs_r);

    pcout << "  Number of DoFs = " << dof_handler_r.n_dofs() << std::endl;
  }

  pcout << "-------------------------------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    // TrilinosWrappers::SparsityPattern sparsity_r(locally_owned_dofs_r,
    //                                              MPI_COMM_WORLD);
    // DoFTools::make_sparsity_pattern(dof_handler_r, sparsity_r);
    // sparsity_r.compress();

    DynamicSparsityPattern dsp(dof_handler_r.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_r, dsp);
    sparsity_pattern_r.copy_from(dsp);

    // Note that in the reduced order system there's no need to assemble the mass matrix, stiffness matrix and right-hand side
    // matrix. Since the reduced left-hand side matrix and the reduced right-hand side vector are obtained by projection.
    pcout << "  Initializing the matrices" << std::endl;
    reduced_system_lhs_aux.reinit(sparsity_pattern_r);
    reduced_system_lhs.copy_from(reduced_system_lhs_aux);

    pcout << "  Initializing the system right-hand side" << std::endl;
    // reduced_system_rhs.reinit(locally_owned_dofs_r, MPI_COMM_WORLD);
    reduced_system_rhs.reinit(dof_handler_r.n_dofs());
    
    pcout << "  Initializing the solution vector" << std::endl;
    // reduced_solution_owned.reinit(locally_owned_dofs_r, MPI_COMM_WORLD);
    // reduced_solution.reinit(locally_owned_dofs_r, locally_relevant_dofs_r, MPI_COMM_WORLD);
    // reduced_solution_owned.reinit(dof_handler.n_dofs());
    reduced_solution.reinit(dof_handler_r.n_dofs());
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
AdvDiffPOD::convert_modes(PETScWrappers::FullMatrix &transformation_matrix)
{

  unsigned int rows_per_process = modes.size()/mpi_size;
  unsigned int rows_remainder = modes.size()%mpi_size;

  unsigned int local_row_size, local_row_start;

  if (mpi_rank < rows_remainder) {
    local_row_size = rows_per_process + 1;
    local_row_start = mpi_rank * local_row_size;
  } else {
    local_row_size = rows_per_process;
    local_row_start = mpi_rank * local_row_size + rows_remainder;
  } 

  for (unsigned int i = local_row_start; i < local_row_start+local_row_size; ++i)
    for (unsigned int j = 0; j < modes[0].size(); ++j)
      transformation_matrix.set(i, j, modes[i][j]);
  transformation_matrix.compress(VectorOperation::insert); 

  // for (unsigned int i = 0; i < modes.size(); ++i)
  //   for (unsigned int j = 0; j < modes[0].size(); ++j)
  //     transformation_matrix.set(i, j, modes[i][j]);
  // transformation_matrix.compress(VectorOperation::insert); // Changed from add to insert

  // This print is commented to save time and space in the output.
  pcout << "  Check transformation_matrix values:" << std::endl;
  pcout << "    modes[0][0] = " << modes[0][0] << std::endl;
  pcout << "    transformation_matrix(0, 0) = " << transformation_matrix(0, 0) << std::endl;
  pcout << "    modes[1][0] = " << modes[1][0] << std::endl;
  pcout << "    transformation_matrix(1, 0) = " << transformation_matrix(1, 0) << std::endl;
  pcout << "    modes[1][1] = " << modes[1][1] << std::endl;
  pcout << "    transformation_matrix(1, 1) = " << transformation_matrix(1, 1) << std::endl;
  pcout << "    modes[40][1] = " << modes[40][1] << std::endl;
  pcout << "    transformation_matrix(40, 1) = " << transformation_matrix(40, 1) << std::endl;
  pcout << "    modes[44][1] = " << modes[44][1] << std::endl;
  pcout << "    transformation_matrix(44, 1) = " << transformation_matrix(44, 1) << std::endl;
  // pcout << "    modes[89][1] = " << modes[89][1] << std::endl;
  // pcout << "    transformation_matrix(89, 1) = " << transformation_matrix(89, 1) << std::endl;
  // pcout << "    modes[108][1] = " << modes[108][1] << std::endl;
  // pcout << "    transformation_matrix(108, 1) = " << transformation_matrix(108, 1) << std::endl;
  pcout << "    modes[27][0] = " << modes[27][0] << std::endl;
  pcout << "    transformation_matrix(27, 0) = " << transformation_matrix(27, 0) << std::endl;

  pcout << "\n  Check transformation_matrix size:" << std::endl;
  pcout << "    modes.size() = " << modes.size() << "\t\ttransformation_matrix.m() = " << transformation_matrix.m() << std::endl;
  pcout << "    modes[0].size() = " << modes[0].size() << "\t\ttransformation_matrix.n() = " << transformation_matrix.n() << std::endl;
}

// Projection of the initial condition.
void
AdvDiffPOD::project_u0(PETScWrappers::FullMatrix &transformation_matrix)
{

  PETScWrappers::MPI::Vector dst(MPI_COMM_WORLD, transformation_matrix.n(), transformation_matrix.n()/mpi_size); // (transformation_matrix.n());
  std::pair<unsigned int, unsigned int> solution_owned_lr = solution_owned.local_range();
  PETScWrappers::MPI::Vector solution_owned_copy(MPI_COMM_WORLD, solution_owned.size(), solution_owned.local_size()); // (solution_owned.size());
  // for(unsigned int i = 0; i < solution_owned.size(); ++i)
  for(unsigned int i = solution_owned_lr.first; i < solution_owned_lr.second; ++i)
    solution_owned_copy(i) = solution_owned(i);
  solution_owned_copy.compress(VectorOperation::insert);

  pcout << "solowned" << solution_owned.size() << std::endl;
  pcout << "Check solution owned" << solution_owned(8) << "  " << solution_owned_copy(8) << std::endl;
  pcout << "Check solution owned" << solution_owned(80) << "  " << solution_owned_copy(80) << std::endl;

  // Note that at time 0 solution_owned is defined and contains the initial condition.
  // reduced_solution_owned = 0.0;
  reduced_solution = 0.0;
  assert(transformation_matrix.m() == solution_owned_copy.size());
  transformation_matrix.Tvmult(dst, solution_owned_copy);
  dst.compress(VectorOperation::add);

  std::pair<unsigned int, unsigned int> dst_lr = dst.local_range();
  // for (unsigned int i = 0; i < transformation_matrix.n(); ++i)
  for (unsigned int i = dst_lr.first; i < dst_lr.second; ++i)
    // reduced_solution_owned(i) = dst(i); // riprova con questo
    reduced_solution(i) = dst(i);
  // reduced_solution_owned = dst; // Copy, mi sa che però bisogna fare attenzione

  // Projection: reduced_solution_owned = T^T * solution_owned
  // transformation_matrix.Tvmult(reduced_solution_owned, solution_owned);
  reduced_solution.compress(VectorOperation::insert);

  // This print is commented to save time and space in the output.
  // pcout << "  Check reduced_solution_owned size:\t" << reduced_solution_owned.size() << std::endl;
  // pcout << "  Check reduced_solution_owned values:" << std::endl;
  // pcout << "    reduced_solution_owned(0) = "  << reduced_solution_owned(0) << std::endl;
  // pcout << "    reduced_solution_owned(1) = "  << reduced_solution_owned(1) << std::endl;
  pcout << "  Check reduced_solution size:\t" << reduced_solution.size() << std::endl;
  pcout << "  Check reduced_solution values:" << std::endl;
  pcout << "    reduced_solution(0) = "  << reduced_solution(0) << std::endl;
  pcout << "    reduced_solution(1) = "  << reduced_solution(1) << std::endl;
}

// Projection of the left-hand side matrix.
void
AdvDiffPOD::project_lhs(PETScWrappers::FullMatrix &transformation_matrix)
{
  // PETScWrappers::FullMatrix aux(transformation_matrix.n(), lhs_matrix.n()); // (Tn * Tm) * (Lm * Ln) = Tn * Ln
  PETScWrappers::MPI::Vector aux_col;
  // pcout << "BLOCCO QUI 1" << std::endl;
  PETScWrappers::MPI::Vector dst_col;
  // pcout << "BLOCCO QUI 1" << std::endl;
  PETScWrappers::FullMatrix intermediate_mat(transformation_matrix.n(), lhs_matrix.n());
   // (transformation_matrix.n(), lhs_matrix.n()); // (Tn * Tm) * (Lm * Ln) = Tn * Ln
  // PETScWrappers::FullMatrix dst; // (transformation_matrix.n(), transformation_matrix.n()); // (Tn * Ln) * (Tm * Tn) = Tn * Tn
  // PETScWrappers::FullMatrix lhs_matrix_copy; // (lhs_matrix.m(), lhs_matrix.n());
  // for (unsigned int i = 0; i < lhs_matrix.m(); ++i)
  //   for (unsigned int j = 0; j < lhs_matrix.n(); ++j)
      // lhs_matrix_copy(i, j) = lhs_matrix(i, j);
      // lhs_matrix_copy.set(i, j, lhs_matrix(i, j));
  // lhs_matrix_copy.copy_from(lhs_matrix); // non mi sembra funzionare

  // pcout << "  Check lhs_matrix_copy: " << std::endl;
  // pcout << lhs_matrix(0, 0) << std::endl;
  // pcout << lhs_matrix_copy(0, 0) << std::endl;
  // pcout << lhs_matrix(1, 0) << std::endl;
  // pcout << lhs_matrix_copy(1, 0) << std::endl;
  // pcout << lhs_matrix(100, 70) << std::endl;
  // pcout << lhs_matrix_copy(100, 70) << std::endl;
  // pcout << lhs_matrix(120, 70) << std::endl;
  // pcout << lhs_matrix_copy(120, 70) << std::endl;

  // for 
  // aux_row = prima col di transformation
  // aux_col = prima colonna di lhs

  // pcout << "BLOCCO QUI 1" << std::endl;
  reduced_system_lhs = 0.0;
  // pcout << "BLOCCO QUI 1" << std::endl;
  pcout <<"lhs" << lhs_matrix.m() << " " << lhs_matrix.n() << std::endl;
  pcout << lhs_matrix.el(0, 0) << std::endl;
  pcout << lhs_matrix.el(1, 0) << std::endl;
  pcout << lhs_matrix.el(2, 0) << std::endl;

  // ciclo in cui si moltiplica la transformation trasposta per la singola colonna di lhs
  // Intermediate step of projection: aux = T^T * lhs_matrix
  for (unsigned int j = 0; j < lhs_matrix.n(); ++j)
  {
    std::pair<unsigned int, unsigned int> lhs_matrix_lr = lhs_matrix.local_range();
    aux_col.reinit(MPI_COMM_WORLD, lhs_matrix.m(), lhs_matrix.local_size());
    pcout << "  Check aux_col size:\t\t" << aux_col.size() << std::endl;

    // for (unsigned int i = 0; i < lhs_matrix.m(); ++i)
    for (unsigned int i = lhs_matrix_lr.first; i < lhs_matrix_lr.second; ++i)
    {
      assert(i < aux_col.size());
      // assert(i < lhs_matrix.m());
      assert(i >= lhs_matrix_lr.first && i < lhs_matrix_lr.second);
      assert(j < lhs_matrix.n());
      // if (lhs_matrix(i, j))
      aux_col(i) = lhs_matrix.el(i, j);
      // else
      //   aux_col(i) = 0.0;
    }
          pcout << "BLOCCO 1" << std::endl;
    aux_col.compress(VectorOperation::insert);

    // pcout << "BLOCCO QUI 1" << std::endl;

    // pcout << "  Check aux_col size:\t\t" << aux_col.size() << std::endl;
    // pcout << "  Check aux_col values:" << std::endl;
    // pcout << "    lhs_matrix(0, j) = " << lhs_matrix(0, j) << std::endl;
    // pcout << "    aux_col(0)       = " << aux_col(0) << std::endl;
    // pcout << "    lhs_matrix(1, j) = " << lhs_matrix(1, j) << std::endl;
    // pcout << "    aux_col(1)       = " << aux_col(1) << std::endl;

    assert(transformation_matrix.m() == aux_col.size()); // Check on sizes
    // std::pair<unsigned int, unsigned int> transformation_matrix_lr = transformation_matrix.local_range();
    dst_col.reinit(MPI_COMM_WORLD, transformation_matrix.n(), transformation_matrix.n()/mpi_size);
    transformation_matrix.Tvmult(dst_col, aux_col);
    dst_col.compress(VectorOperation::add); // cambiato in add
    pcout << "BLOCCO 2" << std::endl;

    // pcout << "  Check dst_col size:\t" << dst_col.size() << std::endl;

    std::pair<unsigned int, unsigned int> dst_col_lr = dst_col.local_range();
    // for (unsigned int i = 0; i < transformation_matrix.n(); ++i)
    for (unsigned int i = dst_col_lr.first; i < dst_col_lr.second; ++i)
      intermediate_mat.set(i, j, dst_col(i));
    intermediate_mat.compress(VectorOperation::insert); // questo magari da portar fuori?

    pcout << "  Check intermediate_mat size:\t" << intermediate_mat.m() << " * " << intermediate_mat.n() << std::endl;

    aux_col.clear();
    dst_col.clear();
    pcout << "BLOCCO 3" << std::endl;
  }
  pcout << "BLOCCO 3A" << std::endl;

  unsigned int rows_per_process = transformation_matrix.m()/mpi_size;
  unsigned int rows_remainder = transformation_matrix.m()%mpi_size;

  unsigned int local_row_size, local_row_start;

  if (mpi_rank < rows_remainder) {
    local_row_size = rows_per_process + 1;
    local_row_start = mpi_rank * local_row_size;
  } else {
    local_row_size = rows_per_process;
    local_row_start = mpi_rank * local_row_size + rows_remainder;
  } 

  unsigned int cols_per_process = transformation_matrix.n() / mpi_size;
  unsigned int cols_remainder = transformation_matrix.n() % mpi_size;

  unsigned int local_col_size, local_col_start;

  if (mpi_rank < cols_remainder) {
    local_col_size = cols_per_process + 1;
    local_col_start = mpi_rank * local_col_size;
  } else {
    local_col_size = cols_per_process;
    local_col_start = mpi_rank * local_col_size + cols_remainder;
  }

  // Projection: reduced_system_lhs = aux * T = T^T * lhs_matrix * T
  for (unsigned int j = 0; j < transformation_matrix.n(); ++j)
  {
    // std::pair<unsigned int, unsigned int> transformation_matrix_lr = transformation_matrix.local_range();

            // pcout << "check transf local size" << transformation_matrix.local_size();
            // pcout << "check transf lr" << transformation_matrix_lr.first << " " << transformation_matrix_lr.second;

    aux_col.reinit(MPI_COMM_WORLD, transformation_matrix.m(), local_row_size);
    
    pcout << "  Check aux_col size:\t\t" << aux_col.size() << std::endl;
    pcout << "  Check aux_col local size:\t\t" << aux_col.locally_owned_size() << std::endl;
    pcout << "check transf local size" << transformation_matrix.local_size();

    // for (unsigned int i = 0; i < transformation_matrix.m(); ++i)
    for (unsigned int i = local_row_start; i < local_row_start+local_row_size; ++i)
    {
      assert(i < aux_col.size());
      // assert(i < lhs_matrix.m());
      assert(i >= transformation_matrix_lr.first && i < transformation_matrix_lr.second);
      assert(j < transformation_matrix.n());
      aux_col(i) = transformation_matrix(i, j);
    }
    aux_col.compress(VectorOperation::insert);

    pcout << "  Check aux_col size:\t\t" << aux_col.size() << std::endl;
    pcout << "  Check aux_col values:" << std::endl;
    pcout << "    transformation_matrix(0, j) = " << transformation_matrix(0, j) << std::endl;
    pcout << "    aux_col(0)                  = " << aux_col(0) << std::endl;
    pcout << "    transformation_matrix(1, j) = " << transformation_matrix(1, j) << std::endl;
    pcout << "    aux_col(1)                  = " << aux_col(1) << std::endl;
    pcout << "    transformation_matrix(2, j) = " << transformation_matrix(2, j) << std::endl;
    pcout << "    aux_col(2)                  = " << aux_col(2) << std::endl;
    pcout << "    transformation_matrix(17, j) = " << transformation_matrix(17, j) << std::endl;
    pcout << "    aux_col(17)                  = " << aux_col(17) << std::endl;
    // pcout << "    transformation_matrix(80, j) = " << transformation_matrix(80, j) << std::endl;
    // pcout << "    aux_col(80)                  = " << aux_col(80) << std::endl;

    assert(intermediate_mat.n() == aux_col.size()); // Check on sizes

    // dst_col.reinit(MPI_COMM_WORLD, transformation_matrix.n(), local_col_size); // tolto mpi size
    // dst_col.reinit(MPI_COMM_WORLD, transformation_matrix.n(), transformation_matrix.n()/2);
    dst_col.reinit(MPI_COMM_WORLD, intermediate_mat.m(), intermediate_mat.m()/mpi_size);
    intermediate_mat.vmult(dst_col, aux_col);
    dst_col.compress(VectorOperation::add);
        pcout << "BLOCCO 4" << std::endl;


    // pcout << "  Check dst_col size:\t" << dst_col.size() << std::endl;

    std::pair<unsigned int, unsigned int> dst_col_lr = dst_col.local_range();
    // for (unsigned int i = 0; i < transformation_matrix.n(); ++i)
    for (unsigned int i = dst_col_lr.first; i < dst_col_lr.second; ++i)
    { 
      // if (reduced_system_lhs.el(i, j) == 0.0) // In questo modo stai andando a prendere solamente quelli che esistono e scartando gli altri
      // perchè è necessario che la matrice sia sparsa e quindi mi sa anche che rispetti dof_handler
        reduced_system_lhs.set(i, j, dst_col(i)); // qui mi sa che funziona perché matrice non più sparsa
    }
    // reduced_system_lhs.compress(VectorOperation::insert);

    aux_col.clear();
    dst_col.clear();
        pcout << "BLOCCO 5" << std::endl;

  }


  // assert(transformation_matrix.m() == lhs_matrix_copy.m()); // Check on sizes
  // pcout << "  Check lhs_matrix size:\t\t" << lhs_matrix_copy.m() << " * " << lhs_matrix.n() << std::endl;

  // Intermediate step of projection: aux = T^T * lhs_matrix

  // transformation_matrix.Tmmult(aux, lhs_matrix_copy, v);
  
  // pcout << "  Check auxiliary matrix size:\t\t" << aux.m() << " * " << aux.n() << std::endl;

  // assert(aux.n() == transformation_matrix.m()); // Check on sizes
  // Projection: reduced_system_lhs = aux * T = T^T * lhs_matrix * T
  // aux.mmult(dst, transformation_matrix, v);
  // for (unsigned int i = 0; i < dst.m(); ++i)
  //   for (unsigned int j = 0; j < dst.n(); ++j) // Dimensioni giuste??
  //     reduced_system_lhs.set(i, j, dst(i, j));
  pcout << "  Check reduced_system_lhs size:\t" << reduced_system_lhs.m() << " * " << reduced_system_lhs.n() << std::endl;

  // reduced_system_lhs.compress(VectorOperation::insert);

  // PRIMADI AGGIUNGERE 
  // reduced_system_lhs = 0.0;
  // assert(transformation_matrix.m() == lhs_matrix_copy.m()); // Check on sizes
  // pcout << "  Check lhs_matrix size:\t\t" << lhs_matrix_copy.m() << " * " << lhs_matrix.n() << std::endl;

  // // Intermediate step of projection: aux = T^T * lhs_matrix
  // transformation_matrix.Tmmult(aux, lhs_matrix);
  // pcout << "  Check auxiliary matrix size:\t\t" << aux.m() << " * " << aux.n() << std::endl;

  // assert(aux.n() == transformation_matrix.m()); // Check on sizes
  // // Projection: reduced_system_lhs = aux * T = T^T * lhs_matrix * T
  // aux.mmult(reduced_system_lhs, transformation_matrix);
  // pcout << "  Check reduced_system_lhs size:\t" << reduced_system_lhs.m() << " * " << reduced_system_lhs.n() << std::endl;

  // reduced_system_lhs.compress(VectorOperation::add);

  // This print is commented to save time and space in the output.
  pcout << "  Check reduced_system_lhs values:" << std::endl;
  pcout << "    reduced_system_lhs(0, 0) = " << reduced_system_lhs(0, 0) << std::endl;
  pcout << "    reduced_system_lhs(1, 0) = " << reduced_system_lhs(1, 0) << std::endl;
}

// Projection of the right-hand side vector.
void
AdvDiffPOD::project_rhs(PETScWrappers::FullMatrix &transformation_matrix)
{

  PETScWrappers::MPI::Vector dst(MPI_COMM_WORLD, transformation_matrix.n(), transformation_matrix.n()/mpi_size); // (transformation_matrix.n());
  // Vector<double> system_rhs_copy(system_rhs); // Copy constructor
  PETScWrappers::MPI::Vector system_rhs_copy(MPI_COMM_WORLD, system_rhs.size(), system_rhs.local_size()); // (system_rhs.size());
  // Vector<double> system_rhs_copy(transformation_matrix.m());

  // // FAI CHECK SU DIMENSIONI CON ERRORE
  std::pair<unsigned int, unsigned int> system_rhs_lr = system_rhs.local_range();
  // for (unsigned int i = 0; i < transformation_matrix.m(); ++i)
  for (unsigned int i = system_rhs_lr.first; i < system_rhs_lr.second; ++i)
    system_rhs_copy(i) = system_rhs(i);
  system_rhs_copy.compress(VectorOperation::insert);
  // system_rhs_copy.compress(VectorOperation::add);
  pcout << "  Check rhs_matrix_copy: " << std::endl;
  pcout << system_rhs(40) << std::endl;
  pcout << system_rhs_copy(40) << std::endl;
  pcout << system_rhs(41) << std::endl;
  pcout << system_rhs_copy(41) << std::endl;
  // PROBLEMINO QUI
  reduced_system_rhs = 0.0; // TENERE?
  
  assert(transformation_matrix.m() == system_rhs_copy.size());
  transformation_matrix.Tvmult(dst, system_rhs_copy);
  dst.compress(VectorOperation::add);

  std::pair<unsigned int, unsigned int> dst_lr = dst.local_range();
  // for (unsigned int i = 0; i < transformation_matrix.n(); ++i)
  for (unsigned int i = dst_lr.first; i < dst_lr.second; ++i)
    reduced_system_rhs(i) = dst(i);
  // reduced_system_rhs = dst; // Copy
    // reduced_system_rhs.set(i, dst(i));
  reduced_system_rhs.compress(VectorOperation::insert);

  // PRIMA DI FULL
  // reduced_system_rhs = 0.0;
  // // Projection: reduced_system_rhs = T^T * system_rhs
  // transformation_matrix.Tvmult(reduced_system_rhs, system_rhs);
  // reduced_system_rhs.compress(VectorOperation::add);

  // This print is commented to save time and space in the output.
  pcout << "  Check reduced_system_rhs values:" << std::endl;
  pcout << "    reduced_system_rhs(0) = "  << reduced_system_rhs(0) << std::endl;
  pcout << "    reduced_system_rhs(1) = "  << reduced_system_rhs(1) << std::endl;
}

// Expansion of the reduced order solution to the full order solution.
void
AdvDiffPOD::expand_solution(PETScWrappers::FullMatrix &transformation_matrix)
{

  unsigned int rows_per_process = transformation_matrix.m()/mpi_size;
  unsigned int remainder = transformation_matrix.m()%mpi_size;

  unsigned int local_row_size, local_row_start;

  if (mpi_rank < remainder) {
      local_row_size = rows_per_process + 1;
      local_row_start = mpi_rank * local_row_size;
  } 
  else 
  {
      local_row_size = rows_per_process;
      local_row_start = mpi_rank * local_row_size + remainder;
  } 


  PETScWrappers::MPI::Vector dst(MPI_COMM_WORLD, transformation_matrix.m(), local_row_size); 
  PETScWrappers::MPI::Vector reduced_solution_copy(MPI_COMM_WORLD, reduced_solution.size(), reduced_solution.size()/mpi_size);

  // std::pair<unsigned int, unsigned int> reduced_solution_lr = reduced_solution.local_range();
  for(unsigned int i = 0; i < reduced_solution.size(); ++i) // ...
  // for(unsigned int i = reduced_solution_lr.first; i < reduced_solution_lr.second; ++i)
    reduced_solution_copy(i) = reduced_solution(i);
  reduced_solution_copy.compress(VectorOperation::insert);

  pcout << "Check reduced solution" << reduced_solution(0) << "  " << reduced_solution_copy(0) << std::endl;
  pcout << "Check reduced solution" << reduced_solution(1) << "  " << reduced_solution_copy(1) << std::endl;

  // Note that at time 0 solution_owned is defined and contains the initial condition.
  // reduced_solution_owned = 0.0;
  // transformation_matrix.Tvmult(dst, solution_owned_copy);
  fom_solution = 0.0;
  // Expansion: fom_solution = T * reduced_solution
  assert(transformation_matrix.n() == reduced_solution_copy.size()); // cambiato in n
  transformation_matrix.vmult(dst, reduced_solution_copy);
  dst.compress(VectorOperation::add);
  // for (unsigned int i = 0; i < transformation_matrix.n(); ++i)
  //   dst_aux(i) = dst(i);
  Vector<double> dst_aux(dst); // Copy constructor 

  // dst_aux.compress(VectorOperation::insert); 
  fom_solution = dst_aux;

  // Projection: reduced_solution_owned = T^T * solution_owned
  // transformation_matrix.Tvmult(reduced_solution_owned, solution_owned);
  fom_solution.compress(VectorOperation::insert);
  // assert(fom_solution == dst);
  // pcout << "Check fom_solution" << fom_solution(0) << "  " << dst(0) << std::endl;
  // pcout << "Check fom_solution" << fom_solution(1) << "  " << dst(1) << std::endl;



  // PRIMA 
  // fom_solution = 0.0;
  // // Expansion: fom_solution = T * reduced_solution
  // transformation_matrix.vmult(fom_solution, reduced_solution);
  // fom_solution.compress(VectorOperation::add);

  // pcout << "  Check fom_solution size: " << fom_solution.size() << std::endl;
  // pcout << "  Check fom_solution values: " << std::endl;
  // pcout << "    fom_solution(17) = " << fom_solution(17) << std::endl;
  // pcout << "    fom_solution(80) = " << fom_solution(80) << std::endl;

  // COMMENTATO
  pcout << "  Check fom_solution size: " << fom_solution.size() << std::endl;
  pcout << "  Check fom_solution values: " << std::endl;
  pcout << "    fom_solution(2)  = " << fom_solution(2) << std::endl;
  pcout << "    fom_solution(17) = " << fom_solution(17) << std::endl;
  pcout << "    fom_solution(50) = " << fom_solution(50) << std::endl;
  pcout << "    fom_solution(80) = " << fom_solution(80) << std::endl;
}

// Solve the reduced order system.
void
AdvDiffPOD::solve_time_step_reduced()
{
  // SolverControl solver_control(1000, 1e-6 * reduced_system_rhs.l2_norm());

  // SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  // TrilinosWrappers::PreconditionSSOR      preconditioner;
  // preconditioner.initialize(
  //   reduced_system_lhs, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  // solver.solve(reduced_system_lhs, reduced_solution_owned, reduced_system_rhs, preconditioner);
  // This print is commented to save time and space in the output.
  // pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;

  // reduced_solution = reduced_solution_owned;

  SolverControl solver_control(1000, 1e-6 * reduced_system_rhs.l2_norm());
  // SolverControl solver_control(2000, 1e-5 * reduced_system_rhs.l2_norm());

  // SolverCG<Vector<double>> solver(solver_control);
  SolverGMRES<Vector<double>> solver(solver_control); // magari non simmetrico

  // PROVARE CON FULL MATRIX SENZA PRECONDITIONER E SENZA SPARSITY PATTERN
  // PreconditionSOR preconditioner;
  // preconditioner.initialize(
  //   reduced_system_lhs, PreconditionSOR<SparseMatrix<double>>::AdditionalData(1.0));

  // solver.solve(reduced_system_lhs, reduced_solution, reduced_system_rhs, preconditioner); // qui da sistemare owned
  solver.solve(reduced_system_lhs, reduced_solution, reduced_system_rhs, PreconditionIdentity());
  // pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;
  pcout << "  " << solver_control.last_step() << " GMRES iterations" << std::endl;
}

void
AdvDiffPOD::output(const unsigned int &time_step) const
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler_r, reduced_solution, "u");

  // std::vector<unsigned int> partition_int(mesh_r.n_active_cells());
  // GridTools::get_subdomain_association(mesh_r, partition_int);
  // const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  // data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::string output_file_name =
    "output-" + std::to_string(modes[0].size()-1) + ".vtk";
  std::ofstream output_file(output_file_name);
  // data_out.write_vtk(output_file);

  data_out.write_vtu_with_pvtu_record(
    "./", "output", time_step, MPI_COMM_WORLD, 3);
}

void
AdvDiffPOD::solve_reduced()
{
  assemble_matrices();
  setup_reduced();

  // TrilinosWrappers::SparseMatrix transformation_matrix(modes.size(), modes[0].size(), modes[0].size());
  PETScWrappers::FullMatrix transformation_matrix; // (modes.size(), modes[0].size());
  transformation_matrix.reinit(modes.size(), modes[0].size());
  // EVENTUALMENTE CAPIRE FULLMATRIX SIZETYPE PER INIZIALIZZARE CON RIGHE E COLONNE, TIPO ROW = MODES.SIZE() O MAGARI CAST?
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
    pcout << "  Check fom_solution values:" << std::endl;
    pcout << "    solution_owned(0)   = " << solution_owned(0) << std::endl;
    pcout << "    fom_solution(0)     = " << fom_solution(0) << std::endl;
    pcout << "    solution_owned(17)  = " << solution_owned(17) << std::endl;
    pcout << "    fom_solution(17)    = " << fom_solution(17) << std::endl;
    pcout << "    solution_owned(100) = " << solution_owned(100) << std::endl;
    pcout << "    fom_solution(100)   = " << fom_solution(100) << std::endl;

    project_u0(transformation_matrix);
    // reduced_solution = reduced_solution_owned;

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

      auto start_reduced = high_resolution_clock::now();
      solve_time_step_reduced();
      auto stop_reduced = high_resolution_clock::now();
      auto duration_reduced = duration_cast<milliseconds>(stop_reduced - start_reduced);
      duration_reduced_vec.push_back(duration_reduced);

      expand_solution(transformation_matrix);

      output(time_step);
    }

  // Compute the average duration of solving a single time step.
  duration_reduced_avg = std::reduce(duration_reduced_vec.begin(), duration_reduced_vec.end())/static_cast<double>(duration_reduced_vec.size());

  // duration_reduced_avg = duration_reduced_vec.accumulate() / duration_reduced_vec.size();
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