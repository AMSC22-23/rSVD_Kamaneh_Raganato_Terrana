#include "AdvDiff1D_POD.hpp"

template class AdvDiffPOD<1>;
// template class AdvDiff<2>; per questo ti servono altri elementi finiti
// template class AdvDiff<3>;
template <int dim>
void AdvDiffPOD<dim>::setup()
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
  }
}

/**
// The reduced mesh, the reduced left-hand side matrix, the reduced right-hand side vector and the reduced solution vector are not
// distributed. This choice is made to avoid complications in assembling the reduced components and solving the reduced order problem.
*/
template <int dim>
void AdvDiffPOD<dim>::setup_reduced()
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

    const unsigned int r = parameters.get_integer("degree");
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

    pcout << "  Number of DoFs = " << dof_handler_r.n_dofs() << std::endl;
  }

  pcout << "-------------------------------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;
    // pcout << "  Initializing the sparsity pattern" << std::endl;

    DynamicSparsityPattern dsp(dof_handler_r.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_r, dsp);
    sparsity_pattern_r.copy_from(dsp);

    /** 
    // Note that in the reduced order system there's no need to assemble the mass matrix, stiffness matrix and right-hand side
    // matrix. Since the reduced left-hand side matrix and the reduced right-hand side vector are obtained by projection.
    */
    pcout << "  Initializing the matrices" << std::endl;
    reduced_system_lhs_aux.reinit(sparsity_pattern_r);
    reduced_system_lhs.copy_from(reduced_system_lhs_aux);

    pcout << "  Initializing the system right-hand side" << std::endl;
    reduced_system_rhs.reinit(dof_handler_r.n_dofs());
    
    pcout << "  Initializing the solution vector" << std::endl;
    reduced_solution.reinit(dof_handler_r.n_dofs());
  }
}

template <int dim>
void AdvDiffPOD<dim>::assemble_matrices()
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
void AdvDiffPOD<dim>::assemble_rhs(const double &time)
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

  /**
  // Add the term that comes from the old solution: rhs_matrix.vmult_add(system_rhs, solution_owned);
  // Note that we don't have solution_owned at each time step but we can use fom_solution instead. When we assemble the full order
  // right-hand side, we use the approximated full order solution at the previous time step, that is fom_solution. Moreover
  // fom_solution is used some lines below to apply the boundary values.
  */
  rhs_matrix.vmult_add(system_rhs, fom_solution);
  
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
      boundary_values, lhs_matrix, fom_solution, system_rhs, false);
  }
}

/**
// Copy of the standard matrix of modes in a PETScWrappers::FullMatrix. The transformation matrix is then used for projecting
// the full order space into the reduced order space.
*/
template <int dim>
void AdvDiffPOD<dim>::convert_modes(PETScWrappers::FullMatrix &transformation_matrix)
{
  for (unsigned int i = 0; i < modes.size(); ++i)
    for (unsigned int j = 0; j < modes[0].size(); ++j)
    { 
      if (std::isnan(modes[i][j]))
        transformation_matrix.set(i, j, 0.0);
      else
        transformation_matrix.set(i, j, modes[i][j]);
    }
  transformation_matrix.compress(VectorOperation::insert); // Changed from add to insert

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

  pcout << "  Check transformation_matrix size:" << std::endl;
  pcout << "    modes.size() = " << modes.size() << "\t\ttransformation_matrix.m() = " << transformation_matrix.m() << std::endl;
  pcout << "    modes[0].size() = " << modes[0].size() << "\t\ttransformation_matrix.n() = " << transformation_matrix.n() << std::endl;
}

// Projection of the initial condition.
template <int dim>
void AdvDiffPOD<dim>::project_u0(PETScWrappers::FullMatrix &transformation_matrix)
{
  PETScWrappers::MPI::Vector dst(MPI_COMM_WORLD, transformation_matrix.n(), transformation_matrix.n());
  PETScWrappers::MPI::Vector solution_owned_copy(MPI_COMM_WORLD, solution_owned.size(), solution_owned.size());
  for(unsigned int i = 0; i < solution_owned.size(); ++i)
    solution_owned_copy(i) = solution_owned(i);
  solution_owned_copy.compress(VectorOperation::insert);

  // This print is commented to save time and space in the output.
  // pcout << "  Check solution_owned_copy values:" << std::endl;
  // pcout << "    solution_owned(8)       = " << solution_owned(8) << std::endl;
  // pcout << "    solution_owned_copy(8)  = " << solution_owned_copy(8) << std::endl;
  // pcout << "    solution_owned(80)      = " << solution_owned(80) << std::endl;
  // pcout << "    solution_owned_copy(80) = " << solution_owned_copy(80) << std::endl;

  /**
  // Note that at time 0 solution_owned is defined and contains the initial condition.
  // reduced_solution_owned = 0.0;
  */
  reduced_solution = 0.0;

  /**
  // Projection: reduced_solution = T^T * solution_owned
  */
  assert(transformation_matrix.m() == solution_owned_copy.size());
  transformation_matrix.Tvmult(dst, solution_owned_copy);
  dst.compress(VectorOperation::insert);

  for (unsigned int i = 0; i < transformation_matrix.n(); ++i)
    reduced_solution(i) = dst(i);

  reduced_solution.compress(VectorOperation::insert);

  pcout << "  Check reduced_solution size:\t\t" << reduced_solution.size() << std::endl;
  // This print is commented to save time and space in the output.
  // pcout << "  Check reduced_solution values:" << std::endl;
  // pcout << "    reduced_solution(0) = "  << reduced_solution(0) << std::endl;
  // pcout << "    reduced_solution(1) = "  << reduced_solution(1) << std::endl;
}

// Projection of the left-hand side matrix.
template <int dim>
void AdvDiffPOD<dim>::project_lhs(PETScWrappers::FullMatrix &transformation_matrix)
{
  PETScWrappers::MPI::Vector aux_col;
  PETScWrappers::MPI::Vector dst_col;
  PETScWrappers::FullMatrix intermediate_mat(transformation_matrix.n(), lhs_matrix.n());

  reduced_system_lhs = 0.0;

  pcout << "  Check lhs_matrix size:\t\t" << lhs_matrix.m() << " * " << lhs_matrix.n() << std::endl;
  // This print is commented to save time and space in the output.
  // pcout << "  Check lhs_matrix values:" << std::endl;
  // pcout << "    lhs_matrix.el(0, 0) = " << lhs_matrix.el(0, 0) << std::endl;
  // pcout << "    lhs_matrix.el(1, 0) = " << lhs_matrix.el(1, 0) << std::endl;
  // pcout << "    lhs_matrix.el(2, 0) = " << lhs_matrix.el(2, 0) << std::endl;

  /**
  // Intermediate step of projection: aux = T^T * lhs_matrix
  // The following matrix-matrix multiplications are perfomed multiplying the first matrix by a column vector of the second matrix
  // through void PETScWrappers::MatrixBase::vmult and void PETScWrappers::MatrixBase::Tvmult, since void PETScWrappers::MatrixBase::mmult
  // and void PETScWrappers::MatrixBase::Tmmult are protected member functions.
  */
  for (unsigned int j = 0; j < lhs_matrix.n(); ++j)
  {
    aux_col.reinit(MPI_COMM_WORLD, lhs_matrix.m(), lhs_matrix.m());
    for (unsigned int i = 0; i < lhs_matrix.m(); ++i)
    {
      assert(i < aux_col.size());
      assert(i < lhs_matrix.m());
      assert(j < lhs_matrix.n());
      aux_col(i) = lhs_matrix.el(i, j);
      /**
      // TrilinosScalar SparseMatrix< number >::el "Return the value of the matrix entry (i,j). If this entry does not exist in the
      // sparsity pattern, then zero is returned." This function allows to access any entry of the matrix, even if it doesn't exist
      // in the sparsity pattern. Trying to access an entry with lhs_matrix(i, j), you can get an error if the entry doesn't exist.
      */
    }
    aux_col.compress(VectorOperation::insert);

    // This print is commented to save time and space in the output.
    // pcout << "  Check aux_col size:\t\t" << aux_col.size() << std::endl;
    // pcout << "  Check aux_col values:" << std::endl;
    // pcout << "    lhs_matrix(0, j) = " << lhs_matrix.el(0, j) << std::endl;
    // pcout << "    aux_col(0)       = " << aux_col(0) << std::endl;
    // pcout << "    lhs_matrix(1, j) = " << lhs_matrix.el(1, j) << std::endl;
    // pcout << "    aux_col(1)       = " << aux_col(1) << std::endl;

    assert(transformation_matrix.m() == aux_col.size()); // Check on sizes
    dst_col.reinit(MPI_COMM_WORLD, transformation_matrix.n(), transformation_matrix.n());
    transformation_matrix.Tvmult(dst_col, aux_col);
    dst_col.compress(VectorOperation::insert);
    
    // This print is commented to save time and space in the output.
    // pcout << "  Check dst_col size:\t" << dst_col.size() << std::endl;

    for (unsigned int i = 0; i < transformation_matrix.n(); ++i)
      intermediate_mat.set(i, j, dst_col(i));
    intermediate_mat.compress(VectorOperation::insert);

    aux_col.clear();
    dst_col.clear();
  }

  /**
  // Projection: reduced_system_lhs = aux * T = T^T * lhs_matrix * T
  */
  for (unsigned int j = 0; j < transformation_matrix.n(); ++j)
  {
    aux_col.reinit(MPI_COMM_WORLD, transformation_matrix.m(), transformation_matrix.m());
    for (unsigned int i = 0; i < transformation_matrix.m(); ++i)
      aux_col(i) = transformation_matrix(i, j);
    aux_col.compress(VectorOperation::insert);

    // This print is commented to save time and space in the output.
    // pcout << "  Check aux_col size:\t\t" << aux_col.size() << std::endl;
    // pcout << "  Check aux_col values:" << std::endl;
    // pcout << "    transformation_matrix(0, j) = " << transformation_matrix(0, j) << std::endl;
    // pcout << "    aux_col(0)                  = " << aux_col(0) << std::endl;
    // pcout << "    transformation_matrix(1, j) = " << transformation_matrix(1, j) << std::endl;
    // pcout << "    aux_col(1)                  = " << aux_col(1) << std::endl;
    // pcout << "    transformation_matrix(2, j) = " << transformation_matrix(2, j) << std::endl;
    // pcout << "    aux_col(2)                  = " << aux_col(2) << std::endl;
    // pcout << "    transformation_matrix(17, j) = " << transformation_matrix(17, j) << std::endl;
    // pcout << "    aux_col(17)                  = " << aux_col(17) << std::endl;
    // pcout << "    transformation_matrix(80, j) = " << transformation_matrix(0, j) << std::endl;
    // pcout << "    aux_col(80)                  = " << aux_col(0) << std::endl;

    assert(intermediate_mat.n() == aux_col.size()); // Check on sizes
    dst_col.reinit(MPI_COMM_WORLD, transformation_matrix.n(), transformation_matrix.n());
    intermediate_mat.vmult(dst_col, aux_col);
    dst_col.compress(VectorOperation::insert);

    // This print is commented to save time and space in the output.
    // pcout << "  Check dst_col size:\t" << dst_col.size() << std::endl;

    for (unsigned int i = 0; i < transformation_matrix.n(); ++i)
      reduced_system_lhs.set(i, j, dst_col(i));

    aux_col.clear();
    dst_col.clear();
  }

  pcout << "  Check reduced_system_lhs size:\t" << reduced_system_lhs.m() << " * " << reduced_system_lhs.n() << std::endl;
  // This print is commented to save time and space in the output.
  // pcout << "  Check reduced_system_lhs values:" << std::endl;
  // pcout << "    reduced_system_lhs(0, 0) = " << reduced_system_lhs(0, 0) << std::endl;
  // pcout << "    reduced_system_lhs(1, 0) = " << reduced_system_lhs(1, 0) << std::endl;

  // Check if the reduced_system_lhs matrix is symmetric.
  unsigned int r = 0, c = 0, flag = 0;
  while(r<reduced_system_lhs.m() && flag == 0)
  {
    c = 0;
    while(c<reduced_system_lhs.n() && flag == 0)
    {
      if(reduced_system_lhs(r, c) != reduced_system_lhs(c, r))
        flag = 1;
      c++;
    }
    r++;
  }

  if(flag == 0)
    pcout << "  The reduced_system_lhs matrix is symmetric!" << std::endl;
  else
    pcout << "  The reduced_system_lhs matrix is not symmetric!" << std::endl;
}

// Projection of the right-hand side vector.
template <int dim>
void AdvDiffPOD<dim>::project_rhs(PETScWrappers::FullMatrix &transformation_matrix)
{
  PETScWrappers::MPI::Vector dst(MPI_COMM_WORLD, transformation_matrix.n(), transformation_matrix.n());
  PETScWrappers::MPI::Vector system_rhs_copy(MPI_COMM_WORLD, system_rhs.size(), system_rhs.size());
  for (unsigned int i = 0; i < transformation_matrix.m(); ++i)
    system_rhs_copy(i) = system_rhs(i);
  system_rhs_copy.compress(VectorOperation::insert);

  // This print is commented to save time and space in the output.
  // pcout << "  Check system_rhs_copy values: " << std::endl;
  // pcout << "    system_rhs(40)      = " << system_rhs(40) << std::endl;
  // pcout << "    system_rhs_copy(40) = " << system_rhs_copy(40) << std::endl;
  // pcout << "    system_rhs(41)      = " << system_rhs(41) << std::endl;
  // pcout << "    system_rhs_copy(41) = " << system_rhs_copy(41) << std::endl;

  reduced_system_rhs = 0.0;
  
  assert(transformation_matrix.m() == system_rhs_copy.size());
  transformation_matrix.Tvmult(dst, system_rhs_copy);
  dst.compress(VectorOperation::insert);
  for (unsigned int i = 0; i < transformation_matrix.n(); ++i)
    reduced_system_rhs(i) = dst(i);
  reduced_system_rhs.compress(VectorOperation::insert);

  // This print is commented to save time and space in the output.
  // pcout << "  Check reduced_system_rhs size:\t" << reduced_system_rhs.size() << std::endl;
  // pcout << "  Check reduced_system_rhs values:" << std::endl;
  // pcout << "    reduced_system_rhs(0) = "  << reduced_system_rhs(0) << std::endl;
  // pcout << "    reduced_system_rhs(1) = "  << reduced_system_rhs(1) << std::endl;
}

// Expansion of the reduced order solution to the full order solution.
template <int dim>
void AdvDiffPOD<dim>::expand_solution(PETScWrappers::FullMatrix &transformation_matrix)
{
  PETScWrappers::MPI::Vector dst(MPI_COMM_WORLD, transformation_matrix.m(), transformation_matrix.m()); 
  PETScWrappers::MPI::Vector reduced_solution_copy(MPI_COMM_WORLD, reduced_solution.size(), reduced_solution.size()); 
  for(unsigned int i = 0; i < reduced_solution.size(); ++i)
    reduced_solution_copy(i) = reduced_solution(i);
  reduced_solution_copy.compress(VectorOperation::insert);

  // This print is commented to save time and space in the output.
  // pcout << "  Check reduced_solution_copy values: " << std::endl;
  // pcout << "    reduced_solution(0)      = " << reduced_solution(0) << std::endl;
  // pcout << "    reduced_solution_copy(0) = " << reduced_solution_copy(0) << std::endl;
  // pcout << "    reduced_solution(1)      = " << reduced_solution(1) << std::endl;
  // pcout << "    reduced_solution_copy(1) = " << reduced_solution_copy(1) << std::endl;

  // Note that at time 0 fom_solution is defined and contains the initial condition.
  fom_solution = 0.0;

  /**
  // Expansion: fom_solution = T * reduced_solution
  */
  assert(transformation_matrix.n() == reduced_solution_copy.size());
  transformation_matrix.vmult(dst, reduced_solution_copy);
  dst.compress(VectorOperation::insert);

  Vector<double> dst_aux(dst); // Copy constructor 

  fom_solution = dst_aux;

  fom_solution.compress(VectorOperation::insert);

  // This print is commented to save time and space in the output.
  // pcout << "  Check fom_solution size:\t" << fom_solution.size() << std::endl;
  // pcout << "  Check fom_solution values: " << std::endl;
  // pcout << "    fom_solution(2)  = " << fom_solution(2) << std::endl;
  // pcout << "    fom_solution(17) = " << fom_solution(17) << std::endl;
  // pcout << "    fom_solution(50) = " << fom_solution(50) << std::endl;
  // pcout << "    fom_solution(80) = " << fom_solution(80) << std::endl;
}

/**
// Solve the reduced order system. The matrix reduced_system_lhs is a FullMatrix<double> and it is not symmetric for construction,
// it is indeed obtained through the projection T^T * lhs_matrix * T. For this reason, we can't use the Conjugate Gradient solver,
// but we choose the GMRES solver. We can't explore different preconditioners because they are built for sparse matrices.
*/
template <int dim>
void AdvDiffPOD<dim>::solve_time_step_reduced()
{
  SolverControl solver_control(1000, 1e-6 * reduced_system_rhs.l2_norm());
  SolverGMRES<Vector<double>> solver(solver_control);

  solver.solve(reduced_system_lhs, reduced_solution, reduced_system_rhs, PreconditionIdentity());

  // This print is commented to save time and space in the output.
  // pcout << "  " << solver_control.last_step() << " GMRES iterations" << std::endl;
}

// template <int dim>
// void AdvDiffPOD<dim>::output(const unsigned int &time_step) const
// {
//   DataOut<dim> data_out;
//   data_out.add_data_vector(dof_handler_r, reduced_solution, "u");

//   // std::vector<unsigned int> partition_int(mesh_r.n_active_cells());
//   // GridTools::get_subdomain_association(mesh_r, partition_int);
//   // const Vector<double> partitioning(partition_int.begin(), partition_int.end());
//   // data_out.add_data_vector(partitioning, "partitioning");

//   data_out.build_patches();

//   const std::string output_file_name =
//     "output-" + std::to_string(modes[0].size()-1) + ".vtk";
//   std::ofstream output_file(output_file_name);
//   // data_out.write_vtk(output_file);

//   data_out.write_vtu_with_pvtu_record(
//     "./", "output", time_step, MPI_COMM_WORLD, 3);
// }

template <int dim>
void AdvDiffPOD<dim>::solve_reduced()
{
  assemble_matrices();
  setup_reduced();

  PETScWrappers::FullMatrix transformation_matrix;
  transformation_matrix.reinit(modes.size(), modes[0].size());

  pcout << "-------------------------------------------------------------------" << std::endl;
  pcout << "Assembling the transformation matrix" << std::endl;
  convert_modes(transformation_matrix);
  try
  {
    pcout << "-------------------------------------------------------------------" << std::endl;
    pcout << "Projecting the left-hand side" << std::endl;
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

    // Output the initial solution.
    // output(0);
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
  pcout << "Solving the reduced order system" << std::endl;
  while (time < T)
    {
      time += deltat;
      ++time_step;

      // This print is commented to save time and space in the output.
      // pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
      //       << time << ":" << std::flush;

      assemble_rhs(time);
      project_rhs(transformation_matrix);
      if (time_step == 1)
        pcout << "  Check reduced_system_rhs size:\t" << reduced_system_rhs.size() << std::endl;

      auto start_reduced = high_resolution_clock::now();
      solve_time_step_reduced();
      auto stop_reduced = high_resolution_clock::now();
      auto duration_reduced = duration_cast<milliseconds>(stop_reduced - start_reduced);
      duration_reduced_vec.push_back(duration_reduced);

      expand_solution(transformation_matrix);
      if (time_step == 1)
        pcout << "  Check fom_solution size:\t\t" << fom_solution.size() << std::endl;

      // output(time_step);
    }
  pcout << "===================================================================" << std::endl;

  // Compute the average duration of solving a single time step.
  duration_reduced_avg = std::reduce(duration_reduced_vec.begin(), duration_reduced_vec.end())/static_cast<double>(duration_reduced_vec.size());
}

template <int dim>
double AdvDiffPOD<dim>::compute_error(const VectorTools::NormType &norm_type)
{
  FE_Q<dim> fe_linear(1);
  MappingFE mapping(fe_linear);

  const unsigned int r = parameters.get_integer("degree");
  const QGauss<dim> quadrature_error = QGauss<dim>(r + 2);

  exact_solution.set_time(time);

  Vector<double> error_per_cell;
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    fom_solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  const double error =
    VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}
