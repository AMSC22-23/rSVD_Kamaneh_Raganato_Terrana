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

  pcout << "-----------------------------------------------" << std::endl;

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

    // quadrature_boundary = std::make_unique<QGauss<dim - 1>>(r + 1);

    // pcout << "  Quadrature points per boundary cell = "
    //           << quadrature_boundary->size() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;

    // MODIFIED FROM HERE
    // // Count the number of boundary DoFs.
    // dof_handler.n_boundary_dofs();
    // pcout << "  Number of boundary Dofs = " << dof_handler.n_boundary_dofs() << std::endl;

    // // Count the number of internal DoFs that will be Nh.
    // unsigned int n_internal_dofs = dof_handler.n_dofs() - dof_handler.n_boundary_dofs();
    // pcout << "  Number of internal Dofs = " << n_internal_dofs << std::endl;

    // // Try to extract the boundary DoFs.
    // // https://www.dealii.org/current/doxygen/deal.II/namespaceDoFTools.html#a06b3c33925c1a1f15de20deda20b4d21
    // const ComponentMask component_mask = ComponentMask();
    // const std::set<types::boundary_id> boundary_ids = {0, 1, 2, 3}; // #include <set>
    // const IndexSet boundary_dofs = DoFTools::extract_boundary_dofs(dof_handler, component_mask, boundary_ids);
    // std::vector<types::global_dof_index> boundary_dofs_idx;

    // for (auto it = boundary_dofs.begin(); it != boundary_dofs.end(); it++)
    //   boundary_dofs_idx.push_back(*it);

    // pcout << "  Check boundary_dofs_idx.size()    = " << boundary_dofs_idx.size() << std::endl;
    // for (const auto& dof : boundary_dofs_idx)
    //   pcout << "    " << dof << std::endl;
    // pcout << std::endl;

    // boundary_dofs_idx_int.assign(boundary_dofs_idx.begin(), boundary_dofs_idx.end());
    // pcout << "  Check boundary_dofs_idx_int.size() = " << boundary_dofs_idx_int.size() << std::endl;
    // TO HERE
  }

  pcout << "-----------------------------------------------" << std::endl;

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
    // solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
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

  pcout << "-----------------------------------------------" << std::endl;

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

    // quadrature_boundary = std::make_unique<QGauss<dim - 1>>(r + 1);

    // pcout << "  Quadrature points per boundary cell = "
    //           << quadrature_boundary->size() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler_r.reinit(mesh_r);
    dof_handler_r.distribute_dofs(*fe_r);

    locally_owned_dofs_r = dof_handler_r.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler_r, locally_relevant_dofs_r);

    pcout << "  Number of DoFs = " << dof_handler_r.n_dofs() << std::endl;

    // MODIFIED FROM HERE
    // // Count the number of boundary DoFs.
    // dof_handler.n_boundary_dofs();
    // pcout << "  Number of boundary Dofs = " << dof_handler.n_boundary_dofs() << std::endl;

    // // Count the number of internal DoFs that will be Nh.
    // unsigned int n_internal_dofs = dof_handler.n_dofs() - dof_handler.n_boundary_dofs();
    // pcout << "  Number of internal Dofs = " << n_internal_dofs << std::endl;

    // // Try to extract the boundary DoFs.
    // // https://www.dealii.org/current/doxygen/deal.II/namespaceDoFTools.html#a06b3c33925c1a1f15de20deda20b4d21
    // const ComponentMask component_mask = ComponentMask();
    // const std::set<types::boundary_id> boundary_ids = {0, 1, 2, 3}; // #include <set>
    // const IndexSet boundary_dofs = DoFTools::extract_boundary_dofs(dof_handler, component_mask, boundary_ids);
    // std::vector<types::global_dof_index> boundary_dofs_idx;

    // for (auto it = boundary_dofs.begin(); it != boundary_dofs.end(); it++)
    //   boundary_dofs_idx.push_back(*it);

    // pcout << "  Check boundary_dofs_idx.size()    = " << boundary_dofs_idx.size() << std::endl;
    // for (const auto& dof : boundary_dofs_idx)
    //   pcout << "    " << dof << std::endl;
    // pcout << std::endl;

    // boundary_dofs_idx_int.assign(boundary_dofs_idx.begin(), boundary_dofs_idx.end());
    // pcout << "  Check boundary_dofs_idx_int.size() = " << boundary_dofs_idx_int.size() << std::endl;
    // TO HERE
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    TrilinosWrappers::SparsityPattern sparsity_r(locally_owned_dofs_r,
                                                 MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler_r, sparsity_r);
    sparsity_r.compress();

    pcout << "  Initializing the matrices" << std::endl;
    // mass_matrix.reinit(sparsity_r);
    // stiffness_matrix.reinit(sparsity_r);
    reduced_system_lhs.reinit(sparsity_r); // prova ad aggiungere questo per risolvere il problema di mpi
    // rhs_matrix.reinit(sparsity_r);

    pcout << "  Initializing the system right-hand side" << std::endl;
    reduced_system_rhs.reinit(locally_owned_dofs_r, MPI_COMM_WORLD); // SCOMMENTANDO QUESTO IL CODICE FUNZIONA SENZA MPI
    pcout << "  Initializing the solution vector" << std::endl;
    reduced_solution_owned.reinit(locally_owned_dofs_r, MPI_COMM_WORLD);
    reduced_solution.reinit(locally_owned_dofs_r, locally_relevant_dofs_r, MPI_COMM_WORLD);
  }
}

void
AdvDiffPOD::assemble_matrices()
{
  pcout << "===============================================" << std::endl;
  pcout << "Assembling the system matrices" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  // FEFaceValues<dim> fe_values_boundary(*fe,
  //                                      *quadrature_boundary,
  //                                      update_values |
  //                                        update_quadrature_points |
  //                                        update_JxW_values);

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

          // Convert the transport term to a tensor (so that we can use it with
          // scalar_product).
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

  // We build the matrix on the left-hand side of the algebraic problem (the one
  // that we'll invert at each timestep).
  lhs_matrix.copy_from(mass_matrix);
  lhs_matrix.add(theta, stiffness_matrix);

  // We build the matrix on the right-hand side (the one that multiplies the old
  // solution un).
  rhs_matrix.copy_from(mass_matrix);
  rhs_matrix.add(-(1.0 - theta), stiffness_matrix);
}

void
AdvDiffPOD::assemble_rhs(const double &time, TrilinosWrappers::SparseMatrix &snapshot_matrix_trilinos)
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
          // We need to compute the forcing term at the current time (tn+1) and
          // at the old time (tn). deal.II Functions can be computed at a
          // specific time by calling their set_time method.

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
  TrilinosWrappers::MPI::Vector aux(locally_owned_dofs, MPI_COMM_WORLD);
  // pcout << "  Check aux.size() = " << aux.size() << std::endl;
  // pcout << "  Check solution_owned.size() = " << solution_owned.size() << std::endl;
  for (unsigned int i = 0; i < aux.size(); ++i)
    // aux.add(i, snapshot_matrix[i][(time-deltat)/deltat]);
    aux(i) = snapshot_matrix_trilinos(i, (time-deltat)/deltat);
  aux.compress(VectorOperation::add);
  pcout << "BLOCCO QUI" << std::endl;

  rhs_matrix.vmult_add(system_rhs, aux); // QUI TI SERVIREBBE NUOVA SOLUTION
  
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
      boundary_values, lhs_matrix, solution_owned, system_rhs, false);
  }
}

void
AdvDiffPOD::assemble_reduced_rhs(const double &time)
{
  const unsigned int dofs_per_cell_r = fe_r->dofs_per_cell;
  const unsigned int n_q_r           = quadrature_r->size();

  FEValues<dim> fe_values_r(*fe_r,
                            *quadrature_r,
                            update_values | update_quadrature_points |
                            update_JxW_values);

  Vector<double> cell_rhs_r(dofs_per_cell_r);

  std::vector<types::global_dof_index> dof_indices_r(dofs_per_cell_r);

  reduced_system_rhs = 0.0;

  for (const auto &cell : dof_handler_r.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values_r.reinit(cell);

      cell_rhs_r = 0.0;

      for (unsigned int q = 0; q < n_q_r; ++q)
        {
          // We need to compute the forcing term at the current time (tn+1) and
          // at the old time (tn). deal.II Functions can be computed at a
          // specific time by calling their set_time method.

          // Compute f(tn+1)
          forcing_term.set_time(time);
          const double f_new_loc =
            forcing_term.value(fe_values_r.quadrature_point(q));

          // Compute f(tn)
          forcing_term.set_time(time - deltat);
          const double f_old_loc =
            forcing_term.value(fe_values_r.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell_r; ++i)
            {
              cell_rhs_r(i) += (theta * f_new_loc + (1.0 - theta) * f_old_loc) *
                               fe_values_r.shape_value(i, q) * fe_values_r.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices_r);
      reduced_system_rhs.add(dof_indices_r, cell_rhs_r);
    }

  reduced_system_rhs.compress(VectorOperation::add);

  // Add the term that comes from the old solution.
  reduced_rhs_matrix.vmult_add(reduced_system_rhs, reduced_solution_owned); // QUI TI SERVIREBBE NUOVA SOLUTION
  
  // Boundary conditions.
  {
    std::map<types::global_dof_index, double> boundary_values;

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    boundary_functions[0] = &function_g;
    boundary_functions[1] = &function_g;

    VectorTools::interpolate_boundary_values(dof_handler_r,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
      boundary_values, reduced_system_lhs, reduced_solution_owned, reduced_system_rhs, false);
  }
}

// // COMMENTA PROIEZIONE
// void
// AdvDiffPOD::convert_modes(FullMatrix<double> &transformation_matrix)
// {
//   // transformation_matrix.copy_from(modes);
//   // transformation_matrix.reinit(modes.size(), modes[0].size());
//   // transformation_matrix.m() = modes.size();
//   // transformation_matrix.n() = modes[0].size();
//   // TrilinosWrappers::SparseMatrix transformation_matrix(modes.size(), modes[0].size());
//   // TrilinosWrappers::SparseMatrix transformation_matrix(static_cast<unsigned int>(modes.size()), 
//   //                                                      static_cast<unsigned int>(modes[0].size()));
//   for (unsigned int i = 0; i < modes.size(); ++i)
//     for (unsigned int j = 0; j < modes[0].size(); ++j)
//   //     // transformation_matrix.set(i, j, modes[i][j]);
//       transformation_matrix(i, j) = modes[i][j];
//   // transformation_matrix.copy_from(modes); // vedi se l'errore di .m e .n è qui e mi sa che era qui ...

//   pcout << "  Check transformation_matrix: " << std::endl;
//   pcout << modes[0][0] << std::endl;
//   pcout << transformation_matrix(0, 0) << std::endl;
//   pcout << modes[1][0] << std::endl;
//   pcout << transformation_matrix(1, 0) << std::endl;
//   pcout << modes[1][1] << std::endl;
//   pcout << transformation_matrix(1, 1) << std::endl;
//   pcout << modes[1][40] << std::endl;
//   pcout << transformation_matrix(1, 40) << std::endl;
//   pcout << "  Check transformation_matrix: size" << std::endl;
//   pcout << modes.size() << std::endl;
//   pcout << transformation_matrix.m() << std::endl;
//   pcout << modes[0].size() << std::endl;
//   pcout << transformation_matrix.n() << std::endl;
// }

// void
// AdvDiffPOD::project_u0(FullMatrix<double> &transformation_matrix)
// {
//   // QUI MI SA PROBLEMA CHE u_0 non è vettore ma funzione
//   // reduced_u_0.Tvmult(transformation_matrix, u_0);
//   // reduced_u_0 = transformation_matrixT * u_0;

//   Vector<double> dst(solution_owned.size());
//   Vector<double> solution_owned_copy(solution_owned.size());
//   for(unsigned int i = 0; i < solution_owned.size(); ++i)
//     solution_owned_copy(i) = solution_owned(i);
//   // solution_owned_copy.copy_from(solution_owned);
// //   std::vector<Point<dim>> &points;


// // // template <int dim>
// // // void function_to_vector(const Function<dim> &function, Vector<double> &vector, const std::vector<Point<dim>> &points)
// // Point<dim> & p
// // // {
// // //   const unsigned int n_points = points.size();
// // //   vector.reinit(n_points);

// // //   for (unsigned int i = 0; i < n_points; ++i)
// // //   {
// // //     vector[i] = function.value(points[i]);
// // //   }
// // // }


//   // Matrix-vector multiplication: let dst = MT*src with M being this matrix. This function does the same as vmult() but takes
//   // the transposed matrix.
//   // aux.reinit(solution_owned.size());
//   // aux.reinit(solution_owned); // DIMENSIONE

//   transformation_matrix.Tvmult(dst, solution_owned_copy);
//   // solution_owned.reinit(dst);
//   for (unsigned int i = 0; i < dst.size(); ++i)
//     solution_owned(i) = dst(i);
//   // solution_owned.set(); // MAGARI PER AVERE MENO CONFLITTI DATO CHE NON SO COME VIENE FATTA OPERAZIONE
//   // solution_owned.Tvmult(transformation_matrix, solution_owned);
// }

// void
// AdvDiffPOD::project_lhs(FullMatrix<double> &transformation_matrix)
// {
//   // TrilinosWrappers::SparseMatrix aux(static_cast<unsigned int>(lhs_matrix.m()), static_cast<unsigned int>(lhs_matrix.n())); // ANCHE QUI SERVE DIMENSIONE?
//   // aux.Tmmult(transformation_matrix, lhs_matrix);
//   // reduced_system_lhs.mmult(aux, transformation_matrix);
//   // ATTENZIONE SISTEMA CAPENDO INPUT E OUTPUT
//   // aux.reinit(lhs_matrix.size());
//   FullMatrix<double> aux(transformation_matrix.n(), lhs_matrix.n()); // (Tn * Tm) * (Lm * Ln) = Tn * Ln
//   FullMatrix<double> dst(transformation_matrix.n(), transformation_matrix.n()); // (Tn * Ln) * (Tm * Tn) = Tn * Tn
//   FullMatrix<double> lhs_matrix_copy(lhs_matrix.m(), lhs_matrix.n());
//   for (unsigned int i = 0; i < lhs_matrix.m(); ++i)
//     for (unsigned int j = 0; j < lhs_matrix.n(); ++j)
//       lhs_matrix_copy(i, j) = lhs_matrix(i, j);
//   // lhs_matrix_copy.copy_from(lhs_matrix); // non mi sembra funzionare

//   pcout << "  Check lhs_matrix_copy: " << std::endl;
//   pcout << lhs_matrix(0, 0) << std::endl;
//   pcout << lhs_matrix_copy(0, 0) << std::endl;
//   pcout << lhs_matrix(1, 0) << std::endl;
//   pcout << lhs_matrix_copy(1, 0) << std::endl;

//   transformation_matrix.Tmmult(aux, lhs_matrix_copy);
//   aux.mmult(dst, transformation_matrix);





//   // PROVA A COMPORRE VETTORE PER VETTORE
//   // std::vector<size_t> indexes(transformation_matrix.n());
//   // for (size_t i = 0; i < transformation_matrix.n(); i++)
//   //   indexes.push_back(i);


//   // TrilinosWrappers::MPI::Vector aux_vec(transformation_matrix.n());
//   // for (unsigned int i = 0; i < transformation_matrix.n(); ++i)
//   // {
//   //   for (unsigned int j = 0; j < transformation_matrix.n(); ++j)
//   //   {
//   //     aux_vec(i) = dst(i, j);
//   //   }
//   //   reduced_system_lhs.set(indexes, aux_vec);
//   // }

//   //   reduced_system_lhs.compress(VectorOperation::insert);





//   std::vector<unsigned int> indexes(transformation_matrix.n());
//   for (unsigned int i = 0; i < transformation_matrix.n(); i++)
//     indexes.push_back(i);

//   // std::vector<size_t> indexes(transformation_matrix.n());
//   // for (size_t i = 0; i < transformation_matrix.n(); i++)
//     // indexes.push_back(i);
//   // reduced_system_rhs.set(indexes, indexes, dst);

//   // IndexSet all_indexes(transformation_matrix.n());
//   // for (unsigned int i=0; i<transformation_matrix.n(); ++i)
//   //   all_indexes.add_index(i);
//   // TrilinosWrappers::SparsityPattern all(all_indexes, MPI_COMM_WORLD);
//   // all.compress();
//   // reduced_system_lhs.reinit(all);
//   // reduced_system_lhs.set(all_indexes, dst);
//   // reduced_system_lhs.compress(VectorOperation::add); // prova a vedere se codì non escono gli zeri
// // .......
//   //   std::vector<unsigned int> row_indices;
//   //   std::vector<unsigned int> col_indices;
//   //   std::vector<double> values;

//   //     for (unsigned int i = 0; i < transformation_matrix.n(); ++i)
//   //   {
//   //       for (unsigned int j = 0; j < transformation_matrix.n(); ++j)
//   //       {
//   //           if (dst(i, j) != 0.0)
//   //           {
//   //               row_indices.push_back(i);
//   //               col_indices.push_back(j);
//   //               values.push_back(dst(i, j));
//   //           }
//   //       }
//   //   }

//   //   // Set values in reduced_system_lhs
//   //   reduced_system_lhs.set(row_indices, col_indices, values);

//   //   // Compress to finalize the matrix assembly
//   //   reduced_system_lhs.compress(VectorOperation::insert);
// // :::::::::::::::

//   // for (unsigned int i = 0; i < transformation_matrix.n(); i++)
//   //   for (unsigned int j = 0; j < transformation_matrix.n(); j++)
//   //   {
//   //     double aux = dst(i, j);
//   //     reduced_system_lhs.add(i, j, aux); // MI SA CHE STO COSO NON FUNZIONA PERCHÉ STAMPANDO ESCONO ZERI
//   //     // prova con add perché tanto matrice inizializzata a zero
//   //     // reduced_system_lhs(i, j) = dst(i, j);
//   //   }
//     // reduced_system_lhs.add(indexes, dst);

//     reduced_system_lhs.compress(VectorOperation::add); // prova a vedere se codì non escono gli zeri
//         reduced_system_lhs.add(indexes, dst);


  
//     pcout << "  Check reduced_system_lhs: " << std::endl;
//   pcout << dst(0, 0) << std::endl;
//   pcout << reduced_system_lhs(0, 0) << std::endl;
//   pcout << dst(1, 0) << std::endl;
//   pcout << reduced_system_lhs(1, 0) << std::endl;


//   // check dim
//   // if
//   // MPI_ABORT(MPI_COMM_WORLD, 1);
// }

// void
// AdvDiffPOD::project_rhs(FullMatrix<double> &transformation_matrix)
// {
//   // reduced_system_rhs.Tvmult(transformation_matrix, system_rhs);
//   Vector<double> dst(transformation_matrix.n());
//   Vector<double> system_rhs_copy(transformation_matrix.m());

//   // FAI CHECK SU DIMENSIONI CON ERRORE
//   for (unsigned int i = 0; i < transformation_matrix.m(); ++i)
//     system_rhs_copy(i) = system_rhs(i);

//     pcout << "  Check rhs_matrix_copy: " << std::endl;
//   pcout << system_rhs(40) << std::endl;
//   pcout << system_rhs_copy(40) << std::endl;
//   pcout << system_rhs(41) << std::endl;
//   pcout << system_rhs_copy(41) << std::endl;

//   transformation_matrix.Tvmult(dst, system_rhs_copy);
//   // reduced_system_rhs.reinit(dst);
//   for (unsigned int i = 0; i < transformation_matrix.n(); ++i)
//     reduced_system_rhs(i) = dst(i); // RICERCATI POI DEFINIZIONE DELLE VARIE FUNZIONI PER I VARI OGGETTI

//       pcout << "  Check reduced_system_rhs: " << std::endl;
//   pcout << dst(40) << std::endl;
//   pcout << reduced_system_rhs(40) << std::endl;
//   pcout << dst(41) << std::endl;
//   pcout << reduced_system_rhs(41) << std::endl;
// }

// COMMENTA PROIEZIONE
void
AdvDiffPOD::convert_modes(TrilinosWrappers::SparseMatrix &transformation_matrix)
{
  // transformation_matrix.copy_from(modes);
  // transformation_matrix.reinit(modes.size(), modes[0].size());
  // transformation_matrix.m() = modes.size();
  // transformation_matrix.n() = modes[0].size();
  // TrilinosWrappers::SparseMatrix transformation_matrix(modes.size(), modes[0].size());
  // TrilinosWrappers::SparseMatrix transformation_matrix(static_cast<unsigned int>(modes.size()), 
  //                                                      static_cast<unsigned int>(modes[0].size()));
  for (unsigned int i = 0; i < modes.size(); ++i)
    for (unsigned int j = 0; j < modes[0].size(); ++j)
  //     // transformation_matrix.set(i, j, modes[i][j]);
      // transformation_matrix(i, j) = modes[i][j];
      transformation_matrix.set(i, j, modes[i][j]);
  transformation_matrix.compress(VectorOperation::add);
  // transformation_matrix.copy_from(modes); // vedi se l'errore di .m e .n è qui e mi sa che era qui ...

  pcout << "  Check transformation_matrix: " << std::endl;
  pcout << modes[0][0] << std::endl;
  pcout << transformation_matrix(0, 0) << std::endl;
  pcout << modes[1][0] << std::endl;
  pcout << transformation_matrix(1, 0) << std::endl;
  pcout << modes[1][1] << std::endl;
  pcout << transformation_matrix(1, 1) << std::endl;
  pcout << modes[40][1] << std::endl;
  pcout << transformation_matrix(40, 1) << std::endl;
  pcout << modes[44][1] << std::endl;
  pcout << transformation_matrix(44, 1) << std::endl;
  pcout << modes[27][0] << std::endl;
  pcout << transformation_matrix(27, 0) << std::endl;
  pcout << "  Check transformation_matrix: size" << std::endl;
  pcout << modes.size() << std::endl;
  pcout << transformation_matrix.m() << std::endl;
  pcout << modes[0].size() << std::endl;
  pcout << transformation_matrix.n() << std::endl;
}

void
AdvDiffPOD::project_u0(TrilinosWrappers::SparseMatrix &transformation_matrix)
{
  // QUI MI SA PROBLEMA CHE u_0 non è vettore ma funzione
  // reduced_u_0.Tvmult(transformation_matrix, u_0);
  // reduced_u_0 = transformation_matrixT * u_0;

  // Vector<double> dst(solution_owned.size());
  // Vector<double> solution_owned_copy(solution_owned.size());
  // for(unsigned int i = 0; i < solution_owned.size(); ++i)
  //   solution_owned_copy(i) = solution_owned(i);
  // solution_owned_copy.copy_from(solution_owned);
//   std::vector<Point<dim>> &points;


// // template <int dim>
// // void function_to_vector(const Function<dim> &function, Vector<double> &vector, const std::vector<Point<dim>> &points)
// Point<dim> & p
// // {
// //   const unsigned int n_points = points.size();
// //   vector.reinit(n_points);

// //   for (unsigned int i = 0; i < n_points; ++i)
// //   {
// //     vector[i] = function.value(points[i]);
// //   }
// // }


  // Matrix-vector multiplication: let dst = MT*src with M being this matrix. This function does the same as vmult() but takes
  // the transposed matrix.
  // aux.reinit(solution_owned.size());
  // aux.reinit(solution_owned); // DIMENSIONE

  // TrilinosWrappers::MPI::Vector dst;
  reduced_solution_owned = 0.0;
  transformation_matrix.Tvmult(reduced_solution_owned, solution_owned);
  reduced_solution_owned.compress(VectorOperation::add);
  // solution_owned.reinit(dst);
  // for (unsigned int i = 0; i < dst.size(); ++i)
  //   solution_owned(i) = dst(i);
  // solution_owned.set(); // MAGARI PER AVERE MENO CONFLITTI DATO CHE NON SO COME VIENE FATTA OPERAZIONE
  // solution_owned.Tvmult(transformation_matrix, solution_owned);
}

void
AdvDiffPOD::project_lhs(TrilinosWrappers::SparseMatrix &transformation_matrix)
{
  // TrilinosWrappers::SparseMatrix aux(static_cast<unsigned int>(lhs_matrix.m()), static_cast<unsigned int>(lhs_matrix.n())); // ANCHE QUI SERVE DIMENSIONE?
  // aux.Tmmult(transformation_matrix, lhs_matrix);
  // reduced_system_lhs.mmult(aux, transformation_matrix);
  // ATTENZIONE SISTEMA CAPENDO INPUT E OUTPUT
  // aux.reinit(lhs_matrix.size());
  // FullMatrix<double> aux(transformation_matrix.n(), lhs_matrix.n()); // (Tn * Tm) * (Lm * Ln) = Tn * Ln
  // FullMatrix<double> dst(transformation_matrix.n(), transformation_matrix.n()); // (Tn * Ln) * (Tm * Tn) = Tn * Tn
  // FullMatrix<double> lhs_matrix_copy(lhs_matrix.m(), lhs_matrix.n());
  // for (unsigned int i = 0; i < lhs_matrix.m(); ++i)
  //   for (unsigned int j = 0; j < lhs_matrix.n(); ++j)
  //     lhs_matrix_copy(i, j) = lhs_matrix(i, j);
  // lhs_matrix_copy.copy_from(lhs_matrix); // non mi sembra funzionare

  // pcout << "  Check lhs_matrix_copy: " << std::endl;
  // pcout << lhs_matrix(0, 0) << std::endl;
  // pcout << lhs_matrix_copy(0, 0) << std::endl;
  // pcout << lhs_matrix(1, 0) << std::endl;
  // pcout << lhs_matrix_copy(1, 0) << std::endl;

  TrilinosWrappers::SparseMatrix aux(locally_owned_dofs, MPI_COMM_WORLD); // AGGIUNTO Locally owned
  reduced_system_lhs = 0.0;
  transformation_matrix.Tmmult(aux, lhs_matrix);
  aux.mmult(reduced_system_lhs, transformation_matrix);
  reduced_system_lhs.compress(VectorOperation::add);




  // PROVA A COMPORRE VETTORE PER VETTORE
  // std::vector<size_t> indexes(transformation_matrix.n());
  // for (size_t i = 0; i < transformation_matrix.n(); i++)
  //   indexes.push_back(i);


  // TrilinosWrappers::MPI::Vector aux_vec(transformation_matrix.n());
  // for (unsigned int i = 0; i < transformation_matrix.n(); ++i)
  // {
  //   for (unsigned int j = 0; j < transformation_matrix.n(); ++j)
  //   {
  //     aux_vec(i) = dst(i, j);
  //   }
  //   reduced_system_lhs.set(indexes, aux_vec);
  // }

  //   reduced_system_lhs.compress(VectorOperation::insert);





  // std::vector<unsigned int> indexes(transformation_matrix.n());
  // for (unsigned int i = 0; i < transformation_matrix.n(); i++)
  //   indexes.push_back(i);

  // std::vector<size_t> indexes(transformation_matrix.n());
  // for (size_t i = 0; i < transformation_matrix.n(); i++)
    // indexes.push_back(i);
  // reduced_system_rhs.set(indexes, indexes, dst);

  // IndexSet all_indexes(transformation_matrix.n());
  // for (unsigned int i=0; i<transformation_matrix.n(); ++i)
  //   all_indexes.add_index(i);
  // TrilinosWrappers::SparsityPattern all(all_indexes, MPI_COMM_WORLD);
  // all.compress();
  // reduced_system_lhs.reinit(all);
  // reduced_system_lhs.set(all_indexes, dst);
  // reduced_system_lhs.compress(VectorOperation::add); // prova a vedere se codì non escono gli zeri
// .......
  //   std::vector<unsigned int> row_indices;
  //   std::vector<unsigned int> col_indices;
  //   std::vector<double> values;

  //     for (unsigned int i = 0; i < transformation_matrix.n(); ++i)
  //   {
  //       for (unsigned int j = 0; j < transformation_matrix.n(); ++j)
  //       {
  //           if (dst(i, j) != 0.0)
  //           {
  //               row_indices.push_back(i);
  //               col_indices.push_back(j);
  //               values.push_back(dst(i, j));
  //           }
  //       }
  //   }

  //   // Set values in reduced_system_lhs
  //   reduced_system_lhs.set(row_indices, col_indices, values);

  //   // Compress to finalize the matrix assembly
  //   reduced_system_lhs.compress(VectorOperation::insert);
// :::::::::::::::

  // for (unsigned int i = 0; i < transformation_matrix.n(); i++)
  //   for (unsigned int j = 0; j < transformation_matrix.n(); j++)
  //   {
  //     double aux = dst(i, j);
  //     reduced_system_lhs.add(i, j, aux); // MI SA CHE STO COSO NON FUNZIONA PERCHÉ STAMPANDO ESCONO ZERI
  //     // prova con add perché tanto matrice inizializzata a zero
  //     // reduced_system_lhs(i, j) = dst(i, j);
  //   }
    // reduced_system_lhs.add(indexes, dst);

    // reduced_system_lhs.compress(VectorOperation::add); // prova a vedere se codì non escono gli zeri
    //     reduced_system_lhs.add(indexes, dst);


  
  pcout << "  Check reduced_system_lhs: " << std::endl;
  // pcout << dst(0, 0) << std::endl;
  pcout << reduced_system_lhs(0, 0) << std::endl;
  // pcout << dst(1, 0) << std::endl;
  pcout << reduced_system_lhs(1, 0) << std::endl;


  // check dim
  // if
  // MPI_ABORT(MPI_COMM_WORLD, 1);
}

void
AdvDiffPOD::project_rhs(TrilinosWrappers::SparseMatrix &transformation_matrix)
{
  // reduced_system_rhs.Tvmult(transformation_matrix, system_rhs);
  // Vector<double> dst(transformation_matrix.n());
  // Vector<double> system_rhs_copy(transformation_matrix.m());

  // // FAI CHECK SU DIMENSIONI CON ERRORE
  // for (unsigned int i = 0; i < transformation_matrix.m(); ++i)
  //   system_rhs_copy(i) = system_rhs(i);

  //   pcout << "  Check rhs_matrix_copy: " << std::endl;
  // pcout << system_rhs(40) << std::endl;
  // pcout << system_rhs_copy(40) << std::endl;
  // pcout << system_rhs(41) << std::endl;
  // pcout << system_rhs_copy(41) << std::endl;
  // PROBLEMINO QUI
  reduced_system_rhs = 0.0;
  transformation_matrix.Tvmult(reduced_system_rhs, system_rhs);
  reduced_system_rhs.compress(VectorOperation::add);
  // reduced_system_rhs.reinit(dst);
  // for (unsigned int i = 0; i < transformation_matrix.n(); ++i)
  //   reduced_system_rhs(i) = dst(i); // RICERCATI POI DEFINIZIONE DELLE VARIE FUNZIONI PER I VARI OGGETTI

  pcout << "  Check reduced_system_rhs: " << std::endl;
  // pcout << dst(40) << std::endl;
  pcout << reduced_system_rhs(0) << std::endl;
  // pcout << dst(41) << std::endl;
  pcout << reduced_system_rhs(1) << std::endl;
}

void
AdvDiffPOD::project_rhs_matrix(TrilinosWrappers::SparseMatrix &transformation_matrix)
{
  // TrilinosWrappers::SparseMatrix aux(static_cast<unsigned int>(lhs_matrix.m()), static_cast<unsigned int>(lhs_matrix.n())); // ANCHE QUI SERVE DIMENSIONE?
  // aux.Tmmult(transformation_matrix, lhs_matrix);
  // reduced_system_lhs.mmult(aux, transformation_matrix);
  // ATTENZIONE SISTEMA CAPENDO INPUT E OUTPUT
  // aux.reinit(lhs_matrix.size());
  // FullMatrix<double> aux(transformation_matrix.n(), lhs_matrix.n()); // (Tn * Tm) * (Lm * Ln) = Tn * Ln
  // FullMatrix<double> dst(transformation_matrix.n(), transformation_matrix.n()); // (Tn * Ln) * (Tm * Tn) = Tn * Tn
  // FullMatrix<double> lhs_matrix_copy(lhs_matrix.m(), lhs_matrix.n());
  // for (unsigned int i = 0; i < lhs_matrix.m(); ++i)
  //   for (unsigned int j = 0; j < lhs_matrix.n(); ++j)
  //     lhs_matrix_copy(i, j) = lhs_matrix(i, j);
  // lhs_matrix_copy.copy_from(lhs_matrix); // non mi sembra funzionare

  // pcout << "  Check lhs_matrix_copy: " << std::endl;
  // pcout << lhs_matrix(0, 0) << std::endl;
  // pcout << lhs_matrix_copy(0, 0) << std::endl;
  // pcout << lhs_matrix(1, 0) << std::endl;
  // pcout << lhs_matrix_copy(1, 0) << std::endl;

  TrilinosWrappers::SparseMatrix aux;
  reduced_rhs_matrix = 0.0;
  transformation_matrix.Tmmult(aux, rhs_matrix);
  aux.mmult(reduced_rhs_matrix, transformation_matrix);
  reduced_rhs_matrix.compress(VectorOperation::add);


  
  pcout << "  Check reduced_rhs_matrix: " << std::endl;
  // pcout << dst(0, 0) << std::endl;
  pcout << reduced_rhs_matrix(0, 0) << std::endl;
  // pcout << dst(1, 0) << std::endl;
  pcout << reduced_rhs_matrix(1, 0) << std::endl;
}



void
AdvDiffPOD::solve_time_step_reduced()
{
  SolverControl solver_control(1000, 1e-6 * reduced_system_rhs.l2_norm());

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionSSOR      preconditioner;
  preconditioner.initialize(
    reduced_system_lhs, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  solver.solve(reduced_system_lhs, reduced_solution_owned, reduced_system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;

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

  // convertire matrice modes
  // FullMatrix<double> transformation_matrix(modes.size(), modes[0].size());
  TrilinosWrappers::SparseMatrix transformation_matrix(modes.size(), modes[0].size(), modes[0].size());

  convert_modes(transformation_matrix);
  project_lhs(transformation_matrix);

  pcout << "  Check lhs:" << reduced_system_lhs.m() << " * " << reduced_system_lhs.n() << std::endl;
  pcout << reduced_system_lhs(0, 0) << std::endl;


  pcout << snapshot_matrix[0][0] << std::endl;

  pcout << "===============================================SEPARA" << std::endl;
  // PROVA, al massimo poi differenzia funzione per non rendere questa troppo lunga
  TrilinosWrappers::SparseMatrix snapshot_matrix_trilinos(snapshot_matrix.size(),
    snapshot_matrix[0].size(), snapshot_matrix[0].size());
  pcout << "===============================================" << std::endl;
  pcout << "Converting the snapshot matrix" << std::endl;

//   const unsigned int dofs_per_cell_s = fe->dofs_per_cell;
//     const unsigned int n_q           = quadrature->size();


//   FullMatrix<double> cell_snapshot_matrix(dofs_per_cell_s, dofs_per_cell_s);

//   std::vector<types::global_dof_index> dof_indices_s(dofs_per_cell_s);

//   snapshot_matrix_trilinos = 0.0;

//   for (const auto &cell : dof_handler.active_cell_iterators())
//     {
//       if (!cell->is_locally_owned())
//         continue;

//       // fe_values.reinit(cell);

//       cell_snapshot_matrix = 0.0;

//       for (unsigned int q = 0; q < n_q; ++q)
//         {
//           for (unsigned int i = 0; i < dofs_per_cell_s; ++i)
//             {
//               for (unsigned int j = 0; j < dofs_per_cell_s; ++j)
//                 {
//                   cell_snapshot_matrix(i, j) = snapshot_matrix[i][j];    
//                 }
//             }
//         }

//       cell->get_dof_indices(dof_indices_s);

//       snapshot_matrix_trilinos.add(dof_indices_s, cell_snapshot_matrix);
//     }

//   snapshot_matrix_trilinos.compress(VectorOperation::add);
// pcout << "FINE PROVA" << std::endl;




  // convert_modes(snapshot_matrix_trilinos); SARÀ CONVERT SNAPSHOTS
  for (unsigned int i = 0; i < snapshot_matrix.size(); ++i)
    for (unsigned int j = 0; j < snapshot_matrix[0].size(); ++j)
  //     // transformation_matrix.set(i, j, modes[i][j]);
      // transformation_matrix(i, j) = modes[i][j];
      snapshot_matrix_trilinos.set(i, j, snapshot_matrix[i][j]);
  snapshot_matrix_trilinos.compress(VectorOperation::add);

  
  pcout << "  Check snapshot_matrix_trilinos: " << std::endl;
  pcout << snapshot_matrix[0][0] << std::endl;
  pcout << snapshot_matrix_trilinos(0, 0) << std::endl;
  pcout << snapshot_matrix[0][1] << std::endl;
  pcout << snapshot_matrix_trilinos(0, 1) << std::endl;







  



  pcout << "===============================================" << std::endl;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;

    // project_u0();
    // VectorTools::interpolate(dof_handler_r, u_0, solution_owned); // dato che u_0 è funzione prova a interpolare direttamente, lasciando perdere per un attimo proiezion
    // solution = solution_owned;

    // DOPO PROVA, mi sembra più giusto concettualmente
    VectorTools::interpolate(dof_handler, u_0, solution_owned);
    project_u0(transformation_matrix);
    reduced_solution = reduced_solution_owned;

    // Output the initial solution.
    output(0);
    pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;
  double       time      = 0;

  while (time < T)
    {
      time += deltat;
      ++time_step;

      pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << time << ":" << std::flush;

      // if (time_step == 1)
      // {
      //   assemble_rhs(time);
      //   project_rhs(transformation_matrix);
      //   project_rhs_matrix(transformation_matrix); // per usarla in tutti gli step successivi un po' come lhs
      // }
      // else
      //   assemble_reduced_rhs(time); // Otherwise we would need solution_owned to assemble the reduced system.

      // Prova usando snapshot_matrix come solution_owned full
      // direi ad ogni tempo di prendere solution_owned come snapshot_matrix colonna al tempo precedente
      assemble_rhs(time, snapshot_matrix_trilinos);
      project_rhs(transformation_matrix);

      solve_time_step_reduced();
      output(time_step);
    }

  // RIPROIETTA SU FULL, ma mi sa scritto fuori, in caso aggiungi solution_fom in hpp
  // for (unsigned int i = 0; i < modes.size(); ++i)
  //   for (unsigned int j = 0; j < modes[0].size(); ++j)
  //     solution_fom += modes[i][j] * solution_owned[j];
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