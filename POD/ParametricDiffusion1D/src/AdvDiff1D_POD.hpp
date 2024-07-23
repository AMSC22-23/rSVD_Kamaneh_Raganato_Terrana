#ifndef ADV_DIFF_1D_POD_HPP
#define ADV_DIFF_1D_POD_HPP

/**
 * The aim of this class is to solve the advection diffusion reduced order problem in 1D.
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
// #include <deal.II/lac/full_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/parameter_handler.h> // https://www.dealii.org/current/doxygen/deal.II/classParameterHandler.html

#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>

// #include <Eigen/Dense>
// #include <unsupported/Eigen/MatrixFunctions>
// #include <Eigen/Sparse>
// #include <unsupported/Eigen/SparseExtra>

// using Mat_m = Eigen::MatrixXd;
// using Vec_v = Eigen::VectorXd;

using namespace dealii;

// Class representing the linear diffusion problem.
class AdvDiffPOD
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 1;

  // Diffusion coefficient.
  class DiffusionCoefficient : public Function<dim>
  {
  public:
    // Constructor.
    DiffusionCoefficient()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      return std::pow(p[0], 4);
    }
  };

  // Reaction coefficient.
  // class ReactionCoefficient : public Function<dim>
  // {
  // public:
  //   // Constructor.
  //   ReactionCoefficient()
  //   {}

  //   // Evaluation.
  //   virtual double
  //   value(const Point<dim> & /*p*/,
  //         const unsigned int /*component*/ = 0) const override
  //   {
  //     return 1.0;
  //   }
  // };

  // Transport coefficient.
  class TransportCoefficient : public Function<dim>
  {
  public:
    // Constructor.
    TransportCoefficient()
    {}

    // Evaluation.
    virtual void
    vector_value(const Point<dim> & /*p*/,
                 Vector<double> &values) const override
    {
      values[0] = 2.0;
    }

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      if (component == 0)
        return 2.0;
      else
        return 0.0;
    }
  };

  // Forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    // Constructor.
    ForcingTerm()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

    // Dirichlet boundary conditions.
  class FunctionG : public Function<dim>
  {
  public:
    // Constructor.
    FunctionG()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> &/*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  // Neumann boundary conditions.
  // class FunctionH : public Function<dim>
  // {
  // public:
  //   // Constructor.
  //   FunctionH()
  //   {}

  //   // Evaluation:
  //   virtual double
  //   value(const Point<dim> &/*p*/,
  //         const unsigned int /*component*/ = 0) const override
  //   {
  //     return 0.0;
  //   }
  // };

  // Function for the initial condition.
  class FunctionU0 : public Function<dim>
  {
  public:
    // Constructor.
    FunctionU0()
    {}

    FunctionU0(const std::vector<double> initial_state) : u0(initial_state)
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      if (u0.empty())
        return std::sin(M_PI*p[0]);
      else
      { // QUESTO SICURAMENTE NON CORRETTO
        for (unsigned int i = 0; i < u0.size(); ++i)
          return u0[i];
      }
        // return u0 * p[0];
    }

    private:
      std::vector<double> u0;

    // // Evaluation.
    // virtual double
    // value(const Point<dim> &p,
    //       const unsigned int /*component*/ = 0) const override
    // {
    //   if (initial_state.empty())
    //     return std::sin(M_PI*p[0]);
    //   else
    //     return initial_state;
    // }
  };

  // Exact solution.
  // class ExactSolution : public Function<dim>
  // {
  // public:
  //   // Constructor.
  //   ExactSolution()
  //   {}

  //   // Evaluation.
  //   virtual double
  //   value(const Point<dim> &p,
  //         const unsigned int /*component*/ = 0) const override
  //   {
  //     return std::cos(M_PI*p[0]) * std::exp(-this->get_time());
  //   }

  //   // Gradient evaluation.
  //   virtual Tensor<1, dim>
  //   gradient(const Point<dim> &p,
  //            const unsigned int /*component*/ = 0) const override
  //   {
  //     Tensor<1, dim> result;

  //     // duex / dx
  //     result[0] = -M_PI * std::sin(M_PI * p[0]) * std::exp(-this->get_time());
  //     result[0] =
  //       std::exp(p[0]) * (std::exp(p[1])-1);
  //     result[1] =
  //       std::exp(p[1]) * (std::exp(p[0])-1);
  //     return result;
  //   }
  // };

  // Default constructor.
  AdvDiffPOD(const unsigned int N_,
             const unsigned int &r_,
             const double       &T_,
             const double       &deltat_,
             const double       &theta_,
             std::vector<std::vector<double>> &modes_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , T(T_)
    , N(N_)
    , r(r_)
    , deltat(deltat_)
    , theta(theta_)
    , modes(modes_)
    , mesh(MPI_COMM_WORLD)
  {}

  // Initialization.
  void
  setup();

  // Solve the problem.
  void
  solve();

  // Compute the error for convergence analysis.
  // double
  // compute_error(const VectorTools::NormType &norm_type);

  // Boundary DOFs indices.
  // std::vector<unsigned int> boundary_dofs_idx_int;

protected:
  // Assemble the mass and stiffness matrices.
  void
  assemble_matrices();

  // Assemble the right-hand side of the problem.
  void
  assemble_rhs(const double &time);

  // Project the full order system to the reduced order system thanks to the transformation matrix.
  void
  convert_modes();

  void
  project_u0();

  void
  project_lhs();

  void
  project_rhs();

  // Solve the problem for one time step.
  void
  solve_time_step();

  // Output.
  void
  output(const unsigned int &time_step) const;

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

  // Diffusion coefficient.
  DiffusionCoefficient mu;

  // Reaction coefficient.
  // ReactionCoefficient reaction_coefficient;

  // Transport coefficient.
  TransportCoefficient beta;

  // Forcing term.
  ForcingTerm forcing_term;

  // g(x).
  FunctionG function_g;

  // Initial condition.
  FunctionU0 u_0;

  // Initial ROM state.
  // const std::vector<double> initial_state;

  // h(x).
  // FunctionH function_h;

  // Exact solution.
  // ExactSolution exact_solution;

  // Current time.
  double time;
  
  // Final time.
  const double T;

  // Number of elements.
  const unsigned int N;

  // Discretization. ///////////////////////////////////////////////////////////

  // Polynomial degree.
  const unsigned int r;

  // Time step.
  const double deltat;

  // Theta parameter of the theta method.
  const double theta;

  // Projection. ///////////////////////////////////////////////////////////

  // Transformation matrix.
  std::vector<std::vector<double>> modes;
  TrilinosWrappers::SparseMatrix transformation_matrix;

// for (unsigned int i = 0; i < modesT.m(); ++i)
//     for (unsigned int j = 0; j < modesT.n(); ++j)
//         modesT(i, j) = modesT_vec[i][j];

// dealii::FullMatrix<double> modes(modes_vec.size(), modes_vec[0].size());
// for (unsigned int i = 0; i < modes.m(); ++i)
//     for (unsigned int j = 0; j < modes.n(); ++j)
//         modes(i, j) = modes_vec[i][j];


  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // Quadrature formula used on boundary lines.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Mass matrix M / deltat.
  TrilinosWrappers::SparseMatrix mass_matrix;

  // Stiffness matrix A.
  TrilinosWrappers::SparseMatrix stiffness_matrix;

  // Matrix on the left-hand side (M / deltat + theta A).
  TrilinosWrappers::SparseMatrix lhs_matrix;
  TrilinosWrappers::SparseMatrix reduced_system_lhs;

  // Matrix on the right-hand side (M / deltat - (1 - theta) A).
  TrilinosWrappers::SparseMatrix rhs_matrix;

  // Right-hand side vector in the linear system.
  TrilinosWrappers::MPI::Vector system_rhs;
  TrilinosWrappers::MPI::Vector reduced_system_rhs;

  // ...
  TrilinosWrappers::MPI::Vector reduced_u_0;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;
};

#endif