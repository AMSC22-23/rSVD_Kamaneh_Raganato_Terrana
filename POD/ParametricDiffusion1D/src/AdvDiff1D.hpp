#ifndef ADV_DIFF_1D_HPP
#define ADV_DIFF_1D_HPP

/**
 * The aim of this class is to solve the advection diffusion full order problem in 1D in order to collect the snapshot matrix.
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

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

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/parameter_handler.h> // https://www.dealii.org/current/doxygen/deal.II/classParameterHandler.html

#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>

#include <chrono>
using namespace std::chrono;

using namespace dealii;

// Class representing the linear diffusion advection problem.
template <int dim>
class AdvDiff
{
public:
  // Physical dimension (1D, 2D, 3D)
  // static constexpr unsigned int dim = 1;

  // Diffusion coefficient.
  class DiffusionCoefficient : public Function<dim>
  {
  public:
    // Constructor.
    DiffusionCoefficient()
    {}

    DiffusionCoefficient(const double prm_diffusion_coefficient) : prm(prm_diffusion_coefficient)
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return prm;
    }

  private:
    double prm;
  };

  // Transport coefficient.
  class TransportCoefficient : public Function<dim>
  {
  public:
    // Constructor.
    TransportCoefficient(const AdvDiff<dim> &advdiff) : advdiff(advdiff)
    {}

    // Evaluation.
    virtual void
    vector_value(const Point<dim> & /*p*/,
                 Vector<double> &values) const override
    {
      const double transport_coefficient = advdiff.parameters.get_double("beta");
      values[0] = transport_coefficient;
    }

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      if (component == 0)
      {
        const double transport_coefficient = advdiff.parameters.get_double("beta");
        return transport_coefficient;
      }
      else
        return 0.0;
    }

    private:
      const AdvDiff<dim> &advdiff;
  };

  // Forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    // Constructor.
    ForcingTerm(const double prm_diffusion_coefficient, const AdvDiff<dim> &advdiff) : prm(prm_diffusion_coefficient), advdiff(advdiff)
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      const double transport_coefficient = advdiff.parameters.get_double("beta");
      const int u0_choice = advdiff.parameters.get_integer("u0_choice");
      const double amplitude = advdiff.parameters.get_double("amplitude");
      if (dim == 1 && u0_choice == 0)
        return (prm*M_PI*M_PI-1)*std::sin(M_PI*p[0])*std::exp(-this->get_time())+
               transport_coefficient*M_PI*std::cos(M_PI*p[0])*std::exp(-this->get_time());
      else if (dim == 1 && u0_choice == 1)
        return std::exp(-this->get_time())*(-2*std::sin(9*M_PI*p[0])+std::sin(4*M_PI*p[0])+
               prm*(162*M_PI*M_PI*std::sin(9*M_PI*p[0])-16*M_PI*M_PI*std::sin(4*M_PI*p[0]))+
               transport_coefficient*(18*M_PI*std::cos(9*M_PI*p[0])-4*M_PI*std::cos(4*M_PI*p[0])));
      else if (dim == 1 && u0_choice == 2)
        return 5*M_PI*std::cos(5*M_PI*this->get_time())*std::sin(amplitude*M_PI*p[0])+
               prm*amplitude*amplitude*M_PI*M_PI*std::sin(5*M_PI*this->get_time())*std::sin(amplitude*M_PI*p[0])+
               transport_coefficient*amplitude*M_PI*std::sin(5*M_PI*this->get_time())*std::cos(amplitude*M_PI*p[0]);
      else if (dim == 1 && u0_choice == 3)
        return std::exp(-this->get_time())*(1+std::exp(p[0])*(transport_coefficient-prm-1));
      else
        return 0.0;
    }

  private:
    double prm;
    const AdvDiff<dim> &advdiff;
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

  // Function for the initial condition.
  class FunctionU0 : public Function<dim>
  {
  public:
    // Constructor.
    FunctionU0(const AdvDiff<dim> &advdiff) : advdiff(advdiff)
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      const int u0_choice = advdiff.parameters.get_integer("u0_choice");
      const double amplitude = advdiff.parameters.get_double("amplitude");

      if (u0_choice == 0)
        return amplitude*std::sin(M_PI*p[0]);
      else if (u0_choice == 1)
        return 2.0*std::sin(9.0*M_PI*p[0])-std::sin(4.0*M_PI*p[0]);
      else if (u0_choice == 2)
        return std::sin(amplitude*M_PI*p[0]);
      else if (u0_choice == 3)
        return std::exp(p[0])-1;
      else
        return amplitude*std::sin(M_PI*p[0]); // Default initial condition
    }

    private:
      const AdvDiff<dim> &advdiff;
  };

  // Exact solution
  class ExactSolution : public Function<dim>
  {
  public:

    ExactSolution(const AdvDiff<dim> &advdiff) : advdiff(advdiff)
    {}

    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      const int u0_choice = advdiff.parameters.get_integer("u0_choice");
      const double amplitude = advdiff.parameters.get_double("amplitude");

      if (dim == 1 && u0_choice == 0) 
        return amplitude*std::sin(M_PI*p[0])*std::exp(-this->get_time());
      else if (dim == 1 && u0_choice == 1)
        return (2.0*std::sin(9.0*M_PI*p[0])-std::sin(4.0*M_PI*p[0]))*std::exp(-this->get_time());
      else if (dim == 1 && u0_choice == 2)
        return std::sin(5*M_PI*this->get_time())*std::sin(amplitude*M_PI*p[0]);
      else if (dim == 1 && u0_choice == 3)
        return (std::exp(p[0])-1)*std::exp(-this->get_time());
      else 
        return 0.0;
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;
      const int u0_choice = advdiff.parameters.get_integer("u0_choice");
      const double amplitude = advdiff.parameters.get_double("amplitude");

      if (dim == 1 && u0_choice == 0)
        result[0] = amplitude*M_PI*std::cos(M_PI*p[0])*std::exp(-this->get_time());
      else if (dim == 1 && u0_choice == 1)
        result[0] = (18.0*M_PI*std::cos(9.0*M_PI*p[0])-4.0*M_PI*std::sin(4.0*M_PI*p[0]))*std::exp(-this->get_time());
      else if (dim == 1 && u0_choice == 2)
        result[0] = amplitude*M_PI*std::sin(5*M_PI*this->get_time())*std::cos(amplitude*M_PI*p[0]);
      else if (dim == 1 && u0_choice == 3)
        result[0] = std::exp(p[0])*std::exp(-this->get_time());
      else
        result[0] = 0.0;

      return result;
    }

  private:
    const AdvDiff<dim> &advdiff;
  };

  // Default constructor.
  AdvDiff(/*const unsigned int N_,
          const unsigned int &r_,
          const double       &T_,
          const double       &deltat_,
          const double       &theta_,
          const unsigned int &sample_every_,*/
          const double       &prm_diffusion_coefficient_,
          const std::string  &prm_file_,
          const double       &convergence_deltat_ = 0.0) // Default value
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , mu(prm_diffusion_coefficient_)
    // , T(T_)
    // , N(N_)
    // , r(r_)
    // , deltat(deltat_)
    // , theta(theta_)
    // , sample_every(sample_every_)
    , beta(*this)
    , forcing_term(prm_diffusion_coefficient_, *this)
    , u_0(*this)
    , exact_solution(*this)
    , convergence_deltat(convergence_deltat_)
    , prm_file(prm_file_)
    , mesh(MPI_COMM_WORLD)
  {
      parameters.declare_entry("beta", "1.0", Patterns::Double(), "dummy");
      parameters.declare_entry("u0_choice", "0", Patterns::Integer(), "dummy");
      parameters.declare_entry("amplitude", "1.0", Patterns::Double(), "dummy");

      parameters.declare_entry("N", "0", Patterns::Integer(), "dummy");
      parameters.declare_entry("degree", "0", Patterns::Integer(), "dummy");
      parameters.declare_entry("T", "0.0", Patterns::Double(), "dummy");
      parameters.declare_entry("deltat", "0.0", Patterns::Double(), "dummy");
      parameters.declare_entry("theta", "0.0", Patterns::Double(), "dummy");
      parameters.declare_entry("sample_every", "0", Patterns::Integer(), "dummy");

      parameters.parse_input(prm_file);
  }

  // Initialization.
  void
  setup();

  // Solve the problem.
  void
  solve();

  // Compute the error for convergence analysis.
  double
  compute_error(const VectorTools::NormType &norm_type);

  // Snapshot matrix. It collects the solution at each time step. It contains the snapshots computed for a single parameter, 
  // in other words, the time evolution for a single parameter.
  std::vector<std::vector<double>> snapshot_matrix;

  // System solution (including ghost elements). 
  TrilinosWrappers::MPI::Vector solution;

  // Average duration of solving a single time step.
  std::chrono::duration<long, std::micro> duration_full_avg;

protected:
  // Assemble the mass and stiffness matrices.
  void
  assemble_matrices();

  // Assemble the right-hand side of the problem.
  void
  assemble_rhs(const double &time);

  // Solve the problem for one time step.
  void
  solve_time_step();

  // Assemble the snapshot matrix.
  void
  assemble_snapshot_matrix(const unsigned int &time_step);

  // Output.
  // void
  // output(const unsigned int &time_step) const;

  // MPI parallel. //////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Diffusion coefficient.
  DiffusionCoefficient mu;

  // Transport coefficient.
  TransportCoefficient beta;

  // Forcing term.
  ForcingTerm forcing_term;

  // g(x).
  FunctionG function_g;

  // Initial condition.
  FunctionU0 u_0;

  // Exact solution.
  ExactSolution exact_solution;

  // Current time.
  double time;

  // Convergence time step.
  const double convergence_deltat;
  
  // Final time.
  // const double T;

  // Number of elements.
  // const unsigned int N;

  // Discretization. ////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Polynomial degree.
  // const unsigned int r;

  // Time step.
  // const double deltat;

  // Theta parameter of the theta method.
  // const double theta;

  // Sample_every parameter for selecting time steps which solution has to be collected in the snapshot matrix.
  // const unsigned int sample_every;

  const std::string prm_file;

  ParameterHandler parameters;

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

  // Matrix on the right-hand side (M / deltat - (1 - theta) A).
  TrilinosWrappers::SparseMatrix rhs_matrix;

  // Right-hand side vector in the linear system.
  TrilinosWrappers::MPI::Vector system_rhs;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // Vector collecting the durations of solving a single time step.
  std::vector<std::chrono::duration<double>> duration_full_vec;

};

#endif