#ifndef POISSON_2D_HPP
#define POISSON_2D_HPP

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/parameter_handler.h> // https://www.dealii.org/current/doxygen/deal.II/classParameterHandler.html

#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class Poisson2D
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 2;

  // Diffusion coefficient.
  // In deal.ii, functions are implemented by deriving the dealii::Function
  // class, which provides an interface for the computation of function values
  // and their derivatives.
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
  // class TransportCoefficient : public Function<dim>
  // {
  // public:
  //   // Constructor.
  //   TransportCoefficient()
  //   {}

  //   // Evaluation.
  //   virtual void
  //   vector_value(const Point<dim> & /*p*/,
  //                Vector<double> &values) const override
  //   {
  //     values[0] = 1.0;
  //     values[1] = 1.0;
  //   }

  //   virtual double
  //   value(const Point<dim> & /*p*/,
  //         const unsigned int component = 0) const override
  //   {
  //     if (component == 0)
  //       return 1.0;
  //     else
  //       return 1.0;
  //   }
  // };

  // Forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    // Constructor.
    ForcingTerm()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> &/*p*/,
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
  class FunctionH : public Function<dim>
  {
  public:
    // Constructor.
    FunctionH()
    {}

    // Evaluation:
    virtual double
    value(const Point<dim> &/*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
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
  //     return (std::exp(p[0])-1) * (std::exp(p[1])-1);
  //   }

  //   // Gradient evaluation.
  //   virtual Tensor<1, dim>
  //   gradient(const Point<dim> &p,
  //            const unsigned int /*component*/ = 0) const override
  //   {
  //     Tensor<1, dim> result;

  //     result[0] =
  //       std::exp(p[0]) * (std::exp(p[1])-1);
  //     result[1] =
  //       std::exp(p[1]) * (std::exp(p[0])-1);

  //     return result;
  //   }
  // };

  // Constructor.
  Poisson2D(const std::string &mesh_file_name_, const unsigned int &r_,
            std::vector<unsigned int> &boundary_dofs_idx_int_,
            std::vector<double> &snapshot_array_, const double &prm_diffusion_coefficient_)
    : mesh_file_name(mesh_file_name_)
    , r(r_)
    , boundary_dofs_idx_int(boundary_dofs_idx_int_)
    , snapshot_array(snapshot_array_)
    , diffusion_coefficient(prm_diffusion_coefficient_)
  {}

  // Initialization.
  void
  setup();

  // System assembly.
  void
  assemble();

  // System solution.
  void
  solve();

  // Output.
  void
  output() const;

  // Compute the error.
  double
  compute_error(const VectorTools::NormType &norm_type) const;

protected:
  // Path to the mesh file.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Boundary DOFs indices.
  std::vector<unsigned int> boundary_dofs_idx_int;

  // Snapshot array.
  std::vector<double> &snapshot_array;

  // const double prm_diffusion_coefficient;

  // Diffusion coefficient.
  DiffusionCoefficient diffusion_coefficient;

  // Reaction coefficient.
  // ReactionCoefficient reaction_coefficient;

  // Transport coefficient.
  // TransportCoefficient transport_coefficient;

  // Forcing term.
  ForcingTerm forcing_term;

  // g(x).
  FunctionG function_g;

  // h(x).
  FunctionH function_h;

  // Triangulation.
  Triangulation<dim> mesh;

  // Finite element space.
  // We use a unique_ptr here so that we can choose the type and degree of the
  // finite elements at runtime (the degree is a constructor parameter). The
  // class FiniteElement<dim> is an abstract class from which all types of
  // finite elements implemented by deal.ii inherit.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  // We use a unique_ptr here so that we can choose the type and order of the
  // quadrature formula at runtime (the order is a constructor parameter).
  std::unique_ptr<Quadrature<dim>> quadrature;

  // Quadrature formula used on boundary lines.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // Sparsity pattern.
  SparsityPattern sparsity_pattern;

  // System matrix.
  SparseMatrix<double> system_matrix;

  // System right-hand side.
  Vector<double> system_rhs;

  // System solution.
  Vector<double> solution;
};

#endif