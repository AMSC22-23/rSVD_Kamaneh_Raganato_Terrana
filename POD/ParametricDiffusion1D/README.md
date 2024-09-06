# Advection Diffusion Problem with POD

### Folder Structure

* `input`: each test needs a pair of parameter files:
  * `test_pod_0#.txt` is used in the main file, it contains parameters for both the advection-diffusion problem and the POD;
  * `test_advdiff_0#.prm` is used in the classes `AdvDiff` and `AdvDiffPOD`, it contains parameters only for the advection-diffusion problem.
* `output`: once the main file is executed, three matrices in MatrixMarket format and one vector are exported in this folder. The matrices `full.mtx`, `reconstruction.mtx`, `errors.mtx` are used in `plot_solution.py`, the vector `sigma.txt` is used in `plot_singular_values.py`.
* `results`: this folder is organized in subfolders, one for each test. Its aim is to collect the outputs of the python scripts.
* `scripts`: three python scripts for plotting some results:
  * `plot_solution.py` to plot both the FOM solution with its approximations for different `rom_sizes` and the relative error in $L^2$ norm;
  * `plot_singular_values.py` to plot the singular values and visualize their decay in semilogarithmic scale;
  * `plot_convergence.py` to plot the convergence order in case of performance analysis.
* `src`: in the main files `Diff1D_<name>.cpp` we create `AdvDiff`, `POD` and `AdvDiffPOD` objects.

### POD Class
* **Public members**
  * **Constructors**
    * `POD()` default constructor for POD;
    * `POD(Mat_m &S, const int r, const int svd_type)` constructor for naive POD;
    * `POD(Mat_m &S, const int r, const double tol, const int svd_type)` constructor for standard POD;
    * `POD(Mat_m &S, Mat_m &Xh, const int r, const double tol, const int svd_type)` constructor for energy POD;
    * `POD(Mat_m &S, Mat_m &Xh, Mat_m &D, const int r, const double tol, const int svd_type)` constructor for weight POD;
  * `Mat_m W` matrix containing the POD modes;
  * `Vec_v sigma` vector containing the singular values.
* **Private members**
  * `void perform_SVD(Mat_m &A, Mat_m &U, Vec_v &sigma, Mat_m &V, const int r, const int svd_type)` performs the singular value decomposition given the svd_type: Power, Jacobi, Parallel Jacobi;
  * `std::tuple<Mat_m, Vec_v> naive_POD(Mat_m &S, const int r, const int svd_type)` naive POD Algorithm, it only performs the SVD on the snapshot matrix S;
  * `std::tuple<Mat_m, Vec_v> standard_POD(Mat_m &S, const int r, double const tol, const int svd_type)` Algorithm 6.1 page 126 – POD Algorithm
  * `std::tuple<Mat_m, Vec_v> energy_POD(Mat_m &S, Mat_m &Xh, const int r, const double tol, const int svd_type)` Algorithm 6.2 page 128 – POD Algorithm with energy norm
  * `std::tuple<Mat_m, Vec_v> weight_POD(Mat_m &S, Mat_m &Xh, Mat_m &D, const int r, const double tol, const int svd_type)` Algorithm 6.3 page 134 – POD Algorithm with energy norm and quadrature weights.

### AdvDiff Class
* **Relevant public members**
  * `class DiffusionCoefficient : public Function<dim>`: it assumes as value the argument `prm_diffusion_coefficient`;
  * `class TransportCoefficient : public Function<dim>`: it assumes as value the parameter `beta`;
  * `class ForcingTerm : public Function<dim>`: it is derived accordingly to the exact solution for the parameter `u0_choice`, it can use `beta` and `amplitude` parameters;
  * `class FunctionG : public Function<dim>`: homogeneous Dirichlet boundary conditions;
  * `class FunctionU0 : public Function<dim>`: it is selected with `u0_choice` and can use `amplitude` parameter;
  * `class ExactSolution : public Function<dim>`: it is selected with `u0_choice` and can use `amplitude` parameter;
  * `AdvDiff(const double &prm_diffusion_coefficient_, const std::string  &prm_file_, const double &convergence_deltat_ = 0.0)`: constructor;
  * `std::vector<std::vector<double>> snapshot_matrix`: it collects the solution at each time step. It contains the snapshots computed for a single parameter, in other words, the time evolution for a single parameter;
  * `TrilinosWrappers::MPI::Vector solution`: system solution, including ghost elements;
  * `std::chrono::duration<long, std::micro> duration_full_avg`: average duration of solving a single time step.
* **Relevant protected members**
  * `void assemble_snapshot_matrix(const unsigned int &time_step)`: it assembles the snapshot matrix;
  * `std::vector<std::chrono::duration<double>> duration_full_vec`: vector collecting the durations of solving a single time step.

### AdvDiffPOD Class
* **Relevant public members**
  * `AdvDiffPOD(const std::vector<std::vector<double>> &modes_, const double &prm_diffusion_coefficient_, const std::string  &prm_file_, const double &convergence_deltat_ = 0.0)`: constructor;
  * `void solve_reduced()`: it setups and solves the reduced order problem;
  * `TrilinosWrappers::MPI::Vector fom_solution`: system solution, including ghost elements. It collects the full order approximated solution that is obtained by projecting (expanding) the reduced order solution;
  * `std::chrono::duration<long, std::micro> duration_reduced_avg`: average duration of solving a single time step.
* **Relevant protected members**
  * `void setup_reduced()`: it setups the reduced mesh, the reduced left-hand side matrix, the reduced right-hand side vector and the reduced solution vector;
  * `void convert_modes(PETScWrappers::FullMatrix &transformation_matrix)`: it copies the standard matrix of modes in a PETScWrappers::FullMatrix. The transformation matrix is then used for projecting the full order space into the reduced order space;
  * `void project_u0(PETScWrappers::FullMatrix &transformation_matrix)`: it projects the initial condition: `reduced_solution = T^T * solution_owned`;
  * `void project_lhs(PETScWrappers::FullMatrix &transformation_matrix)`: it projects the left-hand side matrix: `reduced_system_lhs = aux * T = T^T * lhs_matrix * T`;
  * `void project_rhs(PETScWrappers::FullMatrix &transformation_matrix)`: it projects the right-hand side vector: `reduced_system_rhs = T^T * system_rhs_copy`;
  * `void expand_solution(PETScWrappers::FullMatrix &transformation_matrix)`: it projects the rom solution: `fom_solution = T * reduced_solution`;
  * `void solve_time_step_reduced()`: solution of the reduced system for the GMRES solver;
  * `const std::vector<std::vector<double>> modes`: it is the matrix that contains the POD modes, which will be the columns of the transformation matrix used as projector from full order to reduced order model;
  * `std::vector<std::chrono::duration<double>> duration_reduced_vec`: vector collecting the durations of solving a single time step.

### Compiling
To build the executable, make sure you have loaded the needed modules with
```bash
$ module load gcc-glibc dealii
```
Then run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```

This will create the executable in the `build` directory.

### Execution
To run the executable, navigate to the `build` directory and use the following command, replacing # with the test number:

```bash
$ ./Diff1D ../input/test_pod_0#.txt ../input/test_advdiff_0#.prm
```

To plot the solution, the error and the singular values, navigate to the `results/test0#` and use the following commands:

```bash
$ ../../scripts/plot_solution.py ../../output/full.mtx ../../output/reconstruction.mtx ../../output/errors.mtx ../../input/test_pod_0#.txt
$ ../../scripts/plot_singular_values.py ../../output/sigma.txt
```


