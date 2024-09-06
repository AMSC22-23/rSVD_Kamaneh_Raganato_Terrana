### Folder Structure

* `input`: each test needs a pair of parameter files:
  * `test_pod_0#.txt` is used in the main file, it contains parameters for both the advection-diffusion problem and the POD;
  * `test_advdiff_0#.txt` is used in the classes `AdvDiff` and `AdvDiffPOD`, it contains parameters only for the advection-diffusion problem.
* `output`: once the main file is executed, three matrices in MatrixMarket format and one vector are exported in this folder. The matrices `full.mtx`, `reconstruction.mtx`, `errors.mtx` are used in `plot_solution.py`, the vector `sigma.txt` is used in `plot_singular_values.py`.
* `results`: this folder is organized in subfolders, one for each test. Its aim is to collect the outputs of the python scripts.
* `scripts`: three python scripts for plotting some results:
  * `plot_solution.py` to plot both the FOM solution with its approximations for different `rom_sizes` and the relative error in $L^2$ norm;
  * `plot_singular_values.py` to plot the singular values and visualize their decay in semilogarithmic scale;
  * `plot_convergence.py` to plot the convergence order in case of performance analysis.
* `src`: in the main files `Diff1D_<name>.cpp` we create `AdvDiff`, `POD` and `AdvDiffPOD` objects.

PIU DETTAGLI

### POD Class
* Public members
  * Constructors
    * `POD()` default constructor for POD;
    * `POD(Mat_m &S, const int r, const int svd_type)` constructor for naive POD;
    * `POD(Mat_m &S, const int r, const double tol, const int svd_type)` constructor for standard POD;
    * `POD(Mat_m &S, Mat_m &Xh, const int r, const double tol, const int svd_type)` constructor for energy POD;
    * `POD(Mat_m &S, Mat_m &Xh, Mat_m &D, const int r, const double tol, const int svd_type)` constructor for weight POD;
  * `Mat_m W` matrix containing the POD modes;
  * `Vec_v sigma` vector containing the singular values.
* Private members  
  * `void perform_SVD(Mat_m &A, Mat_m &U, Vec_v &sigma, Mat_m &V, const int r, const int svd_type)` performs the singular value decomposition given the svd_type: Power, Jacobi, Parallel Jacobi;
  * `std::tuple<Mat_m, Vec_v> naive_POD(Mat_m &S, const int r, const int svd_type)` naive POD Algorithm, it only performs the SVD on the snapshot matrix S;
  * `std::tuple<Mat_m, Vec_v> standard_POD(Mat_m &S, const int r, double const tol, const int svd_type)` Algorithm 6.1 page 126 – POD Algorithm
  * `std::tuple<Mat_m, Vec_v> energy_POD(Mat_m &S, Mat_m &Xh, const int r, const double tol, const int svd_type)` Algorithm 6.2 page 128 – POD Algorithm with energy norm
  * `std::tuple<Mat_m, Vec_v> weight_POD(Mat_m &S, Mat_m &Xh, Mat_m &D, const int r, const double tol, const int svd_type)` Algorithm 6.3 page 134 – POD Algorithm with energy norm and quadrature weights.


    


### AdvDiff Class

### AdvDiffPOD Class

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


