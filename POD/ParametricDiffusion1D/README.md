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

