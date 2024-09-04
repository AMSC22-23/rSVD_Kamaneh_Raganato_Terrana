

```
./Diff1D ../input/test_pod_more_prm.txt ../input/test_advdiff.prm 
../scripts/plot_solution.py ../output/full.mtx ../output/reconstruction.mtx ../output/errors.mtx ../input/test_pod_more_prm.txt
./Diff1D_convergence ../input/test_pod_convergence.txt ../input/test_advdiff_convergence.prm 
../scripts/plot_singular_values.py ../output/sigma.txt
./Diff1D ../input/test_pod_00.txt ../input/test_advdiff_00.prm
```