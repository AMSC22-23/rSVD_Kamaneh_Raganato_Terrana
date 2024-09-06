

```
./Diff1D ../input/test_pod_more_prm.txt ../input/test_advdiff.prm 
../scripts/plot_solution.py ../output/full.mtx ../output/reconstruction.mtx ../output/errors.mtx ../input/test_pod_more_prm.txt
./Diff1D_convergence ../input/test_pod_convergence.txt ../input/test_advdiff_convergence.prm 

./Diff1D ../input/test_pod_00.txt ../input/test_advdiff_00.prm


./Diff1D ../input/test_pod_01.txt ../input/test_advdiff_01.prm
../scripts/plot_solution.py ../output/full.mtx ../output/reconstruction.mtx ../output/errors.mtx ../input/test_pod_01.txt
../scripts/plot_singular_values.py ../output/sigma.txt

in results/test00
../../scripts/plot_solution.py ../../output/full.mtx ../../output/reconstruction.mtx ../../output/errors.mtx ../../input_new/test_pod_00.txt

TEST 02     RACCOGLI TEMPI
./Diff1D_new_prm ../input_new/test_pod_02.txt ../input_new/test_advdiff_02.prm

TEST 03     RACCOGLI TEMPI
./Diff1D_openmp ../input_new/test_pod_03.txt ../input_new/test_advdiff_03.prm

```