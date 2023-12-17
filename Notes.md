# Some comments #

Stroring a dynamic memory matrix as vectors of vectors is not a good idea. It is better to store it as a single vector. This is because the vectors of vectors are not contiguous in memory and hence the cache is not utilized efficiently.

I do not understand why you have vector and vector_modified as wella as matrix and matrix_modified. Maybe they were just experiments?

The code is not fully parallelizable. A few things that can be done however. for instance matrix operations.

The generation of the random matrix can be made parallel by generating multiple seeds with std::seed_seq and then use the different seeds to fill with random numbers different parts of the matrix in parallel.

The code does not compile becouse there are still parts missing (PM1.hpp)

the values returned for the number of columns/rows are int. This is not good because the matrix can be very large. It is better to use size_t.

The code is not very readable. It is better to use more descriptive names for the variables and functions. Also, it is better to use more comments.

