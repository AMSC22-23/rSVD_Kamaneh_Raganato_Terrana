# Singular Value Decomposition


## Overview

The primary goal of this project is to implement the SVD and the randomized SVD (rSVD) algorithm, focusing on matrix decompositions such as pivoted QR factorization, eigenvalue decomposition, and singular value decomposition (SVD).
The project emphasizes the use of randomization as an efficient tool for low-rank matrix approximation. 
Additionally, multiple SVD methods (Power, Jacobi, and Parallel Jacobi) are implemented, offering flexibility for different use cases and computational trade-offs.


## Prerequisites

Before you begin, ensure you have met the following requirements:

### For C++ Components:

- **C++ Compiler:** This project requires a C++ compiler to build the source code. You can use `g++` or any other C++ compiler that supports C++11 or later.

- **MPI (Message Passing Interface):** MPI is needed since it is used for parallelization of some tasks.

- **Eigen Library:** This project depends on the Eigen library for linear algebra operations. Download and install Eigen from [eigen.tuxfamily.org](https://eigen.tuxfamily.org/dox/GettingStarted.html).


## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/AMSC22-23/rSVD_Kamaneh_Raganato_Terrana.git
    ```

2. Navigate to the project directory:

    ```bash
    cd rSVD_Kamaneh_Raganato_Terrana
    ```


### Running Tests:

There is no main program in this part of the project. The focus is on the PCA (Principal Component Analysis) and POD (Proper Orthogonal Decomposition) applications. Therefore, there are only tests to verify the correct functionality of the core implementations (SVD and rSVD).

Compile the tests with the following command:

```bash
make test
```
Two tests will be executed—one for SVD and one for rSVD. Both tests use input matrices located in the input folder. Since the project includes three different methods for performing SVD (Power, Jacobi, and Parallel Jacobi), the default method used in the tests is Jacobi. To explore the other methods, you can modify the test configuration in the tests folder.



### To clean up generated files:

```bash
make clean
```
