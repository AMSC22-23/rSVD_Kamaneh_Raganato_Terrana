# Image Compression with randomized SVD


## Overview

The primary goal of this project is to implement the Random SVD (rSVD) algorithm, which addresses standard matrix decompositions. These decompositions include the pivoted QR factorization, eigenvalue decomposition, and singular value decomposition (SVD).
The project emphasizes the use of randomization as a powerful tool for low-rank matrix approximation. This approach not only enhances the efficiency of utilizing modern computational architectures but also enables effective handling of massive data sets.


## Prerequisites

Before you begin, ensure you have met the following requirements:

### For C++ Components:

- **C++ Compiler:** This project requires a C++ compiler to build the source code. You can use `g++` or any other C++ compiler that supports C++11 or later.

- **MPI (Message Passing Interface):** MPI is needed since it is used for parallelization of some tasks

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

3. Navigate to the image_compression directory:

    ```bash
    cd image_compression
    ```


5. Build the project:

    ```bash
    make main
    ```


## Usage

### Running the Main Program:

To execute the main program, follow these steps:
1. Navigate to the program's root directory.
2. Run the program by providing the name of the image you wish to compress from the `./data/input/img` directory.
3. Specify the number of processors to be used.
Note: For the parallel version, the number of processors must be a perfect square (e.g., 1, 4, 9, ...).


#### Example: 

```bash
mpirun -np 4 bin/main 1024_01.jpg
```


### To clean up generated files:

```bash
make clean
```

### Running Tests:

To assess different components of the program, various tests are available in the  `./tests` directory. 

Compile the tests with the following command:

```bash
make test
```

There are two types of tests for each component:

1. A test ending with the number `1` takes matrices from `./data/input/`mat as input and writes the corresponding output to `./data/output`.

#### Example:

```bash
mpirun -np 4 bin/QR_test1
```

A test file ending with the number `2` uses hardcoded matrices as input, allowing users to modify them according to their needs.

#### Example:

```bash
mpirun -np 4 bin/rSVD_test2 
```

Feel free to explore and adapt the tests based on your specific requirements.