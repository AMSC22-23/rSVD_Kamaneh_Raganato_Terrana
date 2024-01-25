# Image Compression with randomized SVD


## Overview

The primary goal of this project is to implement the Random SVD (rSVD) algorithm, which addresses standard matrix decompositions. These decompositions include the pivoted QR factorization, eigenvalue decomposition, and singular value decomposition (SVD).
The project emphasizes the use of randomization as a powerful tool for low-rank matrix approximation. This approach not only enhances the efficiency of utilizing modern computational architectures but also enables effective handling of massive data sets.


## Prerequisites

Before you begin, ensure you have met the following requirements:

### For C++ Components:

- **C++ Compiler:** This project requires a C++ compiler to build the source code. You can use `g++` or any other C++ compiler that supports C++11 or later.

- **CMake:** CMake is used for building the project. Ensure you have CMake installed on your system. You can download it from [cmake.org](https://cmake.org/download/).

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

1. To run the main program:

Execute the program by passing the name of the image you want to compress from the 'data/input/img' directory, and specify the number of processors.
Note: In the parallel version the number of processors must be square (e.g., 1, 4, 9, ..)
Example: 

```bash
mpirun -np 4 bin/main 1024_01.jpg
```


To clean up generated files:

```bash
make clean
```
To enable profiling and generate a profile output:

```bash
make profile
```