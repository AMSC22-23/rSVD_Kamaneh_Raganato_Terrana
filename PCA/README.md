# Principal Components Analysis with SVD


## Overview

This project implements Principal Component Analysis (PCA) using Singular Value Decomposition (SVD) to reduce the dimensionality of large datasets while preserving as much variability as possible. PCA is widely used in areas such as image compression, data analysis, and more to simplify complex data sets.

The core of this project is built in C++ and utilizes the Eigen library for efficient matrix operations, ensuring high performance for large-scale computations.

This specific setup is designed to handle a pre-defined dataset tourists.txt, ideal for showcasing the PCA's capability in handling real-world data. Additionally, the project structure allows for easy adaptation to other datasets by incorporating custom data reading functions that cater to different formats and structures.

Through this project, users can gain insights into the practical implementation of mathematical concepts in data processing and how these can be applied to solve real-world problems efficiently.

## Prerequisites

Before you begin, ensure you have met the following requirements:

### For C++ Components:

- **C++ Compiler:** This project requires a C++ compiler to build the source code. You can use `g++` or any other C++ compiler that supports C++11 or later.

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
    cd PCA
    ```


5. Build the project:

    ```bash
    make run dataset=tourists.txt normalize=yes/no
    ```
NB: this project is meant to run for this specific dataset,however you can upload your own dataset in data/input to run PCA with a different dataset. Be aware that in order to build correctly the project with another dataset you have to implement a specific function in the main that reads correctly your dataset if it is not formatted like tourists.txt

### Running Test:

There is a test on an additional dataset in the  `./tests` directory. This dataset is athletic.txt .

Compile the test with the following command:

```bash
make test
```

