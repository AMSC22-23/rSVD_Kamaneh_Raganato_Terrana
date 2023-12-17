// matrix.h
#pragma once //@note pragma once is not standard C++, even if it is supported by most compilers

#include <vector>

class Matrix {
private:
    //@note Storing the matrix as a vector of vectors is not the most efficient way to do it.
    //      Normally youstore the matrix linearly in a single vector and access the elements
    //      using the formula: index = row * num_cols + col (if the matrix is stored in row-major order
    //      or index = col * num_rows + row (if the matrix is stored in column-major order)
    std::vector<std::vector<double>> data;
    int rows;
    int cols;

public:
    Matrix(int rows, int cols);
    Matrix(const std::vector<std::vector<double>>& input_data);
    Matrix genRandom() const;

    // Getter methods
    int getRows() const;
    int getCols() const;
    double getElement(int row, int col) const;
    Matrix extractCol(int col) const; // Vector extractCol(int col);
    Matrix extractRow(int row) const; // Vector extractRow(int row);

    // Setter methods
    void setElement(int row, int col, double value);
    void setRow(int row, const Matrix& input_data); // void setRow(int row, const Vector& input_data);
    void setCol(int col, const Matrix& input_data); // void setCol(int col, const Vector& input_data);

    // Matrix operations
    Matrix transpose() const;
    Matrix multiply(const Matrix& other) const;
    std::vector<double> mat_vet_multiply(const std::vector<double>& other); // Vector mat_vet_multiply(const Vector& other);
    // Additional operations specific to the SVD implementation
    // ...

    // Overloaded operators for convenience
    //@note The addition and subtraction are normally mplemented as free functions
    //      (i.e. not member functions) to allow for implicit conversions of the operands
    //      and the functions are made friend to allow them to access the private members
    //      of the class and be more efficient.
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    // @note The multiplication by a scalar is normally implemented as a free function
    //       so that you can overload the operator on both sides of the scalar 
    //      (i.e. scalar * matrix and matrix*scalar)
    Matrix operator*(double scalar) const;
    // ...

    // Display method
    void display() const;
}; 
