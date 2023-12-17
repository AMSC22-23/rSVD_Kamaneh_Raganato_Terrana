#include "../include/powerMethod/PM.hpp"
//@note you shoud write some comments to help readers understand what the function does
// You could use Doxygen to have comemnts that are automatically parsed and generate
// documentation for your code. See: https://www.doxygen.nl/manual/docblocks.html
void PM(Mat &A, Mat &B, double &sigma, Vec &u, Vec &v) {
    // Generate a random initial guess x0
    Vec x0 = Vec::Zero(A.cols()); // To multiply B with x0

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> distribution(0.0, 1.0);

    for (unsigned int i=0; i<x0.size(); i++) {
        x0(i) = distribution(gen);
    }
    x0.normalize();

    // Define the number of iterations
    //@note Algorithm iperparameters (epsilon, delta, lambda) are better put
    //      in an aggregate passed to the function. So it is easier for the user
    //      to change them and the function is more general. Avoid having iperparameters hardcoded..
    double epsilon = 1.e-10;
    double delta = 0.05;
    double lambda = 0.1;
    int s = ceil( log(4*log(2*A.cols()/delta)/(epsilon*delta)) /(2*lambda));
    // cout << "Check the number of iterations: " << s << endl;

    for (unsigned int i=1; i<=s; i++) {
        x0 = B*x0; // B = A^T*A
        x0.normalize();
    }

    // Compute the left singlular vector
    v = x0;
    v.normalize();

    // Compute the singular value
    sigma = (A*v).norm();

    // Compute the right singular vector
    u = A*v/sigma;

}



