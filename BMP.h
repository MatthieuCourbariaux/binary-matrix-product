/*****************************************************************
BMP
Written by Matthieu Courbariaux in 2015
*******************************************************************/

#ifndef binary_mp
#define binary_mp

#include <Eigen/Core>
using Eigen::MatrixXf;
using Eigen::Matrix;
typedef Matrix<unsigned long int,Eigen::Dynamic,Eigen::Dynamic> MatrixXdi;

unsigned long int float2bool(float* array);
MatrixXdi binarize(MatrixXf A);
MatrixXf BMP(MatrixXf A,MatrixXf B);

#endif