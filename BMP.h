/*****************************************************************
BMP
Written by Matthieu Courbariaux in 2015
*******************************************************************/

#ifndef binary_mp
#define binary_mp

#include <Eigen/Core>
using Eigen::MatrixXf;
using Eigen::Matrix;
typedef Matrix<unsigned int,Eigen::Dynamic,Eigen::Dynamic> MatrixXi;

unsigned int concatenate(float* array);
MatrixXi concatenate(MatrixXf A);
float* deconcatenate(unsigned int x);
MatrixXf deconcatenate(MatrixXi A);

MatrixXf BMP(MatrixXf A,MatrixXf B);

#endif