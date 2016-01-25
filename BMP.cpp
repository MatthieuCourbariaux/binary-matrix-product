
#include <iostream>
#include "BMP.h"
#include <math.h>
#include <ctime>
#include <Eigen/Core>
using Eigen::MatrixXf;
// using Eigen::MatrixXi;
using Eigen::Matrix;
// typedef Matrix<unsigned long int,Eigen::Dynamic,Eigen::Dynamic> MatrixXdi;
typedef Matrix<bi64,Eigen::Dynamic,Eigen::Dynamic> MatrixXb;
using namespace std;

// TODO = use Eigen matrix multiply with custom scalar type -> better memory usage
// http://eigen.tuxfamily.org/dox/TopicCustomizingEigen.html#user_defined_scalars

// 64 single float array -> 64 bools -> 64 bits unsigned int
unsigned long int float2bool(float* array)
{
    unsigned long int rvalue=0;
    unsigned long int sign;
    
    for (int i = 0; i < 64; i++)
    {
        sign = (array[i]>0);
        rvalue = rvalue & (sign<<i);
    }
    
    return rvalue;
}

MatrixXb binarize(MatrixXf A)
{
    int I = (int)A.rows();
    int J = (int)A.cols();
    int i, j;
    
    MatrixXb B(I,J/64);
    
    float * ptA = &A(0,0);
    bi64 * ptB = &B(0,0);
    float* ptAi;
    bi64* ptBi;
    
    for(i=0;i<I;i+=1)
    {   
        ptAi = ptA+i*J;
        ptBi = ptB+i*J/64;
        
        for(j=0;j<J;j+=64)
        {
            ptBi[j/64] = float2bool(ptAi+j);
        }
    }
    return B;
}

// Arithmetic gain = 64 (nb of bits) /3 (no fused popcnt add) /256 (avx) x32 (single float)= x2.7
// Memory bandwidth gain = 256/64 *3(no fused popcnt add) = x12
// Actual gain = x4 :)
MatrixXf BMP2(MatrixXf A,MatrixXf B)
{ 
    // Binarization
    MatrixXb Ab = binarize(A);
    MatrixXb Bb = binarize(B);
    
    return (Ab*Bb).cast<float>();    
}