
#include <iostream>
#include "BMP.h"
#include <math.h>
#include <ctime>
#include <Eigen/Core>
using Eigen::MatrixXf;
// using Eigen::MatrixXi;
using Eigen::Matrix;
typedef Matrix<unsigned int,Eigen::Dynamic,Eigen::Dynamic> MatrixXi;
using namespace std;

// 32 single float array -> 32 bools -> 32 bits unsigned int
unsigned int concatenate(float* array)
{
    unsigned int rvalue=0;
    unsigned int sign;
    
    for (int i = 0; i < 32; i++)
    {
        sign = (array[i]>0);
        rvalue = rvalue | (sign<<i);
    }
    
    return rvalue;
}

MatrixXi concatenate(MatrixXf A)
{
    int I = (int)A.rows();
    int J = (int)A.cols();
    int i, j;
    
    float * ptA = &A(0,0);
    float * ptAi;
    
    MatrixXi B(I,J/32);
    unsigned int * ptB = &B(0,0);
    unsigned int * ptBi;
    
    for(i=0;i<I;i+=1)
    {   
        ptAi = ptA+i*J;
        ptBi = ptB+i*J/32;
        
        for(j=0;j<J;j+=32)
        {
            ptBi[j/32] = concatenate(ptAi+j);
        }
    }
    return B;
}

// 32 bits unsigned int -> 32 bools -> 32 single float array
float* deconcatenate(unsigned int x)
{
    float * array = new float[32];
    
    for (int i = 0; i < 32; i++)    
    {   
        array[i] = (x & ( 1 << i )) >> i;
    }
    
    return array;
}

MatrixXf deconcatenate(MatrixXi A)
{
    int I = (int)A.rows();
    int J = (int)A.cols();
    int i, j;
    
    unsigned int * ptA = &A(0,0);
    unsigned int * ptAi;
    
    MatrixXf B(I,J*32);
    float * ptB = &B(0,0);
    float * ptBi;
    float * ptBij;
    float * array;
    
    for(i=0;i<I;i+=1)
    {   
        ptAi = ptA+i*J;
        ptBi = ptB+i*J*32;
        
        for(j=0;j<J;j+=1)
        {   
            ptBij = ptBi + j*32;
            array = deconcatenate(ptAi[j]);
            for (int k=0;k<32;k++) ptBij[k] = array[k];
        }
    }
    return B;
}

// Arithmetic gain = 32 (nb of bits) /3 (no fused and popcnt add) /256 (no avx) x32 (single float)= x1.35
// Memory bandwidth gain = 256(no avx) /32 (nb of bits) *3(no fused and popcnt add) = x24
// Actual gain = x2.13
MatrixXf BMP(MatrixXf A,MatrixXf B)
{ 
    // Binarization
    MatrixXi Ab = concatenate(A);
    MatrixXf B_transpose = B.transpose();
    MatrixXi Bb = concatenate(B_transpose);
    
    int I = (int)Ab.rows();
    int J = (int)Bb.rows();
    int K = (int)Bb.cols();
    int i, j, k;

    MatrixXf C(I,J);
    C.setZero();
    
    // cout<<"Ab.rows() = "<<Ab.rows()<<endl;
    // cout<<"Ab.cols() = "<<Ab.cols()<<endl;
    // cout<<"Bb.rows() = "<<Bb.rows()<<endl;
    // cout<<"Bb.cols() = "<<Bb.cols()<<endl;
    // cout<<"C.rows() = "<<C.rows()<<endl;
    // cout<<"C.cols() = "<<C.cols()<<endl;

    unsigned int *ptA = &Ab(0,0);
    unsigned int *ptB = &Bb(0,0);
    unsigned int *ptAi, *ptBj;
    
    float *ptC = &C(0,0);
    float *ptCi;
    float Cij;
    
    // default is shared for openmp
    // #pragma omp parallel for private(i,j,k,Cij, ptAi, ptCi, ptBj) num_threads(4)
    for(i=0;i<I;i+=1)
    {   
        ptAi = ptA+i*K;
        ptCi = ptC+i*J;
        
        for(j=0;j<J;j+=1)
        {   
            ptBj = ptB+j*K;
            Cij = ptCi[j];
            
            for(k=0;k<K;k+=1)
            {   
                Cij += (float)__builtin_popcount(ptAi[k]&ptBj[k]);
            }
            ptCi[j] = Cij;
        }
    }
    
    return C;
}

MatrixXf no_block(MatrixXf A,MatrixXf B)
{ 
    int I = (int)A.rows();
    int J = (int)B.cols();
    int K = (int)B.rows();
    
    MatrixXf C(I,J);
    C.setZero();
    
    float * ptA = &A(0,0);
    float * ptB = &B(0,0);
    float * ptC = &C(0,0);
    
    int i, j, k;
    float* ptAi, *ptCi, *ptBk;
    float Aik;
    
    // default is shared for openmp
    #pragma omp parallel for private(i,j,k,Aik, ptAi, ptCi, ptBk) num_threads(2)
    for(i=0;i<I;i+=1)
    {   
        ptAi = ptA+i*K;
        ptCi = ptC+i*J;
        for(k=0;k<K;k+=1)
        {   
            ptBk = ptB+k*J;
            Aik = ptAi[k]; 
            
            // actually g++ -O3 is capable of vectorizing this loop
            #pragma omp simd
            for(j=0;j<J;j+=1)
            {   
                ptCi[j] += Aik*ptBk[j];
            }
        }
    }
    
    return C;
}

// for some reason, my block matrix mult is slower than my no block 
MatrixXf block(MatrixXf A,MatrixXf B)
{   

    // block matrix multiplication
    // in order to optimize memory accesses, I am using loop tiling and sequential accesses
    // https://en.wikipedia.org/wiki/Loop_tiling
    // https://en.wikipedia.org/wiki/Locality_of_reference
    
    int I = (int)A.rows();
    int J = (int)B.cols();
    int K = (int)B.rows();
    
    MatrixXf C(I,J);
    C.setZero();
    MatrixXf B_transpose = B.transpose();
    
    float * ptA = &A(0,0);
    float * ptB = &B_transpose(0,0);
    float * ptC = &C(0,0);
    
    int b = 32;
    
    int i, j, k, jb, kb;
    float* ptAi, *ptCi, *ptBj;
    float Cij;
    
    // default is shared for openmp
    // #pragma omp parallel for private(i,j,k,Aik) num_threads(4)
    for(kb=0;kb<K;kb+=b)
    {   
        for(jb=0;jb<J;jb+=b)
        {   
            // default is shared for openmp
            // i and ib are merged
            // #pragma omp parallel for private(i,j,k,Cij, ptAi, ptCi, ptBj) num_threads(4)
            for(i=0;i<I;i+=1)
            {   
                ptAi = ptA+i*K+jb;
                ptCi = ptC+i*J+jb;
                
                for(j=0;j<b;j+=1)
                {   
                    ptBj = ptB+(jb+j)*K;
                    Cij = ptCi[j];
                    
                    // actually g++ -O3 is capable of vectorizing this loop
                    #pragma omp simd reduction (+:Cij)
                    for(k=0;k<b;k+=1)
                    {   
                        // ptCi[j] += Aik*ptBk[j];
                        // Aik += Aik * 2;
                        Cij += ptAi[k]*ptBj[k];
                    }
                    ptCi[j] = Cij;
                }
            }
        }
    }
    
    return C;
}
