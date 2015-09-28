
#include <iostream>
#include "BMP.h"
#include <math.h>
#include <ctime>
#include <Eigen/Core>
using Eigen::MatrixXf;
// using Eigen::MatrixXi;
using Eigen::Matrix;
typedef Matrix<unsigned long int,Eigen::Dynamic,Eigen::Dynamic> MatrixXdi;
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

MatrixXdi binarize(MatrixXf A)
{
    int I = (int)A.rows();
    int J = (int)A.cols();
    int i, j;
    
    MatrixXdi B(I,J/64);
    
    float * ptA = &A(0,0);
    unsigned long int * ptB = &B(0,0);
    float* ptAi;
    unsigned long int* ptBi;
    
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
MatrixXf BMP(MatrixXf A,MatrixXf B)
{ 
    // Binarization
    MatrixXdi Ab = binarize(A);
    MatrixXf B_transpose = B.transpose();
    MatrixXdi Bb = binarize(B_transpose);
    
    int I = (int)Ab.rows();
    int J = (int)Bb.rows();
    int K = (int)Bb.cols();
    int i, j, k;

    MatrixXdi C(I,J);
    C.setZero();
    
    // cout<<"Ab.rows() = "<<Ab.rows()<<endl;
    // cout<<"Ab.cols() = "<<Ab.cols()<<endl;
    // cout<<"Bb.rows() = "<<Bb.rows()<<endl;
    // cout<<"Bb.cols() = "<<Bb.cols()<<endl;
    // cout<<"C.rows() = "<<C.rows()<<endl;
    // cout<<"C.cols() = "<<C.cols()<<endl;

    unsigned long int * ptA = &Ab(0,0);
    unsigned long int * ptB = &Bb(0,0);
    unsigned long int * ptC = &C(0,0);
    unsigned long int* ptAi, *ptCi, *ptBj;
    unsigned long int Cij;
    
    // default is shared for openmp
    #pragma omp parallel for private(i,j,k,Cij, ptAi, ptCi, ptBj) num_threads(4)
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
                Cij += __builtin_popcount(ptAi[k]&ptBj[k]);
            }
            ptCi[j] = Cij;
        }
    }
    
    return C.cast<float>();
}

MatrixXf approx(MatrixXf A,MatrixXf B)
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
    float scaler = .25/K;
    
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
            // only works on g++ >4.9
            #pragma omp simd
            for(j=0;j<J;j+=1)
            {   
                ptCi[j]+=scaler*pow(abs(Aik+ptBk[j])-abs(Aik-ptBk[j]),2);
            }
        }
    }
    
    return C;
}

MatrixXf not_approx(MatrixXf A,MatrixXf B)
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
    float scaler = .25;
    
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
            // only works on g++ >4.9
            #pragma omp simd
            for(j=0;j<J;j+=1)
            {   
                ptCi[j]+=scaler*(pow(Aik+ptBk[j],2)-pow(Aik-ptBk[j],2));
            }
        }
    }
    
    return C;
}

MatrixXf no_block2(MatrixXf A,MatrixXf B)
{ 
    int I = (int)A.rows();
    int J = (int)B.cols();
    int K = (int)B.rows();
    
    MatrixXf C(I,J);
    C.setZero();
    MatrixXf B_transpose = B.transpose();
    
    float * ptA = &A(0,0);
    float * ptB = &B_transpose(0,0);
    float * ptC = &C(0,0);
    
    int i, j, k;
    float* ptAi, *ptCi, *ptBj;
    float Cij;
    
    // default is shared for openmp
    #pragma omp parallel for private(i,j,k,Cij, ptAi, ptCi, ptBj) num_threads(2)
    for(i=0;i<I;i+=1)
    {   
        ptAi = ptA+i*K;
        ptCi = ptC+i*J;
        
        for(j=0;j<J;j+=1)
        {   
            ptBj = ptB+j*K;
            Cij = ptCi[j];
            
            // only works on g++ >4.9
            #pragma omp simd reduction (+:Cij)
            for(k=0;k<K;k+=1)
            {   
                // ptCi[j] += Aik*ptBk[j];
                // Aik += Aik * 2;
                Cij += ptAi[k]*ptBj[k];
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