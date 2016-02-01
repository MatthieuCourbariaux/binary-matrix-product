
#include <iostream>
#include "BMP.h"
#include <Eigen/Core>
using Eigen::MatrixXf;
using namespace std;

float float_sign(float x)
{
    // return (x>=0);
    return 2. * (x>=0) - 1.;
}

int main(int argc, char* argv[])
{   
    /*
    cout << endl << "-- Testing concatenate and deconcatenate part 1 --" << endl;
    
    float * A = new float[32];
    for(int i = 0; i<32; i++)
    {
        A[i] = rand() % 2;
    }
    cout<<endl<<"A = ["<<endl;
    for(int i = 0; i<32; i++)
    {
        cout<<A[i]<<"  ";
    }
    cout<<endl<<"]"<<endl;
    
    unsigned long int a = concatenate(A);
    cout<<endl<<"concatenate(A) = "<<a<<endl;
    
    A = deconcatenate(a);
    cout<<endl<<"deconcatenate(concatenate(A)) = ["<<endl;
    for(int i = 0; i<32; i++)
    {
        cout<<A[i]<<"  ";
    }
    cout<<endl<<"]"<<endl;
    
    
    cout << endl << "-- Testing concatenate and deconcatenate part 2 --" << endl;
    
    int N = 1024;
    
    // A is a matrix filled with 1 and 0s.
    MatrixXf B(N,N);
    B.setRandom();
    B = B.unaryExpr(ptr_fun(float_sign));
    
    cout<<endl<<"B.maxCoeff() = " << B.maxCoeff();
    cout<<endl<<"B.minCoeff() = " << B.minCoeff();
    cout<<endl<<"B.sum() = " << B.sum();
    // cout<<endl<<"deconcatenate(concatenate(B)).maxCoeff() = " << deconcatenate(concatenate(B)).maxCoeff();
    // cout<<endl<<"deconcatenate(concatenate(B)).minCoeff() = " << deconcatenate(concatenate(B)).minCoeff();
    cout<<endl<<"deconcatenate(concatenate(B)).sum() = " << deconcatenate(concatenate(B)).sum();
    cout<<endl<<"(B-deconcatenate(concatenate(B))).sum() = " << (B-deconcatenate(concatenate(B))).sum()<<endl<<endl;
    
    */
    
	cout << endl << "-- Testing Binary Matrix Product --" << endl;	
    
    // Eigen::setNbThreads(4);
    Eigen::setNbThreads(1);
        
    // int N = 8192;
    int N = 4096;
    
    MatrixXf A(N,N);
    // A.setZero();
    A.setRandom();
    A = A.unaryExpr(ptr_fun(float_sign));
    // cout <<endl<<"A max = " <<A.maxCoeff();
    // cout <<endl<<"A min = " <<A.minCoeff();
    // cout <<endl<<"A sum = " <<A.sum();

    MatrixXf B(N,N);
    // B.setZero();
    B.setRandom();
    B = B.unaryExpr(ptr_fun(float_sign));
    // cout <<endl<<"B max = " <<B.maxCoeff();
    // cout <<endl<<"B min = " <<B.minCoeff();
    
    // cout <<endl<<"A B diff = " <<(A-B).sum();
    
    MatrixXf C1(N,N);
    MatrixXf C2(N,N);
    
    double elapsed_time = omp_get_wtime();
    C1 = BMP(A,B);
    // C1 = A*B;
    elapsed_time = omp_get_wtime()-elapsed_time;
    cout <<endl<<"BPM elapsed_time = " << elapsed_time<<"s";
    
    elapsed_time = omp_get_wtime();
    C2 = A*B;
    // C2 = BMP(A,B);
    elapsed_time = omp_get_wtime()-elapsed_time;
    cout <<endl<<"Eigen SGEMM elapsed_time = " << elapsed_time<<"s";

    cout<<endl<<"C1 sum = " << C1.sum();
    cout<<endl<<"C2 sum = " << C2.sum();
    cout<<endl<<"Mean difference = " << (C1-C2).mean()<<endl<<endl;
}    
    
