
#include <iostream>
#include "BMP.h"
#include <Eigen/Core>
using Eigen::MatrixXf;
using namespace std;

int main(int argc, char* argv[])
{
	cout << endl << "-- BEGINNING OF PROGRAM --" << endl<<endl;	
    
    // Eigen::setNbThreads(4);
    Eigen::setNbThreads(4);
    
    int N = 8192;
    // int N = 4096;
    MatrixXf A(N,N);
    A.setZero();
    MatrixXf B(N,N);
    B.setZero();
    MatrixXf C(N,N);
    
    double elapsed_time = omp_get_wtime();
    C = BMP(A,B);
    elapsed_time = omp_get_wtime()-elapsed_time;
    cout <<endl<<"	BPM elapsed_time = " << elapsed_time<<"s";
    
    elapsed_time = omp_get_wtime();
    C = A*B;
    elapsed_time = omp_get_wtime()-elapsed_time;
    cout <<endl<<"	Eigen SGEMM elapsed_time = " << elapsed_time<<"s"<<endl<<endl;
}    
    
