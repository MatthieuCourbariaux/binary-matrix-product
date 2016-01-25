/*****************************************************************
BMP
Written by Matthieu Courbariaux in 2015
*******************************************************************/

#ifndef binary_mp
#define binary_mp

#include <Eigen/Core>
using Eigen::MatrixXf;
using Eigen::Matrix;
// typedef Matrix<unsigned long int,Eigen::Dynamic,Eigen::Dynamic> MatrixXdi;

class bi64
{

public:
	
	unsigned long int value;
    
    bi64(){}
	~bi64(){}
    
    bi64(int x)
    {   
        value = (unsigned long int) x;
    }
    
    bi64(float x)
    {
        value = (unsigned long int) x;
    }
    
    void operator=(const unsigned long int & x)
    {
        value = x;
    }
    
    void operator=(const int & x)
    {
        value = x;
    }
    
    void operator=(const bi64 & x)
    {
        value = x.value;
    }
    
    operator float()
	{
		return (float)value;
	}
    
    operator int()
	{
		return (int)value;
	}
    
    operator long int()
	{
		return (long int)value;
	}
    
    operator unsigned long int()
	{
		return (unsigned long int)value;
	}
    
    operator unsigned int()
	{
		return (unsigned int)value;
	}
    
    void operator+=(const bi64 & x)
    {
        value += __builtin_popcount(x.value);
    }

    bi64 operator*(const bi64& x)
	{
        bi64 result;
		result.value = ~ (value ^ x.value);
		return result;
	}
    
};



namespace Eigen {
template<> struct NumTraits<bi64>
{
  typedef bi64 Real;
  typedef bi64 NonInteger;
  typedef bi64 Nested;
  enum {
    IsComplex = 0,
    IsInteger = 1,
    IsSigned = 0,
    RequireInitialization = 1,
    ReadCost = 1,
    AddCost = 5,
    MulCost = 2
  };
};
};

typedef Matrix<bi64,Eigen::Dynamic,Eigen::Dynamic> MatrixXb;

unsigned long int float2bool(float* array);
MatrixXb binarize(MatrixXf A);
MatrixXf BMP2(MatrixXf A,MatrixXf B);

#endif