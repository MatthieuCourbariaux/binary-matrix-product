


all: BMP_test.cpp BMP.h BMP.cpp
	g++ -I/u/courbarm/Eigen/ -mpopcnt -O3 -fopenmp -o out $^

clean:
	rm -rf *.gch *.o out
