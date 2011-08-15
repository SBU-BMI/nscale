#ifndef CONSTANT_H_
#define CONSTANT_H_

using namespace std;
using namespace cv;

class Constant {
public:
	//! Number of angles in which the coocurrence matrices are calculated
	static const int NUM_ANGLES=4;

	//! Constants used to calculate coocurrence matrices
	static const int ANGLE_0=0;
	static const int ANGLE_45=1;
	static const int ANGLE_90=2;
	static const int ANGLE_135=3;

	//! Defining what processor should be used when invoking the functions
	static const int CPU=1;
	static const int GPU=2;

	static const int N_INTENSITY_FEATURES=6;
};

#endif
