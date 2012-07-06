/*
 * Segment.cpp
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#include "Segment.h"
#include "Debug.h"
#include "opencv2/opencv.hpp"

namespace cci {
namespace rt {
namespace adios {

Segment::Segment(MPI_Comm const * _parent_comm, int const _gid) :
				Action_I(_parent_comm, _gid) {
	// TODO Auto-generated constructor stub

}

Segment::~Segment() {
	Debug::print("%s destructor called.\n", getClassName());
}
int Segment::compute(int const &input_size , void * const &input,
			int &output_size, void * &output) {
	if (input_size == 0 || input == NULL) return -1;

	//	output_size = input_size;
	//	output = malloc(output_size);
	//	memcpy(output, input, input_size);


	const char *filename = (const char *)input;
	Debug::print("%s READING %s\n", getClassName(), filename);

	cv::Mat im = cv::imread(std::string(filename), -1);

	assert(im.isContinuous());
	assert(!im.empty());

	output_size = im.rows * im.cols * im.elemSize();
	output = malloc(output_size);
	memcpy(output, im.data, output_size);

	im.release();

	return 1;
}

int Segment::run() {

	if (!canAddOutput()) return output_status;

	call_count++;

	int output_size, input_size;
	void *output, *input;

	int istatus = getInputStatus();

	if (istatus == READY) {
		input = NULL;
		int result = getInput(input_size, input);
//		Debug::print("%s READY and getting input:  call count= %d\n", getClassName(), call_count);

		result = compute(input_size, input, output_size, output);
		if (input != NULL) {
			free(input);
			input = NULL;
		}

		if (result >= 0) {
//			Debug::print("%s bufferring output:  call count= %d\n", getClassName(), call_count);

			result = addOutput(output_size, output);
//			free(output);
		} else {
			Debug::print("%s no output.  call count = %d\n", getClassName(), call_count);
			result = WAIT;
		}
		return result;
	} else if (istatus == WAIT) {
//		Debug::print("%s READY and waiting for input: call count= %d\n", getClassName(), call_count);
		return WAIT;
	} else {  // done or error //
		// output already changed.
		Debug::print("%s Done for input:  call count= %d\n", getClassName(), call_count);
		return output_status;
	}

}

}
} /* namespace rt */
} /* namespace cci */
