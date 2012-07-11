/*
 * Segment.cpp
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#include "Segment.h"
#include "Debug.h"
#include "opencv2/opencv.hpp"
#include "CVImage.h"
#include "FileUtils.h"
#include <string>

namespace cci {
namespace rt {
namespace adios {

Segment::Segment(MPI_Comm const * _parent_comm, int const _gid, cciutils::SCIOLogSession *_logger) :
				Action_I(_parent_comm, _gid, _logger) {
}

Segment::~Segment() {
//	Debug::print("%s destructor called.\n", getClassName());
}
int Segment::compute(int const &input_size , void * const &input,
			int &output_size, void * &output) {
	if (input_size == 0 || input == NULL) return -1;


	std::string fn = std::string((char const *)input);
	//Debug::print("%s READING %s\n", getClassName(), fn.c_str());


	// parse the input string
	FileUtils futils;
	std::string filename = futils.getFile(const_cast<std::string&>(fn));
	// get the image name
	size_t pos = filename.rfind('.');
	if (pos == std::string::npos) printf("ERROR:  file %s does not have extension\n", fn.c_str());
	string prefix = filename.substr(0, pos);
	pos = prefix.rfind("-");
	if (pos == std::string::npos) printf("ERROR:  file %s does not have a properly formed x, y coords\n", fn.c_str());
	string ystr = prefix.substr(pos + 1);
	prefix = prefix.substr(0, pos);
	pos = prefix.rfind("-");
	if (pos == std::string::npos) printf("ERROR:  file %s does not have a properly formed x, y coords\n", fn.c_str());
	string xstr = prefix.substr(pos + 1);

	string imagename = prefix.substr(0, pos);
	int tilex = atoi(xstr.c_str());
	int tiley = atoi(ystr.c_str());

	cv::Mat im = cv::imread(fn, -1);

	// simulate computation
	//sleep(10);

	CVImage *img = new CVImage(im, imagename, fn, tilex, tiley);
	img->serialize(output_size, output);

	// clean up
	delete img;

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
