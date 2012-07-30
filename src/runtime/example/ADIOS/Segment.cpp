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
#include "utils.h"
#include "SCIOHistologicalEntities.h"

namespace cci {
namespace rt {
namespace adios {

Segment::Segment(MPI_Comm const * _parent_comm, int const _gid,
		std::string &proctype, int gpuid,
		cciutils::SCIOLogSession *_logsession) :
				Action_I(_parent_comm, _gid, _logsession) {
	if (strcmp(proctype.c_str(), "cpu")) proc_code = cciutils::DEVICE_CPU;
	else if (strcmp(proctype.c_str(), "cpu")) {
		proc_code = cciutils::DEVICE_GPU;
	}
}

Segment::~Segment() {
//	Debug::print("%s destructor called.\n", getClassName());
}
int Segment::compute(int const &input_size , void * const &input,
			int &output_size, void * &output) {
	if (input_size == 0 || input == NULL) return -1;

	long long t1, t2;

	t1 = ::cciutils::event::timestampInUS();

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
	t2 = ::cciutils::event::timestampInUS();
	char len[21];  // max length of uint64 is 20 digits
	memset(len, 0, 21);
	sprintf(len, "%lu", (long)(im.dataend) - (long)(im.datastart));
	if (logsession != NULL) logsession->log(cciutils::event(0, std::string("read"), t1, t2, std::string(len), ::cciutils::event::FILE_I));


	if (!im.data) {
		im.release();
		return -1;
	}

	t1 = ::cciutils::event::timestampInUS();

	// real computation:
	int status;
	int *bbox = NULL;
	int compcount;
	cv::Mat mask;
//	if (proc_code == cciutils::DEVICE_GPU ) {
//		nscale::gpu::SCIOHistologicalEntities *seg = new nscale::gpu::SCIOHistologicalEntities(fn);
//		status = seg->segmentNuclei(std::string(input), std::string(mask), compcount, bbox, NULL, session, writer);
//		delete seg;
//
//	} else {

	nscale::SCIOHistologicalEntities *seg = new nscale::SCIOHistologicalEntities(fn);
	status = seg->segmentNuclei(im, mask, compcount, bbox, logsession, NULL);
	delete seg;
//	}
	if (bbox != NULL) free(bbox);
	im.release();

	t2 = ::cciutils::event::timestampInUS();
	if (logsession != NULL) logsession->log(cciutils::event(90, std::string("compute"), t1, t2, std::string("1"), ::cciutils::event::COMPUTE));

	if (status == ::nscale::SCIOHistologicalEntities::SUCCESS) {
		t1 = ::cciutils::event::timestampInUS();
		CVImage *img = new CVImage(mask, imagename, fn, tilex, tiley);
		img->serialize(output_size, output);
		// clean up
		delete img;


		t2 = ::cciutils::event::timestampInUS();
		memset(len, 0, 21);
		sprintf(len, "%lu", (long)output_size);
		if (logsession != NULL) logsession->log(cciutils::event(90, std::string("serialize"), t1, t2, std::string(len), ::cciutils::event::MEM_IO));

	}
	mask.release();
	return status;
}

int Segment::run() {

	if (!canAddOutput()) return output_status;

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

		if (result == ::nscale::SCIOHistologicalEntities::SUCCESS) {
//			Debug::print("%s bufferring output:  call count= %d\n", getClassName(), call_count);

			result = addOutput(output_size, output);
			call_count++;

//			free(output);
		} else {
//			Debug::print("%s no output.  entries processed = %d\n", getClassName(), call_count);
			result = WAIT;
		}
		return result;
	} else if (istatus == WAIT) {
//		Debug::print("%s READY and waiting for input: call count= %d\n", getClassName(), call_count);
		return WAIT;
	} else {  // done or error //
		// output already changed.
		Debug::print("%s DONE.  entries processed = %d\n", getClassName(), call_count);
		return output_status;
	}

}

}
} /* namespace rt */
} /* namespace cci */
