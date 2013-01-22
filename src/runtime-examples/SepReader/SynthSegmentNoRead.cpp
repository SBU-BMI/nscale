/*
 * SynthSegmentNoRead.cpp
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#include "SynthSegmentNoRead.h"
#include "Debug.h"
#include "opencv2/opencv.hpp"
#include "CVImage.h"
#include <string>
#include "SCIOHistologicalEntities.h"
#include "FileUtils.h"
#include "TypeUtils.h"
#include "MathUtils.h"
#include "CmdlineParser.h"

namespace cci {
namespace rt {
namespace adios {

bool SynthSegmentNoRead::initParams() {
	params.add_options()
		("compute_time_distro,d", boost::program_options::value< std::string >(), "synthetic compute time distributions: p_bg,mean_bg,stdev_bg;p_nu,mean_nu,stdev_nu;p_full,mean_full,stdev_full")
			;
	return true;
}

boost::program_options::options_description SynthSegmentNoRead::params("Synth Compute Options");
bool SynthSegmentNoRead::param_init = SynthSegmentNoRead::initParams();


SynthSegmentNoRead::SynthSegmentNoRead(MPI_Comm const * _parent_comm, int const _gid,
		DataBuffer *_input, DataBuffer *_output,
		boost::program_options::variables_map &_vm,
		cci::common::LogSession *_logsession) :
				Action_I(_parent_comm, _gid, _input, _output, _logsession), output_count(0)
 	 {
	assert(_input != NULL);
	assert(_output != NULL);

	compressing = cci::rt::CmdlineParser::getParamValueByName<bool>(_vm, cci::rt::DataBuffer::PARAM_COMPRESSION);

	std::string distro;
	p_bg = 0.3;
	p_nu = 0.1;
	p_full = 0.6;
	mean_bg = 1;
	mean_nu = 1;
	mean_full = 1;
	stdev_bg = 0.3;
	stdev_nu = 0.3;
	stdev_full = 0.3;
	if (_vm.count("compute_time_distro")) {
		distro = cci::rt::CmdlineParser::getParamValueByName< std::string >(_vm, "compute_time_distro");



		int spos = 0, epos = 0;
		epos = distro.find(',', spos);
		p_bg = atof(distro.substr(spos, epos - spos).c_str());
		spos = epos + 1;
		epos = distro.find(',', spos);
		mean_bg = atof(distro.substr(spos, epos - spos).c_str());
		spos = epos + 1;
		epos = distro.find(';', spos);
		stdev_bg = atof(distro.substr(spos, epos - spos).c_str());
		spos = epos + 1;
		epos = distro.find(',', spos);
		p_nu = atof(distro.substr(spos, epos - spos).c_str());
		spos = epos + 1;
		epos = distro.find(',', spos);
		mean_nu = atof(distro.substr(spos, epos - spos).c_str());
		spos = epos + 1;
		epos = distro.find(';', spos);
		stdev_nu = atof(distro.substr(spos, epos - spos).c_str());
		spos = epos + 1;
		epos = distro.find(',', spos);
		p_full = atof(distro.substr(spos, epos - spos).c_str());
		spos = epos + 1;
		epos = distro.find(',', spos);
		mean_full = atof(distro.substr(spos, epos - spos).c_str());
		spos = epos + 1;
		stdev_full = atof(distro.substr(spos).c_str());

	}

}

SynthSegmentNoRead::~SynthSegmentNoRead() {
	cci::common::Debug::print("%s destructor called.\n", getClassName());
}

int SynthSegmentNoRead::compute(int const &input_size , void * const &input,
			int &output_size, void * &output) {
	if (input_size == 0 || input == NULL) return -1;

	long long t1, t2;

	t1 = ::cci::common::event::timestampInUS();

	CVImage * img;
	if (compressing) img = new CVImage(input_size, input, CVImage::ENCODE_Z);
	else img = new CVImage(input_size, input, CVImage::ENCODE_RAW);

	int dummy1, dummy2;
	std::string fn(img->getSourceFileName(dummy1, dummy2));
	std::string imagename(img->getImageName(dummy1, dummy2));
	int tilex = img->getMetadata().info.x_offset;
	int tiley = img->getMetadata().info.y_offset;

	cv::Mat im = img->getImage();

	//sleep(rand() % 3 + 1);
	t2 = ::cci::common::event::timestampInUS();
	char len[21];  // max length of uint64 is 20 digits
	memset(len, 0, 21);
	sprintf(len, "%lu", (long)(im.dataend) - (long)(im.datastart));
	if (logsession != NULL) logsession->log(cci::common::event(0, std::string("deserialize"), t1, t2, std::string(len), ::cci::common::event::FILE_I));


	if (!im.data) {
		im.release();
		return -1;
	}

	t1 = ::cci::common::event::timestampInUS();

	// real computation:
	int status = ::nscale::SCIOHistologicalEntities::SUCCESS;
	cv::Mat mask = cv::Mat::zeros(im.size(), CV_32SC1);

	double p = (double)rand() / ((double)RAND_MAX);
	double mean, stdev;
	std::string eventName;
	if (p < p_bg) {
		// background
		mean = mean_bg;
		stdev = stdev_bg;
		eventName.assign("computeNoFG");
		status =  ::nscale::SCIOHistologicalEntities::BACKGROUND;
	} else if (p < (p_bg + p_nu)) {
		// not enough nuclei
		mean = mean_nu;
		stdev = stdev_nu;
		eventName.assign("computeNoNU");
		status =  ::nscale::SCIOHistologicalEntities::NO_CANDIDATES_LEFT;
	} else if (p_bg + p_nu + p_full <= 1.0) {
		// process finished completely.
		mean = mean_full;
		stdev = stdev_full;
		eventName.assign("computeFull");
		status =  ::nscale::SCIOHistologicalEntities::SUCCESS;
	} else {
		eventName.assign("computeOTHER");
		mean = 0;
		stdev = 0;
		status =  ::nscale::SCIOHistologicalEntities::INVALID_IMAGE;
	}
	double q = cci::common::MathUtils::randn(mean, stdev);
	if (q > 0) usleep((unsigned int)round(q * 1000000));


	t2 = ::cci::common::event::timestampInUS();
	if (logsession != NULL) logsession->log(cci::common::event(90, eventName, t1, t2, std::string("1"), ::cci::common::event::COMPUTE));

	if (status == ::nscale::SCIOHistologicalEntities::SUCCESS) {
		t1 = ::cci::common::event::timestampInUS();
		CVImage *img = new CVImage(mask, imagename, fn, tilex, tiley);
//		CVImage *img = new CVImage(im, imagename, fn, tilex, tiley);
		if (compressing) img->serialize(output_size, output, CVImage::ENCODE_Z);
		else img->serialize(output_size, output);
		// clean up
		delete img;


		t2 = ::cci::common::event::timestampInUS();
		memset(len, 0, 21);
		sprintf(len, "%lu", (long)output_size);
		if (logsession != NULL) logsession->log(cci::common::event(90, std::string("serialize"), t1, t2, std::string(len), ::cci::common::event::MEM_IO));

	}
	im.release();

	mask.release();
	return status;
}

int SynthSegmentNoRead::run() {

	if (this->inputBuf->isFinished()) {
		cci::common::Debug::print("%s input DONE.  input count = %d, output count = %d\n", getClassName(), call_count, output_count);
		this->outputBuf->stop();

		return Communicator_I::DONE;
	} else if (this->outputBuf->isStopped()) {
		cci::common::Debug::print("%s output DONE.  input count = %d, output count = %d\n", getClassName(), call_count, output_count);
		this->inputBuf->stop();

		if (!this->inputBuf->isFinished()) cci::common::Debug::print("WARNING: %s input buffer is not empty.\n", getClassName());
		return Communicator_I::DONE;
	} else if (!this->inputBuf->canPop() || !this->outputBuf->canPush()) {
		return Communicator_I::WAIT;
	}

	DataBuffer::DataType data;
	int output_size, input_size;
	void *output = NULL, *input = NULL;


	int bstat = this->inputBuf->pop(data);
	if (bstat == DataBuffer::EMPTY) {
		return Communicator_I::WAIT;
	}
	input_size = data.first;
	input = data.second;

//		cci::common::Debug::print("%s READY and getting input:  call count= %d\n", getClassName(), call_count);

	int result = compute(input_size, input, output_size, output);
	call_count++;


	if (result == ::nscale::SCIOHistologicalEntities::SUCCESS) {
//			cci::common::Debug::print("%s bufferring output:  call count= %d\n", getClassName(), call_count);
		++output_count;
		bstat = this->outputBuf->push(std::make_pair(output_size, output));

		if (bstat == DataBuffer::STOP) {
			cci::common::Debug::print("ERROR: %s can't push into buffer.  status STOP.  Should have caught this earlier. \n", getClassName());
			this->inputBuf->push(data);
			this->inputBuf->stop();
			free(output);
			return Communicator_I::DONE;
		} else if (bstat == DataBuffer::FULL) {
			cci::common::Debug::print("WARNING: %s can't push into buffer.  status FULL.  Should have caught this earlier.\n", getClassName());
			this->inputBuf->push(data);
			free(output);
			return Communicator_I::WAIT;
		} else {
			if (input != NULL) {
				free(input);
				input = NULL;
			}
			return Communicator_I::READY;
		}
	} else {
		if (input != NULL) {
			free(input);
			input = NULL;
		}
		return Communicator_I::READY;
	}


}

}
} /* namespace rt */
} /* namespace cci */
