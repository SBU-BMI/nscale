/*
 * ReadTiles.cpp
 *
 *  Created on: Jun 19, 2012
 *      Author: tcpan
 */

#include "ReadTiles.h"
#include "Debug.h"
#include "FileUtils.h"
#include <dirent.h>
#include <string.h>
#include <algorithm>
#include "CmdlineParser.h"
#include "CVImage.h"

namespace cci {
namespace rt {
namespace adios {

const std::string ReadTiles::PARAM_READSIZE = "read_size";


bool ReadTiles::initParams() {
	params.add_options()
		("input_directory,i", boost::program_options::value< std::string >()->required(), "input directory.  REQUIRED")
		("read_size,Q", boost::program_options::value<int>()->required(), "input node count.  REQUIRED")
			;
	return true;
}

boost::program_options::options_description ReadTiles::params("Input Options");
bool ReadTiles::param_init = ReadTiles::initParams();

ReadTiles::ReadTiles(MPI_Comm const * _parent_comm, int const _gid,
		DataBuffer *_input, DataBuffer *_output,
		boost::program_options::variables_map &_vm,
		cci::common::LogSession *_logsession)  :
	Action_I(_parent_comm, _gid, _input, _output, _logsession) {

	long long t1, t2;
	char *fndata=NULL;
	int scount=0, maxLen=0, sbyte=0;

	assert(_output != NULL);

	compressing = cci::rt::CmdlineParser::getParamValueByName<bool>(_vm, cci::rt::DataBuffer::PARAM_COMPRESSION);


	if (this->rank == 0) {

		std::string dirName = cci::rt::CmdlineParser::getParamValueByName<std::string>(_vm, "input_directory");
		int count = cci::rt::CmdlineParser::getParamValueByName<int>(_vm, cci::rt::CmdlineParser::PARAM_INPUTCOUNT);

		t1 = ::cci::common::event::timestampInUS();

		// check to see if it's a directory or a file
		std::vector<std::string> exts;
		exts.push_back(std::string(".tif"));
		exts.push_back(std::string(".tiff"));

		cci::common::FileUtils futils(exts);
		futils.traverseDirectory(dirName, filenames, cci::common::FileUtils::FILE, true);

		std::string dirname = dirName;
		if (filenames.size() == 1) {
			// if the maskname is actually a file, then the dirname is extracted from the maskname.
			if (strcmp(filenames[0].c_str(), dirName.c_str()) == 0) {
				dirname = dirName.substr(0, dirName.find_last_of("/\\"));
			}
		}

		// randomize the file order.
		srand(0);
		std::random_shuffle( filenames.begin(), filenames.end() );
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		srand(rank * 113 + 1);

		int cc = filenames.size();
		if (count == -1) count = cc;
		if (count < filenames.size()) {
			// resize the list
			filenames.resize(count);
		}
		cci::common::Debug::print("%s total number of items found=%d, processing limited to %d\n", getClassName(), cc, count);

		t2 = ::cci::common::event::timestampInUS();
		char len[21];  // max length of uint64 is 20 digits
		memset(len, 0, 21);
		sprintf(len, "%ld", (long)(cc));
		if (this->logsession != NULL) this->logsession->log(cci::common::event(0, std::string("List Files"), t1, t2, std::string(len), ::cci::common::event::FILE_I));


		if (this->size > 1) {
			cci::common::Debug::print("%s constructor exchange data.\n", getClassName());

			// now compute how many each is going to get
			t1 = ::cci::common::event::timestampInUS();

			// get the max length
			int len = 0;
			maxLen = 0;
			for (std::vector<std::string>::iterator iter = filenames.begin(); iter != filenames.end(); ++iter) {
				len = (*iter).length()+1;
				if ( len > maxLen ) maxLen = len;
			}

			// compute number of filenames to send
			int *scounts = new int[this->size];
			scount = filenames.size() / this->size;
			int remains = filenames.size() % this->size;
			for (int i = 0; i < remains; ++i) {
				scounts[i] = (scount + 1);
			}
			for (int i = remains; i < this->size; ++i) {
				scounts[i] = scount;
			}
			scount = scounts[0];
			cci::common::Debug::print("maxlen = %d, scount = %d, size = %d\n", maxLen, scount, this->size);


			// put filenames in an array
			char *filenamedata = new char[filenames.size() * maxLen];
			memset(filenamedata, 0, filenames.size() * maxLen);
			char *ptr = filenamedata;
			for (std::vector<std::string>::iterator iter = filenames.begin(); iter != filenames.end(); ++iter) {
				strcpy(ptr, (*iter).c_str());
				ptr += maxLen;
			}

			// send the size to the peers;
			MPI_Bcast(&maxLen, 1, MPI_INT, 0, this->comm);
			MPI_Scatter(scounts, 1, MPI_INT, &scount, 1, MPI_INT, 0, this->comm);

			int *sbytes = new int[this->size];
			int *soffsets = new int[this->size];
			soffsets[0] = 0;
			sbytes[0] = scounts[0] * maxLen;
			for (int i = 1; i < this->size; ++i) {
				sbytes[i] = scounts[i] * maxLen;
				soffsets[i] = soffsets[i-1] + sbytes[i-1];
			}
			fndata = new char[scount * maxLen];
			memset(fndata, 0, scount * maxLen);
			sbyte = scount * maxLen;
			// send the filenames to the peers;
			MPI_Scatterv(filenamedata, sbytes, soffsets, MPI_CHAR, fndata, sbyte, MPI_CHAR, 0, this->comm);

			// resize the filenames vector
			filenames.resize(scount);

			delete [] scounts;
			delete [] filenamedata;
			delete [] fndata;
			delete [] sbytes;
			delete [] soffsets;
			t2 = ::cci::common::event::timestampInUS();
			if (this->logsession != NULL) this->logsession->log(cci::common::event(0, std::string("Partition Filenames"), t1, t2, std::string(), ::cci::common::event::NETWORK_IO));
		}
	} else {
		if (this->size > 1) {
			cci::common::Debug::print("%s constructor exchange data.\n", getClassName());

			t1 = ::cci::common::event::timestampInUS();

			// get the size from the head
			MPI_Bcast(&maxLen, 1, MPI_INT, 0, this->comm);
			MPI_Scatter(NULL, 1, MPI_INT, &scount, 1, MPI_INT, 0, this->comm);

			cci::common::Debug::print("maxlen = %d, scount = %d, size = %d\n", maxLen, scount, this->size);


			fndata = new char[scount * maxLen];
			memset(fndata, 0, scount * maxLen);
			sbyte = scount * maxLen;

			// get the filenames from the head
			MPI_Scatterv(NULL, NULL, NULL, MPI_CHAR, fndata, sbyte, MPI_CHAR, 0, this->comm);

			// construct the filenames vector
			filenames.clear();
			char *ptr = fndata;
			for (int i = 0; i < scount; ++i) {
				filenames.push_back(std::string(ptr));
				ptr += maxLen;
			}

			delete [] fndata;
			t2 = ::cci::common::event::timestampInUS();
			if (this->logsession != NULL) this->logsession->log(cci::common::event(0, std::string("partitioned"), t1, t2, std::string(), ::cci::common::event::NETWORK_IO));
		}
	}

	cci::common::Debug::print("%s constructor done.\n", getClassName());

}

ReadTiles::~ReadTiles() {

	cci::common::Debug::print("%s destructor called.\n", getClassName());
	filenames.clear();
}

/**
 * generate some results.  if no more, set the done flag.
 */
int ReadTiles::compute(int const &input_size , void * const &input,
			int &output_size, void * &output) {

	long long t1, t2;

	while (filenames.size() > 0) {
		t1 = ::cci::common::event::timestampInUS();

		std::string fn = filenames.back();
		filenames.pop_back();

		// parse the input string
		std::string filename = cci::common::FileUtils::getFile(const_cast<std::string&>(fn));
		// get the image name
		size_t pos = filename.rfind('.');
		if (pos == std::string::npos) printf("ERROR:  file %s does not have extension\n", fn.c_str());
		std::string prefix = filename.substr(0, pos);
		pos = prefix.rfind("-");
		if (pos == std::string::npos) printf("ERROR:  file %s does not have a properly formed x, y coords\n", fn.c_str());
		std::string ystr = prefix.substr(pos + 1);
		prefix = prefix.substr(0, pos);
		pos = prefix.rfind("-");
		if (pos == std::string::npos) printf("ERROR:  file %s does not have a properly formed x, y coords\n", fn.c_str());
		std::string xstr = prefix.substr(pos + 1);

		std::string imagename = prefix.substr(0, pos);
		int tilex = atoi(xstr.c_str());
		int tiley = atoi(ystr.c_str());

		cv::Mat im = cv::imread(fn, -1);

		t2 = ::cci::common::event::timestampInUS();
		char len[21];  // max length of uint64 is 20 digits
		memset(len, 0, 21);
		sprintf(len, "%lu", (long)(im.dataend) - (long)(im.datastart));
		if (logsession != NULL) logsession->log(cci::common::event(0, std::string("read"), t1, t2, std::string(len), ::cci::common::event::FILE_I));

		if (im.data != NULL) {

			t1 = ::cci::common::event::timestampInUS();

			CVImage *img = new CVImage(im, imagename, fn, tilex, tiley);
			if (compressing) img->serialize(output_size, output, CVImage::ENCODE_Z);
			else img->serialize(output_size, output);
			// clean up
			delete img;
			im.release();

			t2 = ::cci::common::event::timestampInUS();
			memset(len, 0, 21);
			sprintf(len, "%lu", (long)output_size);
			if (logsession != NULL) logsession->log(cci::common::event(90, std::string("serialize"), t1, t2, std::string(len), ::cci::common::event::MEM_IO));

			return Communicator_I::READY;

		} else {
			im.release();
		}

	}

	// else no more so done.
	output = NULL;
	output_size = 0;

	return Communicator_I::DONE;

}

int ReadTiles::run() {


	if (outputBuf->isStopped()) {
		cci::common::Debug::print("%s STOPPED. call count %d \n", getClassName(), call_count);
		return Communicator_I::DONE;
	} else if (!outputBuf->canPush()){
		//cci::common::Debug::print("%s FULL. call count %d \n", getClassName(), call_count);
		return Communicator_I::WAIT;
	} // else has room, and not stopped, so can push.

	int output_size = 0;
	void *output = NULL;

	int result = compute(-1, NULL, output_size, output);

//	if (output != NULL)
//		cci::common::Debug::print("%s iter %d output var passed back at address %x, value %s, size %d, result = %d\n", getClassName(), call_count, output, output, output_size, result);
//	else
//		cci::common::Debug::print("%s iter %d output var passed back at address %x, size %d, result = %d\n", getClassName(), call_count, output, output_size, result);

	int bstat;
	if (result == Communicator_I::READY) {
		++call_count;
		bstat = outputBuf->push(std::make_pair(output_size, output));

		if (bstat == DataBuffer::STOP) {
			cci::common::Debug::print("ERROR: %s can't push into buffer.  status STOP.  Should have caught this earlier. \n", getClassName());
			free(output);
			return Communicator_I::DONE;
		} else if (bstat == DataBuffer::FULL) {
			cci::common::Debug::print("ERROR: %s can't push into buffer.  status FULL.  Should have caught this earlier.\n", getClassName());
			free(output);
			return Communicator_I::WAIT;
		} else {
			return Communicator_I::READY;
		}

	} else if (result == Communicator_I::DONE) {

		// no more, so done.
		outputBuf->stop();
	}
	return result;

}

}
} /* namespace rt */
} /* namespace cci */
