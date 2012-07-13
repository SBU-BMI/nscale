/*
 * AssignTiles.cpp
 *
 *  Created on: Jun 19, 2012
 *      Author: tcpan
 */

#include "AssignTiles.h"
#include "Debug.h"
#include "FileUtils.h"
#include <dirent.h>
#include <string.h>
#include <algorithm>

namespace cci {
namespace rt {
namespace adios {


AssignTiles::AssignTiles(MPI_Comm const * _parent_comm, int const _gid,
		std::string &dirName, int count, cciutils::SCIOLogSession *_logsession)  :
	Action_I(_parent_comm, _gid, _logsession) {

	long long t1, t2;
	t1 = ::cciutils::event::timestampInUS();

	// check to see if it's a directory or a file
	std::string suffix;
	suffix.assign(".tif");

	FileUtils futils(suffix);
	futils.traverseDirectoryRecursive(dirName, filenames);

	std::string dirname = dirName;
	if (filenames.size() == 1) {
		// if the maskname is actually a file, then the dirname is extracted from the maskname.
		if (strcmp(filenames[0].c_str(), dirName.c_str()) == 0) {
			dirname = dirName.substr(0, dirName.find_last_of("/\\"));
		}
	}

	// randomize the file order.
	std::random_shuffle( filenames.begin(), filenames.end() );

	int cc = filenames.size();
	if (count == -1) count = cc;
	if (count < filenames.size()) {
		// resize the list
		filenames.resize(count);
	}
	Debug::print("total number of items found=%d, processing limited to %d\n", filenames.size(), count);

	t2 = ::cciutils::event::timestampInUS();
	char len[21];  // max length of uint64 is 20 digits
	memset(len, 0, 21);
	sprintf(len, "%ld", (long)(cc));
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("List Files"), t1, t2, std::string(len), ::cciutils::event::FILE_I));
}

AssignTiles::~AssignTiles() {

//	Debug::print("%s destructor called.\n", getClassName());
	filenames.clear();
}

/**
 * generate some results.  if no more, set the done flag.
 */
int AssignTiles::compute(int const &input_size , void * const &input,
			int &output_size, void * &output) {

	if (filenames.size() > 0) {
		long long t1, t2;
		t1 = ::cciutils::event::timestampInUS();

		output_size = filenames.back().length() + 1;
		output = malloc(output_size);
		memset(output, 0, output_size);
		memcpy(output, filenames.back().c_str(), output_size - 1);

		filenames.pop_back();

		t2 = ::cciutils::event::timestampInUS();
		if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("Assign"), t1, t2, std::string(), ::cciutils::event::MEM_IO));

		return READY;
	} else {
		output = NULL;
		output_size = 0;

		return DONE;
	}

}

int AssignTiles::run() {

	if (!canAddOutput()) {
		Debug::print("%s DONE. call count %d \n", getClassName(), call_count);
		return output_status;
	}


	int output_size = 0;
	void *output = NULL;


	int result = compute(-1, NULL, output_size, output);
	call_count++;

//	if (output != NULL)
//		Debug::print("%s iter %d output var passed back at address %x, value %s, size %d, result = %d\n", getClassName(), call_count, output, output, output_size, result);
//	else
//		Debug::print("%s iter %d output var passed back at address %x, size %d, result = %d\n", getClassName(), call_count, output, output_size, result);


	if (result == READY) {
		output_status = addOutput(output_size, output);
	} else if (result == WAIT) {
		if (this->outputSizes.empty()) output_status = WAIT;
		else output_status = READY;
	} else {
		Debug::print("%s DONE. call count %d \n", getClassName(), call_count);

		output_status = result;
	}


	return output_status;
}

}
} /* namespace rt */
} /* namespace cci */
