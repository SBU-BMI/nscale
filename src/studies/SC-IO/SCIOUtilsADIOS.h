/*
 * for outputting intermediate images.
 * since the intermediate results may only be needed for certain executables, instead of relying on include to pickup the #defines,
 * and risk that a library may not have been compiled with the right set, I am creating 2 versions of the class (so as to not to have branch).
 *
 *  Created on: Jul 15, 2011
 *      Author: tcpan
 */

#ifndef SCIO_UTILS_ADIOS_H_
#define SCIO_UTILS_ADIOS_H_

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "adios.h"
#include "mpi.h"
#include <vector>

namespace cciutils {

class SCIOADIOSWriter;

class ADIOSManager {
private:
	MPI_Comm *comm;
	int rank;

	std::vector<SCIOADIOSWriter *> writers;

public:
	ADIOSManager(const char* configfilename,  int _rank, MPI_Comm *_comm);
	virtual ~ADIOSManager();

	virtual SCIOADIOSWriter *allocateWriter(const std::string &pref, const std::string &suf, const bool newfile, const std::string &_group_name, std::vector<int> &selStages, int _local_rank, MPI_Comm *_local_comm);
	virtual void freeWriter(SCIOADIOSWriter *w);
};



class SCIOADIOSWriter {

	friend SCIOADIOSWriter* ADIOSManager::allocateWriter(const std::string &pref, const std::string &suf, const bool newfile, const std::string &_group_name, std::vector<int> &selStages, int _local_rank, MPI_Comm *_local_comm);
	friend void ADIOSManager::freeWriter(SCIOADIOSWriter *w);

private:
    std::vector<int> selectedStages;
    int64_t adios_handle;
    std::string group_name;
    std::string filename;

    int local_count;
    bool hasData;
    bool newfile;

	MPI_Comm *local_comm;
	int local_rank;

	bool selected(const int stage);

	SCIOADIOSWriter() : local_count(0), hasData(false) {};
	virtual ~SCIOADIOSWriter() {};

public:

	virtual int open();
	virtual int close();

	virtual void saveIntermediate(const ::cv::Mat& intermediate, const int stage, const char *image_name, const int offsetX, const int offsetY);

	virtual void saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage);

};



}


#endif /* UTILS_ADIOS_H_ */
