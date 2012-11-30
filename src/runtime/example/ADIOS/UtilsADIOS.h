/*
 * for outputting intermediate images.
 * since the intermediate results may only be needed for certain executables, instead of relying on include to pickup the #defines,
 * and risk that a library may not have been compiled with the right set, I am creating 2 versions of the class (so as to not to have branch).
 *
 *  Created on: Jul 15, 2011
 *      Author: tcpan
 */

#ifndef UTILS_ADIOS_H_
#define UTILS_ADIOS_H_

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "mpi.h"
#include <vector>
#include "UtilsCVImageIO.h"
#include "SCIOUtilsLogger.h"
#include "CVImage.h"
#include "boost/program_options.hpp"

namespace cci {
namespace rt {
namespace adios {

class ADIOSWriter;
class CVImage;


class ADIOSManager {
private:
	MPI_Comm comm;
	int rank;

	std::string transport;
	bool grouped;
	bool gapped;

	std::vector<ADIOSWriter *> writers;
	cciutils::SCIOLogSession * logsession;
	boost::program_options::variables_map vm;

public:
	ADIOSManager(const char* configfilename, boost::program_options::variables_map &_vm,
			int _rank, MPI_Comm &_comm, cciutils::SCIOLogSession * session = NULL);
	ADIOSManager(const char* configfilename, std::string _transport,
			int _rank, MPI_Comm &_comm, cciutils::SCIOLogSession * session,
			bool _grouped, bool _gapped) __attribute__ ((deprecated));
	virtual ~ADIOSManager();

	virtual ADIOSWriter *allocateWriter(std::string const &pref,
			std::string const &suf, bool _newfile,
			bool _appendInTime, std::vector<int> const &selStages,
			int max_image_count, int local_image_count,
			int mx_image_bytes, int mx_imagename_bytes, int mx_sourcetilefile_bytes,
			MPI_Comm const &_local_comm, int _local_group) __attribute__ ((deprecated));
	virtual ADIOSWriter *allocateWriter(boost::program_options::variables_map &_vm,
			int max_image_count,
			int mx_image_bytes, int mx_imagename_bytes, int mx_sourcetilefile_bytes,
			MPI_Comm const &_local_comm, int _local_group);
	virtual void freeWriter(ADIOSWriter *w);
};



class ADIOSWriter : public cciutils::cv::IntermediateResultHandler {
	friend ADIOSWriter *ADIOSManager::allocateWriter(std::string const &pref,
			std::string const &suf, bool _newfile,
			bool _appendInTime, std::vector<int> const &selStages,
			int max_image_count, int local_image_count,
			int mx_image_bytes, int mx_imagename_bytes, int mx_sourcetilefile_bytes,
			MPI_Comm const &_local_comm, int _local_group);
	friend ADIOSWriter *ADIOSManager::allocateWriter(boost::program_options::variables_map &_vm,
			int max_image_count,
			int mx_image_bytes, int mx_imagename_bytes, int mx_sourcetilefile_bytes,
			MPI_Comm const &_local_comm, int _local_group);


private:
    std::string prefix;
    std::string suffix;
    bool newfile;
    std::string transport;

    bool appendInTime;
    std::vector<int> selected_stages;

   	int tileInfo_buffer_capacity;
	bool gapped;

    int mx_image_bytes;
	int mx_imagename_bytes;
	int mx_filename_bytes;

	MPI_Comm comm;
	bool grouped;
	int comm_group;
	int comm_rank;
	int comm_size;


    int64_t adios_handle;

    // tracking how much has a process been writing out.
//    long pg_tileInfo_count;
//    long pg_imageName_bytes;
//    long pg_filename_bytes;
//    long pg_image_bytes;

    // tracking how much has been written out TOTAL at the end of each step from all processes 
    long tileInfo_total;
    long imageName_total; 
    long sourceTileFile_total;
    long tile_total;

    uint32_t write_session_id;

    // set the capacity of the adios arrays, total across all processes
    long tileInfo_capacity;
    long imageName_capacity;
    long sourceTileFile_capacity;
    long tile_capacity;

	long local_tile_capacity;
	int local_imagename_capacity;
	int local_sourceTileFile_capacity;


	std::vector<CVImage *> buffer;
	unsigned char *tile;
	char *imageName;
	char *sourceTileFile;
	long data_pos;
	int imageNames_pos;
	int filenames_pos;

	cciutils::SCIOLogSession *logsession;
	std::string event_name_prefix;

protected:

	ADIOSWriter(std::string const &_prefix, std::string const &_suffix, bool _newfile,
			bool _appendInTime, std::vector<int> const &_selected_stages,
			int _mx_image_capacity, int _mx_local_image_capacity, bool _gapped,
			int _mx_image_bytes, int _mx_imagename_bytes, int _mx_filename_bytes,
			MPI_Comm const &_comm, bool _grouped, int _comm_group);
	ADIOSWriter(boost::program_options::variables_map &_vm,
			int _mx_image_capacity,	int _mx_image_bytes, int _mx_imagename_bytes, int _mx_filename_bytes,
			MPI_Comm const &_comm, int _comm_group);
	bool selected(const int stage) __attribute__ ((deprecated));

	CVImage *saveCVImage(CVImage const *img);

	virtual int open(const char* groupName);
	virtual int close(uint32_t time_index = 0);

public:
	virtual ~ADIOSWriter();

	void clearBuffer();

	virtual int persist(int iter);  // return the session id at the end
	virtual int persistCountInfo();
	virtual int benchmark(int id);

	virtual void saveIntermediate(CVImage const *img, const int stage);

	// write out with raw
	virtual void saveIntermediate(const ::cv::Mat& intermediate, const int stage,
			const char *_image_name = NULL, const int _offsetX = 0, const int _offsetY = 0, const char* _source_tile_file_name = NULL);

	virtual void saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage,
			const char *_image_name = NULL, const int _offsetX = 0, const int _offsetY = 0, const char* _source_tile_file_name = NULL);

	virtual int currentLoad() {
		return buffer.size();
	}

	virtual void setLogSession(cciutils::SCIOLogSession *_logsession) {
		this->logsession = _logsession;
	}
};


}  // ns adios
}  // ns rt
}  // ns cci


#endif /* UTILS_ADIOS_H_ */
