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
#include "mpi.h"
#include <vector>
#include "UtilsCVImageIO.h"
#include "SCIOUtilsLogger.h"

namespace cciutils {

class SCIOADIOSWriter;

struct Tile {
public:
	std::string image_name;
	int x_offset;
	int y_offset;
	std::string source_tile_file_name;
	::cv::Mat tile;
};

class ADIOSManager {
private:
	MPI_Comm *comm;
	int rank;
	bool gapped;
	bool grouped;

	std::vector<SCIOADIOSWriter *> writers;
	cciutils::SCIOLogSession * logsession;

public:
	ADIOSManager(const char* configfilename,  int _rank, MPI_Comm *_comm, cciutils::SCIOLogSession * session, bool _gapped = false, bool _groupped = true);
	virtual ~ADIOSManager();

	virtual SCIOADIOSWriter *allocateWriter(const std::string &pref, const std::string &suf,
				const bool _appendInTime, const bool _newfile, std::vector<int> &selStages,
				long mx_tileinfo_count, long mx_imagename_bytes, long mx_sourcetilefile_bytes, long mx_tile_bytes,
				int _chunkNumTiles, long _tileSize,
				int _local_group, MPI_Comm *_local_comm);
	virtual void freeWriter(SCIOADIOSWriter *w);
};


#define ENC_RAW 1

class SCIOADIOSWriter : public cv::IntermediateResultHandler {

	friend SCIOADIOSWriter* ADIOSManager::allocateWriter(const std::string &pref, const std::string &suf,
					const bool _appendInTime, const bool _newfile, std::vector<int> &selStages,
					long mx_tileinfo_count, long mx_imagename_bytes, long mx_sourcetilefile_bytes, long mx_tile_bytes,
					int _chunkNumTiles, long _tileSize,
					int _local_group, MPI_Comm *_local_comm);

	friend void ADIOSManager::freeWriter(SCIOADIOSWriter *w);

private:
    std::vector<int> selectedStages;
    int64_t adios_handle;
    std::string filename;

    // tracking how much has a process been writing out.
    long pg_tileInfo_count;
    long pg_imageName_bytes;
    long pg_sourceTileFile_bytes;
    long pg_tile_bytes;

    // tracking how much has been written out TOTAL at the end of each step from all processes 
    long tileInfo_total;
    long imageName_total; 
    long sourceTileFile_total;
    long tile_total;

    // set the capacity of the adios arrays
    long tileInfo_capacity;
    long imageName_capacity;
    long sourceTileFile_capacity;
    long tile_capacity;

    std::string prefix;
    std::string suffix;
    bool newfile;
    bool appendInTime;
    std::vector<Tile> tile_cache;
    uint32_t write_session_id;

	MPI_Comm *local_comm;
	int local_rank;
	int local_size;
	int local_group;
	cciutils::SCIOLogSession *logsession;

	bool grouped;
	bool gapped;
	int chunkNumTiles;
	long tileSize;



protected:
	bool selected(const int stage);

	SCIOADIOSWriter() : gapped(false), tileInfo_total(0), tile_total(0), imageName_total(0), sourceTileFile_total(0),
		pg_tile_bytes(0), pg_tileInfo_count(0), pg_imageName_bytes(0), pg_sourceTileFile_bytes(0),
		tileInfo_capacity(0), tile_capacity(0), imageName_capacity(0), sourceTileFile_capacity(0),
		logsession(NULL), tileSize(0), chunkNumTiles(0), local_group(0) {};
	virtual ~SCIOADIOSWriter();

	virtual int open(const char* groupName);
	virtual int close(uint32_t time_index = 0);

	virtual int persistGapped(int iter);

public:

	virtual int persist(int iter);
	virtual int persistCountInfo();
	virtual int currentLoad() {
		return tile_cache.size();
	}
	virtual int benchmark(int id);

	virtual void saveIntermediate(const ::cv::Mat& intermediate, const int stage,
			const char *_image_name, const int _offsetX, const int _offsetY, const char* _source_tile_file_name);

	virtual void saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage,
			const char *_image_name, const int _offsetX, const int _offsetY, const char* _source_tile_file_name);

	virtual void setLogSession(cciutils::SCIOLogSession *_logsession) {
		this->logsession = _logsession;
	}
};



}


#endif /* UTILS_ADIOS_H_ */
