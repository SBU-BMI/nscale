/*
 * CVImage.h
 *
 *  Created on: Jul 6, 2012
 *      Author: tcpan
 */

#ifndef CVIMAGE_H_
#define CVIMAGE_H_

#include "opencv2/opencv.hpp"
#include <string>
#include "UtilsADIOS.h"

namespace cci {
namespace rt {
namespace adios {

#define CVIMAGE_METADATA_SIZE 48

class CVImage {
public:


	typedef union {
		struct {
			int32_t x_offset;
			int32_t y_offset;
			uint32_t x_size;
			uint32_t y_size;
			uint32_t nChannels;
			uint32_t elemSize1;
			int32_t cvDataType;

			int32_t encoding;
			uint32_t data_size;
			uint32_t image_name_size;   // does not include null termination character
			uint32_t source_file_name_size;

			uint32_t step;  // for managing non-continuous data.
		} info;
		unsigned char bytes[CVIMAGE_METADATA_SIZE];
	} MetadataType;



	virtual ~CVImage();

	CVImage(cv::Mat const &img, std::string const &_image_name, std::string const &_source_tile_file_name, int const _offsetX, int const _offsetY);
	CVImage(int _data_max = 0, int _name_max = 0, int _source_name_max = 0);  // empty one
	CVImage(int const size, void const *d, bool decode=false);  // data is referenced after the call.  point to a block of memory
	CVImage(MetadataType *metadata,
			unsigned char *d, int data_max_size,
			char * image_name, int image_name_max_size,
			char * source_file_name, int source_file_name_max_size);
	// point to memory already allocated externally, and possibly populated.

	bool copy(CVImage const &other, bool decode=false);  // if error, return false;

	void serialize(int &size, void* &d, int encoding = ENCODE_RAW);
//	bool deserialize(int const size, void const *data);  // data is not referenced after the call

	static const int READWRITE; // mapped to buffer memory.  can't new or delete pointers.  content is free to be changed
	static const int READ;	    // mapped to object members. can't new or delete pointers.  content is not free to be changed
	static const int MANAGE;	// allocated denovo.  can new or delete points, can change contents.

	static const int ENCODE_RAW;	// raw encoding type
	static const int ENCODE_Z;

	// works because data, imagename, and sourcefilenames are separate arrays.
	void compact() {
		data_max_size = metadata.info.data_size;
		image_name_max_size = metadata.info.image_name_size;
		source_file_name_max_size = metadata.info.source_file_name_size;
	}

	MetadataType const &getMetadata() { return this->metadata; };
	unsigned char const *getData(int &max_size, int &size) { max_size = data_max_size; size = metadata.info.data_size; return this->data; };
	char const *getImageName(int &max_size, int &size) { max_size = image_name_max_size; size = metadata.info.image_name_size; return this->image_name; };
	char const *getSourceFileName(int &max_size, int &size) { max_size = source_file_name_max_size; size = metadata.info.source_file_name_size; return this->source_file_name; };
	int const getType() { return this->type; };

	static MetadataType *allocMetadata() {
		MetadataType *meta = new ::cci::rt::adios::CVImage::MetadataType[1];
		memset(meta, 0, CVIMAGE_METADATA_SIZE);
		return meta;
	}
	static void freeMetadata(MetadataType* &meta) {
		delete [] meta;
	}

	static void freeSerializedData(void *d) {
		free(d);
	}

protected:
	MetadataType metadata;
	unsigned char * data;  // not necessarily contiguous
	char * image_name;
	char * source_file_name;

	// some other metadata
	int type; // for pointing data, image_name, and source_file_name to some preallocated space.
	int stage;

	int data_max_size;
	int image_name_max_size;
	int source_file_name_max_size;

};

} /* namespace adios */
} /* namespace rt */
} /* namespace cci */
#endif /* CVIMAGE_H_ */
