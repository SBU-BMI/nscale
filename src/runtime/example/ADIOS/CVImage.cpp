/*
 * CVImage.cpp
 *
 *  Created on: Jul 6, 2012
 *      Author: tcpan
 */

#include "CVImage.h"

namespace cci {
namespace rt {
namespace adios {

const int CVImage::READWRITE = 2;
const int CVImage::READ = 1;
const int CVImage::MANAGE = 0;

const int CVImage::ENCODE_RAW = 0;


CVImage::CVImage(cv::Mat const &img, std::string const &_image_name,
		std::string const &_source_file_name,
		int const _offsetX, int const _offsetY) {
	this->type = READ;

	this->metadata.info.x_offset = _offsetX;
	this->metadata.info.y_offset = _offsetY;

	this->metadata.info.x_size = img.cols;
	this->metadata.info.y_size = img.rows;
	this->metadata.info.nChannels = img.channels();
	this->metadata.info.elemSize1 = img.elemSize1();
	this->metadata.info.cvDataType = img.type();

	this->metadata.info.encoding = ENCODE_RAW; // RAW
	this->metadata.info.data_size = img.rows * img.cols * img.elemSize();
	this->metadata.info.image_name_size = _image_name.length() + 1;
	this->metadata.info.source_file_name_size = _source_file_name.length() + 1;

	this->metadata.info.step = img.step;  // in bytes
	this->data = img.data;
	this->image_name = const_cast<char *>(_image_name.c_str());
	this->source_file_name = const_cast<char*>(_source_file_name.c_str());

	this->data_max_size = this->metadata.info.data_size;
	this->image_name_max_size = this->metadata.info.image_name_size;
	this->source_file_name_max_size = this->metadata.info.source_file_name_size;
}

CVImage::CVImage(int _data_max, int _name_max, int _source_name_max) :
		data(NULL), image_name(NULL), source_file_name(NULL) {
	this->type = MANAGE;

	this->metadata.info.x_offset = 0;
	this->metadata.info.y_offset = 0;

	this->metadata.info.x_size = 0;
	this->metadata.info.y_size = 0;
	this->metadata.info.nChannels = 0;
	this->metadata.info.elemSize1 = 0;
	this->metadata.info.cvDataType = 0;

	this->metadata.info.encoding = ENCODE_RAW;
	this->metadata.info.data_size = 0;
	this->metadata.info.image_name_size = 0;
	this->metadata.info.source_file_name_size = 0;

	this->metadata.info.step = 0;

	this->data_max_size = _data_max;
	this->image_name_max_size = _name_max;
	this->source_file_name_max_size = _source_name_max;
}

// wrapping a contiguous block of memory.
CVImage::CVImage(int const size, void const *data) {
	if (size < CVIMAGE_METADATA_SIZE + 2) return;

	// map the memory.
	this->type = READ;

	this->metadata.info.x_offset = 0;
	this->metadata.info.y_offset = 0;

	this->metadata.info.x_size = 0;
	this->metadata.info.y_size = 0;
	this->metadata.info.nChannels = 0;
	this->metadata.info.elemSize1 = 0;
	this->metadata.info.cvDataType = 0;

	this->metadata.info.encoding = ENCODE_RAW;
	this->metadata.info.data_size = 0;
	this->metadata.info.image_name_size = 0;
	this->metadata.info.source_file_name_size = 0;

	unsigned char const * cdata = (unsigned char const*)data;
	unsigned char * tdata = const_cast<unsigned char*>(cdata);
	int temp = 0;
	memcpy(this->metadata.bytes, tdata + temp, CVIMAGE_METADATA_SIZE); temp += CVIMAGE_METADATA_SIZE;

	if (this->metadata.info.data_size < 0) this->metadata.info.data_size = 0;
	if (this->metadata.info.image_name_size < 1) this->metadata.info.image_name_size = 1;
	if (this->metadata.info.source_file_name_size < 1) this->metadata.info.source_file_name_size = 1;

	this->data = tdata+temp; temp += this->metadata.info.data_size;
	this->data_max_size = this->metadata.info.data_size;

	char *sdata = (char *)tdata;
	this->image_name = sdata+temp; temp += this->metadata.info.image_name_size;
	this->image_name_max_size = this->metadata.info.image_name_size;

	this->source_file_name = sdata + temp; temp += this->metadata.info.source_file_name_size;
	this->source_file_name_max_size = this->metadata.info.source_file_name_size;

	this->metadata.info.step = this->metadata.info.x_size * this->metadata.info.nChannels * this->metadata.info.elemSize1;
	this->stage = -1;
}

CVImage::CVImage(MetadataType *_metadata,
			unsigned char *_data, int _data_max_size,
			char * _image_name, int _image_name_max_size,
			char * _source_file_name, int _source_file_name_max_size) {
	this->type = READWRITE;

	this->metadata.info.x_offset = 0;
	this->metadata.info.y_offset = 0;

	this->metadata.info.x_size = 0;
	this->metadata.info.y_size = 0;
	this->metadata.info.nChannels = 0;
	this->metadata.info.elemSize1 = 0;
	this->metadata.info.cvDataType = 0;

	this->metadata.info.encoding = ENCODE_RAW;
	this->metadata.info.data_size = 0;
	this->metadata.info.image_name_size = 0;
	this->metadata.info.source_file_name_size = 0;

	memcpy(this->metadata.bytes, _metadata, CVIMAGE_METADATA_SIZE);

	this->data_max_size = (_data_max_size < 0 ? 0 : _data_max_size);
	this->data = _data;

	this->image_name_max_size = (_image_name_max_size < 0 ? 0 : _image_name_max_size);
	this->image_name = _image_name;

	this->source_file_name_max_size = (_source_file_name_max_size< 0 ? 0 : _source_file_name_max_size);
	this->source_file_name = _source_file_name;

	this->metadata.info.step = this->metadata.info.x_size * this->metadata.info.nChannels * this->metadata.info.elemSize1;

	this->stage = -1;
}


CVImage::~CVImage() {
	if (type == MANAGE) {
		if (data != NULL) {
			delete [] data;
			data = NULL;
		}
		if (image_name != NULL) {
			delete [] image_name;
			image_name = NULL;
		}
		if (source_file_name != NULL) {
			delete [] source_file_name;
			source_file_name = NULL;
		}
	}
}


bool CVImage::copy(CVImage const &other) {

	// not allowing modifying of object members.
	if (type == READ) {
		printf("cannot copy.  read only target CVIMAGE\n");
		return false;
	}

	// for readwrite, can't go beyond the size allocated, and can't delete the memory
	if (type == READWRITE) {
		if (other.data != NULL) {
			if (data == NULL) // can't allocate, so fail
				return false;
			else if (other.metadata.info.data_size > data_max_size)
				return false;  // can't grow.  so fail
		}
		if (data != NULL) {// other is null or not null.  either way, clear local.
//			printf("CVIMAGE INFO: data at %p, data_max_size = %d\n", data, data_max_size);
			memset(data, 0, data_max_size);
		}
		if (other.image_name != NULL) {
			if (image_name == NULL) // can't allocate, so fail
				return false;
			else if (other.metadata.info.image_name_size > image_name_max_size)
				return false;  // can't grow.  so fail
		}
		if (image_name != NULL) // other is null or not null.  either way, clear local.
			memset(image_name, 0, image_name_max_size);

		if (other.source_file_name != NULL) {
			if (source_file_name == NULL) // can't allocate, so fail
				return false;
			else if (other.metadata.info.source_file_name_size > source_file_name_max_size)
				return false;  // can't grow.  so fail
		}
		if (source_file_name != NULL) // other is null or not null.  either way, clear local.
			memset(source_file_name, 0, source_file_name_max_size);

		// keep the current *_max_size;
	} else {
		// managed.  if memory is big enough, reuse.  else allocate new.
		if (other.data == NULL) {
			if (data != NULL) {
				delete [] data;
				data = NULL;
			}
			data_max_size = 0;
		} else {
			if (data != NULL && data_max_size < other.data_max_size) {
				delete [] data;
				data = NULL;
			}
			if (data == NULL) {
				data_max_size = other.data_max_size;
				data = new unsigned char[data_max_size];
			}
			memset(data, 0, data_max_size);
		}
		if (other.image_name == NULL) {
			if (image_name != NULL) {
				delete [] image_name;
				image_name = NULL;
			}
			image_name_max_size = 0;
		} else {
			if (image_name != NULL && image_name_max_size < other.image_name_max_size) {
				delete [] image_name;
				image_name = NULL;
			}
			if (image_name == NULL) {
				image_name_max_size = other.image_name_max_size;
				image_name = new char[image_name_max_size];
			}
			memset(image_name, 0, image_name_max_size);
		}
		if (other.source_file_name == NULL) {
			if (source_file_name != NULL) {
				delete [] source_file_name;
				source_file_name = NULL;
			}
			source_file_name_max_size = 0;
		} else {
			if (source_file_name != NULL && source_file_name_max_size < other.source_file_name_max_size) {
				delete [] source_file_name;
				source_file_name = NULL;
			}
			if (source_file_name == NULL) {
				source_file_name_max_size = other.source_file_name_max_size;
				source_file_name = new char[source_file_name_max_size];
			}
			memset(source_file_name, 0, source_file_name_max_size);
		}
	}


	// keep my type
	memcpy(metadata.bytes, other.metadata.bytes, CVIMAGE_METADATA_SIZE);
	metadata.info.step = other.metadata.info.step;
	stage = other.stage;

	if (other.data != NULL && data != NULL) {
		int row_size = metadata.info.x_size * metadata.info.nChannels * metadata.info.elemSize1;
		if (other.metadata.info.step > row_size) {  // there are gaps
			// compacting
			metadata.info.step = row_size;
			int temp = 0, temp2 = 0;
			for (int i = 0; i < metadata.info.y_size; ++i) {
				memcpy(data + temp, other.data + temp2, metadata.info.step);
				temp += metadata.info.step;
				temp2 += other.metadata.info.step;
			}
		} else {
			memcpy(data, other.data, metadata.info.data_size);
		}
	} // else this data is not null, so okay to copy
	  // or other data has null, so don't copy

	if (other.image_name != NULL && image_name != NULL) {
		memcpy(image_name, other.image_name, metadata.info.image_name_size);
	}
	if (other.source_file_name != NULL && source_file_name != NULL) {
		memcpy(source_file_name, other.source_file_name, metadata.info.source_file_name_size);
	}

	return true;

}

void CVImage::serialize(int &size, void* &data) {

	size = CVIMAGE_METADATA_SIZE +
			this->metadata.info.data_size +
			this->metadata.info.image_name_size +
			this->metadata.info.source_file_name_size;

	data = malloc(size);
	unsigned char * tdata = (unsigned char*) data;

	int temp = 0, temp2 = 0;
	memcpy(tdata + temp, this->metadata.bytes, CVIMAGE_METADATA_SIZE);  temp += CVIMAGE_METADATA_SIZE;

	int row_size = this->metadata.info.x_size * this->metadata.info.nChannels * this->metadata.info.elemSize1;
	if (this->metadata.info.step > row_size) {  // there are gaps
		for (int i = 0; i < this->metadata.info.y_size; ++i) {
			memcpy(tdata + temp, this->data + temp2, row_size);
			temp += row_size;
			temp2 += this->metadata.info.step;
		}
	} else {
		memcpy(tdata + temp, this->data, this->metadata.info.data_size); temp += this->metadata.info.data_size;
	}

	memcpy(tdata + temp, this->image_name, this->metadata.info.image_name_size); temp += this->metadata.info.image_name_size;
	memcpy(tdata + temp, this->source_file_name, this->metadata.info.source_file_name_size); temp += this->metadata.info.source_file_name_size;
}




//bool CVImage::deserialize(int const size, void const *data) {
//	CVImage *other = new CVImage(size, data);  // no mem alloc and minimal copy.
//
//	bool result = this->copy(other);  // now real alloc and copy
//	delete other;
//
//	return result;
//}



} /* namespace adios */
} /* namespace rt */
} /* namespace cci */
