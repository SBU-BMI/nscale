/*
 * CVImage.cpp
 *
 *  Created on: Jul 6, 2012
 *      Author: tcpan
 */

#include "CVImage.h"
#include "zlib.h"

namespace cci {
namespace rt {
namespace adios {

const int CVImage::READWRITE = 2;
const int CVImage::READ = 1;
const int CVImage::MANAGE = 0;

const int CVImage::ENCODE_RAW = 0;
const int CVImage::ENCODE_Z = 1;


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
CVImage::CVImage(int const size, void const *d, bool decode) {
	if (size < CVIMAGE_METADATA_SIZE + 2) return;  // minimum size is metadata, zero data, 2 null terminated strings

	if (decode) {
		// if decoding, then first create a copy that is a memory map, then copy it.


		this->type = MANAGE;

		CVImage temp(size, d, false);
		this->copy(temp, true);

	} else {
		// if not decoding, then just map the memory.
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

		unsigned char const * cdata = (unsigned char const*)d;
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
}

CVImage::CVImage(MetadataType *_metadata,
			unsigned char *_d, int _data_max_size,
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
	this->data = _d;

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


bool CVImage::copy(CVImage const &other, bool decode) {

	// not allowing modifying of object members.
	if (type == READ) {
		printf("cannot copy.  read only target CVIMAGE\n");
		return false;
	}

	int dataSize;
	if (decode) dataSize = other.metadata.info.x_size * other.metadata.info.y_size * other.metadata.info.nChannels * other.metadata.info.elemSize1;
	else dataSize = other.data_max_size;

	// for readwrite, can't go beyond the size allocated, and can't delete the memory
	if (type == READWRITE) {
		if (other.data != NULL) {
			if (data == NULL) // can't allocate, so fail
				return false;
			else if (dataSize > data_max_size)
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
			if (data != NULL && dataSize > data_max_size) {
				delete [] data;
				data = NULL;
			}
			if (data == NULL) {
				data_max_size = dataSize;
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

	if (other.data != NULL && data != NULL && other.metadata.info.data_size > 0) {
		if (decode && other.metadata.info.encoding != ENCODE_RAW) {
			unsigned long destsize = data_max_size;
			int status = uncompress(data, &destsize, other.data, (unsigned long)other.metadata.info.data_size);
			data_max_size = destsize;
			if (status == Z_MEM_ERROR) {
				printf("CVImage Copy uncompress : memory error\n");
				return false;
			} else if (status == Z_BUF_ERROR) {
				printf("CVImage Copy uncompress : dest buffer too small\n");
				return false;
			}
			if (status == Z_DATA_ERROR) {
				printf("CVImage Copy uncompress : data error\n");
				return false;
			}
		} else {
			memcpy(data, other.data, dataSize);
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

void CVImage::serialize(int &size, void* &d, int encoding) {

// first check to see if the data needs to be compressed.  If yes, check to see if the data needs to be compacted first.
	unsigned long s2 = this->metadata.info.data_size;
	unsigned long row_size = this->metadata.info.x_size * this->metadata.info.nChannels * this->metadata.info.elemSize1;

	int temp = 0;

	if (encoding == ENCODE_Z) {
		// if compressing, then compress and put into the output directly.
		MetadataType meta;
		memcpy(meta.bytes, this->metadata.bytes, CVIMAGE_METADATA_SIZE);
		meta.info.encoding = ENCODE_Z;

		// first we figure out the likely compressed size:
		unsigned long destsize = compressBound(s2);
		// estimated size
		size = CVIMAGE_METADATA_SIZE +
				destsize +
				this->metadata.info.image_name_size +
				this->metadata.info.source_file_name_size;
		// allocate the right size
		d = malloc(size);

		// see if we need compaction
		void * d2;
		if (this->metadata.info.step > row_size ) {
			// if need to compact, then compact first before compress
			d2 = malloc(s2);
			for (int i = 0; i < this->metadata.info.y_size; ++i) {
				memcpy((unsigned char*)d2 + i * row_size, this->data + i * this->metadata.info.step, row_size);
			}
		} else {
			// else just use the original pointer.
			d2 = this->data;
		}

		// compress into the new buffer
		compress(((unsigned char *)d) + CVIMAGE_METADATA_SIZE, &destsize, (unsigned char*)d2, s2);

		// clean up.
		if (this->metadata.info.step > row_size) free(d2);


		// realloc to get the exact right size
		size = CVIMAGE_METADATA_SIZE +
				destsize +
				this->metadata.info.image_name_size +
				this->metadata.info.source_file_name_size;
		d2 = realloc(d, size);
		if (d2 == NULL) {
			printf("ERROR:  should not have NULL from realloc!\n");
		} else {
			d = d2;
		}
		meta.info.data_size = destsize;

		// copy in the metadata
		memcpy(((unsigned char *)d), meta.bytes, CVIMAGE_METADATA_SIZE);

		temp = CVIMAGE_METADATA_SIZE + destsize;

	} else {
		size = CVIMAGE_METADATA_SIZE +
				this->metadata.info.data_size +
				this->metadata.info.image_name_size +
				this->metadata.info.source_file_name_size;
		d = malloc(size);

		// if not compressing, then put into the output directly.
		memcpy(((unsigned char *)d), this->metadata.bytes, CVIMAGE_METADATA_SIZE);
		temp = CVIMAGE_METADATA_SIZE;


		// copy the data into the buffer.
		if (this->metadata.info.step > row_size) {  // there are gaps
			for (int i = 0; i < this->metadata.info.y_size; ++i) {
				memcpy(((unsigned char *)d) + temp, this->data + i * this->metadata.info.step, row_size);
				temp += row_size;
			}
		} else {
			memcpy(((unsigned char *)d) + temp, this->data, this->metadata.info.data_size);
			temp += this->metadata.info.data_size;
		}

	}

	// copy the file and image names
	memcpy(((unsigned char *)d) + temp, this->image_name, this->metadata.info.image_name_size);
	temp += this->metadata.info.image_name_size;
	memcpy(((unsigned char *)d) + temp, this->source_file_name, this->metadata.info.source_file_name_size);

	// DONE.
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
