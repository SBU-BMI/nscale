/*
 * ParameterDatagram.cpp
 *
 *  Created on: Apr 11, 2012
 *      Author: tcpan
 */

#include "ParameterDatagram.h"

namespace cci {

namespace runtime {

void buffer(void* &buf, const void *src, int &offset, int &capacity, const int size) {
	if (buf + offset + size > capacity) {
		capacity += size + capacity;
		void *temp = malloc(capacity);
		memset(temp, 0, capacity);
		memcpy(temp, buf, offset);
		free(buf);
		buf = temp;
	}
	memcpy(buf + offset, src, size);
	offset += size;
	return capacity;
}

/**
 * format is :
 * 		numEntries (int)
 * 		{	workType (int)
 * 			numEntries (int)
 * 			{	name_length (int)
 * 				name (str/unsigned char)
 * 				value_type (int)
 * 				value_length (int)
 * 				value (T)
 * 			}
 * 		}
 */
int ParameterDatagram::serialize(const Parameters &params, void* &output, int &output_size) {
	// initial allocation
	int capacity = 1024;
	void* temp = malloc(capacity);
	int size = 0;
	void* temp2;
	int addition;

	int temp3 = params.size();
	buffer(temp, &temp3, size, capacity, sizeof(int));

	for (std::tr1::unordered_map<int, std::tr1::unordered_map<std::string, ValueBase &> >::iterator iter = params.params.begin();
			iter != params.params.end(); ++iter) {

		temp3 = iter->first;
		buffer(temp, &temp3, size, capacity, sizeof(int));

		temp3 = iter->second.size();
		buffer(temp, &temp3, size, capacity, sizeof(int));

		for (std::tr1::unordered_map<std::string, ValueBase &>::iterator iter2 = iter->second.begin();
				iter2 != iter->second.end(); ++iter2) {

			temp3 = iter2->first.length();
			buffer(temp, &temp3, size, capacity, sizeof(int));

			buffer(temp, iter2->first.c_str(), size, capacity, temp3);

			// copy the type info into the datagram
			temp3 = iter2->second.getType();
			buffer(temp, &temp3, size, capacity, sizeof(int));

			temp3 = iter2->second.getSize();
			buffer(temp, &temp3, size, capacity, sizeof(int));

			buffer(temp, iter2->second.getBytes(), size, capacity, temp3);
		}
	}

	output_size = size;
	output = temp;
	return 0;
}

int ParameterDatagram::deserialize(const void* input, const int input_size, Parameters &params) {

	int count = 0;
	int offset = 0;

	// read how many
	memcpy(&count, input + offset, sizeof(int) );
	offset += sizeof(int);

	for (int i = 0; i < count; ++i) {
		int worktype;
		// read work type
		memcpy(&worktype, input + offset, sizeof(int) );
		offset += sizeof(int);

		params.insert(worktype, std::tr1::unordered_map<std::string, ValueBase &>());

		// read how many are in this worktype
		int count2;
		memcpy(&count2, input + offset, sizeof(int) );
		offset += sizeof(int);

		for (int j = 0; j < count2; ++j) {
			int name_len;
			std::string name;

			// read name length
			memcpy(&name_len, input + offset, sizeof(int) );
			offset += sizeof(int);

			// read name
			name.assign((const char*) input+offset, name_len);
			offset += name_len;

			// read the type
			int type;
			memcpy(&type, input + offset, sizeof(int) );
			offset += sizeof(int);

			int val_len;
			memcpy(&val_len, input + offset, sizeof(int) );
			offset += sizeof(int);

			// read the data
			ValueBase * val = ValueFactory::createValue(type, val_len, input+offset);
			offset += val_len;

			params.insert(worktype, name, *val);
		}
	}
	return 0;
}

}

}
