/*
 * ParameterDatagram.h
 *
 *  Created on: Apr 11, 2012
 *      Author: tcpan
 */

#ifndef PARAMETERDATAGRAM_H_
#define PARAMETERDATAGRAM_H_

#include "ParameterSerialization.h"

namespace cci {

namespace runtime {

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
class ParameterDatagram: public cci::runtime::ParameterSerialization {
public:
	/**
	 * serialize parameters to some in memory space.
	 * return status code
	 */
	static int serialize(const Parameters &params, void* &output, int &output_size);
	static int deserialize(const void* input, const int input_size, Parameters &params);

};

}

}

#endif /* PARAMETERDATAGRAM_H_ */
