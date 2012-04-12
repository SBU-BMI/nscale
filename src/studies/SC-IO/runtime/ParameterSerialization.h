/*
 * ParameterSerialization.h
 *
 *  Created on: Apr 11, 2012
 *      Author: tcpan
 */

#ifndef PARAMETERSERIALIZATION_H_
#define PARAMETERSERIALIZATION_H_

namespace cci {

namespace runtime {

/**
 * for serialization and deserialization
 *
 * example subclasses may be
 * datagram
 * xml
 * text file
 *
 */

class ParameterSerialization {
public:
	/**
	 * serialize parameters to some in memory space.
	 * return status code
	 */
	static int serialize(const Parameters &params, void* &output, int &output_size) = 0;
	static int deserialize(const void* input, const int input_size, Parameters &params) = 0;
};

}

}

#endif /* PARAMETERSERIALIZATION_H_ */
