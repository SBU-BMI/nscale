/*
 * Segment.h
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#ifndef SEGMENT_H_
#define SEGMENT_H_

#include <Action_I.h>

namespace cci {
namespace rt {

class Segment: public cci::rt::Action_I {
public:
	Segment(MPI_Comm const * _parent_comm, int const _gid,
			DataBuffer *_input, DataBuffer *_output,
			cci::common::LogSession *_logsession = NULL);
	virtual ~Segment();
	virtual int run();
	virtual const char* getClassName() { return "Segment"; };

protected:
	virtual int compute(int const &input_size , void * const &input,
				int &output_size, void * &output);

	int output_count;

};

} /* namespace rt */
} /* namespace cci */
#endif /* SEGMENT_H_ */
