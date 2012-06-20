/*
 * Segment.h
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#ifndef SEGMENT_H_
#define SEGMENT_H_

#include <Worker_I.h>

namespace cci {
namespace rt {

class Segment: public cci::rt::Worker_I {
public:
	Segment(MPI_Comm const * _parent_comm, int const _gid);
	virtual ~Segment();

	virtual int compute(int const &input_size , char* const &input,
				int &output_size, char* &output);
};

} /* namespace rt */
} /* namespace cci */
#endif /* SEGMENT_H_ */
