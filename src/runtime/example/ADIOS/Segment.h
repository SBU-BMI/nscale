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
namespace adios {

class Segment: public cci::rt::Action_I {
public:
	Segment(MPI_Comm const * _parent_comm, int const _gid);
	virtual ~Segment();
	virtual int run();
	virtual const char* getClassName() { return "Segment"; };

protected:
	virtual int compute(int const &input_size , void * const &input,
				int &output_size, void * &output);


};

}
} /* namespace rt */
} /* namespace cci */
#endif /* SEGMENT_H_ */
