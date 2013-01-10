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
	Segment(MPI_Comm const * _parent_comm, int const _gid,
			DataBuffer *_input, DataBuffer *_output,
			boost::program_options::variables_map &_vm,
			cci::common::LogSession *_logsession = NULL);
	virtual ~Segment();
	virtual int run();
	virtual const char* getClassName() { return "Segment"; };

	static boost::program_options::options_description params;


protected:
	virtual int compute(int const &input_size , void * const &input,
				int &output_size, void * &output);

	int proc_code;
	int output_count;
	bool compressing;

private:
	static bool param_init;
	static bool initParams();

};

}
} /* namespace rt */
} /* namespace cci */
#endif /* SEGMENT_H_ */
