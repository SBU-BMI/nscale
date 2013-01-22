/*
 * SegmentNoRead.h
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#ifndef SegmentNoRead_H_
#define SegmentNoRead_H_

#include <Action_I.h>

namespace cci {
namespace rt {
namespace adios {

class SegmentNoRead: public cci::rt::Action_I {
public:
	SegmentNoRead(MPI_Comm const * _parent_comm, int const _gid,
			DataBuffer *_input, DataBuffer *_output,
			boost::program_options::variables_map &_vm,
			cci::common::LogSession *_logsession = NULL);
	virtual ~SegmentNoRead();
	virtual int run();
	virtual const char* getClassName() { return "SegmentNoRead"; };

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
#endif /* SegmentNoRead_H_ */
