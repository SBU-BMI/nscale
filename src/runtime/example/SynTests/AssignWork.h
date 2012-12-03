/*
 * AssignWork.h
 *
 *  Created on: Jun 19, 2012
 *      Author: tcpan
 */

#ifndef AssignWork_H_
#define AssignWork_H_

#include <Action_I.h>
#include <string>
#include <vector>

namespace cci {
namespace rt {
namespace adios {

class AssignWork: public cci::rt::Action_I {
public:
	AssignWork(MPI_Comm const * _parent_comm, int const _gid,
			DataBuffer *_input, DataBuffer *_output,
			boost::program_options::variables_map &_vm,
			cciutils::SCIOLogSession *_logsession = NULL);
	virtual ~AssignWork();
	virtual int run();
	virtual const char* getClassName() { return "AssignWork"; };

	static boost::program_options::options_description params;

protected:

	virtual int compute(int const &input_size , void * const &input,
				int &output_size, void * &output);

	std::vector<std::string> filenames;

private:
	static bool param_init;
	static bool initParams();
};


}
} /* namespace rt */
} /* namespace cci */
#endif /* AssignWork_H_ */
