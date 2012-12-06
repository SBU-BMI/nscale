/*
 * AssignTiles.h
 *
 *  Created on: Jun 19, 2012
 *      Author: tcpan
 */

#ifndef ASSIGNTILES_H_
#define ASSIGNTILES_H_

#include <Action_I.h>
#include <string>
#include <vector>

namespace cci {
namespace rt {
namespace adios {

class AssignTiles: public cci::rt::Action_I {
public:
	AssignTiles(MPI_Comm const * _parent_comm, int const _gid,
			DataBuffer *_input, DataBuffer *_output,
			boost::program_options::variables_map &_vm,
			cci::common::LogSession *_logsession = NULL);
	virtual ~AssignTiles();
	virtual int run();
	virtual const char* getClassName() { return "AssignTiles"; };

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
#endif /* ASSIGNTILES_H_ */
