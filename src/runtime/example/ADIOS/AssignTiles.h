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
			std::string &dirName, int count, cciutils::SCIOLogSession *_logsession = NULL);
	virtual ~AssignTiles();
	virtual int run();
	virtual const char* getClassName() { return "AssignTiles"; };


protected:

	virtual int compute(int const &input_size , void * const &input,
				int &output_size, void * &output);

	std::vector<std::string> filenames;
};


}
} /* namespace rt */
} /* namespace cci */
#endif /* ASSIGNTILES_H_ */
