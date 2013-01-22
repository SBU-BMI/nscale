/*
 * ReadTiles.h
 *
 *  Created on: Jun 19, 2012
 *      Author: tcpan
 */

#ifndef ReadTiles_H_
#define ReadTiles_H_

#include <Action_I.h>
#include <string>
#include <vector>

namespace cci {
namespace rt {
namespace adios {

class ReadTiles: public cci::rt::Action_I {
public:
	ReadTiles(MPI_Comm const * _parent_comm, int const _gid,
			DataBuffer *_input, DataBuffer *_output,
			boost::program_options::variables_map &_vm,
			cci::common::LogSession *_logsession = NULL);
	virtual ~ReadTiles();
	virtual int run();
	virtual const char* getClassName() { return "ReadTiles"; };

	static boost::program_options::options_description params;
	static const std::string PARAM_READSIZE;

protected:

	virtual int compute(int const &input_size , void * const &input,
				int &output_size, void * &output);

	std::vector<std::string> filenames;
	bool compressing;

private:
	static bool param_init;
	static bool initParams();
};


}
} /* namespace rt */
} /* namespace cci */
#endif /* ReadTiles_H_ */
