/*
 * GenerateOutputPush.h
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#ifndef GENERATE_OUTPUT_PUSH_H_
#define GENERATE_OUTPUT_PUSH_H_

#include <Action_I.h>


namespace cci {
namespace rt {
namespace syntest {

class GenerateOutputPush: public cci::rt::Action_I {
public:
	GenerateOutputPush(MPI_Comm const * _parent_comm, int const _gid,
			DataBuffer *_input, DataBuffer *_output,
			boost::program_options::variables_map &_vm,
			cci::common::LogSession *_logsession = NULL);
	virtual ~GenerateOutputPush();
	virtual int run();
	virtual const char* getClassName() { return "GenerateOutputPush"; };

	static boost::program_options::options_description params;

protected:
	virtual int compute(int const &input_size , void * const &input,
				int &output_size, void * &output);

	int output_dim;
	int output_count;
	int count;
	bool compress;
	int world_rank;

	double p_bg, p_nu, p_full;
	double mean_bg, stdev_bg;
	double mean_nu, stdev_nu;
	double mean_full, stdev_full;

private:
	static bool param_init;
	static bool initParams();
};

}
} /* namespace rt */
} /* namespace cci */
#endif /* GENERATE_OUTPUT_PUSH_H_ */
