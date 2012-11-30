/*
 * CmdlineParser.h
 *
 *  Created on: Jul 12, 2012
 *      Author: tcpan
 */

#ifndef CMDLINEPARSER_H_
#define CMDLINEPARSER_H_

#include "boost/program_options.hpp"
#include <cstdio>
#include <iostream>
#include <string>
#include "mpi.h"

namespace cci {
namespace rt {

class CmdlineParser {
public:

	CmdlineParser();
	virtual ~CmdlineParser() {};

	bool parse(int argc, char** argv);

	void addParams(const boost::program_options::options_description &_desc) {
		all_options.add(_desc);
	};

	boost::program_options::options_description &getParams() { return all_options; };
	boost::program_options::variables_map &getParamValues() { return vm; };

	static const std::string PARAM_INPUTCOUNT;
	static const std::string PARAM_OUTPUTDIR;
	static const std::string PARAM_MAXIMGSIZE;
	static const std::string PARAM_IOTRANSPORT;
	static const std::string PARAM_IOSIZE;
	static const std::string PARAM_IOINTERLEAVE;
	static const std::string PARAM_IOGROUPSIZE;
	static const std::string PARAM_IOGROUPINTERLEAVE;

	template <typename T>
	static T getParamValueByName(boost::program_options::variables_map &_vm, const std::string &name);

protected:
	boost::program_options::variables_map vm;
	boost::program_options::options_description all_options;
};

} /* namespace rt */
} /* namespace cci */
#endif /* CMDLINEPARSER_H_ */
