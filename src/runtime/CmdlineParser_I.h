/*
 * CmdlineParser_I.h
 *
 *  Created on: Jul 12, 2012
 *      Author: tcpan
 */

#ifndef CMDLINEPARSER_I_H_
#define CMDLINEPARSER_I_H_

#include <tr1/array>
#include <string>

namespace cci {
namespace rt {

template <size_t N>
class CmdlineParser_I {
public:
	typedef std::tr1::array<std::string, N> ParamsType;


	CmdlineParser_I() {};
	virtual ~CmdlineParser_I() {};

	virtual bool parse(int argc, char** argv) = 0;
	virtual void printUsage(char *cmd) = 0;

	ParamsType &getParams() {
		return params;
	}
	std::string &getParam(int const & name) {
		return params[name];
	}


protected:
	ParamsType params;

};

} /* namespace rt */
} /* namespace cci */
#endif /* CMDLINEPARSER_I_H_ */
