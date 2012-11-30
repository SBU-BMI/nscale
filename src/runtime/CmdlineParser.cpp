/*
 * CmdlineParser.cpp
 *
 *  Created on: Nov 29, 2012
 *      Author: tcpan
 */
#include "CmdlineParser.h"
#include "Debug.h"
#include <iostream>
#include <iterator>
#include <vector>
#include <string>

namespace cci {
namespace rt {

const std::string CmdlineParser::PARAM_INPUTCOUNT = "input_count";
const std::string CmdlineParser::PARAM_OUTPUTDIR = "output_directory";
const std::string CmdlineParser::PARAM_MAXIMGSIZE = "max_image_size";
const std::string CmdlineParser::PARAM_IOTRANSPORT = "transport";
const std::string CmdlineParser::PARAM_IOSIZE = "io_size";
const std::string CmdlineParser::PARAM_IOINTERLEAVE = "io_interleave";
const std::string CmdlineParser::PARAM_IOGROUPSIZE = "io_group_size";
const std::string CmdlineParser::PARAM_IOGROUPINTERLEAVE = "io_group_interleave";

CmdlineParser::CmdlineParser() : all_options("Options (defaults in parens)"), vm() {
	boost::program_options::options_description common("Generic Options");
	common.add_options()
			("help,h", "this help message")
			;
	all_options.add(common);

	boost::program_options::options_description output_desc("Output Options");
	output_desc.add_options()
			("input_count,n", boost::program_options::value<int>()->default_value(-1), "input count.  -1 == all")
			("output_directory,o", boost::program_options::value< std::string >()->required(), "output directory. REQUIRED")
			("max_image_size,m", boost::program_options::value<int>()->default_value(4096), "output image size, pixels on one size of image.")
			("transport,t", boost::program_options::value< std::string >()->default_value(std::string("na-NULL")), "transport mechanism.  ADIOS transports or na-NULL, na-POSIX")
			("io_size,P", boost::program_options::value<int>()->required(), "output node count.  REQUIRED")
			("io_interleave,V", boost::program_options::value<int>()->default_value(1), "output node to compute interleave.")
			("io_group_size,p", boost::program_options::value<int>()->default_value(1), "output group size.")
			("io_group_interleave,v", boost::program_options::value<int>()->default_value(1), "output group interleave.")
			;
	all_options.add(output_desc);

}

bool CmdlineParser::parse(int argc, char** argv) {
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	try {
		boost::program_options::parsed_options parsed =
				boost::program_options::command_line_parser(argc, argv).options(all_options).allow_unregistered().run();
		boost::program_options::store(parsed, vm);
		boost::program_options::notify(vm);

		// now get the unrecognized ones and print them out
		std::vector<std::string> to_pass_further = boost::program_options::collect_unrecognized(parsed.options, boost::program_options::include_positional);
		if (rank == 0) {
			std::cout << "Unrecognized Options:\n\t";
			std::ostream_iterator<std::string> out_it (std::cout,"\n\t");
			copy ( to_pass_further.begin(), to_pass_further.end(), out_it );
			std::cout << std::endl;
		}

	} catch (const boost::program_options::error& e) {
		if (vm.count("help")) {
			if (rank == 0) {
				std::cout << all_options << std::endl;
			}
			return false;
		}

		if (rank == 0) {
			std::cout << "ERROR parsing command line: " << e.what() << std::endl;
		    std::cout << all_options << std::endl;
		}
		return false;
	}

	if (vm.count("help")) {
		if (rank == 0) {
			std::cout << all_options << std::endl;
		}
		return false;
	}

	return true;
}



template <typename T>
T CmdlineParser::getParamValueByName(boost::program_options::variables_map &_vm, const std::string &name) {
	try {
		//Debug::print("STATUS: getting parameter value with name %s\n", name.c_str());
		return _vm[name].as<T>();
	} catch (const boost::program_options::error& e) {
		Debug::print("ERROR: can't get parameter value with name %s\n", name.c_str());
		return T();
	} catch (const boost::bad_any_cast& e) {
		Debug::print("ERROR: can't cast parameter value with name %s\n", name.c_str());
		return T();
	}
}

template std::string CmdlineParser::getParamValueByName< std::string >(boost::program_options::variables_map &_vm, const std::string &name);
template int CmdlineParser::getParamValueByName<int>(boost::program_options::variables_map &_vm, const std::string &name);
template bool CmdlineParser::getParamValueByName<bool>(boost::program_options::variables_map &_vm, const std::string &name);


}
}

