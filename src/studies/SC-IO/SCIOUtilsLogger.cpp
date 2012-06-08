/*
 * utils.h
 *
 *  Created on: Jul 15, 2011
 *      Author: tcpan
 */
#include "SCIOUtilsLogger.h"

const int ::cciutils::event::COMPUTE = 0;
const int ::cciutils::event::MEM_IO = 11;
const int ::cciutils::event::GPU_MEM_IO = 12;
const int ::cciutils::event::NETWORK_IO = 21;
const int ::cciutils::event::NETWORK_WAIT = 22;
const int ::cciutils::event::FILE_I = 31;
const int ::cciutils::event::FILE_O = 32;
const int ::cciutils::event::ADIOS_INIT = 41;
const int ::cciutils::event::ADIOS_OPEN = 42;
const int ::cciutils::event::ADIOS_ALLOC = 43;
const int ::cciutils::event::ADIOS_WRITE = 44;
const int ::cciutils::event::ADIOS_CLOSE = 45;
const int ::cciutils::event::ADIOS_BENCH_OPEN = 52;
const int ::cciutils::event::ADIOS_BENCH_ALLOC = 53;
const int ::cciutils::event::ADIOS_BENCH_WRITE = 54;
const int ::cciutils::event::ADIOS_BENCH_CLOSE = 55;
const int ::cciutils::event::ADIOS_FINALIZE = 46;
const int ::cciutils::event::OTHER = -1;

