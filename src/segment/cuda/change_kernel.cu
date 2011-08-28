/*
 * initialization of the change flag
 */

#include "internal_shared.hpp"

namespace nscale { namespace gpu {


__global__ void init_change( bool *change ) {
	*change = false;
}

}}
