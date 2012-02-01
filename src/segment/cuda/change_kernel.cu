/*
 * initialization of the change flag
 */

namespace nscale { namespace gpu {


__global__ void init_change( bool *change ) {
	*change = false;
}

}}
