/*
 * kernel codes for the reconstruction by dilation (reconstruction_by_dilation_kernel)
 * and the reconstruction by erosion (reconstruction_by_erosion_kernel)
 */
namespace nscale { namespace gpu {


void reconstruction_by_dilation_kernel( unsigned char* g_marker, const unsigned char* g_mask, const int sx, const int sy, const int sz );

}}