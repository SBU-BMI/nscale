/*
 * kernel codes for the reconstruction by dilation (reconstruction_by_dilation_kernel)
 * and the reconstruction by erosion (reconstruction_by_erosion_kernel)
 */

void reconstruction_by_dilation_kernel( float* g_marker, const float* g_mask, const int sx, const int sy, const int sz );

void reconstruction_by_erosion_kernel( float* g_marker, const float* g_mask, const int sx, const int sy, const int sz );
