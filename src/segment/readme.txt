1) export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig 
	-> This updates the pkg config path, which will suppose to "see" opencv paths

2) Changes in the Makefile: Comment lines 7 and 9 that refer to CUDA seetings, these lines
should be commented if you do not want to use CUDA. 
	-> #include Makefile-cuda.inc
	-> CUDA_SRC

3) Adding path of precomp.hpp to Makefile.inc
	->  /usr/local/src/OpenCV-2.3.0/modules/gpu/src/

4) Change LD_LIBRARY_PATH to find OpenCV libraries, as well as to link to
libraries in the current directory
