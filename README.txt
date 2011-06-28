Build requirements:
1. autoconf, automake, libtool 
2. make
3. a suitable c/c++ compiler
4. cmake

This package requires the following external libraries
Intel TBB 3.0 update 7 oss for lin (through ubuntu)
CUDA  (need drivers, toolkit, and SDK from NVIDIA)
OpenCV 2.3.0 (require cmake, and optionally CUDA, TBB).  Do make install at the end
#dlib 17.41
libtiff 3.9.4-5ubuntu6 (through ubuntu)
HDF5 4.1.2 shared 64bit for lnux 2.6

editing:
using eclipse is recommended but not required.
installed valgrind
eclipse Linux Tools is recommended for building with autoconf/automake


compiling:
1. need to have LD_LIBRARY_PATH point to the location where the opencv libs are installed
2. use `pkg-config --cflags --libs opencv` in Makefile.

TODO:
switch to autoconf and/or CMake