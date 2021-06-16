THIS IS NOT UP TO DATE.

Build requirements:
1. cmake 
2. make
3. a suitable c/c++ compiler
4. libeigen2-dev, libpng-dev, libpng++-dev, libtiff 3.9.4-5ubuntu6 (through ubuntu)
5. hdf5 (for pipeline), adios (for SCIO), boost, MPI

This package requires the following external libraries
CUDA  (need drivers, toolkit, and SDK from NVIDIA)
OpenCV 2.3.0 (require cmake, and optionally CUDA, TBB).  use system libpnginstead of the built-in. - important.  else get png version issues.  turned off video stuff, on with TBB and CUDA.  turn off QT-opengl. Do make install at the end
HDF5 4.1.2 shared 64bit for lnux 2.6

editing:
using eclipse is recommended but not required.
installed valgrind
eclipse Linux Tools is recommended for building with autoconf/automake


compiling:

=====================================================================================

Instructions for building as a Region Templates dependency, without CUDA support.
Tested for gcc 7 and 9 on ubuntu 20.04.
Some other libs may be required for different systems

# libs required
sudo apt-get install libavformat-dev libavcodec-dev

# opencv 2.7
git clone https://github.com/opencv/opencv.git
mv opencv opencv-2.4.9
cd opencv-2.4.9
git checkout 2.4.9.1
sed -i 's/dumpversion/dumpfullversion/g' cmake/OpenCVDetectCXXCompiler.cmake
sed -i '1111,1130d' modules/contrib/src/chamfermatching.cpp
sed -i '1016,1019d' modules/contrib/src/chamfermatching.cpp
sed -i '969,972d' modules/contrib/src/chamfermatching.cpp
sed -i '225d' modules/highgui/src/cap_v4l.cpp
sed -i '245,250d' modules/highgui/src/cap_libv4l.cpp
sed -i '131,132d' cmake/OpenCVFindLibsVideo.cmake
mkdir build
cd build
cmake ../ -D WITH_LIBV4L=OFF -D WITH_V4L=OFF -D WITH_FFMPEG=OFF -DWITH_CUDA=OFF -D ENABLE_PRECOMPILED_HEADERS=OFF
make -j8

# nscale
cd ../../
git clone https://github.com/SBU-BMI/nscale.git
cd nscale
mkdir build
cd build
cmake ../ -D OpenCV_DIR=../../opencv-2.4.9/build/
make -j8
