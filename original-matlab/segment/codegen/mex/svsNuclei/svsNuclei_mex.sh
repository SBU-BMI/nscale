MATLAB="/usr/local/MATLAB/R2011a"
Arch=glnxa64
ENTRYPOINT=mexFunction
MAPFILE=$ENTRYPOINT'.map'
PREFDIR="/home/tcpan/.matlab/R2011a"
OPTSFILE_NAME="./mexopts.sh"
. $OPTSFILE_NAME
COMPILER=$CC
. $OPTSFILE_NAME
echo "# Make settings for svsNuclei" > svsNuclei_mex.mki
echo "CC=$CC" >> svsNuclei_mex.mki
echo "CFLAGS=$CFLAGS" >> svsNuclei_mex.mki
echo "CLIBS=$CLIBS" >> svsNuclei_mex.mki
echo "COPTIMFLAGS=$COPTIMFLAGS" >> svsNuclei_mex.mki
echo "CDEBUGFLAGS=$CDEBUGFLAGS" >> svsNuclei_mex.mki
echo "CXX=$CXX" >> svsNuclei_mex.mki
echo "CXXFLAGS=$CXXFLAGS" >> svsNuclei_mex.mki
echo "CXXLIBS=$CXXLIBS" >> svsNuclei_mex.mki
echo "CXXOPTIMFLAGS=$CXXOPTIMFLAGS" >> svsNuclei_mex.mki
echo "CXXDEBUGFLAGS=$CXXDEBUGFLAGS" >> svsNuclei_mex.mki
echo "LD=$LD" >> svsNuclei_mex.mki
echo "LDFLAGS=$LDFLAGS" >> svsNuclei_mex.mki
echo "LDOPTIMFLAGS=$LDOPTIMFLAGS" >> svsNuclei_mex.mki
echo "LDDEBUGFLAGS=$LDDEBUGFLAGS" >> svsNuclei_mex.mki
echo "Arch=$Arch" >> svsNuclei_mex.mki
echo OMPFLAGS= >> svsNuclei_mex.mki
echo OMPLINKFLAGS= >> svsNuclei_mex.mki
echo "EMC_COMPILER=unix" >> svsNuclei_mex.mki
echo "EMC_CONFIG=debug" >> svsNuclei_mex.mki
"/usr/local/MATLAB/R2011a/bin/glnxa64/gmake" -B -f svsNuclei_mex.mk
