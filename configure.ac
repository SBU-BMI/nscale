AC_INIT([nscale],[0.1])
AC_CONFIG_SRCDIR([src/Makefile.am])
AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE(nscale,0.1)
AC_CONFIG_HEADERS([config.h])
AM_MAINTAINER_MODE
AM_PROG_LIBTOOL
LT_INIT
AC_CANONICAL_HOST


AC_ARG_ENABLE([debug],
	[AS_HELP_STRING([--enable-debug],[Enable a debug build])],
	[AC_DEFINE(DEBUG_BUILD, 1, [Debug build enabled])])

AC_ARG_ENABLE([tests],
	[AS_HELP_STRING([--enable-tests],[Build test programs])],
	[AC_DEFINE(TESTS_BUILD, 1, [Build with test programs])])

AC_ARG_ENABLE([mpi-build],
	[AS_HELP_STRING([--disable-mpi-build],[Do not build MPI-based programs])],
	[case "${enableval}" in 
	  yes) mpi_build=true ;;
	  no)  mpi_build=false ;;
	  *)   AC_MSG_ERROR([bad value ${enableval} for --disable-mpi-build]) ;;
	esac],[mpi_build=false])
	AM_CONDITIONAL([MPIBUILD], [test x$mpi_build = xtrue])

AC_ARG_ENABLE([openmp-build],
	[AS_HELP_STRING([--disable-openmp-build],[Do not use OpenMP code])],
	[case "${enableval}" in 
	  yes) openmp_build=true ;;
	  no)  openmp_build=false ;;
	  *)   AC_MSG_ERROR([bad value ${enableval} for --disable-openmp-build]) ;;
	esac],[openmp_build=false])
	AM_CONDITIONAL([OPENMPBUILD], [test x$openmp_build = xtrue])

AC_ARG_WITH([extra-tree],
        [AS_HELP_STRING([--with-extra-tree=DIR],
        [Add DIR as an extra tree to search for headers/libs])],
        [
        CPPFLAGS="$CPPFLAGS -I$withval/include"
        LDFLAGS="$LDFLAGS -L$withval/lib"
        AC_MSG_NOTICE([Adding $withval as extra header/lib search paths])
        ],
        [])

AC_ARG_WITH([intrinsics],
	[AS_HELP_STRING([--with-intrinsics={off | MMX | SSE}],
			[Use MMX or SSE intrinsic isets])],
	[
	if test ".$withval" = ".MMX" ; then
	   AC_DEFINE([OCVM_USE_MMX_INTRINSICS],[],[Use MMX intrinsic instructions]) 
	   if test "$CXX" = "g++" ; then
	      CXXFLAGS="$CXXFLAGS -march=pentium3"
	   fi
	fi
	if test ".$withval" = ".SSE" ; then 
	   AC_DEFINE([OCVM_USE_SSE_INTRINSICS],[],[Use SSE intrinsic instructions])
	   if test "$CXX" = "g++" ; then
	      CXXFLAGS="$CXXFLAGS -march=pentium4"
	   fi
	fi
	if test ".$withval" != ".off" ; then
	   AC_MSG_WARN([Invalid argument to --with-intrinsics={off|MMX|SSE}])
	fi
	],[])


AC_LANG([C++])
AC_PROG_CXX
AX_CXX_NAMESPACES
AX_CXX_NAMESPACE_STD

#DX_HTML_FEATURE(ON)
#DX_INIT_DOXYGEN(dcmpi,Doxyfile,doc/doxygen-doc)

AC_C_CONST
AC_CHECK_SIZEOF([void *])

AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_TYPE_UID_T
AC_TYPE_INT8_T
AC_TYPE_UINT8_T
AC_TYPE_INT16_T
AC_TYPE_UINT16_T
AC_TYPE_INT32_T
AC_TYPE_UINT32_T
AC_TYPE_INT64_T
AC_TYPE_UINT64_T
AC_TYPE_OFF_T
AC_TYPE_INTMAX_T
AC_TYPE_UINTMAX_T
AC_EXEEXT


AC_CHECK_HEADERS(unistd.h malloc.h sys/types.h sys/sysctl.h sched.h execinfo.h)
# dlib/matrix.h)

#AC_CHECK_FUNCS()
AC_CHECK_DECLS([_SC_PHYS_PAGES,_SC_PAGE_SIZE,HW_PAGESIZE,HW_MEMSIZE],
	[],
	[],
	[#include <unistd.h>
	  #include <sys/types.h>
	  #include <sys/sysctl.h>
	  ])

ACX_PTHREAD

AC_CHECK_HEADER([libconfig.h++])

AC_LANG_PUSH([C++])
AC_SEARCH_LIBS([config_init],[config++])
AC_SEARCH_LIBS([TIFFOpen],[tiff])
#AC_CHECK_LIB([config],[config_init],[goodlib=yes; LIBS="-lconfig++ $LIBS"], [goodlib=no])

AC_LANG_POP


AX_LIB_HDF5
if test x$openmp_build = xtrue; then
   AX_OPENMP
fi
LX_FIND_MPI
AX_PROG_GENGETOPT

AC_SUBST(OPENMP_CXXFLAGS)
AC_SUBST(OPENMP_CFLAGS)

AC_CONFIG_FILES([Makefile src/Makefile src/segment/Makefile src/features/Makefile src/dimreduc/Makefile src/classify/Makefile])
AC_OUTPUT




