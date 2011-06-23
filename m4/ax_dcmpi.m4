AC_DEFUN([AX_DCMPI],[

AC_MSG_NOTICE([Looking for a DCMPI installation])
AC_ARG_WITH([dcmpi],
	[AS_HELP_STRING([--with-dcmpi=DIR],
			[Use DCMPI installation from DIR/{include,lib,bin} [REQUIRED] ])],
	[
	dcmpi_path=$withval
	],
	[
	AC_MSG_ERROR([--with-dcmpi=DIR is required])
	])


OLDCPPFLAGS="$CPPFLAGS"
OLDLDFLAGS="$LDFLAGS"
OLDLIBS="$LIBS"

CPPFLAGS="$CPPFLAGS -I$dcmpi_path/include"
LDFLAGS="$LDFLAGS -L$dcmpi_path/lib"
LIBS="$LIBS -ldcmpi"

AC_CHECK_HEADER([dcmpi.h])

AC_LANG_PUSH([C++])
AC_CHECK_LIB([dcmpi],[main],[goodlib=yes; LIBS="-ldcmpi $LIBS"], [goodlib=no])

if test $goodlib = no ; then
   AC_MSG_ERROR([cannot use DCMPI installation from $dcmpi_path])
fi

AC_MSG_RESULT([OK])


]) dnl AX_DCMPI



