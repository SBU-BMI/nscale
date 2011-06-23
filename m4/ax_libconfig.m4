AC_DEFUN([AX_LIBCONFIG],[

AC_MSG_NOTICE([Looking for libconfig])
AC_ARG_WITH([libconfig],
	[AS_HELP_STRING([--with-libconfig=DIR],
			[Use Libconfig installation from DIR/{include,lib,bin} [OPTIONAL] ])],
	[
	libconfig_path=$withval
	])


OLDCPPFLAGS="$CPPFLAGS"
OLDLDFLAGS="$LDFLAGS"
OLDLIBS="$LIBS"

CPPFLAGS="$CPPFLAGS -I$libconfig_path/include"
LDFLAGS="$LDFLAGS -L$libconfig_path/lib"
LIBS="$LIBS -lconfig++"

AC_CHECK_HEADER([libconfig.h++])

AC_LANG_PUSH([C++])
AC_SEARCH_LIBS([config_init],[config++])
#AC_CHECK_LIB([config],[config_init],[goodlib=yes; LIBS="-lconfig++ $LIBS"], [goodlib=no])

AC_LANG_POP


]) dnl AX_LIBCONFIG



