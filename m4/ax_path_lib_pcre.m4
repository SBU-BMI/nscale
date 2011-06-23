# ===========================================================================
#     http://www.gnu.org/software/autoconf-archive/ax_path_lib_pcre.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_PATH_LIB_PCRE [(A/NA)]
#
# DESCRIPTION
#
#   check for pcre lib and set PCRE_LIBS and PCRE_CFLAGS accordingly.
#
#   also provide --with-pcre option that may point to the $prefix of the
#   pcre installation - the macro will check $pcre/include and $pcre/lib to
#   contain the necessary files.
#
#   the usual two ACTION-IF-FOUND / ACTION-IF-NOT-FOUND are supported and
#   they can take advantage of the LIBS/CFLAGS additions.
#
# LICENSE
#
#   Copyright (c) 2008 Guido U. Draheim <guidod@gmx.de>
#
#   This program is free software; you can redistribute it and/or modify it
#   under the terms of the GNU General Public License as published by the
#   Free Software Foundation; either version 2 of the License, or (at your
#   option) any later version.
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
#   Public License for more details.
#
#   You should have received a copy of the GNU General Public License along
#   with this program. If not, see <http://www.gnu.org/licenses/>.
#
#   As a special exception, the respective Autoconf Macro's copyright owner
#   gives unlimited permission to copy, distribute and modify the configure
#   scripts that are the output of Autoconf when processing the Macro. You
#   need not follow the terms of the GNU General Public License when using
#   or distributing such scripts, even though portions of the text of the
#   Macro appear in them. The GNU General Public License (GPL) does govern
#   all other use of the material that constitutes the Autoconf Macro.
#
#   This special exception to the GPL applies to versions of the Autoconf
#   Macro released by the Autoconf Archive. When you make and distribute a
#   modified version of the Autoconf Macro, you may extend this special
#   exception to the GPL to apply to your modified version as well.

#serial 6

AC_DEFUN([AX_PATH_LIB_PCRE],[dnl
AC_MSG_NOTICE([checking for usable PCRE library])
AC_ARG_WITH([pcre],
[AS_HELP_STRING([--with-pcre[[=prefix]] ],[use a PCRE library (defaults to yes)])],[],
     with_pcre="yes")
if test ".$with_pcre" = ".no" || test ".$with_pcre" = "." ; then
  AC_MSG_RESULT([PCRE support disabled])
else
  OLDLDFLAGS="$LDFLAGS" ; OLDCPPFLAGS="$CPPFLAGS"
  if test ".$with_pcre" != ".yes" ; then
     PCRE_LDFLAGS="$LDFLAGS -L$with_pcre/lib"
     PCRE_CPPFLAGS="$CPPFLAGS -I$with_pcre/include"
  fi
  LDFLAGS="$LDFLAGS $PCRE_LDFLAGS"
  CPPFLAGS="$CPPFLAGS $PCRE_CPPFLAGS"
  AC_CHECK_LIB(pcre, pcre_study)
  AC_CHECK_HEADER(pcre.h)
  if test "$ac_cv_lib_pcre_pcre_study" = "yes" && test "$ac_cv_header_pcre_h" = "yes" ; then
     PCRE_LIBS="-lpcre"
     if test ".$with_pcre" != ".yes" ; then 
        AC_MSG_NOTICE([libpcre found; headers and libs under $with_pcre])
     else
        AC_MSG_NOTICE([libpcre found])
     fi
  else
     AC_MSG_ERROR([no usable PCRE library found and PCRE not explicitly disabled])
     CPPFLAGS="$OLDCPPFLAGS"
     LDFLAGS="$LDFLAGS"
  fi
fi

AC_SUBST([PCRE_LIBS])
AC_SUBST([PCRE_LDFLAGS])
AC_SUBST([PCRE_CPPFLAGS])
])