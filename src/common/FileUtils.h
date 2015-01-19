/*
 * FileUtils.h
 *
 *  Created on: Sep 26, 2011
 *      Author: tcpan
 *
 *
 */

#ifndef FileUtils_H_
#define FileUtils_H_

#include <string>
#include <vector>

#ifndef _GNUC
	#define __attribute__(A) /* do nothing */
#endif

namespace cci {
namespace common {


/**
 *
 *
 * notes:  stat() follows symlink in reporting.  perfect.
 */
#ifdef _MSC_VER
	class	__declspec(dllexport) FileUtils 
#else
class FileUtils
#endif
{
    public:
        FileUtils();
        FileUtils(const std::string & _suffix);
        FileUtils(const std::vector<std::string> _suffixes);
        virtual ~FileUtils();

        static const int DIRECTORY;
        static const int FILE;
        static const char *DIR_SEPARATOR;

		static int getDIRECTORY();
		static int getFILE();

        /**
         * get parts of the filename
         * TODO: change to static methods
         */
        static std::string getDir(const std::string& name);
        static std::string getFile(const std::string& filename);
        static std::string getExt(const std::string& filename);

        /**
         * for matching names.  extension is at the end of string, directory is at beginning of string
         * common directory is basically the least common leading directories of the name.
         * TODO: change to static methods
         */
        static bool hasExt(const std::string &filename, const std::string &ext);
        static bool inDir(const std::string &name, const std::string &dir);
        /**
         * input - if dir name, need to terminate with /
         * TODO: change to static methods
         */
        static std::string getCommonDir(const std::string& name1, const std::string& name2);

        /**
         * for constructing new names
         * if filename does not contain the old string, then new string is prepended or appended respectively.
         * if filename contains the old string as extensions or prefix, then new string replaces the old string
         * the delimiter characters for dir, or for ext should be present.
         * TODO: change to static methods
         */
        static std::string replaceDir(const std::string& filename, const std::string& oldDir, const std::string& newDir);
        static std::string replaceExt(const std::string& filename, const std::string& oldExt, const std::string& newExt);

        /**
         * same as the other replaceExt, but uses the preset extensions.
         */
        std::string replaceExt(const std::string& filename, const std::string& newExt);


        /**
         * traverse the directory, recursively if specified.
         * type is the capture type - directories and/or files.
         * types can be a bit-wise AND of DIRECTORY and FILE.
         */
        bool traverseDirectory(const std::string& directory, std::vector<std::string> &list, int types, bool recursive);


        /**
         * for backward compatibility.  use TraverseDirectory if possible.
         */
        void traverseDirectoryRecursive(const std::string& directory, std::vector<std::string> & fullList) __attribute__ ((deprecated));
        void getFilesInDirectory(const std::string& directory, std::vector<std::string> &fileList) __attribute__ ((deprecated));
        void getDirectoriesInDirectory(const std::string& directory, std::vector<std::string> &dirList) __attribute__ ((deprecated));

        /**
         * TODO: change to static methods
         */
        static bool mkdirs(const std::string& dirname);

    protected:
        std::vector<std::string> exts;
    private:
};

}
}

#endif /* FileUtils_H_ */
