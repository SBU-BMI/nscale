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

using namespace std;

/**
 *
 *
 * notes:  stat() follows symlink in reporting.  perfect.
 */
class FileUtils
{
    public:
        FileUtils();
        FileUtils(const std::string & _suffix);
        FileUtils(const std::vector<std::string> _suffixes);
        virtual ~FileUtils();

        static const int DIRECTORY;
        static const int FILE;
        static const char *DIR_SEPARATOR;

        /**
         * get parts of the filename
         * TODO: change to static methods
         */
        static string getDir(const string& name);
        static string getFile(const string& filename);
        static string getExt(const string& filename);

        /**
         * for matching names.  extension is at the end of string, directory is at beginning of string
         * common directory is basically the least common leading directories of the name.
         * TODO: change to static methods
         */
        static bool hasExt(const string &filename, const string &ext);
        static bool inDir(const string &name, const string &dir);
        /**
         * input - if dir name, need to terminate with /
         * TODO: change to static methods
         */
        static string getCommonDir(const string& name1, const string& name2);

        /**
         * for constructing new names
         * if filename does not contain the old string, then new string is prepended or appended respectively.
         * if filename contains the old string as extensions or prefix, then new string replaces the old string
         * the delimiter characters for dir, or for ext should be present.
         * TODO: change to static methods
         */
        static string replaceDir(const string& filename, const string& oldDir, const string& newDir);
        static string replaceExt(const string& filename, const string& oldExt, const string& newExt);

        /**
         * same as the other replaceExt, but uses the preset extensions.
         */
        string replaceExt(const string& filename, const string& newExt);


        /**
         * traverse the directory, recursively if specified.
         * type is the capture type - directories and/or files.
         * types can be a bit-wise AND of DIRECTORY and FILE.
         */
        bool traverseDirectory(const string& directory, vector<string> &list, int types, bool recursive);


        /**
         * for backward compatibility.  use TraverseDirectory if possible.
         */
        void traverseDirectoryRecursive(const string& directory, vector<string> & fullList) __attribute__ ((deprecated));;
        void getFilesInDirectory(const string& directory, vector<string> &fileList) __attribute__ ((deprecated));;
        void getDirectoriesInDirectory(const string& directory, vector<string> &dirList) __attribute__ ((deprecated));;

        /**
         * TODO: change to static methods
         */
        static bool mkdirs(const string& dirname);

    protected:
        std::vector<std::string> exts;
    private:
};

#endif /* FileUtils_H_ */
