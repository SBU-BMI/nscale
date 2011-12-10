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


class FileUtils
{
    public:
        FileUtils();
        FileUtils(const std::string & suffix);
        virtual ~FileUtils();

        void traverseDirectoryRecursive(const string& directory, vector<string> & fullList);

        string getDir(string& filename);
        string getRelativePath(string& filename, const string& dir);
        string replaceDir(string& filename, const string& oldDir, const string& newDir);
        string replaceExt(string& filename, const string& oldExt, const string& newExt);
        bool mkdirs(const string& dirname);

    protected:
        const std::string ext;
    private:
};

#endif /* FileUtils_H_ */
