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
        FileUtils(std::string & suffix);
        virtual ~FileUtils();

        void traverseDirectoryRecursive(string directory, vector<string> *fullList);
    protected:
        const std::string ext;
    private:
};

#endif /* FileUtils_H_ */
