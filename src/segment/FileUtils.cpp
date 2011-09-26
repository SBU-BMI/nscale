/*
 * FileUtils.cpp
 *
 *  Created on: Sep 26, 2011
 *      Author: tcpan
 *
 */

#include "FileUtils.h"

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <sys/types.h>
#include <dirent.h>
#include <vector>
#include <queue>

using namespace std;

FileUtils::FileUtils() : ext()
{
}

FileUtils::FileUtils(std::string & suffix) : ext(suffix)
{
}


FileUtils::~FileUtils()
{
    //dtor
}


void FileUtils::traverseDirectoryRecursive(string directory, vector<string> *fullList)
{
	queue<string> dirList;
	dirList.push(directory);

	DIR *dir;
    struct dirent *ent;
    string d, s;

    while (!dirList.empty()) {
    	d = dirList.front();
    	dirList.pop();

		// open a directory
    	if ((dir=opendir(d.c_str())) != NULL)
    	{
			while((ent=readdir(dir)) != NULL) // loop until the directory is traveled thru
			{
				// push directory or filename to the list
	            stringstream fullname;

				if (strcmp(ent->d_name, ".") &&
			            strcmp(ent->d_name, "..")) {
					fullname << d << "/" << ent->d_name;
					s = fullname.str();
					dirList.push(s);
				}
			}
			// close up
			closedir(dir);
		} else {
			// a file.  add to the fullList
            if (ext.empty() || d.find_last_of(ext) != std::string::npos) {
        		fullList->push_back(d);
        	}
		}
    }
}

