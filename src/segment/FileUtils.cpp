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
#include <sys/stat.h>

using namespace std;

FileUtils::FileUtils() : ext()
{
}

FileUtils::FileUtils(const std::string & suffix) : ext(suffix)
{
}


FileUtils::~FileUtils()
{
    //dtor
}

string FileUtils::replaceDir(string& filename, const string& oldDir, const string& newDir) {
	int pos = filename.find(oldDir);

    stringstream newname;
    newname << newDir << filename.substr(pos + oldDir.length());
	return newname.str();
}

string FileUtils::replaceExt(string& filename, const string& oldExt, const string& newExt) {
	int pos = filename.rfind(oldExt);

    stringstream newname;
    newname << filename.substr(0, pos) << newExt;
	return newname.str();
}



void FileUtils::traverseDirectoryRecursive(const string & directory, vector<string> & fullList)
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
            if (ext.empty() || d.rfind(ext) != std::string::npos) {
        		fullList.push_back(d);
        	}
		}
    }
}

bool FileUtils::mkdirs(const string & d)
{
	DIR *dir = opendir(d.c_str());
	if (dir == 0 && d.size() > 0) {
		// find the parent
		int currPos = d.find_last_of("/\\");
		string parent = d.substr(0, currPos);
		printf("dir to create: %s, parent: %s\n", d.c_str(), parent.c_str());
		if (FileUtils::mkdirs(parent)) {
			int n = mkdir(d.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
			if (n != 0) return false;
		} else {
			return false;
		}
	} else {
		closedir(dir);
	}
	return true;
}
