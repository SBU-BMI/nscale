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

string FileUtils::getDir(string& filename) {
	int pos = filename.rfind('/');
	if (pos > 0) return filename.substr(0, pos);
	else if (pos == 0) return string("/");
	else return string("");
}
string FileUtils::getFile(string& filename) {
	int pos = filename.rfind('/');
	if (pos >= filename.length() - 1) return string("");
	else if (pos >= 0) return filename.substr(pos + 1);
	else return filename;
}
string FileUtils::getExt(string& filename) {
	string fn = getFile(filename);
	int pos = fn.rfind('.');
	if (pos >= filename.length() - 1) return string("");
	else if (pos >= 0) return filename.substr(pos + 1);
	else return string("");
}

string FileUtils::getRelativePath(string& filename, const string& dir) {
	int pos = filename.find(dir);

	string result = filename;
	if (pos > 0) {
		result = filename.substr(pos + dir.length());
		if (result.find('/') == 0) {
			result = result.substr(1);
		}
	}  // else give an absolute path
	return result;
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

void FileUtils::getFilesInDirectory(const string & directory, vector<string> & fileList)
{

	DIR *dir, *dir2;
    struct dirent *ent;
    string s;


	// open the name to see if it's a directory
	if ((dir=opendir(directory.c_str())) != NULL)
	{
		while((ent=readdir(dir)) != NULL) // loop until the directory is traveled thru
		{
			// push directory or filename to the list
			stringstream fullname;

			if (strcmp(ent->d_name, ".") &&
					strcmp(ent->d_name, "..")) {
				fullname << directory << "/" << ent->d_name;
				s = fullname.str();

				// now check to see if it's a directory.
				if ((dir2=opendir(s.c_str())) == NULL) {
					if (ext.empty() || s.rfind(ext) != std::string::npos) {
						fileList.push_back(s);
						//printf("TESTING: %s\n", s.c_str());
					}

				} else {
					// entry is a directory, so don't keep it
					closedir(dir2);
				}

			}
		}
		// close up
		closedir(dir);
	} else {
		// a file.  don't touch the file list
		return;
	}
}

void FileUtils::getDirectoriesInDirectory(const string & directory, vector<string> & dirList)
{

	DIR *dir, *dir2;
    struct dirent *ent;
    string s;


	// open the name to see if it's a directory
	if ((dir=opendir(directory.c_str())) != NULL)
	{
		while((ent=readdir(dir)) != NULL) // loop until the directory is traveled thru
		{
			// push directory or filename to the list
			stringstream fullname;

			if (strcmp(ent->d_name, ".") &&
					strcmp(ent->d_name, "..")) {
				fullname << directory << "/" << ent->d_name;
				s = fullname.str();

				//printf("here:  found %s\n", s.c_str());

				// now check to see if it's a directory.
				if ((dir2=opendir(s.c_str())) != NULL) {
					dirList.push_back(s);
					//printf("TESTING: %s\n", s.c_str());
					// entry is a directory, so keep it
					closedir(dir2);
				}

			}
		}
		// close up
		closedir(dir);
	} else {
		// a file.  don't touch the file list
		return;
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
