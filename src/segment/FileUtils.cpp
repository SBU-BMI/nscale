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
	if (pos >= fn.length() - 1) return string("");
	else if (pos >= 0) return fn.substr(pos + 1);
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

/***  directories are assume to have a trailing directory separator
 * basically, use string replace.
 */
string FileUtils::replaceDir(string& filename, const string& oldStr, const string& newStr) {
	if (filename.length() == 0) {
		printf("WARNING: filename is empty.\n");
		return std::string();
	}

	if (strcmp(oldStr.c_str(), newStr.c_str()) == 0 ) {
		printf("Overwrite WARNING: old %s and new %s are the same.  filename %s unchanged\n", oldStr.c_str(), newStr.c_str(), filename.c_str());
		return std::string();
}

	int pos;
	std::string output;
	if (oldStr.length() == 0) {
		return output.assign(newStr).append(filename);
	}
	else pos = filename.find(oldStr);

	if (pos == std::string::npos) {
		printf("Overwrite WARNING: %s is not in filename %s.  filename set to empty\n", oldStr.c_str(), filename.c_str());
		return std::string();
	}
	output = filename;
	return output.replace(pos, oldStr.length(), newStr);

}

string FileUtils::replaceExt(string& filename, const string& oldExt, const string& newExt) {
	if (filename.length() == 0) {
		printf("WARNING: filename is empty.\n");
		return std::string();
	}

	if (strcmp(oldExt.c_str(), newExt.c_str()) == 0 ) {
		printf("Overwrite WARNING: old %s and new %s are the same.  filename %s set to empty\n", oldExt.c_str(), newExt.c_str(), filename.c_str());
		return std::string();
	}

	int pos;
	std::string output = filename;
	if (oldExt.length() == 0) {
		return output.append(newExt);
	}

	std::string ex = filename.substr(filename.length() - oldExt.length(), oldExt.length());
	std::string name = filename.substr(0, filename.length() - oldExt.length());
	if (strcmp(ex.c_str(), oldExt.c_str()) == 0) {
		return name.append(newExt);
	} else {
		printf("ERROR: filename %s does not have %s extension\n", filename.c_str(), oldExt.c_str());
		return std::string();
	}
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
			std::string ex = getExt(d);
            if (ext.empty() || strcmp(ex.c_str(), ext.c_str()) == 0) {
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
					std::string ex = getExt(s);
					if (ext.empty() || strcmp(ex.c_str(), ext.c_str()) == 0) {
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
