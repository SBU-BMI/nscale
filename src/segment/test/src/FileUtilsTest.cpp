/*
 * FileUtils_test.cpp
 *
 *  Created on: Jul 11, 2012
 *      Author: tcpan
 */

#include "FileUtils.h"
#include "utils.h"
#include <string>
#include <iostream>
#include <iterator>
#include <vector>
#include <iostream>
#include <sys/stat.h>
#include <dirent.h>
#include <stdio.h>

using namespace std;

void testTraverseDirectory(const string &d1, FileUtils &fu) {
   // time to open a directory vs stat
	printf("PROCESSING %s\n", d1.c_str());

	long c1 = cciutils::ClockGetTime();
	struct stat st_buf;
	int status = stat (d1.c_str(), &st_buf);
	if (status != 0) {
		printf("ERROR: unable to inspect %s\n", d1.c_str());
	} else {
		if (S_ISDIR (st_buf.st_mode)) printf("is a directory. ");
	}
	long c2 = cciutils::ClockGetTime();
	printf("using stat took %ld time\n", c2-c1);

	c1 = cciutils::ClockGetTime();
	DIR *dir = opendir(d1.c_str());
	if (dir != NULL) printf("is a directory too. ");
	c2 = cciutils::ClockGetTime();
	printf("using opendir took %ld time\n", c2-c1);

	vector<string> v1;
	ostream_iterator<string> out_it (cout,"\n");

	cout << "TRAVERSE DIR, DIRECTORY only, no recursion : ";
	fu.traverseDirectory(d1, v1, FileUtils::DIRECTORY, false);
//	copy ( v1.begin(), v1.end(), out_it );
	cout << v1.size() << " entries" << endl;
	v1.clear();

//	// tested,  new works the same.
//	cout << "OLD TRAVERSE DIR, DIR only, no recursion : ";
//	fu.getDirectoriesInDirectory(d1, v1);
////	copy ( v1.begin(), v1.end(), out_it );
//	cout << v1.size() << " entries" << endl;
//	v1.clear();


	cout << "TRAVERSE DIR, FILE only, no recursion : ";
	fu.traverseDirectory(d1, v1, FileUtils::FILE, false);
	//copy ( v1.begin(), v1.end(), out_it );
	cout << v1.size() << " entries" << endl;
	v1.clear();

//	// tested.  new works the same
//	cout << "OLD TRAVERSE DIR, FILE only, no recursion : ";
//	fu.getFilesInDirectory(d1, v1);
////	copy ( v1.begin(), v1.end(), out_it );
//	cout << v1.size() << " entries" << endl;
//	v1.clear();



	cout << "TRAVERSE DIR, BOTH, no recursion : ";
	fu.traverseDirectory(d1, v1, FileUtils::FILE | FileUtils::DIRECTORY, false);
//	copy ( v1.begin(), v1.end(), out_it );
	cout << v1.size() << " entries" << endl;
	v1.clear();

	cout << "TRAVERSE DIR, DIRECTORY only, recursion : ";
	fu.traverseDirectory(d1, v1, FileUtils::DIRECTORY, true);
//	copy ( v1.begin(), v1.end(), out_it );
	cout << v1.size() << " entries" << endl;
	v1.clear();

	cout << "TRAVERSE DIR, FILE only, recursion : ";
	fu.traverseDirectory(d1, v1, FileUtils::FILE, true);
//	copy ( v1.begin(), v1.end(), out_it );
	cout << v1.size() << " entries" << endl;
	v1.clear();

//	// tested. new works the same
//	cout << "OLD TRAVERSE DIR, FILE only, recursion : ";
//	fu.traverseDirectoryRecursive(d1, v1);
////	copy ( v1.begin(), v1.end(), out_it );
//	cout << v1.size() << " entries" << endl;
//	v1.clear();


	cout << "TRAVERSE DIR, BOTH, recursion : ";
	fu.traverseDirectory(d1, v1, FileUtils::FILE | FileUtils::DIRECTORY, true);
//	copy ( v1.begin(), v1.end(), out_it );
	cout << v1.size() << " entries" << endl;


}


int main (int argc, char **argv){
	
	FileUtils fu1;
	FileUtils fu2(".tif");
		
	vector<string> exts;
	exts.push_back(".blah");
	exts.push_back(".txt");
	::FileUtils fu3(exts);
	
	vector<string> exts2;
	exts2.push_back(".cmake");
	exts2.push_back(".txt");
	::FileUtils fu5(exts2);
	
	
// testing the get part
	string s1("");
	
	string s2("/");
	string s3("aa");
	string s3_1("/aa");
	string s3_2("aa/");
	string s4("aa/bb");
	string s5("aa/bb/");
	string s6("/aa/bb");
	string s7("/aa/bb/");
	

	cout << "Dir s1:" <<    FileUtils::getDir(s1).compare("") << endl;
	cout << "Dir s2:" <<    FileUtils::getDir(s2).compare("/") << endl;
	cout << "Dir s3:" <<    FileUtils::getDir(s3).compare("") << endl;
	cout << "Dir s3_1:" <<  FileUtils::getDir(s3_1).compare("/") << endl;
	cout << "Dir s3_2:" <<  FileUtils::getDir(s3_2).compare("aa") << endl;
	cout << "Dir s4:" <<    FileUtils::getDir(s4).compare("aa") << endl;
	cout << "Dir s5:" <<    FileUtils::getDir(s5).compare("aa/bb") << endl;
	cout << "Dir s6:" <<    FileUtils::getDir(s6).compare("/aa") << endl;
	cout << "Dir s7:" <<    FileUtils::getDir(s7).compare("/aa/bb") << endl;
	
	cout << "File s1:" <<   FileUtils::getFile(s1).compare("") << endl;
	cout << "File s2:" <<   FileUtils::getFile(s2).compare("") << endl;
	cout << "File s3:" <<   FileUtils::getFile(s3).compare("aa") << endl;
	cout << "File s3_1:" << FileUtils::getFile(s3_1).compare("aa") << endl;
	cout << "File s3_2:" << FileUtils::getFile(s3_2).compare("") << endl;
	cout << "File s4:" <<   FileUtils::getFile(s4).compare("bb") << endl;
	cout << "File s5:" <<   FileUtils::getFile(s5).compare("") << endl;
	cout << "File s6:" <<   FileUtils::getFile(s6).compare("bb") << endl;
	cout << "File s7:" <<   FileUtils::getFile(s7).compare("") << endl;
	
	cout << "In Dir " << FileUtils::inDir(s1, s2) << " gold = 0" << endl;
	cout << "In Dir " << FileUtils::inDir(s7, s6) << " gold = 1" << endl;
	
	
	string t2(".");
	string t3("cc");
	string t3_1(".cc");
	string t3_2("cc.");
	string t4("aa.cc");
	string t5("aa.cc.");
	string t6(".aa.cc");
	string t7(".aa.cc.");
		
	cout << "Ext t1:" <<   FileUtils::getExt(s1).compare("") << endl;
	cout << "Ext s2:" <<   FileUtils::getExt(t2).compare("") << endl;
	cout << "Ext s3:" <<   FileUtils::getExt(t3).compare("") << endl;
	cout << "Ext s3_1:" << FileUtils::getExt(t3_1).compare("cc") << endl;
	cout << "Ext s3_2:" << FileUtils::getExt(t3_2).compare("") << endl;
	cout << "Ext s4:" <<   FileUtils::getExt(t4).compare("cc") << endl;
	cout << "Ext s5:" <<   FileUtils::getExt(t5).compare("") << endl;
	cout << "Ext s6:" <<   FileUtils::getExt(t6).compare("cc") << endl;
	cout << "Ext s7:" <<   FileUtils::getExt(t7).compare("") << endl;

	cout << "Ext compare " << FileUtils::hasExt(t3_1, "cc") << " gold = 1 " << endl;
	cout << "Ext compare " << FileUtils::hasExt(t3_1, "aa") << " gold = 0 " << endl;
		

	printf("no filter:\n");
	testTraverseDirectory(string("blah"), fu1);
	testTraverseDirectory(string("CMakeCache.txt"), fu1);
	testTraverseDirectory(string("CMakeFiles"), fu1);

	printf("filter by extension .tif:\n");
	testTraverseDirectory(string("."), fu2);

	printf("filter by extension .blah and .txt:\n");
	testTraverseDirectory(string("."), fu3);

	printf("filter by extension .cmake and .txt:\n");
	testTraverseDirectory(string("."), fu5);


}
