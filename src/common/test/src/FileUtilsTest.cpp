/*
 * cci::common::FileUtils_test.cpp
 *
 *  Created on: Jul 11, 2012
 *      Author: tcpan
 */

#include "FileUtils.h"
#include "Logger.h"
#include <string>
#include <iostream>
#include <iterator>
#include <vector>
#include <iostream>
#include <sys/stat.h>

#ifdef _MSC_VER
#include "direntWin.h"
#define long long long
#else
#include <dirent.h>
#endif

#include <stdio.h>


void testTraverseDirectory(const std::string &d1, cci::common::FileUtils &fu) {
   // time to open a directory vs stat
	printf("PROCESSING %s\n", d1.c_str());

	long c1 = cci::common::event::timestampInUS();
	struct stat st_buf;
	int status = stat (d1.c_str(), &st_buf);
	if (status != 0) {
		printf("ERROR: unable to inspect %s\n", d1.c_str());
	} else {
		if (S_ISDIR (st_buf.st_mode)) printf("is a directory. ");
	}
	long c2 = cci::common::event::timestampInUS();
	printf("using stat took %ld time\n", c2-c1);

	c1 = cci::common::event::timestampInUS();
	DIR *dir = opendir(d1.c_str());
	if (dir != NULL) printf("is a directory too. ");
	c2 = cci::common::event::timestampInUS();
	printf("using opendir took %ld time\n", c2-c1);

	std::vector<std::string> v1;
	std::ostream_iterator<std::string> out_it (std::cout,"\n");

	std::cout << "TRAVERSE DIR, DIRECTORY only, no recursion : ";
	fu.traverseDirectory(d1, v1, cci::common::FileUtils::getDIRECTORY(), false);
//	copy ( v1.begin(), v1.end(), out_it );
	std::cout << v1.size() << " entries" << std::endl;
	v1.clear();

//	// tested,  new works the same.
//	std::cout << "OLD TRAVERSE DIR, DIR only, no recursion : ";
//	fu.getDirectoriesInDirectory(d1, v1);
////	copy ( v1.begin(), v1.end(), out_it );
//	std::cout << v1.size() << " entries" << std::endl;
//	v1.clear();


	std::cout << "TRAVERSE DIR, FILE only, no recursion : ";
	fu.traverseDirectory(d1, v1, cci::common::FileUtils::getFILE(), false);
	//copy ( v1.begin(), v1.end(), out_it );
	std::cout << v1.size() << " entries" << std::endl;
	v1.clear();

//	// tested.  new works the same
//	std::cout << "OLD TRAVERSE DIR, FILE only, no recursion : ";
//	fu.getFilesInDirectory(d1, v1);
////	copy ( v1.begin(), v1.end(), out_it );
//	std::cout << v1.size() << " entries" << std::endl;
//	v1.clear();



	std::cout << "TRAVERSE DIR, BOTH, no recursion : ";
	fu.traverseDirectory(d1, v1, cci::common::FileUtils::getFILE() | cci::common::FileUtils::getDIRECTORY(), false);
//	copy ( v1.begin(), v1.end(), out_it );
	std::cout << v1.size() << " entries" << std::endl;
	v1.clear();

	std::cout << "TRAVERSE DIR, DIRECTORY only, recursion : ";
	fu.traverseDirectory(d1, v1, cci::common::FileUtils::getDIRECTORY(), true);
//	copy ( v1.begin(), v1.end(), out_it );
	std::cout << v1.size() << " entries" << std::endl;
	v1.clear();

	std::cout << "TRAVERSE DIR, FILE only, recursion : ";
	fu.traverseDirectory(d1, v1, cci::common::FileUtils::getFILE(), true);
//	copy ( v1.begin(), v1.end(), out_it );
	std::cout << v1.size() << " entries" << std::endl;
	v1.clear();

//	// tested. new works the same
//	std::cout << "OLD TRAVERSE DIR, FILE only, recursion : ";
//	fu.traverseDirectoryRecursive(d1, v1);
////	copy ( v1.begin(), v1.end(), out_it );
//	std::cout << v1.size() << " entries" << std::endl;
//	v1.clear();


	std::cout << "TRAVERSE DIR, BOTH, recursion : ";
	fu.traverseDirectory(d1, v1, cci::common::FileUtils::getFILE() | cci::common::FileUtils::getDIRECTORY(), true);
//	copy ( v1.begin(), v1.end(), out_it );
	std::cout << v1.size() << " entries" << std::endl;


}


int main (int argc, char **argv){
	
	cci::common::FileUtils fu1;
	cci::common::FileUtils fu2(".tif");
		
	std::vector<std::string> exts;
	exts.push_back(".blah");
	exts.push_back(".txt");
	::cci::common::FileUtils fu3(exts);
	
	std::vector<std::string> exts2;
	exts2.push_back(".cmake");
	exts2.push_back(".txt");
	::cci::common::FileUtils fu5(exts2);
	
	
// testing the get part
	std::string s1("");
	
	std::string s2("/");
	std::string s3("aa");
	std::string s3_1("/aa");
	std::string s3_2("aa/");
	std::string s4("aa/bb");
	std::string s5("aa/bb/");
	std::string s6("/aa/bb");
	std::string s7("/aa/bb/");
	

	std::cout << "Dir s1:" <<    cci::common::FileUtils::getDir(s1).compare("") << std::endl;
	std::cout << "Dir s2:" <<    cci::common::FileUtils::getDir(s2).compare("/") << std::endl;
	std::cout << "Dir s3:" <<    cci::common::FileUtils::getDir(s3).compare("") << std::endl;
	std::cout << "Dir s3_1:" <<  cci::common::FileUtils::getDir(s3_1).compare("/") << std::endl;
	std::cout << "Dir s3_2:" <<  cci::common::FileUtils::getDir(s3_2).compare("aa") << std::endl;
	std::cout << "Dir s4:" <<    cci::common::FileUtils::getDir(s4).compare("aa") << std::endl;
	std::cout << "Dir s5:" <<    cci::common::FileUtils::getDir(s5).compare("aa/bb") << std::endl;
	std::cout << "Dir s6:" <<    cci::common::FileUtils::getDir(s6).compare("/aa") << std::endl;
	std::cout << "Dir s7:" <<    cci::common::FileUtils::getDir(s7).compare("/aa/bb") << std::endl;
	
	std::cout << "File s1:" <<   cci::common::FileUtils::getFile(s1).compare("") << std::endl;
	std::cout << "File s2:" <<   cci::common::FileUtils::getFile(s2).compare("") << std::endl;
	std::cout << "File s3:" <<   cci::common::FileUtils::getFile(s3).compare("aa") << std::endl;
	std::cout << "File s3_1:" << cci::common::FileUtils::getFile(s3_1).compare("aa") << std::endl;
	std::cout << "File s3_2:" << cci::common::FileUtils::getFile(s3_2).compare("") << std::endl;
	std::cout << "File s4:" <<   cci::common::FileUtils::getFile(s4).compare("bb") << std::endl;
	std::cout << "File s5:" <<   cci::common::FileUtils::getFile(s5).compare("") << std::endl;
	std::cout << "File s6:" <<   cci::common::FileUtils::getFile(s6).compare("bb") << std::endl;
	std::cout << "File s7:" <<   cci::common::FileUtils::getFile(s7).compare("") << std::endl;
	
	std::cout << "In Dir " << cci::common::FileUtils::inDir(s1, s2) << " gold = 0" << std::endl;
	std::cout << "In Dir " << cci::common::FileUtils::inDir(s7, s6) << " gold = 1" << std::endl;
	
	
	std::string t2(".");
	std::string t3("cc");
	std::string t3_1(".cc");
	std::string t3_2("cc.");
	std::string t4("aa.cc");
	std::string t5("aa.cc.");
	std::string t6(".aa.cc");
	std::string t7(".aa.cc.");
		
	std::cout << "Ext t1:" <<   cci::common::FileUtils::getExt(s1).compare("") << std::endl;
	std::cout << "Ext s2:" <<   cci::common::FileUtils::getExt(t2).compare("") << std::endl;
	std::cout << "Ext s3:" <<   cci::common::FileUtils::getExt(t3).compare("") << std::endl;
	std::cout << "Ext s3_1:" << cci::common::FileUtils::getExt(t3_1).compare("cc") << std::endl;
	std::cout << "Ext s3_2:" << cci::common::FileUtils::getExt(t3_2).compare("") << std::endl;
	std::cout << "Ext s4:" <<   cci::common::FileUtils::getExt(t4).compare("cc") << std::endl;
	std::cout << "Ext s5:" <<   cci::common::FileUtils::getExt(t5).compare("") << std::endl;
	std::cout << "Ext s6:" <<   cci::common::FileUtils::getExt(t6).compare("cc") << std::endl;
	std::cout << "Ext s7:" <<   cci::common::FileUtils::getExt(t7).compare("") << std::endl;

	std::cout << "Ext compare " << cci::common::FileUtils::hasExt(t3_1, "cc") << " gold = 1 " << std::endl;
	std::cout << "Ext compare " << cci::common::FileUtils::hasExt(t3_1, "aa") << " gold = 0 " << std::endl;
		

	printf("no filter:\n");
	testTraverseDirectory(std::string("blah"), fu1);
	testTraverseDirectory(std::string("CMakeCache.txt"), fu1);
	testTraverseDirectory(std::string("CMakeFiles"), fu1);

	printf("filter by extension .tif:\n");
	testTraverseDirectory(std::string("."), fu2);

	printf("filter by extension .blah and .txt:\n");
	testTraverseDirectory(std::string("."), fu3);

	printf("filter by extension .cmake and .txt:\n");
	testTraverseDirectory(std::string("."), fu5);


}
