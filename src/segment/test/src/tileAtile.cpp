

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <iostream>
#include <stdio.h>
#include <vector>
#include "Logger.h"
#include "FileUtils.h"
#ifdef _MSC_VER
#include "direntWin.h"
#else
#include <dirent.h>
#endif
#include "UtilsLogger.h"
#include "UtilsCVImageIO.h"
#include <algorithm>    // std::min

using namespace cv;

#define SSTR( x ) dynamic_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()


void getFiles(const std::string &imageName, std::vector<std::string> &filenames, std::vector< std::pair<int,int> > &coordinates);



void getFiles(const std::string &imageName, std::vector<std::string> &filenames) {
	// check to see if it's a directory or a file
	std::vector<std::string> exts;
	exts.push_back(std::string(".tif"));
	exts.push_back(std::string(".tiff"));

	cci::common::FileUtils futils(exts);
	futils.traverseDirectory(imageName, filenames, cci::common::FileUtils::getFILE(), true);
}

int main (int argc, char **argv){
	std::vector<std::string> filenames;

	if(argc != 4){
		std::cout << "Usage: ./tileAtile.exe inDir outDir tileSize" << std::endl;
		exit(1); 
	}
	// get parameters
     	std::string imageName = argv[1];
	std::string outDir = argv[2];
	int tileSize = atoi(argv[3]);

	// make sure to create output dir
	cci::common::FileUtils::mkdirs(outDir);

    	uint64_t t0 = 0, t1 = 0;

	// get file names in directory
    	t0 = cci::common::event::timestampInUS();
    	getFiles(imageName, filenames);

    	t1 = cci::common::event::timestampInUS();
    	std::cout << "file name read took "  << t1-t0 << " us." << std::endl;
	std::cout << " NFiles: "<< filenames.size() << std::endl;

	// extensions used by input file name
	std::vector<std::string> exts;
	exts.push_back(std::string(".tif"));
	exts.push_back(std::string(".tiff"));
//	exts.push_back(std::string(".png"));
	// initialize helper objects with extensions to be processed
	cci::common::FileUtils futils(exts);

	// process each input image file
	for(int i = 0; i < filenames.size(); i++){
		// remove extension from file name
		std::string temp = futils.replaceExt(filenames[i], "");
		// remove subdirectories part of the name
		unsigned int found = temp.find_last_of("/");
		temp = temp.substr(found, temp.size());
		std::string newFileSuffix = outDir;
		newFileSuffix.append(temp);

		// read input tile
		cv::Mat tile = cv::imread(filenames[i], -1);

		// iterate in each subtile, creating a name for it and writting it to file
		for(int x = 0; x < tile.cols; x+=tileSize){
			for(int y = 0; y < tile.rows; y+=tileSize){
				std::string tileName = newFileSuffix;
				tileName.append("-");
				tileName.append(SSTR(x));
				tileName.append("-");
				tileName.append(SSTR(y));
				tileName.append(".tif");
				cv::Mat roi(tile, Rect(x, y, min(tileSize,tile.cols-x), min(tileSize,tile.rows-y)));
				std::cout << "roi name: "<< tileName << std::endl;
				imwrite(tileName, roi);
			}
		}

		tile.release();
	}
    	return 0;
}
