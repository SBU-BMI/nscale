#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <iostream>
#include <stdio.h>
#include <vector>
#include "Logger.h"
#include "FileUtils.h"
#include <dirent.h>
#include "UtilsLogger.h"
#include "UtilsCVImageIO.h"
#include <algorithm>    // std::min

using namespace cv;

#define SSTR( x ) dynamic_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()


#define	COORDINATE_SIZE	10
#define TILE_SIZE	4096
void getFiles(const std::string &imageName, std::vector<std::string> &filenames, std::vector< std::pair<int,int> > &coordinates);



void getFiles(const std::string &imageName, std::vector<std::string> &filenames, std::vector< std::pair<int,int> > &coordinates) {
	// check to see if it's a directory or a file
	std::vector<std::string> exts;
	exts.push_back(std::string(".tif"));
	exts.push_back(std::string(".tiff"));

	cci::common::FileUtils futils(exts);
	futils.traverseDirectory(imageName, filenames, cci::common::FileUtils::FILE, true);

	std::string temp, x, y;
	for(unsigned int i = 0; i < filenames.size(); i++){
		temp = futils.replaceExt(filenames[i], "");
		unsigned found = temp.find_last_of("-");
		
		x = temp.substr(found-COORDINATE_SIZE, COORDINATE_SIZE);
		y = temp.substr(found+1, COORDINATE_SIZE);

		std::pair<int,int> coordinate = std::make_pair(atoi(x.c_str()), atoi(y.c_str()));
		coordinates.push_back(coordinate);
//		std::cout <<  temp << " x:"<< x << " y:" << y << " found:" << found <<std::endl;
	}
}

int main (int argc, char **argv){
	std::vector<std::string> filenames;
	std::vector< std::pair<int,int> > coordinates;

	if(argc != 4){
		std::cout << "Usage: ./tiler.exe inDir outDir tileSize" << std::endl;
		exit(1); 
	}
	// get parameters
     	std::string imageName = argv[1];
	std::string outDir = argv[2];
	int tileSize = atoi(argv[3]);

	// make sure to create output dir
	cci::common::FileUtils::mkdirs(outDir);

    	uint64_t t0 = 0, t1 = 0;

    	t0 = cci::common::event::timestampInUS();
    	getFiles(imageName, filenames, coordinates);
	assert(filenames.size() == coordinates.size());

    	t1 = cci::common::event::timestampInUS();
    	std::cout << "file name read took "  << t1-t0 << " us." << std::endl;
	std::cout << " NFiles: "<< filenames.size() << " #coordinates:" << coordinates.size()  << std::endl;
	int max_x=0, max_y=0;
	for(int i = 0; i < coordinates.size(); i++){
//		std::cout << filenames[i] << " x:" << coordinates[i].first << " y:" << coordinates[i].second <<std::endl;
		if(coordinates[i].first > max_x)
			max_x = coordinates[i].first;

		if(coordinates[i].second > max_y)
			max_y = coordinates[i].second;
	}
	std::cout << "max_x:"<< max_x << " max_y:"<< max_y << std::endl;
	cv::Mat image(max_y+TILE_SIZE, max_x+TILE_SIZE, CV_8UC3);
	image.setTo(cv::Scalar(256,256,256));

	std::cout << "rebuilding image" << std::endl; 
	for(int i = 0; i < coordinates.size(); i++){
		cv::Mat tile = cv::imread(filenames[i], -1);
//		std::cout << "rows:"<<tile.rows << " cols:"<< tile.cols<< " channels:"<< tile.channels() <<std::endl;
		cv::Mat roi (image, Rect(coordinates[i].first, coordinates[i].second, TILE_SIZE, TILE_SIZE));
		tile.copyTo(roi);

		tile.release();
	}

	std::cout << "end rebuilding image" << std::endl << "Starting retiling phase"<< std::endl ; 


	for(int x = 0; x < image.cols; x+=tileSize){
		for(int y = 0; y < image.rows; y+=tileSize){
			std::string tileName = outDir;
			tileName.append("/");
			tileName.append(SSTR(x));
			tileName.append("-");
			tileName.append(SSTR(y));
			tileName.append(".tiff");
			cv::Mat roi(image, Rect(x, y, min(tileSize,image.cols-x), min(tileSize,image.rows-y)));
			std::cout << "roi name: "<< tileName << std::endl;
			imwrite(tileName, roi);
		}
	}

//	imwrite("image.jpg", image);	
	image.release();
    	return 0;
}
