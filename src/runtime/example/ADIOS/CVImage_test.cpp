/*
 * CVImagetest.cpp
 *
 *  Created on: Jul 9, 2012
 *      Author: tcpan
 */

#include "CVImage.h"
#include <string>
#include "FileUtils.h"

using namespace std;

int main (int argc, char **argv){

	char const *input = "/home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1/astroII.1.ndpi-0000028672-0000012288.tif";

	std::string fn = std::string(input);
	//Debug::print("%s READING %s\n", getClassName(), fn.c_str());


	// parse the input string
	FileUtils futils;
	std::string filename = futils.getFile(const_cast<std::string&>(fn));
	// get the image name
	size_t pos = filename.rfind('.');
	if (pos == std::string::npos) printf("ERROR:  file %s does not have extension\n", fn.c_str());
	string prefix = filename.substr(0, pos);
	pos = prefix.rfind("-");
	if (pos == std::string::npos) printf("ERROR:  file %s does not have a properly formed x, y coords\n", fn.c_str());
	string ystr = prefix.substr(pos + 1);
	prefix = prefix.substr(0, pos);
	pos = prefix.rfind("-");
	if (pos == std::string::npos) printf("ERROR:  file %s does not have a properly formed x, y coords\n", fn.c_str());
	string xstr = prefix.substr(pos + 1);

	string imagename = prefix.substr(0, pos);
	int tilex = atoi(xstr.c_str());
	int tiley = atoi(ystr.c_str());



	cv::Mat im = cv::imread(fn, -1);
	int dummy;
	cci::rt::adios::CVImage *img = new cci::rt::adios::CVImage(im, imagename, fn, tilex, tiley);
	printf("image name %s, filename %s, data size %d\n", img->getImageName(dummy), img->getSourceFileName(dummy), img->getMetadata().info.data_size);

	void *output;
	int output_size;

	img->serialize(output_size, output);

	// clean up
	delete img;
	im.release();


	cci::rt::adios::CVImage *img2 = new cci::rt::adios::CVImage(output_size, output);
	printf("image name %s, filename %s, data size %d\n", img2->getImageName(dummy), img2->getSourceFileName(dummy), img2->getMetadata().info.data_size);

	cci::rt::adios::CVImage *img3 = new cci::rt::adios::CVImage();
	img3->copy(*img2);
	delete img2;
	//	img3->deserialize(output_size, output);
	printf("image name %s, filename %s, data size %d\n", img3->getImageName(dummy), img3->getSourceFileName(dummy), img3->getMetadata().info.data_size);

	delete img3;

	return 0;


}
