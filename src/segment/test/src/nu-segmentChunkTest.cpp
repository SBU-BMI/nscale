/*
 * test.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */

#include "opencv2/opencv.hpp"
#ifdef WITH_CUDA
#include "opencv2/gpu/gpu.hpp"
#endif
#include <iostream>
#ifdef _MSC_VER
#include "direntWin.h"
#else
#include <dirent.h>
#endif
#include <vector>
#include "HistologicalEntities.h"
#include "MorphologicOperations.h"
#include <time.h>
#include "Logger.h"
#include "FileUtils.h"


using namespace cv;
/*
Mat computeGroundtruth(const Mat & img, cciutils::SimpleCSVLogger& logger, const int & modecode) {
	Mat out3(img.size(), CV_8U);
	int status;
	switch (modecode) {
	case cci::common::type::DEVICE_CPU :
	case cci::common::type::DEVICE_MCORE :

		status = nscale::HistologicalEntities::segmentNuclei(img, out3, &logger, 25);
		break;
	case cci::common::type::DEVICE_GPU :
		status = nscale::gpu::HistologicalEntities::segmentNuclei(img, out3, &logger, 25);
		break;
	default :
		break;
	}


	Mat groundtruth = Mat::zeros(img.size(), CV_8UC3);
	cvtColor(out3, groundtruth, CV_GRAY2BGR);
	return groundtruth;
}


Mat render(const std::vector<Mat> & chunks, const std::vector<Mat> & outputs, const Size & s, const int & stage, const int & b, const Mat& gridtruth)
{
	std::vector<Mat>::const_iterator it = chunks.begin();
	std::vector<Mat>::const_iterator last = chunks.end();
    std::vector<Mat>::const_iterator oit = outputs.begin();
    std::vector<Mat>::const_iterator olast = outputs.end();
    std::vector<std::vector<Point> > contours;
    std::vector<Vec4i> hierarchy;

    Mat nu, nu2, nu3, chunk, chunkout;
    Size s2;
    Point ofs;
    Mat outputROI;
    int startx, startx2, starty, starty2, endx, endx2, endy, endy2;
    Rect bbox, inside, normal, outside;
	Scalar redc, greenc, bluec, green2c;
	redc = Scalar(0, 0, 255);
	greenc = Scalar(0, 255, 0);
	green2c = Scalar(0, 96, 0);
	bluec = Scalar(255, 0, 0);

    Mat output = Mat::zeros(s, CV_8UC3);
    //logger.consoleOn();
    for ( ; it < last && oit < olast; ++it, ++oit) {
		contours.clear();
		hierarchy.clear();

		nu = *oit;
		chunk = *it;
		chunk.locateROI(s2, ofs);
		s2 = chunk.size();


		// FOR testing stages.
		if (stage > -1 && stage < 25) {
			outputROI = output(Rect(ofs.x, ofs.y, s2.width, s2.height));
			if (nu.channels() == 1) {
				nu.convertTo(nu2, CV_8U);
				cvtColor(nu2, nu3, CV_GRAY2BGR);
			} else {
				nu3 = nu;
			}
			nu3.copyTo(outputROI);
			continue;
		}


		// need to pad before finding contours?
		copyMakeBorder(nu, nu2, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));

		findContours(nu2, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(-1, -1));
		//logger.log("contour size ", contours.size());
		if (contours.size() > 0) {


			if (ofs.x == 0) {
				startx = 0;
				startx2 = startx;
			} else {
				startx = 2*b;
				startx2 = b;
			}
			if (ofs.y == 0) {
				starty = 0;
				starty2 = starty;
			} else {
				starty = 2*b;
				starty2 = b;
			}
			if (ofs.x + s2.width >= s.width) {
				endx = s.width;
				endx2 = endx;
			} else {
				endx = s2.width - 2*b;
				endx2 = s2.width - b;
			}
			if (ofs.y + s2.height >= s.height) {
				endy = s.height;
				endy2 = endy;
			} else {
				endy = s2.height - 2*b;
				endy2 = s2.height - b;
			}
			inside = Rect(startx, starty, endx-startx, endy - starty);
			normal = Rect(startx2, starty2, endx2 - startx2, endy2 - starty2);
			outside = Rect(0, 0, s2.width, s2.height);

			chunkout = Mat::zeros(s2, CV_8UC3);
			for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
				bbox = boundingRect(contours[idx]);
//						std::cout << "bbox: " << bbox.x << "," << bbox.y << " - " << (bbox.x + bbox.width) << "," << (bbox.y + bbox.height) << std::endl;
//						std::cout << "inside: " << inside.x << "," << inside.y << " - " << (inside.x + inside.width) << "," << (inside.y + inside.height) << std::endl;
//						std::cout << "outside: " << outside.x << "," << outside.y << " - " << (outside.x + outside.width) << "," << (outside.y + outside.height) << std::endl;
//						std::cout << "chunksize: " << s2.width << "," << s2.height << " outsize " << nu.rows << "," << nu.cols << std::endl;

				if ((bbox.x <= outside.x) ||
						(bbox.y <= outside.y) ||
						(bbox.x + bbox.width >= outside.x + outside.width) ||
						(bbox.y + bbox.height >= outside.y + outside.height)) {
//								logger.log("on padded boundary ", idx);
					drawContours( chunkout, contours, idx, redc, 1, 8, hierarchy);

				} else if ((bbox.x <= normal.x) ||
						(bbox.y <= normal.y) ||
						(bbox.x + bbox.width >= normal.x + normal.width) ||
						(bbox.y + bbox.height >= normal.y + normal.height)) {
//								logger.log("on normal boundary ", idx);
					drawContours( chunkout, contours, idx, greenc, 1, 8, hierarchy);

				} else if ((bbox.x <= inside.x) ||
						(bbox.y <= inside.y) ||
						(bbox.x + bbox.width >= inside.x + inside.width) ||
						(bbox.y + bbox.height >= inside.y + inside.height)) {
//								logger.log("on neighbor's boundary ", idx);
					drawContours( chunkout, contours, idx, bluec, 1, 8, hierarchy);

				// on the outside boundary - color it red
				} else  {
					// inside
					//logger.log("in padding ", idx);

					drawContours( chunkout, contours, idx, green2c, CV_FILLED, 8, hierarchy);
				}
			}

			// now paste it in.
			outputROI = output(Rect(ofs.x, ofs.y, s2.width, s2.height));

			addWeighted(outputROI, 1.0, chunkout, 1.0, 0, outputROI);
			//chunkout.copyTo(outputROI);
		}
	}
    //logger.consoleOff();
    addWeighted(gridtruth, 0.3, output, 1.0, 0.0, output);
    return output;
}


void write(const int & w, const int & b, const int & stage, const char *& runid, const char *& mode, const Mat & output)
{
	char prefix[80];

    char ws[10];
    sprintf(ws, "%d", w);
    char bs[10];
    sprintf(bs, "%d", b);
    char ss[10];
    sprintf(ss, "%d", stage);
    if(stage > -1 && stage < 25){
        strcpy(prefix, "test/Stages/out-segment-");
    }else{
        strcpy(prefix, "test/Segmented/out-segment-");
    }
    strcat(prefix, runid);
    strcat(prefix, "-");
    strcat(prefix, mode);
    strcat(prefix, "-chunk-");
    strcat(prefix, ws);
    strcat(prefix, "-border-");
    strcat(prefix, bs);
    strcat(prefix, "-stage-");
    strcat(prefix, ss);
    strcat(prefix, ".pgm");
    imwrite(prefix, output);
}

Mat renderGrid(const std::vector<Mat> & chunks, const int & b, const Mat & groundtruth)
{
	int startx, endx, starty, endy;
	std::vector<Mat>::const_iterator it = chunks.begin();
	std::vector<Mat>::const_iterator last = chunks.end();
	Size s2;
	Point ofs;
	Scalar grayc = Scalar(127, 127, 127);
	Scalar lightredc = Scalar(255, 255, 255);
	std::vector<Point> box;
	std::vector<std::vector<Point> > contours;
	Mat chunk;
	Mat gridtruth(groundtruth.size(), groundtruth.type());
	Size s = groundtruth.size();

    for(;it < last;++it){
    	chunk = *it;

        chunk.locateROI(s2, ofs);
        s2 = chunk.size();
        box.push_back(Point(ofs.x, ofs.y));
        box.push_back(Point(ofs.x, ofs.y + s2.height - 1));
        box.push_back(Point(ofs.x + s2.width - 1, ofs.y + s2.height - 1));
        box.push_back(Point(ofs.x + s2.width - 1, ofs.y));
        contours.push_back(box);
        drawContours(gridtruth, contours, 0, grayc, 1, 8);
        contours.clear();
        box.clear();
        if(ofs.x == 0){
            startx = 0;
        }else{
            startx = ofs.x + b;
        }
        if(ofs.y == 0){
            starty = 0;
        }else{
            starty = ofs.y + b;
        }
        if(ofs.x + s2.width >= s.width){
            endx = s.width - 1;
        }else{
            endx = ofs.x + s2.width - b - 1;
        }
        if(ofs.y + s2.height >= s.height){
            endy = s.height - 1;
        }else{
            endy = ofs.y + s2.height - b - 1;
        }
        box.push_back(Point(startx, starty));
        box.push_back(Point(startx, endy));
        box.push_back(Point(endx, endy));
        box.push_back(Point(endx, starty));
        contours.push_back(box);
        drawContours(gridtruth, contours, 0, lightredc, 1, 8);
        contours.clear();
        box.clear();
    }

    addWeighted(groundtruth, 0.5, gridtruth, 1.0, 0., gridtruth);
    return gridtruth;
}
*/
int main (int argc, char **argv){

	printf("TODO: his code needs to be updated so that the chunks can be stitched together appropriately\n");
	return -1;

/*
	bool timingOnly = true;

	if (argc < 8) {
		std::cout << "Usage:  " << argv[0] << " image_filename " << "run-id minw minb maxb minstage maxstage [cpu | mcore | gpu [id]]" << std::endl;
		return -1;
	}
	string imagename(argv[1]);
	const char* runid = argv[2];
	int minw = atoi(argv[3]);
	int minb = atoi(argv[4]);
	int maxb = atoi(argv[5]);
	int minstage = atoi(argv[6]);
	int maxstage = atoi(argv[7]);

	const char* mode = argc > 8 ? argv[8] : "cpu";

	int modecode = 0;
	if (strcasecmp(mode, "cpu") == 0) modecode = cci::common::type::DEVICE_CPU;
	else if (strcasecmp(mode, "mcore") == 0) {
		modecode = cci::common::type::DEVICE_MCORE;
		// get core count
	} else if (strcasecmp(mode, "gpu") == 0) {
		modecode = cci::common::type::DEVICE_GPU;
		// get device count
		int numGPU = gpu::getCudaEnabledDeviceCount();
		if (numGPU < 1) {
			std::cout << "gpu requested, but no gpu available.  please use cpu or mcore option."  << std::endl;
			return -2;
		}
		if (argc > 9) {
			gpu::setDevice(atoi(argv[9]));
		}
		std::cout << " number of cuda enabled devices = " << gpu::getCudaEnabledDeviceCount() << std::endl;
	} else {
		std::cout << "Usage:  " << argv[0] << " image_filename " << "run-id minw minb maxb minstage maxstage [cpu | mcore | gpu [id]]" << std::endl;
		return -1;
	}


	std::vector<Mat>::const_iterator it;
	std::vector<Mat>::const_iterator last;

	// need to go through filesystem
	char prefix[80];
	strcpy(prefix, "results/");
	strcat(prefix, mode);
	strcat(prefix, "-per-chunk");
	cciutils::SimpleCSVLogger logger(prefix);
	logger.consoleOff();
	//logger.off();

	// check to see if it's a directory or a file
    DIR *dir;
    std::vector<std::string> filenames;
    std::string suffix;
    suffix.assign(".tif");

	FileUtils traveler(suffix);
	traveler.traverseDirectoryRecursive(imagename, filenames);
    std::vector<std::string>::iterator niter = filenames.begin();
    std::vector<std::string>::iterator nlast = filenames.end();

    niter = filenames.begin();
    nlast = filenames.end();


    for (; niter < nlast; ++niter) {


	Mat img = imread(*niter);
	if (!img.data) return -1;

	logger.log("run-id", runid);
	logger.log("filename", *niter);
	logger.log("w", 4096);
	logger.log("b", 0);
//	uint64_t t1 = cci::common::event::timestampInUS();
//	logger.log("time", cci::common::event::timestampInUS());
	logger.log("type", "cpu");
	logger.log("chunk x", 0);
	logger.log("chunk y", 0);


	Mat groundtruth = computeGroundtruth(img, logger, modecode);

	logger.endSession();
	std::cout << "groundtruth segmentation completed" << std::endl;

	Size s = img.size();
	std::vector<Mat> chunks;
	std::vector<Mat> outputs;
	Mat chunk;
	Range rx;
	Range ry;
	Size s2;
	Point ofs;
	int status;
	uint64_t t1, t2;


	int b;
	Mat gridtruth;
	int stage = 25;
	for (int w = minw; w < s.width && w < s.height; w = w* 2) {
		for (int b2 = minb; b2 > 0 && b2 <= maxb && b2 <= (w / 2); b2 = b2 * 2) {
			b = b2 / 2;

			// break it apart
			gridtruth = Mat::zeros(s, CV_8UC3);
			chunks.clear();

			for (int i = 0; i < s.width; i += w) {
				for (int j = 0; j < s.height; j+= w) {
					rx = Range((i-b > 0 ? i-b : 0), (i+w+b < s.width ? i+w+b : s.width));
					ry = Range((j-b > 0 ? j-b : 0), (j+w+b < s.height ? j+w+b : s.height));
//					std::cout << "ranges: " << rx.start << "-" << rx.end << ", " << ry.start << "-" << ry.end << std::endl;
					chunks.push_back(img(rx, ry));

				}
			}

			if (!timingOnly) gridtruth = renderGrid(chunks, b, groundtruth);

			//			std::cout << "groundtruth rendered" << std::endl;
			for (int stage = minstage; stage <= maxstage; ++stage) {
	
				std::cout << "***** mode = " << mode << " w = " << w << ", b = " << b << ", stage= " << stage << std::endl;

				outputs.clear();


				it = chunks.begin();
				last = chunks.end();
				t1 = cci::common::event::timestampInUS();
				switch (modecode) {
				case cci::common::type::DEVICE_CPU :
				case cci::common::type::DEVICE_MCORE :
	//				logger.consoleOn();
					for (; it < last; ++it) {
						logger.log("run-id", runid);
						logger.log("filename", *niter);
						logger.log("w", w);
						logger.log("b", b);
						logger.log("time", cci::common::event::timestampInUS());
						logger.log("type", "cpu");

						chunk = *it;
						chunk.locateROI(s2, ofs);
						logger.log("chunk x", ofs.x);
						logger.log("chunk y", ofs.y);
						Mat out(chunk.size(), CV_8U, Scalar(0));
//						std::cout << " segmenting cpu chunk size: " << chunk.size().width << "x" << chunk.size().height << std::endl;
						status = nscale::HistologicalEntities::segmentNuclei(chunk, out, &logger, stage);
						outputs.push_back(out);
						logger.endSession();
					}
	//				logger.consoleOff();
					break;
				case cci::common::type::DEVICE_GPU :
					for (; it < last; ++it) {
						logger.log("run-id", runid);
						logger.log("filename", *niter);
						logger.log("w", w);
						logger.log("b", b);
						logger.log("time", cci::common::event::timestampInUS());
						logger.log("type", "gpu");

						chunk = *it;
						chunk.locateROI(s2, ofs);
						logger.log("chunk x", ofs.x);
						logger.log("chunk y", ofs.y);
						Mat out(chunk.size(), CV_8U, Scalar(0));
//						std::cout << " segmenting cpu chunk size: " << chunk.size().width << "x" << chunk.size().height << std::endl;
						status = nscale::gpu::HistologicalEntities::segmentNuclei(chunk, out, &logger, stage);
						outputs.push_back(out);
						logger.endSession();
					}
					break;
				default :
					break;
				}
				t2 = cci::common::event::timestampInUS();
				std::cout << "**** SEGMENTATION took " << t2-t1 << "ms" << std::endl;

				// concat.

				if (timingOnly) continue;

				Mat output = render(chunks, outputs, s, stage, b, gridtruth);

				write(w, b, stage, runid, mode, output);

//				if (w == s.width || w == s.height) break;

			}
}
	}
//	waitKey();
    }
	return 0;
	*/
}


