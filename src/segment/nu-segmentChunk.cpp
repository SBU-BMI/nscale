/*
 * test.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "highgui.h"
#include <iostream>
#include <dirent.h>
#include <vector>
#include <string>
#include "HistologicalEntities.h"
#include "MorphologicOperations.h"
#include <time.h>
#include "utils.h"


using namespace cv;

bool areaThreshold1(const std::vector<std::vector<Point> >& contours, const std::vector<Vec4i>& hierarchy, int idx) {
	return nscale::contourAreaFilter(contours, hierarchy, idx, 11, 1000);
}


int main (int argc, char **argv){
/*	// allow walk through of the directory
	const char* impath = argc > 1 ? argv[1];
	// get the files - from http://ubuntuforums.org/showthread.php?t=1409202
	vector<string> files();
	Dir *dir;
	struct dirent *dp;
	if ((dir = std::opendir(impath.c_str())) == NULL) {
		std::cout << "ERROR(" << errno << ") opening" << impath << std::endl;
		return errno;
	}
	while ((dp = readdir(dir)) != NULL) {
		files.push_back(string(dp->d_name));
		if ()
	}
	closedir(dir);


	// set the output path
	const char* resultpath = argc > 2 ? argv[2];
*/
	if (argc < 7) {
		std::cout << "Usage:  " << argv[0] << " image_filename " << "run-id minw maxb minstage maxstage [cpu | mcore | gpu [id]]" << std::endl;
		return -1;
	}
	const char* imagename = argv[1];
	const char* runid = argv[2];
	int minw = atoi(argv[3]);
	int maxb = atoi(argv[4]);
	int minstage = atoi(argv[5]);
	int maxstage = atoi(argv[6]);

	const char* mode = argc > 7 ? argv[7] : "cpu";

	int modecode = 0;
	if (strcasecmp(mode, "cpu") == 0) modecode = cciutils::DEVICE_CPU;
	else if (strcasecmp(mode, "mcore") == 0) {
		modecode = cciutils::DEVICE_MCORE;
		// get core count
	} else if (strcasecmp(mode, "gpu") == 0) {
		modecode = cciutils::DEVICE_GPU;
		// get device count
		int numGPU = gpu::getCudaEnabledDeviceCount();
		if (numGPU < 1) {
			std::cout << "gpu requested, but no gpu available.  please use cpu or mcore option."  << std::endl;
			return -2;
		}
		if (argc > 8) {
			gpu::setDevice(atoi(argv[8]));
		}
		std::cout << " number of cuda enabled devices = " << gpu::getCudaEnabledDeviceCount() << std::endl;
	} else {
		std::cout << "Usage:  " << argv[0] << " image_filename " << "run-id minw maxb minstage maxstage [cpu | mcore | gpu [id]]" << std::endl;
		return -1;
	}



	// need to go through filesystem
	char prefix[80];
	strcpy(prefix, "results/");
	strcat(prefix, mode);
	strcat(prefix, "-chunk");
	cciutils::SimpleCSVLogger logger(prefix);
	logger.consoleOff();
	logger.off();

	Mat img = imread(imagename);
	Size s = img.size();
	if (!img.data) return -1;
	Mat out3(img.size(), CV_8U);
	int status = nscale::HistologicalEntities::segmentNuclei(img, out3, logger, 25);
	Mat out4;
	cvtColor(out3, out4, CV_GRAY2BGR);
			Scalar grayc = Scalar(127, 127, 127);
			Scalar lightredc = Scalar(255, 255, 255);
			int startx, endx, starty, endy;
			int startx2, endx2, starty2, endy2;
			Mat target;
			Mat nu, nu2, nu3, chunkout, outputROI;
			Rect outside, inside, bbox, normal;
			Scalar redc, greenc, yellowc, bluec, cyanc, green2c;
			redc = Scalar(0, 0, 255);
			greenc = Scalar(0, 255, 0);
			green2c = Scalar(0, 96, 0);

			yellowc = Scalar(0, 255, 255);
			bluec = Scalar(255, 0, 0);
			cyanc = Scalar(255, 255, 0);
				std::vector<std::vector<Point> > contours;
				std::vector<Vec4i> hierarchy;  // 3rd entry in the vec is the child - holes.  1st entry in the vec is the next.

	int b;
	int stage = 25;
	for (int w = minw; w < s.width && w < s.height; w = w* 2) {
		for (int b2 = 1; b2 <= maxb && b2 <= (w / 2); b2 = b2 * 2) {

			// break it apart
			std::vector<Mat> chunks;
			Mat output = Mat::zeros(s, CV_8UC3);
			Mat output2 = Mat::zeros(s, CV_8UC3);

			Mat chunk;
			Range rx;
			Range ry;
			for (int i = 0; i < s.width; i += w) {
				for (int j = 0; j < s.height; j+= w) {
					rx = Range((i-b > 0 ? i-b : 0), (i+w+b < s.width ? i+w+b : s.width));
					ry = Range((j-b > 0 ? j-b : 0), (j+w+b < s.height ? j+w+b : s.height));
//					std::cout << "ranges: " << rx.start << "-" << rx.end << ", " << ry.start << "-" << ry.end << std::endl;
					chunks.push_back(img(rx, ry));
				}
			}
			Size s2;
			Point ofs;

				std::vector<Mat>::const_iterator it = chunks.begin();
				std::vector<Mat>::const_iterator last = chunks.end();

			for (; it < last; ++it) {
				chunk = *it;

				// draw the chunk boundary
				chunk.locateROI(s2, ofs);
				s2 = chunk.size();

				std::vector<Point> box;
				box.push_back(ofs);
				box.push_back(Point(ofs.x, ofs.y+s2.height-1));
				box.push_back(Point(ofs.x + s2.width-1, ofs.y+s2.height-1));
				box.push_back(Point(ofs.x + s2.width-1, ofs.y));
				contours.push_back(box);
				drawContours( output2, contours, 0, grayc, 1, 8);
				contours.clear();
				box.clear();
				if (ofs.x == 0) {
					startx = 0;
				} else {
					startx = ofs.x + b;
				}
				if (ofs.y == 0) {
					starty = 0;
				} else {
					starty = ofs.y + b;
				}
				if (ofs.x + s2.width >= s.width) {
					endx = s.width;
				} else {
					endx = ofs.x + s2.width - b;
				}
				if (ofs.y + s2.height >= s.height) {
					endy = s.height;
				} else {
					endy = ofs.y + s2.height - b;
				}
				box.push_back(Point(startx, starty));
				box.push_back(Point(startx, endy));
				box.push_back(Point(endx, endy));
				box.push_back(Point(endx, starty));
				contours.push_back(box);
				drawContours( output2, contours, 0, lightredc, 1, 8);
				contours.clear();
			}
			addWeighted(out4, 0.5, output2, 1.0, 0., output2);


			for (int stage = minstage; stage <= maxstage; ++stage) {
				b = b2 / 2;
	
				std::cout << "***** w = " << w << ", b = " << b << ", stage= " << stage << std::endl;


				logger.log("run-id", runid);
				logger.log("filename", imagename);
				logger.log("w", w);
				logger.log("b", b);
				uint64_t t1 = cciutils::ClockGetTime();
				logger.log("time", cciutils::ClockGetTime());


				Mat out2(img.size(), CV_8U);



				it = chunks.begin();
				last = chunks.end();

				std::vector<Mat> outputs;
				switch (modecode) {
				case cciutils::DEVICE_CPU :
				case cciutils::DEVICE_MCORE :
					logger.log("type", "cpu");
	//				logger.consoleOn();
					for (; it < last; ++it) {
						chunk = *it;
						Mat out(chunk.size(), CV_8U, Scalar(0));
						status = nscale::HistologicalEntities::segmentNuclei(chunk, out, logger, stage);
						outputs.push_back(out);
					}
	//				logger.consoleOff();
					break;
				case cciutils::DEVICE_GPU :
					logger.log("type", "gpu");
					for (; it < last; ++it) {
						chunk = *it;
						Mat out(chunk.size(), CV_8U, Scalar(0));
						status = nscale::gpu::HistologicalEntities::segmentNuclei(chunk, out, logger, stage);
						outputs.push_back(out);
					}
					break;
				default :
					break;
				}
				uint64_t t2 = cciutils::ClockGetTime();
				std::cout << "**** SEGMENTATION took " << t2-t1 << "ms" << std::endl;

				// concat.

				it = chunks.begin();
				last = chunks.end();
				std::vector<Mat>::const_iterator oit = outputs.begin();
				std::vector<Mat>::const_iterator olast = outputs.end();



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
								logger.log("on padded boundary ", idx);
								drawContours( chunkout, contours, idx, redc, 1, 8, hierarchy);

							} else if ((bbox.x <= normal.x) ||
									(bbox.y <= normal.y) ||
									(bbox.x + bbox.width >= normal.x + normal.width) ||
									(bbox.y + bbox.height >= normal.y + normal.height)) {
								logger.log("on normal boundary ", idx);
								drawContours( chunkout, contours, idx, greenc, 1, 8, hierarchy);

							} else if ((bbox.x <= inside.x) ||
									(bbox.y <= inside.y) ||
									(bbox.x + bbox.width >= inside.x + inside.width) ||
									(bbox.y + bbox.height >= inside.y + inside.height)) {
								logger.log("on neighbor's boundary ", idx);
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
				addWeighted(output2, 0.3, output, 1.0, 0.0, output);

				char ws[10];
				sprintf(ws, "%d", w);
				char bs[10];
				sprintf(bs, "%d", b);

				char ss[10];
				sprintf(ss, "%d", stage);
				if (stage > -1 && stage < 25) {
					strcpy(prefix, "test/Stages/out-segment-");
				} else {
					strcpy(prefix, "test/out-segment-");
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

				if (w == s.width || w == s.height) break;

				logger.endSession();
			}  // endif stage
		}
	}
//	waitKey();

	return status;
}


