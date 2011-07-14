/*
 * MorphologicOperation.cpp
 *
 *  Created on: Jul 7, 2011
 *      Author: tcpan
 */

#include "MorphologicOperation.h"
#include <algorithm>
#include <queue>

using namespace cv;


namespace nscale {

/** naive parallel implementation,
 from Vincent paper on "Morphological Grayscale Reconstruction in Image Analysis: Applicaitons and Efficient Algorithms"

 this is the fast hybrid grayscale reconstruction

 connectivity is either 4 or 8, default 4.
 */
template <typename T>
Mat_<T> imreconstruct(const Mat_<T>& image, const Mat_<T>& seeds, int conn) {
	Mat_<T> output = seeds.clone();

	T pval;
	int xminus, xplus, yminus, yplus;
	int maxx = output.cols - 1;
	int maxy = output.rows - 1;
	std::queue<Point> pixQ;
	bool shouldAdd;


	// raster scan
	for (int y = 0; y < output.rows; y++) {
		for (int x = 0; x < output.cols; x++) {

			pval = output[y][x];

			// walk through the neighbor pixels, left and up (N+(p)) only
			if (x > 0) pval = max(pval, output[y][x-1]);
			if (y > 0) {
				pval = max(pval, output[y-1][x]);

				if (conn == 8) {
					if (x < maxx) pval = max(pval, output[y-1][x+1]);
					if (x > 0) pval = max(pval, output[y-1][x-1]);
				}
			}
			output[y][x] = min(pval, image[y][x]);
		}
	}

	// anti-raster scan
	for (int y = output.rows - 1; y >= 0; y--) {
		for (int x = output.cols - 1; x >= 0; x--) {

			pval = output[y][x];

			// walk through the neighbor pixels, right and down (N-(p)) only
			if (x < maxx) pval = max(pval, output[y][x+1]);
			if (y < maxy) {
				pval = max(pval, output[y+1][x]);

				if (conn == 8) {
					if (x < maxx) pval = max(pval, output[y+1][x+1]);
					if (x > 0) pval = max(pval, output[y+1][x-1]);
				}
			}

			output[y][x] = min(pval, image[y][x]);

			// capture the seeds
			shouldAdd = false;
			// walk through the neighbor pixels, right and down (N-(p)) only
			pval = output[y][x];
			if (x < maxx) {
				if ((output[y][x+1] < pval) && (output[y][x+1] < image[y][x+1])) shouldAdd = true;
			}
			if (y < maxy) {
				if ((output[y+1][x] < pval) && (output[y+1][x] < image[y+1][x])) shouldAdd = true;

				if (conn == 8) {
					if (x < maxx) if ((output[y+1][x+1] < pval) && (output[y+1][x+1] < image[y+1][x+1])) shouldAdd = true;
					if (x > 0) if ((output[y+1][x-1] < pval) && (output[y+1][x-1] < image[y+1][x-1])) shouldAdd = true;
				}
			}
			if (shouldAdd) {
				pixQ.push(Point(x, y));
			}
		}
	}

	// now process the queue.
	Point p;
	T qval, ival;
	int x, y;
	while (!(pixQ.empty())) {
		p = pixQ.front();
		pixQ.pop();
		x = p.x;
		y = p.y;
		pval = output[y][x];

		// look at the 4 connected components
		if (y > 0) {
			qval = output[y-1][x];
			ival = output[y-1][x];
			if ((qval < pval) && (ival != qval)) {
				output[y-1][x] = min(pval, ival);
				pixQ.push(Point(x, y-1));
			}
		}
		if (y < maxy) {
			qval = output[y+1][x];
			ival = output[y+1][x];
			if ((qval < pval) && (ival != qval)) {
				output[y+1][x] = min(pval, ival);
				pixQ.push(Point(x, y+1));
			}
		}
		if (x > 0) {
			qval = output[y][x-1];
			ival = output[y][x-1];
			if ((qval < pval) && (ival != qval)) {
				output[y][x-1] = min(pval, ival);
				pixQ.push(Point(x-1, y));
			}
		}
		if (x < maxx) {
			qval = output[y][x+1];
			ival = output[y][x+1];
			if ((qval < pval) && (ival != qval)) {
				output[y][x+1] = min(pval, ival);
				pixQ.push(Point(x+1, y));
			}
		}

		// now 8 connected
		if (conn == 8) {

			if (y > 0) {
				if (x > 0) {
					qval = output[y-1][x-1];
					ival = output[y-1][x-1];
					if ((qval < pval) && (ival != qval)) {
						output[y-1][x-1] = min(pval, ival);
						pixQ.push(Point(x-1, y-1));
					}
				}
				if (x < maxx) {
					qval = output[y-1][x+1];
					ival = output[y-1][x+1];
					if ((qval < pval) && (ival != qval)) {
						output[y-1][x+1] = min(pval, ival);
						pixQ.push(Point(x+1, y-1));
					}
				}

			}
			if (x < maxx) {
				if (x > 0) {
					qval = output[y+1][x-1];
					ival = output[y+1][x-1];
					if ((qval < pval) && (ival != qval)) {
						output[y+1][x-1] = min(pval, ival);
						pixQ.push(Point(x-1, y+1));
					}
				}
				if (x < maxx) {
					qval = output[y+1][x+1];
					ival = output[y+1][x+1];
					if ((qval < pval) && (ival != qval)) {
						output[y+1][x+1] = min(pval, ival);
						pixQ.push(Point(x+1, y+1));
					}
				}

			}
		}
	}


	return output;

}

// do row 1 and col 1 first - avoid boundary case conditionals
// (0, 0) don't need to touch
// do row 1


template Mat_<uchar> imreconstruct(const Mat_<uchar>&, const Mat_<uchar>&, int);

}
