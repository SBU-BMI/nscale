/*
 * MorphologicOperation.cpp
 *
 *  Created on: Jul 7, 2011
 *      Author: tcpan
 */

#include "MorphologicOperation.h"
#include <algorithm>
#include <queue>
#include <iostream>
#include <limits>

using namespace cv;


namespace nscale {


/** slightly optimized serial implementation,
 from Vincent paper on "Morphological Grayscale Reconstruction in Image Analysis: Applicaitons and Efficient Algorithms"

 this is the fast hybrid grayscale reconstruction

 connectivity is either 4 or 8, default 4.

 this is slightly optimized by avoiding conditional where possible.
 */
template <typename T>
Mat_<T> imreconstruct(const Mat_<T>& image, const Mat_<T>& seeds, int conn) {
	Mat_<T> output = seeds.clone();

	T pval;
	int xminus, xplus, yminus, yplus;
	int maxx = output.cols - 1;
	int maxy = output.rows - 1;
	std::queue<int> xQ;
	std::queue<int> yQ;
	bool shouldAdd;


	// raster scan
	// do the first row
	for (int x = 1; x < output.cols; x++) {
		pval = output[0][x];
		// walk through the neighbor pixels, left and up (N+(p)) only
		pval = max(pval, output[0][x-1]);
		output[0][x] = min(pval, image[0][x]);
	}
	// can't do the first col when it's 8 conn.
	for (int y = 1; y < output.rows; y++) {
		for (int x = 0; x < output.cols; x++) {

			pval = output[y][x];

			// walk through the neighbor pixels, left and up (N+(p)) only
			if (x > 0) pval = max(pval, output[y][x-1]);
			pval = max(pval, output[y-1][x]);

			if (conn == 8) {
				if (x < maxx) pval = max(pval, output[y-1][x+1]);
				if (x > 0) pval = max(pval, output[y-1][x-1]);
			}
			output[y][x] = min(pval, image[y][x]);
		}
	}

	// anti-raster scan
	// do the last row
	for (int x = maxx-1; x >= 0; x--) {

		pval = output[maxy][x];

		// walk through the neighbor pixels, right and down (N-(p)) only
		pval = max(pval, output[maxy][x+1]);

		output[maxy][x] = min(pval, image[maxy][x]);

		// capture the seeds
		// walk through the neighbor pixels, right and down (N-(p)) only
		if ((output[maxy][x+1] < output[maxy][x]) && (output[maxy][x+1] < image[maxy][x+1])) {
			xQ.push(x);
			yQ.push(maxy);
		}
	}
	// can't do the last col for 8 conn
	// do the remaining
	for (int y = maxy - 1; y >= 0; y--) {
		for (int x = maxx; x >= 0; x--) {

			pval = output[y][x];

			// walk through the neighbor pixels, right and down (N-(p)) only
			if (x < maxx) pval = max(pval, output[y][x+1]);
			pval = max(pval, output[y+1][x]);

			if (conn == 8) {
				if (x < maxx) pval = max(pval, output[y+1][x+1]);
				if (x > 0) pval = max(pval, output[y+1][x-1]);
			}

			output[y][x] = min(pval, image[y][x]);

			// capture the seeds
			shouldAdd = false;
			// walk through the neighbor pixels, right and down (N-(p)) only
			pval = output[y][x];
			if (x < maxx) {
				if ((output[y][x+1] < pval) && (output[y][x+1] < image[y][x+1])) shouldAdd = true;
			}
			if ((output[y+1][x] < pval) && (output[y+1][x] < image[y+1][x])) shouldAdd = true;

			if (conn == 8) {
				if (x < maxx) if ((output[y+1][x+1] < pval) && (output[y+1][x+1] < image[y+1][x+1])) shouldAdd = true;
				if (x > 0) if ((output[y+1][x-1] < pval) && (output[y+1][x-1] < image[y+1][x-1])) shouldAdd = true;
			}
			if (shouldAdd) {
				xQ.push(x);
				yQ.push(y);
			}
		}
	}

	// now process the queue.
	T qval, ival;
	int x, y;
	while (!(xQ.empty())) {
		x = xQ.front();
		y = yQ.front();
		xQ.pop();
		yQ.pop();

		pval = output[y][x];

		// look at the 4 connected components
		if (y > 0) {
			qval = output[y-1][x];
			ival = image[y-1][x];
			if ((qval < pval) && (ival != qval)) {
				output[y-1][x] = min(pval, ival);
				xQ.push(x);
				yQ.push(y-1);
			}
		}
		if (y < maxy) {
			qval = output[y+1][x];
			ival = image[y+1][x];
			if ((qval < pval) && (ival != qval)) {
				output[y+1][x] = min(pval, ival);
				xQ.push(x);
				yQ.push(y+1);
			}
		}
		if (x > 0) {
			qval = output[y][x-1];
			ival = image[y][x-1];
			if ((qval < pval) && (ival != qval)) {
				output[y][x-1] = min(pval, ival);
				xQ.push(x-1);
				yQ.push(y);
			}
		}
		if (x < maxx) {
			qval = output[y][x+1];
			ival = image[y][x+1];
			if ((qval < pval) && (ival != qval)) {
				output[y][x+1] = min(pval, ival);
				xQ.push(x+1);
				yQ.push(y);
			}
		}

		// now 8 connected
		if (conn == 8) {

			if (y > 0) {
				if (x > 0) {
					qval = output[y-1][x-1];
					ival = image[y-1][x-1];
					if ((qval < pval) && (ival != qval)) {
						output[y-1][x-1] = min(pval, ival);
						xQ.push(x-1);
						yQ.push(y-1);
					}
				}
				if (x < maxx) {
					qval = output[y-1][x+1];
					ival = image[y-1][x+1];
					if ((qval < pval) && (ival != qval)) {
						output[y-1][x+1] = min(pval, ival);
						xQ.push(x+1);
						yQ.push(y-1);
					}
				}

			}
			if (y < maxy) {
				if (x > 0) {
					qval = output[y+1][x-1];
					ival = image[y+1][x-1];
					if ((qval < pval) && (ival != qval)) {
						output[y+1][x-1] = min(pval, ival);
						xQ.push(x-1);
						yQ.push(y+1);
					}
				}
				if (x < maxx) {
					qval = output[y+1][x+1];
					ival = image[y+1][x+1];
					if ((qval < pval) && (ival != qval)) {
						output[y+1][x+1] = min(pval, ival);
						xQ.push(x+1);
						yQ.push(y+1);
					}
				}

			}
		}
	}



	return output;

}


template <typename T>
void propagate(const Mat_<T>& image, Mat_<T>& output, std::queue<T>& xQ, std::queue<T>& yQ,
		const int& x, const int& y, const T& pval) {
	T qval = output[y][x];
	T ival = image[y][x];
	if ((qval < pval) && (ival != qval)) {
		output[y][x] = min(pval, ival);
		xQ.push(x);
		yQ.push(y);
	}
}



/** slightly optimized serial implementation,
 from Vincent paper on "Morphological Grayscale Reconstruction in Image Analysis: Applicaitons and Efficient Algorithms"

 this is the fast hybrid grayscale reconstruction

 connectivity is either 4 or 8, default 4.

 this is slightly optimized by avoiding conditional where possible.
 */
template <typename T>
Mat_<T> imreconstruct2(const Mat_<T>& image, const Mat_<T>& seeds, int conn) {
	Mat_<T> output = seeds.clone();

	T pval;
	int xminus, xplus, yminus, yplus;
	int maxx = output.cols - 1;
	int maxy = output.rows - 1;
	std::queue<int> xQ;
	std::queue<int> yQ;
	bool shouldAdd;


	// raster scan
	// do the first row
	for (int x = 1; x < output.cols; x++) {
		pval = output[0][x];
		// walk through the neighbor pixels, left and up (N+(p)) only
		pval = max(pval, output[0][x-1]);
		output[0][x] = min(pval, image[0][x]);
	}
	// can't do the first col when it's 8 conn.
	for (int y = 1; y < output.rows; y++) {
		for (int x = 0; x < output.cols; x++) {

			pval = output[y][x];

			// walk through the neighbor pixels, left and up (N+(p)) only
			if (x > 0) pval = max(pval, output[y][x-1]);
			pval = max(pval, output[y-1][x]);

			if (conn == 8) {
				if (x < maxx) pval = max(pval, output[y-1][x+1]);
				if (x > 0) pval = max(pval, output[y-1][x-1]);
			}
			output[y][x] = min(pval, image[y][x]);
		}
	}

	// anti-raster scan
	// do the last row
	for (int x = maxx-1; x >= 0; x--) {

		pval = output[maxy][x];

		// walk through the neighbor pixels, right and down (N-(p)) only
		pval = max(pval, output[maxy][x+1]);

		output[maxy][x] = min(pval, image[maxy][x]);

		// capture the seeds
		// walk through the neighbor pixels, right and down (N-(p)) only
		if ((output[maxy][x+1] < output[maxy][x]) && (output[maxy][x+1] < image[maxy][x+1])) {
			xQ.push(x);
			yQ.push(maxy);
		}
	}
	// can't do the last col for 8 conn
	// do the remaining
	for (int y = maxy - 1; y >= 0; y--) {
		for (int x = maxx; x >= 0; x--) {

			pval = output[y][x];

			// walk through the neighbor pixels, right and down (N-(p)) only
			if (x < maxx) pval = max(pval, output[y][x+1]);
			pval = max(pval, output[y+1][x]);

			if (conn == 8) {
				if (x < maxx) pval = max(pval, output[y+1][x+1]);
				if (x > 0) pval = max(pval, output[y+1][x-1]);
			}

			output[y][x] = min(pval, image[y][x]);

			// capture the seeds
			shouldAdd = false;
			// walk through the neighbor pixels, right and down (N-(p)) only
			pval = output[y][x];
			if (x < maxx) {
				if ((output[y][x+1] < pval) && (output[y][x+1] < image[y][x+1])) shouldAdd = true;
			}
			if ((output[y+1][x] < pval) && (output[y+1][x] < image[y+1][x])) shouldAdd = true;

			if (conn == 8) {
				if (x < maxx) if ((output[y+1][x+1] < pval) && (output[y+1][x+1] < image[y+1][x+1])) shouldAdd = true;
				if (x > 0) if ((output[y+1][x-1] < pval) && (output[y+1][x-1] < image[y+1][x-1])) shouldAdd = true;
			}
			if (shouldAdd) {
				xQ.push(x);
				yQ.push(y);
			}
		}
	}

	// now process the queue.
	T qval, ival;
	int x, y;
	while (!(xQ.empty())) {
		x = xQ.front();
		y = yQ.front();
		xQ.pop();
		yQ.pop();

		pval = output[y][x];

		// look at the 4 connected components
		if (y > 0) {
			qval = output[y-1][x];
			ival = image[y-1][x];
			if ((qval < pval) && (ival != qval)) {
				output[y-1][x] = min(pval, ival);
				xQ.push(x);
				yQ.push(y-1);
			}
		}
		if (y < maxy) {
			qval = output[y+1][x];
			ival = image[y+1][x];
			if ((qval < pval) && (ival != qval)) {
				output[y+1][x] = min(pval, ival);
				xQ.push(x);
				yQ.push(y+1);
			}
		}
		if (x > 0) {
			qval = output[y][x-1];
			ival = image[y][x-1];
			if ((qval < pval) && (ival != qval)) {
				output[y][x-1] = min(pval, ival);
				xQ.push(x-1);
				yQ.push(y);
			}
		}
		if (x < maxx) {
			qval = output[y][x+1];
			ival = image[y][x+1];
			if ((qval < pval) && (ival != qval)) {
				output[y][x+1] = min(pval, ival);
				xQ.push(x+1);
				yQ.push(y);
			}
		}

		// now 8 connected
		if (conn == 8) {

			if (y > 0) {
				if (x > 0) {
					qval = output[y-1][x-1];
					ival = image[y-1][x-1];
					if ((qval < pval) && (ival != qval)) {
						output[y-1][x-1] = min(pval, ival);
						xQ.push(x-1);
						yQ.push(y-1);
					}
				}
				if (x < maxx) {
					qval = output[y-1][x+1];
					ival = image[y-1][x+1];
					if ((qval < pval) && (ival != qval)) {
						output[y-1][x+1] = min(pval, ival);
						xQ.push(x+1);
						yQ.push(y-1);
					}
				}

			}
			if (y < maxy) {
				if (x > 0) {
					qval = output[y+1][x-1];
					ival = image[y+1][x-1];
					if ((qval < pval) && (ival != qval)) {
						output[y+1][x-1] = min(pval, ival);
						xQ.push(x-1);
						yQ.push(y+1);
					}
				}
				if (x < maxx) {
					qval = output[y+1][x+1];
					ival = image[y+1][x+1];
					if ((qval < pval) && (ival != qval)) {
						output[y+1][x+1] = min(pval, ival);
						xQ.push(x+1);
						yQ.push(y+1);
					}
				}

			}
		}
	}



	return output;

}



template <typename T>
void propagateBinary(const Mat_<T>& image, Mat_<T>& output, std::queue<T>& xQ, std::queue<T>& yQ,
		const int& x, const int& y, const T& foreground) {
	if ((output[y][x] == 0) && (image[y][x] != 0)) {
		output[y][x] = foreground;
		xQ.push(x);
		yQ.push(y);
	}
}


/** optimized serial implementation for binary,
 from Vincent paper on "Morphological Grayscale Reconstruction in Image Analysis: Applicaitons and Efficient Algorithms"

 connectivity is either 4 or 8, default 4.  background is assume to be 0, foreground is assumed to be NOT 0.

 */
template <typename T>
Mat_<T> imreconstructBinary(const Mat_<T>& image, const Mat_<T>& seeds, int conn) {
	Mat_<T> output = seeds.clone();

	T pval;
	int xminus, xplus, yminus, yplus;
	int maxx = output.cols - 1;
	int maxy = output.rows - 1;
	std::cout<< "max x = " << maxx << "  max y = " << maxy << std::endl;
	//std::queue<Point> pixQ;
	std::queue<int> xQ;
	std::queue<int> yQ;
	bool shouldAdd;


	// contour pixel determination

	// now process the queue.
	T qval, ival;
	T outval = std::numeric_limits<T>::max();
	int x, y;
	while (!(xQ.empty())) {
		x = xQ.front();
		y = yQ.front();
		xQ.pop();
		yQ.pop();

		// look at the 4 connected components
		if (y > 0) {
			nscale::propagateBinary<T>(image, output, xQ, yQ, x, y-1, outval);
		}
		if (y < maxy) {
			nscale::propagateBinary<T>(image, output, xQ, yQ, x, y+1, outval);
		}
		if (x > 0) {
			nscale::propagateBinary<T>(image, output, xQ, yQ, x-1, y, outval);
		}
		if (x < maxx) {
			nscale::propagateBinary<T>(image, output, xQ, yQ, x+1, y, outval);
		}

		// now 8 connected
		if (conn == 8) {

			if (y > 0) {
				if (x > 0) {
					nscale::propagateBinary<T>(image, output, xQ, yQ, x-1, y-1, outval);
				}
				if (x < maxx) {
					nscale::propagateBinary<T>(image, output, xQ, yQ, x+1, y-1, outval);
				}

			}
			if (y < maxy) {
				if (x > 0) {
					nscale::propagateBinary<T>(image, output, xQ, yQ, x-1, y+1, outval);
				}
				if (x < maxx) {
					nscale::propagateBinary<T>(image, output, xQ, yQ, x+1, y+1, outval);
				}

			}
		}
	}

	return output;

}

template Mat_<uchar> imreconstruct(const Mat_<uchar>&, const Mat_<uchar>&, int);
template Mat_<uchar> imreconstruct2(const Mat_<uchar>&, const Mat_<uchar>&, int);
template Mat_<uchar> imreconstructBinary(const Mat_<uchar>&, const Mat_<uchar>&, int);

}



/** fast hybrid serial implementation,
 from Vincent paper on "Morphological Grayscale Reconstruction in Image Analysis: Applicaitons and Efficient Algorithms"

 this is the fast hybrid grayscale reconstruction

 connectivity is either 4 or 8, default 4.

 retired.
 */ /*
template <typename T>
Mat_<T> imreconstruct(const Mat_<T>& image, const Mat_<T>& seeds, int conn) {
	Mat_<T> output = seeds.clone();

	T pval;
	int xminus, xplus, yminus, yplus;
	int maxx = output.cols - 1;
	int maxy = output.rows - 1;
	std::cout<< "max x = " << maxx << "  max y = " << maxy << std::endl;
	//std::queue<Point> pixQ;
	std::queue<int> xQ;
	std::queue<int> yQ;
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
	for (int y = maxy; y >= 0; y--) {
		for (int x = maxx; x >= 0; x--) {

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
				//pixQ.push(Point(x, y));
				xQ.push(x);
				yQ.push(y);
				if (x > maxx) {
					std::cout << "ERROR x too big " << x << "," << y << std::endl;
				}
				if (y > maxy) {
					std::cout << "ERROR y too big " << x << "," << y << std::endl;
				}
			}
		}
	}

	// now process the queue.
	Point p;
	T qval, ival;
	int x, y;
//	while (!(pixQ.empty())) {
	while (!(xQ.empty())) {
		//p = pixQ.front();
		//pixQ.pop();
		//x = p.x;
		//y = p.y;
		x = xQ.front();
		y = yQ.front();
		xQ.pop();
		yQ.pop();
		if (x > maxx) {
			//std::cout << "ERROR processing.  x too big " << x << "," << y << std::endl;
		}
		if (y > maxy) {
			//std::cout << "ERROR processing.  y too big " << x << "," << y << std::endl;
		}

		pval = output[y][x];

		// look at the 4 connected components
		if (y > 0) {
			qval = output[y-1][x];
			ival = image[y-1][x];
			if ((qval < pval) && (ival != qval)) {
				output[y-1][x] = min(pval, ival);
				//pixQ.push(Point(x, y-1));
				xQ.push(x);
				yQ.push(y-1);
				if (x > maxx) {
					std::cout << "ERROR x too big " << x << "," << y-1 << std::endl;
				}
				if (y > maxy) {
					std::cout << "ERROR y+1 too big " << x << "," << y-1 << std::endl;
				}

			}
		}
		if (y < maxy) {
			qval = output[y+1][x];
			ival = image[y+1][x];
			if ((qval < pval) && (ival != qval)) {
				output[y+1][x] = min(pval, ival);
				//pixQ.push(Point(x, y+1));
				xQ.push(x);
				yQ.push(y+1);
				if (x > maxx) {
					std::cout << "ERROR x too big " << x << "," << y+1 << std::endl;
				}
				if (y > maxy) {
					std::cout << "ERROR y+1 too big " << x << "," << y+1 << std::endl;
				}
			}
		}
		if (x > 0) {
			qval = output[y][x-1];
			ival = image[y][x-1];
			if ((qval < pval) && (ival != qval)) {
				output[y][x-1] = min(pval, ival);
				//pixQ.push(Point(x-1, y));
				xQ.push(x-1);
				yQ.push(y);
			}
		}
		if (x < maxx) {
			qval = output[y][x+1];
			ival = image[y][x+1];
			if ((qval < pval) && (ival != qval)) {
				output[y][x+1] = min(pval, ival);
				//pixQ.push(Point(x+1, y));
				xQ.push(x+1);
				yQ.push(y);
			}
		}

		// now 8 connected
		if (conn == 8) {

			if (y > 0) {
				if (x > 0) {
					qval = output[y-1][x-1];
					ival = image[y-1][x-1];
					if ((qval < pval) && (ival != qval)) {
						output[y-1][x-1] = min(pval, ival);
						//pixQ.push(Point(x-1, y-1));
						xQ.push(x-1);
						yQ.push(y-1);
					}
				}
				if (x < maxx) {
					qval = output[y-1][x+1];
					ival = image[y-1][x+1];
					if ((qval < pval) && (ival != qval)) {
						output[y-1][x+1] = min(pval, ival);
						//pixQ.push(Point(x+1, y-1));
						xQ.push(x+1);
						yQ.push(y-1);
					}
				}

			}
			if (y < maxy) {
				if (x > 0) {
					qval = output[y+1][x-1];
					ival = image[y+1][x-1];
					if ((qval < pval) && (ival != qval)) {
						output[y+1][x-1] = min(pval, ival);
						//pixQ.push(Point(x-1, y+1));
						xQ.push(x-1);
						yQ.push(y+1);
					}
				}
				if (x < maxx) {
					qval = output[y+1][x+1];
					ival = image[y+1][x+1];
					if ((qval < pval) && (ival != qval)) {
						output[y+1][x+1] = min(pval, ival);
						//pixQ.push(Point(x+1, y+1));
						xQ.push(x+1);
						yQ.push(y+1);
					}
				}

			}
		}
	}


	return output;

}
*/
