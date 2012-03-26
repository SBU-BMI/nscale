/*
 * ConnComponents.cpp
 *
 *  Created on: Feb 20, 2012
 *      Author: tcpan
 */

#include "ConnComponents.h"
#include <string.h>


namespace nscale {


ConnComponents::ConnComponents() {
	// TODO Auto-generated constructor stub

}

ConnComponents::~ConnComponents() {
	// TODO Auto-generated destructor stub
}

int ConnComponents::find(int *label, int x) {
	if (label[x] == x) {
		return x;
	} else {
		int v = find(label, label[x]);
		label[x] = v;
		return v;
	}
}

void ConnComponents::merge(int *label, int x, int y) {
	x = find(label, x);
	y = find(label, y);

	if (label[x] < label[y]) {
		label[y] = x;
	} else if (label[x] > label[y]) {
		label[x] = y;
	}
}


// adapted from the brazilian cpu union-find CCL.
// 2 sets of improvements:
//		2. use a nested loop instead of x and y - avoids % and /.  also increment i, and precompute i-w, and preload img[i].
//	above does not seem to create a big reduction (about 50ms).  timing for sizePhantom is between 1.1 and 1.3 sec.  for astroII.1's mask, about 1.1 sec.
//		3. early termination - if background, don't go through the merge.  - significant savings.  - down to for size phantom, down to about 300ms.
//	with astroII.1's mask, about 60 ms.  faster than contour based.
void ConnComponents::label(unsigned char *img, int w, int h, int *label, int bgval, int connectivity) {
	int length = w*h;

	// initialize
	for (int i=0; i<length; ++i) {
		label[i] = i;
	}

	// do the labelling
	int i = -1, imw = 0;
	unsigned char p;
	for (int y=0; y<h; ++y) {
		for (int x=0; x<w; ++x) {
			++i;
			p = img[i];
			if (p == 0) {
				label[i] = bgval;
			} else {
				imw = i - w;
				if (x>0 && p==img[i-1]) merge(label, i, i-1);
				if (y>0 && p==img[imw]) merge(label, i, imw);
				if (connectivity == 8) {
					if (y>0 && x>0 && p==img[imw-1]) merge(label, i, imw-1);
					if (y>0 && x<w-1 && p==img[imw+1]) merge(label, i, imw+1);
				}
			}
		}
	}

	// final flatten
	for (int i = 0; i < length; ++i) {
		label[i] = flatten(label, i, bgval);
	}

}

// relabelling to get sequential labels.  assumes labeled image is flattened with background set to bgval
int ConnComponents::relabel(int w, int h, int *label, int bgval) {
	int length = w * h;
	std::tr1::unordered_map<int, int> labelmap;

	int j = 1;
	// first find the roots
	labelmap[bgval] = 0;
	for (int i=0; i<length; ++i) {
		if (label[i] == i) {
			// root.  record its value
			labelmap[i] = j;
//			printf("root: %d: %d\n", j, i);
			++j;
		}
	}

	// Debug
//	printf("labels = ");
//	for (std::tr1::unordered_map<int, int>::iterator iter = labelmap.begin();
//			iter != labelmap.end(); ++iter) {
//		printf("%d->%d ", iter->first, iter->second);
//	}
//	printf("\n");

	// next do a one pass value change.
	for (int i = 0; i < length; ++i) {
		label[i] = labelmap[label[i]];
	}
	return j-1;
}


void ConnComponents::mergeWithArea(int *label, int x, int y, int *areas) {
	x = find(label, x);
	y = find(label, y);
	// update the areas
//	printf("x, y, sum = %d, %d, %d, label[x], label[y] = %d, %d\n", x, y, sum, label[x], label[y]);
	if (label[x] < label[y]) {
		label[y] = x;
		areas[x] += areas[y];
		areas[y] = 0;
	} else if (label[x] > label[y]) {
		label[x] = y;
		areas[y] += areas[x];
		areas[x] = 0;
	}

}
int ConnComponents::flatten(int *label, int x, int bgval) {
	if (label[x] == bgval) {
		return bgval;
	} else if (label[x] == x) {
		return x;
	} else {
		int v = flatten(label, label[x], bgval);
		label[x] = v;
		return v;
	}
}

// inclusive lower, exclusive upper
int ConnComponents::areaThreshold(unsigned char *img, int w, int h, int *label, int bgval, int lower, int upper, int connectivity) {

	int length = w*h;

	// initialize
	int *areas = new int[length];
	for (int i=0; i<length; ++i) {
		label[i] = i;
		areas[i] = 1;
	}

	// do the labelling
	int i = -1, imw = 0;
	unsigned char p;
	for (int y=0; y<h; ++y) {
		for (int x=0; x<w; ++x) {
			++i;
			p = img[i];
			if (p == 0) {
				label[i] = bgval;
			} else {
				imw = i - w;
				if (x>0 && p==img[i-1]) mergeWithArea(label, i, i-1, areas);
				if (y>0 && p==img[imw]) mergeWithArea(label, i, imw, areas);
				if (connectivity == 8) {
					if (y>0 && x>0 && p==img[imw-1]) mergeWithArea(label, i, imw-1, areas);
					if (y>0 && x<w-1 && p==img[imw+1]) mergeWithArea(label, i, imw+1, areas);
				}
			}
		}
	}
	int j = 0;
	//  TONY finally do the threshold and change the value of the
	for (i=0; i<length; ++i) {
		// look at the roots
		if (label[i] == i) {
			//printf("%d, %d area = %d", i/w, i%w, areas[i]);
			if (areas[i] < lower || areas[i] >= upper) {
				// reset the value of the root
				label[i] = bgval;
				//printf(", removed");
			} else ++j;
			//printf("\n");
		}
	}
	// flatten one last time
	for (i = 0; i < length; ++i) {
		label[i] = flatten(label, i, bgval);
	}

	delete [] areas;
	return j;
}


int ConnComponents::areaThresholdLabeled(const int *label, const int w, const int h, int *n_label, const int bgval, const int lower, const int upper) {
	int length = w * h;
	typedef std::tr1::unordered_map<int, int> AreaMap;
	AreaMap areas, thresholdedareas;

	for (int i=0; i<length; ++i) {
		if (label[i] != bgval) {  // not background, then increment area
			areas[label[i]] += 1;
		}
		// initialize output
		n_label[i] = bgval;
	}

	int j = 0;
	// count ones that are within threshold
	for (AreaMap::iterator it = areas.begin();
			it != areas.end(); ++it) {
		if (it->second >= lower && it->second < upper) {
			thresholdedareas[it->first] = 1;
			++j;
		} else {
			thresholdedareas[it->first] = 0;
		}
	}
	areas.clear();

	//  TONY finally do the threshold and change the value of the
	for (int i=0; i<length; ++i) {
		// look at the roots
		if (label[i] != bgval && thresholdedareas[label[i]] == 1) n_label[i] = label[i];
	}

	thresholdedareas.clear();
	return j;
}

int* ConnComponents::boundingBox(const int w, const int h, const int* label, int bgval, int &compcount) {
	int length = w * h;
	std::tr1::unordered_map<int, int> label2index;

	int *tbox = (int*)malloc(sizeof(int) * length * 5);
	int *minx = tbox + length, *maxx = tbox +2*length, *miny = tbox +3*length, *maxy = tbox + 4*length;
	int x, y;
	int j = 0;
	for (int i=0; i<length; ++i) {
		if (label[i] == bgval) continue;

		if (label[i] == -1) { printf("-1 background\n"); break; }
		if (label[i] == 0) { printf("0 background\n"); break; }

		x = i % w;
		y = i / w;
		if (label2index.find(label[i]) == label2index.end()) {
			label2index[label[i]] = j;
			tbox[j] = label[i];
			minx[j] = x;
			maxx[j] = x;
			miny[j] = y;
			maxy[j] = y;
			++j;

		} else {

			minx[label2index[label[i]]] = (minx[label2index[label[i]]] > x ? x : minx[label2index[label[i]]]);
			maxx[label2index[label[i]]] = (maxx[label2index[label[i]]] < x ? x : maxx[label2index[label[i]]]);
			miny[label2index[label[i]]] = (miny[label2index[label[i]]] > y ? y : miny[label2index[label[i]]]);
			maxy[label2index[label[i]]] = (maxy[label2index[label[i]]] < y ? y : maxy[label2index[label[i]]]);
		}
	}
	compcount = j;
	printf("miny = %d maxy=%d\n", miny[0], maxy[0]);
	int *output = (int*)malloc(sizeof(int) * compcount * 5);
	memcpy(output, tbox, compcount * sizeof(int));
	memcpy(output+compcount, minx, compcount * sizeof(int));
	memcpy(output+2*compcount, maxx, compcount * sizeof(int));
	memcpy(output+3*compcount, miny, compcount * sizeof(int));
	memcpy(output+4*compcount, maxy, compcount * sizeof(int));
	free(tbox);
	return output;
}


}
