/*
 * test.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
#include <iostream>
#include <vector>
#include <string.h>
#include "mpi.h"
#include <cstdlib>
#include <string>
#include "adios_read.h"
#include "adios.h"
#include "opencv2/opencv.hpp"
#include "RegionalMorphologyAnalysis.h"
#include "PixelOperations.h"


void readData(ADIOS_GROUP *g,const  uint64_t *start, const uint64_t *count, const int rank, void* &tiledata, ::cv::Mat &img, char* &sourceTileFiledata) {
	int tileOffsetX, tileOffsetY;
	int tileSizeX, tileSizeY, channels, elemSize1, cvDataType, encoding;
	uint64_t sourceTileFile_offset, sourceTileFile_size, tile_offset, tile_size;
	uint64_t bytes_read = 0;

	// read the metadata.
	bytes_read = adios_read_var(g, "tileOffsetX", start, count, &tileOffsetX);
	bytes_read = adios_read_var(g, "tileOffsetY", start, count, &tileOffsetY);
	bytes_read = adios_read_var(g, "tileSizeX", start, count, &tileSizeX);
	bytes_read = adios_read_var(g, "tileSizeY", start, count, &tileSizeY);
	bytes_read = adios_read_var(g, "channels", start, count, &channels);
	bytes_read = adios_read_var(g, "elemSize1", start, count, &elemSize1);
	bytes_read = adios_read_var(g, "cvDataType", start, count, &cvDataType);
	bytes_read = adios_read_var(g, "encoding", start, count, &encoding);
	bytes_read = adios_read_var(g, "sourceTileFile_offset", start, count, &sourceTileFile_offset);
	bytes_read = adios_read_var(g, "sourceTileFile_size", start, count, &sourceTileFile_size);
	bytes_read = adios_read_var(g, "tile_offset", start, count, &tile_offset);
	bytes_read = adios_read_var(g, "tile_size", start, count, &tile_size);
	// printf("tile info at rank %d: at %dx%d, %dx%dx%d, elem %d, cvType %d, encoding %d, sourcetile %ld at %ld, bytes %ld at %ld\n", rank, tileOffsetX, tileOffsetY, tileSizeX, tileSizeY, channels, elemSize1, cvDataType, encoding, sourceTileFile_size, sourceTileFile_offset, tile_size, tile_offset);


	// read the tile
	// TODO: may be able to reuse memory.
	sourceTileFiledata = (char*)malloc(sourceTileFile_size + 1);
	memset(sourceTileFiledata, 0, sourceTileFile_size + 1);
	bytes_read = adios_read_var(g, "sourceTileFile", &sourceTileFile_offset, &sourceTileFile_size, sourceTileFiledata);
	//printf ("rank %d requested read %ld bytes at offset %ld, actual read %ld bytes\n", rank, sourceTileFile_size, sourceTileFile_offset, bytes_read);
	//printf("source tilefile : %s\n", sourceTileFiledata);

	tiledata = malloc(tile_size);
	memset(tiledata, 0, tile_size);
	bytes_read = adios_read_var(g, "tile", &tile_offset, &tile_size, tiledata);
	// printf ("rank %d requested read %ld bytes at offset %ld, actual read %ld bytes\n", rank, tile_size, tile_offset, bytes_read);

	int mat_size[2];

	mat_size[0] = tileSizeX;
	mat_size[1] = tileSizeY;
	img = ::cv::Mat(2, mat_size,cvDataType, tiledata);
}


std::vector<std::vector<float> > computeFeatures(::cv::Mat image, ::cv::Mat mask) {
	::cv::Mat maskMat = mask > 0;

	IplImage originalImageMask(maskMat);

	//bool isNuclei = true;

	// Convert color image to grayscale
//	::cv::Mat grayMat = ::nscale::PixelOperations::bgr2gray(image);
	std::vector<cv::Mat> bgr;
	::cv::split(image, bgr);
	::cv::Mat_<unsigned char> grayMat(image.size());
	unsigned char *rPtr, *gPtr, *bPtr, *grayPtr;
	for (int y = 0; y < image.rows; ++y) {
		rPtr =  bgr[2].ptr<unsigned char>(y);
		gPtr =  bgr[1].ptr<unsigned char>(y);
		bPtr =  bgr[0].ptr<unsigned char>(y);

		grayPtr = grayMat.ptr<unsigned char>(y);
		for (int x = 0; x < image.cols; ++x) {
			grayPtr[x] = (unsigned char)(double(gPtr[x]) / (double(bPtr[x]) + double(gPtr[x]) + double(rPtr[x]) + std::numeric_limits<double>::epsilon()) * 255.0);
		}
	}
	bgr[0].release();
	bgr[1].release();
	bgr[2].release();
	bgr.clear();
	//imwrite("test-g.pgm", grayMat);

	//	cvSaveImage("newGrayScale.png", grayscale);
	IplImage grayscale(grayMat);

	// This is another option for inialize the features computation, where the path to the images are given as parameter
	RegionalMorphologyAnalysis *regional = new RegionalMorphologyAnalysis(&originalImageMask, &grayscale, true);

	// Create H and E images
	//initialize H and E channels
	Mat H = Mat::zeros(image.size(), CV_8UC1);
	Mat E = Mat::zeros(image.size(), CV_8UC1);
	Mat b = (Mat_<char>(1,3) << 1, 1, 0);
	Mat M = (Mat_<double>(3,3) << 0.650, 0.072, 0, 0.704, 0.990, 0, 0.286, 0.105, 0);

	::nscale::PixelOperations::ColorDeconv(image, M, b, H, E);

	IplImage ipl_image_H(H);
	IplImage ipl_image_E(E);


	// This is another option for inialize the features computation, where the path to the images are given as parameter
	//	RegionalMorphologyAnalysis *regional = new RegionalMorphologyAnalysis(argv[1], argv[2]);

	vector<vector<float> > nucleiFeatures;

	/////////////// Compute nuclei based features ////////////////////////
	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
	// 	0)BoundingBox (BB) X; 1) BB.y; 2) BB.width; 3) BB.height; 4) Centroid.x; 5) Centroid.y) 7)Area; 8)Perimeter; 9)Eccentricity;
	//	10)Circularity/Compacteness; 11)MajorAxis; 12)MinorAxis; 13)ExtentRatio; 14)MeanIntensity 15)MaxIntensity; 16)MinIntensity;
	//	17)StdIntensity; 18)EntropyIntensity; 19)EnergyIntensity; 20)SkewnessIntensity;	21)KurtosisIntensity; 22)MeanGrad; 23)StdGrad;
	//	24)EntropyGrad; 25)EnergyGrad; 26)SkewnessGrad; 27)KurtosisGrad; 28)CannyArea; 29)MeanCanny
	regional->doNucleiPipelineFeatures(nucleiFeatures, &grayscale);

	/////////////// Compute cytoplasm based features ////////////////////////
	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
	// 	0)MeanIntensity; 1) MedianIntensity-MeanIntensity; 2)MaxIntensity; 3)MinIntensity; 4)StdIntensity; 5)EntropyIntensity;
	//	6)EnergyIntensity; 7)SkewnessIntensity; 8)KurtosisIntensity; 9)MeanGrad; 10)StdGrad; 11)EntropyGrad; 12)EnergyGrad;
	//	13)SkewnessGrad; 14)KurtosisGrad; 15)CannyArea; 16)MeanCanny;
	vector<vector<float> > cytoplasmFeatures_G;
	regional->doCytoplasmPipelineFeatures(cytoplasmFeatures_G, &grayscale);


	/////////////// Compute cytoplasm based features ////////////////////////
	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
	// 	0)MeanIntensity; 1) MedianIntensity-MeanIntensity; 2)MaxIntensity; 3)MinIntensity; 4)StdIntensity; 5)EntropyIntensity;
	//	6)EnergyIntensity; 7)SkewnessIntensity; 8)KurtosisIntensity; 9)MeanGrad; 10)StdGrad; 11)EntropyGrad; 12)EnergyGrad;
	//	13)SkewnessGrad; 14)KurtosisGrad; 15)CannyArea; 16)MeanCanny;
	vector<vector<float> > cytoplasmFeatures_H;
	regional->doCytoplasmPipelineFeatures(cytoplasmFeatures_H, &ipl_image_H);

	/////////////// Compute cytoplasm based features ////////////////////////
	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
	// 	0)MeanIntensity; 1) MedianIntensity-MeanIntensity; 2)MaxIntensity; 3)MinIntensity; 4)StdIntensity; 5)EntropyIntensity;
	//	6)EnergyIntensity; 7)SkewnessIntensity; 8)KurtosisIntensity; 9)MeanGrad; 10)StdGrad; 11)EntropyGrad; 12)EnergyGrad;
	//	13)SkewnessGrad; 14)KurtosisGrad; 15)CannyArea; 16)MeanCanny;
	vector<vector<float> > cytoplasmFeatures_E;
	regional->doCytoplasmPipelineFeatures(cytoplasmFeatures_E, &ipl_image_E);

	delete regional;

	maskMat.release();
	grayMat.release();

	H.release();
	E.release();
	M.release();
	b.release();

	std::vector<std::vector<float> > features;
	for (int i = 0; i < nucleiFeatures.size(); ++i) {
		std::vector<float> v;
		v.insert(v.end(), nucleiFeatures[i].begin(), nucleiFeatures[i].end());
		v.insert(v.end(), cytoplasmFeatures_G[i].begin(), cytoplasmFeatures_G[i].end());
		v.insert(v.end(), cytoplasmFeatures_H[i].begin(), cytoplasmFeatures_H[i].end());
		v.insert(v.end(), cytoplasmFeatures_E[i].begin(), cytoplasmFeatures_E[i].end());
		features.push_back(v);
		nucleiFeatures[i].clear();
		cytoplasmFeatures_G[i].clear();
		cytoplasmFeatures_H[i].clear();
		cytoplasmFeatures_E[i].clear();
	}
	nucleiFeatures.clear();
	cytoplasmFeatures_G.clear();
	cytoplasmFeatures_H.clear();
	cytoplasmFeatures_E.clear();

	return features;
}

void writeFeatures(std::vector<std::vector<float> > &features, const char* bpfile) {

}


int main (int argc, char **argv) {


	if (argc < 4) {
		printf("usage: %s maskfilename rgbfilename outdirname", argv[0]);
		return -1;
	}

	// init MPI
	int ierr = MPI_Init(&argc, &argv);

	std::string hostname;
	char * temp = (char*)malloc(256);
	gethostname(temp, 255);
	hostname.assign(temp);
	free(temp);

	MPI_Comm comm_world = MPI_COMM_WORLD;
	int size, rank;
	MPI_Comm_size(comm_world, &size);
	MPI_Comm_rank(comm_world, &rank);

	printf("comm-world:  %s: %d of %d\n", hostname.c_str(), rank, size);

//using the world communicator	

	MPI_Barrier(comm_world);


	// open the file
	char * filename = argv[1];
//	printf("opening file: %s\n", filename);
	ADIOS_FILE *fc = adios_fopen(filename, comm_world);
	if (fc == NULL) {
		printf("%s\n", adios_errmsg());
		return -1;
	}

	// then open the tileCount group to get some data.
	ADIOS_GROUP * gc = adios_gopen(fc, "tileCount");
	if (gc == NULL) {
		printf("%s\n", adios_errmsg());
		return -1;
	}


	// all processes read the variables. 
	ADIOS_VARINFO * v = adios_inq_var(gc, "tileInfo_total");
	long tileInfo_total = *((long *)(v->value));
	if (rank == 0) printf("tileinfo total = %ld dims %d timedim %d\n", tileInfo_total, v->ndim, v->timedim );
	adios_free_varinfo(v);

	v = adios_inq_var(gc, "sourceTileFile_total");
	long sourceTileFile_total = *((long *)(v->value));
	if (rank == 0) printf("sourceTileFile total = %ld \n", sourceTileFile_total);
	adios_free_varinfo(v);

	v = adios_inq_var(gc, "tile_total");
	long tile_total = *((long *)(v->value));
	if (rank == 0) 	printf("tile total = %ld \n", tile_total );
	adios_free_varinfo(v);
	
	adios_gclose(gc);
	
	// now partition the space and try to read the content
//	uint64_t slice_size = tileInfo_total / size;  // each read at least these many
//	uint64_t remainder = tileInfo_total % size;  // what's left
//	if (rank == 0) printf("slice_size %d, number of slices %lu, remainder %lu\n", size, slice_size, remainder);

	// each iteration reads 1 per process
	uint64_t start, count = 1;

	// then open the tileCount group to get some data.
	ADIOS_GROUP *g = adios_gopen(fc, "tileInfo");
	if (g == NULL) {
		printf("%s\n", adios_errmsg());
		return -1;
	}

//	// for some reason, it's still trying to compute time based characteristics.   but then adios crashes.
//	ADIOS_VARINFO *v2 = adios_inq_var(g, "tileSizeX");
//	if (v2 == NULL) printf("why is varinfo NULL?\n");
//	printf("RANK %d tileSizeX ndims %d dim n-1 %ld time dim %d\n", rank, v->ndim, v->dims[v->ndim-1], v->timedim);
//	adios_free_varinfo(v2);

	void *tiledata = NULL;
	char *sourceTileFiledata = NULL;
	char fn[256];
	for (uint64_t i = rank; i < tileInfo_total; i += size) {
		start = i;

		// read data and convert to cv MAT
		::cv::Mat mask;
		readData(g, &start, &count, rank, tiledata, mask, sourceTileFiledata);
		if (! mask.data) {
			printf("can't read image mask\n");
			return -1;
		}



		// also read the source tile image.
		printf("read input = %s\n", sourceTileFiledata);
		::cv::Mat img = ::cv::imread(sourceTileFiledata);
		free(sourceTileFiledata);
		if (! img.data) {
			printf("can't read original image\n");
			return -2;
		}


		// do some computation
		std::vector<std::vector<float> > features = computeFeatures(img, mask);

		// release images
		img.release();
		//printf("tiledata pointer %lu\n", (uint64_t)tiledata);
		mask.release();
		free(tiledata);


		// do output
		printf("RANK %d id %lu number of nuclei %lu feature size %lu\n", rank, i, features.size(), (features.size() > 0) ? features[0].size() : 0);

		// clear features.
		features.clear();
	}

	// TODO: reads appears to be async - not all need to participate.  - opportunity for dynamic reads.
	// or is this the right way of doing things?
	// TODO: use a more dynamic array with dynamic number of dimensions for offset, size.

	adios_gclose(g);


	adios_fclose(fc);


	MPI_Barrier(comm_world);
	MPI_Finalize();
	return 0;

}

