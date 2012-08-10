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
#include <algorithm>
#include "adios_read.h"
#include "adios.h"
#include "adios_internals.h"
#include "opencv2/opencv.hpp"
#include "RegionalMorphologyAnalysis.h"
#include "PixelOperations.h"
#include "FileUtils.h"

#include <unistd.h>

void readData(ADIOS_GROUP *g,const  uint64_t *start, const uint64_t *count, const int rank, void* &tiledata, ::cv::Mat &img,
		char* &sourceTileFiledata, int &tileOffsetX, int &tileOffsetY, char* &imageName, long &imageName_offset, long &imageName_size) {

	int tileSizeX, tileSizeY, channels, elemSize1, cvDataType, encoding;
	uint64_t sourceTileFile_offset, sourceTileFile_size, tile_offset, tile_size;
	uint64_t bytes_read = 0;

	// read the metadata.
	// TODO:  tileoffset, etc should be a 1xcount array.
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
	bytes_read = adios_read_var(g, "imageName_offset", start, count, &imageName_offset);
	bytes_read = adios_read_var(g, "imageName_size", start, count, &imageName_size);
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

	// read the tile
	// TODO: may be able to reuse memory.
	imageName = (char*)malloc(imageName_size + 1);
	memset(imageName, 0, imageName_size + 1);
	uint64_t istart = imageName_offset;
	uint64_t isize = imageName_size;
	bytes_read = adios_read_var(g, "imageName", &istart, &isize, imageName);
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


void computeFeatures(::cv::Mat image, ::cv::Mat mask, std::vector<std::vector<float> > &features) {
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
}


int main (int argc, char **argv) {


	if (argc < 5) {
		printf("usage: %s maskfilename rgbfilename outdirname <NULL | POSIX | MPI | MPI_LUSTRE | MPI_AMR>", argv[0]);
		return -1;
	}

	std::string executable(argv[0]);
	FileUtils futils;
	std::string workingDir;
	workingDir.assign(futils.getDir(executable));

	std::string iocode;
	if (argc > 4 && strcmp(argv[4], "NULL") != 0 &&
			strcmp(argv[4], "POSIX") != 0 &&
			strcmp(argv[4], "MPI") != 0 &&
			strcmp(argv[4], "MPI_LUSTRE") != 0 &&
			strcmp(argv[4], "MPI_AMR") != 0) {
		printf("usage: %s maskfilename rgbfilename outdirname <NULL | POSIX | MPI | MPI_LUSTRE | MPI_AMR>", argv[0]);
		return -1;

	} else {
		iocode.assign(argv[4]);
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

	v = adios_inq_var(gc, "imageName_total");
	long imageName_total = *((long *)(v->value));
	if (rank == 0) printf("imageName total = %ld \n", imageName_total);
	adios_free_varinfo(v);

	v = adios_inq_var(gc, "tile_total");
	long tile_total = *((long *)(v->value));
	if (rank == 0) 	printf("tile total = %ld \n", tile_total );
	adios_free_varinfo(v);
	
	adios_gclose(gc);
	adios_fclose(fc);	

	// now partition the space and try to read the content
	uint64_t remainder = tileInfo_total % size;  // what's left
	uint64_t mpi_tileInfo_total = remainder > 0 ? tileInfo_total - remainder + size : tileInfo_total;  // each read at least these many

//	if (rank == 0) printf("slice_size %d, number of slices %lu, remainder %lu\n", size, slice_size, remainder);

	// each iteration reads 1 per process
	uint64_t start, count = 1;




//	// for some reason, it's still trying to compute time based characteristics.   but then adios crashes.
//	ADIOS_VARINFO *v2 = adios_inq_var(g, "tileSizeX");
//	if (v2 == NULL) printf("why is varinfo NULL?\n");
//	printf("RANK %d tileSizeX ndims %d dim n-1 %ld time dim %d\n", rank, v->ndim, v->dims[v->ndim-1], v->timedim);
//	adios_free_varinfo(v2);


	void *tiledata = NULL;
	char *sourceTileFiledata = NULL;
	char fn[256];
	std::vector<std::vector<float> > features;
	bool newfile = true;
	int err;
	int64_t adios_handle;
	uint64_t adios_groupsize, adios_totalsize;

	int tileOffsetX, tileOffsetY;

	long nuFeatureInfo_capacity = std::numeric_limits<long>::max();
	long nuFeatureInfo_pg_offset = 0;
	long nuFeatureInfo_pg_size = 0;
	long nuFeatureInfo_total = 0;
	int feature_count = 74;
	int ndims = 2;
	float *boundingBoxOffset;
	float *boundingBoxSize;
	float *centroid;
	long *imageName_offset;  // replicate for each nu
	long *imageName_size;    // replicate for each nu
	float *feature;

	long imageName_capacity = imageName_total;
	long imageName_pg_offset = 0; // get from the input.  just copy it over.
	long imageName_pg_size = 0; // get from the input.  just copy it over.
	char *imageName;  // get from the input.  just copy it over.

	long nuFeatureInfo_step_total;

	int pos;


	std::string adios_config = workingDir;
	adios_config.append("/../adios_xml/nu-features-globalarray-");
	adios_config.append(iocode);
	adios_config.append(".xml");


	adios_init(adios_config.c_str());

	features.clear();
	for (uint64_t i = rank; i < mpi_tileInfo_total; i += size ) {


		fc = adios_fopen(filename, comm_world);
		if (fc == NULL) {
			printf("%s\n", adios_errmsg());
			return -1;
		}
		// then open the tileCount group to get some data.
		ADIOS_GROUP *g = adios_gopen(fc, "tileInfo");
		if (g == NULL) {
			printf("%s\n", adios_errmsg());
			return -1;
		}

		if (i < tileInfo_total) {  // only compute if there is something to compute....
			start = i;
			// read data and convert to cv MAT
			::cv::Mat mask;

			readData(g, &start, &count, rank, tiledata, mask, sourceTileFiledata,
					tileOffsetX, tileOffsetY, imageName, imageName_pg_offset, imageName_pg_size);



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
			computeFeatures(img, mask, features);

			// release images
			img.release();
			//printf("tiledata pointer %lu\n", (uint64_t)tiledata);
			mask.release();
			free(tiledata);
		}
		adios_gclose(g);
		adios_fclose(fc);
	
		// everyone has to write for adios.
		printf("RANK %d id %lu number of nuclei %lu feature size %lu\n", rank, i, features.size(), (features.size() > 0) ? features[0].size() : 0);

		// can I have a read file and a write file open at the same time?
		// set up the output variables

		if (features.size() > 0) {

			// now the features
			nuFeatureInfo_pg_size = features.size();

			// allocate the member
			boundingBoxOffset = (float *)malloc(nuFeatureInfo_pg_size * ndims * sizeof(float));
			boundingBoxSize = (float *)malloc(nuFeatureInfo_pg_size * ndims * sizeof(float));
			centroid = (float*)malloc(nuFeatureInfo_pg_size * ndims * sizeof(float));
			// image name is not changed.  just copied.
			imageName_offset = (long *)malloc(nuFeatureInfo_pg_size * sizeof(long));
			imageName_size = (long *)malloc(nuFeatureInfo_pg_size * sizeof(long));
			feature = (float *)malloc(nuFeatureInfo_pg_size * feature_count * sizeof(float));

			// now populate
			fill(imageName_offset, imageName_offset+nuFeatureInfo_pg_size, imageName_pg_offset);
			fill(imageName_size, imageName_size+nuFeatureInfo_pg_size, imageName_pg_size);
			for (long j = 0; j < nuFeatureInfo_pg_size; ++j) {

				copy(features[j].begin(), features[j].begin() + ndims, boundingBoxOffset + j *ndims);
				*(boundingBoxOffset + j * ndims) += tileOffsetX;
				*(boundingBoxOffset + j * ndims + 1) += tileOffsetY;

				copy(features[j].begin() + ndims, features[j].begin() + 2*ndims, boundingBoxSize + j *ndims);
				copy(features[j].begin() + 2*ndims, features[j].begin() + 3*ndims, centroid + j *ndims);
				*(centroid + j * ndims) += tileOffsetX;
				*(centroid + j * ndims + 1) += tileOffsetY;

				copy(features[j].begin() + 3*ndims, features[j].end(), feature + j * feature_count);
			}
		}

		/**
		* compute the offset for each step, in global array coordinates
		*/
		// offset within timestep
		MPI_Scan(&nuFeatureInfo_pg_size, &nuFeatureInfo_pg_offset, 1, MPI_LONG, MPI_SUM, comm_world);
		// offset in global coord
		nuFeatureInfo_pg_offset = nuFeatureInfo_pg_offset - nuFeatureInfo_pg_size + nuFeatureInfo_total;
		// number of nuclei in this timestep
		MPI_Allreduce(&nuFeatureInfo_pg_size, &nuFeatureInfo_step_total, 1, MPI_LONG, MPI_SUM, comm_world);
		// all nuclei so far
		nuFeatureInfo_total += nuFeatureInfo_step_total;

		// WRITE OUT
		if (newfile) {
			printf("RANK %d adios handle %ld, filename %s\n", rank, adios_handle, argv[3]);
			err = adios_open(&adios_handle, "nuFeatureInfo", argv[3], "w", &comm_world);
			newfile = false;
		} else {
			err = adios_open(&adios_handle, "nuFeatureInfo", argv[3], "a", &comm_world);
		}

		// do the actual write
		if (nuFeatureInfo_pg_size <= 0) {
			err = adios_group_size (adios_handle, 0, &adios_totalsize);

		} else {
	#include "gwrite_nuFeatureInfo.ch"

		}

		// close the file and flush.
		// if time_index is not specified, then let ADIOS handle it.
	    struct adios_file_struct * fd = (struct adios_file_struct *) adios_handle;
	    struct adios_group_struct *gd = (struct adios_group_struct *) fd->group;
		gd->time_index = 1;
		err = adios_close(adios_handle);

		// once written, clear the feature vector and get ready for next.
		if (features.size() > 0) {
			free(boundingBoxOffset);
			free(boundingBoxSize);
			free(centroid);
			free(imageName_offset);
			free(imageName_size);
			free(feature);
			free(imageName);

		}

		features.clear();
		nuFeatureInfo_pg_size = 0;
		nuFeatureInfo_pg_offset = 0;
	}

	// and now write out the counts (actual size used.)
	if (newfile) {
		printf("RANK %d adios handle %ld, filename %s\n", rank, adios_handle, argv[3]);
		err = adios_open(&adios_handle, "nuFeatureDims", argv[3], "w", &comm_world);
		newfile = false;
	} else {
		err = adios_open(&adios_handle, "nuFeatureDims", argv[3], "a", &comm_world);
	}

#include "gwrite_nuFeatureDims.ch"
	err = adios_close(adios_handle);



	// TODO: reads appears to be async - not all need to participate.  - opportunity for dynamic reads.
	// or is this the right way of doing things?
	// TODO: use a more dynamic array with dynamic number of dimensions for offset, size.
	adios_finalize(rank);

	MPI_Barrier(comm_world);
	MPI_Finalize();
	return 0;

}

