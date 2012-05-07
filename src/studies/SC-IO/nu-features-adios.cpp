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
	printf("opening file: %s\n", filename);
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
	printf("rank %d tileinfo total = %ld dims %d timedim %d\n", rank, tileInfo_total, v->ndim, v->timedim );
	adios_free_varinfo(v);

	v = adios_inq_var(gc, "imageName_total");
	long imageName_total = *((long *)(v->value));
	printf("rank %d imagename total = %ld \n", rank, imageName_total);
	adios_free_varinfo(v);

	v = adios_inq_var(gc, "tile_total");
	long tile_total = *((long *)(v->value));
	printf("rank %d tile total = %ld \n", rank, tile_total );
	adios_free_varinfo(v);
	
	adios_gclose(gc);
	adios_fclose(fc);
	
	// now partition the space and try to read the content
	uint64_t slice_size = tileInfo_total / size;  // each read at least these many
	uint64_t remainder = tileInfo_total % size;  // what's left

	printf("slice_size %lu, remainder %lu\n", slice_size, remainder);

	// each iteration reads 1 per process
	int tileSizeX, tileSizeY, channels, elemSize1, cvDataType, encoding;
	uint64_t tile_offset, tile_size;
	uint64_t start, count = 1, bytes_read = 0;
	void *tiledata = NULL;	
	

	uint64_t adios_groupsize, adios_totalsize, adios_buf_size;
	// then open the tileCount group to get some data.
	printf("opening file: %s\n", filename);
	ADIOS_FILE *f = adios_fopen(filename, comm_world);
	if (f == NULL) {
		printf("%s\n", adios_errmsg());
		return -1;
	}
	printf("opened file\n");

	ADIOS_GROUP *g = adios_gopen(f, "tileInfo");
	if (g == NULL) {
		printf("%s\n", adios_errmsg());
		return -1;
	}
	printf("opened group\n");

	// for some reason, it's still trying to compute time based characteristics.   but then adios crashes.
/*	ADIOS_VARINFO *v2 = adios_inq_var(g, "tileSizeX");
		if (v2 == NULL) printf("why is varinfo NULL?\n");
		printf("RANK %d tileSizeX ndims %d dim 0 ?? time dim %d\n", rank, v->ndim, v->timedim);
		adios_free_varinfo(v2);	
*/	
	char fn[256];
	int mat_size[2];
	int tileOffsetX, tileOffsetY;
	for (int i = 0; i < slice_size; ++i) {
		start = i * size + rank;

		// read the metadata.
		bytes_read = adios_read_var(g, "tileOffsetX", &start, &count, &tileOffsetX); 
		bytes_read = adios_read_var(g, "tileOffsetY", &start, &count, &tileOffsetY); 
		bytes_read = adios_read_var(g, "tileSizeX", &start, &count, &tileSizeX); 
		bytes_read = adios_read_var(g, "tileSizeY", &start, &count, &tileSizeY); 
		bytes_read = adios_read_var(g, "channels", &start, &count, &channels); 
		bytes_read = adios_read_var(g, "elemSize1", &start, &count, &elemSize1); 
		bytes_read = adios_read_var(g, "cvDataType", &start, &count, &cvDataType); 
		bytes_read = adios_read_var(g, "encoding", &start, &count, &encoding); 
		bytes_read = adios_read_var(g, "tile_offset", &start, &count, &tile_offset); 
		bytes_read = adios_read_var(g, "tile_size", &start, &count, &tile_size); 
		printf("tile info at rank %d: at %dx%d, %dx%dx%d, elem %d, cvType %d, encoding %d, bytes %ld at %ld\n", rank, tileOffsetX, tileOffsetY, tileSizeX, tileSizeY, channels, elemSize1, cvDataType, encoding, tile_size, tile_offset);


		// read the tile
		// TODO: may be able to reuse memory.
		tiledata = malloc(tile_size);
		memset(tiledata, 0, tile_size);
		bytes_read = adios_read_var(g, "tiles", &tile_offset, &tile_size, tiledata);
		printf ("rank %d requested read %ld bytes at offset %ld, actual read %ld bytes\n", rank, tile_size, tile_offset, bytes_read);


		mat_size[0] = tileSizeX;
		mat_size[1] = tileSizeY;
		::cv::Mat img(2, mat_size,cvDataType, tiledata);

		// test write out...
		//memset(fn, 0, 256);
		//sprintf(fn, "%s/test%d.png", argv[3], start);
		//imwrite(fn, img);
		
		img.release();
		free(tiledata);

	}



	// at the end, read the remainders.
	if (rank < remainder) {
		start = slice_size * size + rank;

		// read the metadata.
		bytes_read = adios_read_var(g, "tileOffsetX", &start, &count, &tileOffsetX); 
		bytes_read = adios_read_var(g, "tileOffsetY", &start, &count, &tileOffsetY); 
		bytes_read = adios_read_var(g, "tileSizeX", &start, &count, &tileSizeX); 
		bytes_read = adios_read_var(g, "tileSizeY", &start, &count, &tileSizeY); 
		bytes_read = adios_read_var(g, "channels", &start, &count, &channels); 
		bytes_read = adios_read_var(g, "elemSize1", &start, &count, &elemSize1); 
		bytes_read = adios_read_var(g, "cvDataType", &start, &count, &cvDataType); 
		bytes_read = adios_read_var(g, "encoding", &start, &count, &encoding); 
		bytes_read = adios_read_var(g, "tile_offset", &start, &count, &tile_offset); 
		bytes_read = adios_read_var(g, "tile_size", &start, &count, &tile_size); 
		printf("final tile info at rank %d: at %dx%d, %dx%dx%d, elem %d, cvType %d, encoding %d, bytes %ld at %ld\n", rank, tileOffsetX, tileOffsetY, tileSizeX, tileSizeY, channels, elemSize1, cvDataType, encoding, tile_size, tile_offset);


		// read the tile
		tiledata = malloc(tile_size);
		memset(tiledata, 0, tile_size);
		bytes_read = adios_read_var(g, "tiles", &tile_offset, &tile_size, tiledata);
		printf ("final rank %d requested read %ld bytes at offset %ld, actual read %ld bytes\n", rank, tile_size, tile_offset, bytes_read);
		
		mat_size[0] = tileSizeX;
		mat_size[1] = tileSizeY;
		::cv::Mat img(2, mat_size,cvDataType, tiledata);

		// test write out...
		//memset(fn, 0, 256);
		//sprintf(fn, "%s/test%d.png", argv[3], start);
		//imwrite(fn, img);
		
		img.release();
		free(tiledata);

	}

	// TODO: reads appears to be async - not all need to participate.  - opportunity for dynamic reads.
	// or is this the right way of doing things?
	// TODO: use a more dynamic array with dynamic number of dimensions for offset, size.

	adios_gclose (g);
	adios_fclose (f);


	MPI_Barrier (comm_world);
	MPI_Finalize ();
	return 0;

}

