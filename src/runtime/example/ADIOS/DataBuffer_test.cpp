/*
 * CVImagetest.cpp
 *
 *  Created on: Jul 9, 2012
 *      Author: tcpan
 */

#include "CVImage.h"
#include <string>
#include <vector>
#include "FileUtils.h"
#include "DataBuffer.h"
#include "MPISendDataBuffer.h"
#include "MPIRecvDataBuffer.h"
#include <cstdlib>
#include <unistd.h>


using namespace std;
using namespace cci::rt;

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

	cci::rt::adios::CVImage *src, *dest1, *dest2, *dest3;
	int data_pos = 0, mx_image_bytes = im.rows * im.cols * im.elemSize();
	int imageNames_pos = 0, mx_imagename_bytes = 256;
	int filenames_pos = 0, mx_filename_bytes = 1024;
	int data_size, img_name_size, src_fn_size;
	int iterations = 1;

	printf("max image bytes = %d\n", mx_image_bytes);

	const unsigned char* tile;
	const char* imageName;
	const char* sourceTileFile;


	int dummy, dummy2;
	src = new cci::rt::adios::CVImage(im, imagename, fn, tilex, tiley);
	printf("orig image name %s, filename %s, data size %d\n", src->getImageName(dummy, dummy2), src->getSourceFileName(dummy, dummy2), src->getMetadata().info.data_size);


	int output_size = 0;
	void *output = NULL;

	src->serialize(output_size, output);

	///// create buffer with capacity of 4
	DataBuffer buffer(4);

	// add in
	int status;
	DataBuffer::DataType in;
	for (int i = 0; i < 5; ++i) {
		in = std::make_pair(output_size, output);
		status = buffer.push(in);
		printf("buffer testing: iter %d, buffer push status %d, size %d, outdata %d %p\n", i, status, buffer.getBufferSize(), in.first, in.second);
	}

	// pop out;
	DataBuffer::DataType out;
	for (int i = 0; i < 5; ++i) {
		status = buffer.pop(out);
		printf("buffer testing: iter %d, buffer pop status %d, size %d, outdata %d %p\n", i, status, buffer.getBufferSize(), out.first, out.second);
	}

#if defined(WITH_MPI)
	printf("TESTING MPI!!!!!\n");


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

    if (size > 1) {

		int hasMessage;
		char done = 0;
		MPI_Status stat, stat2;

		if (rank == 0) {
			MPIRecvDataBuffer mbuffer(8);

			int input_size = 0;
			void *input = NULL;

			DataBuffer::DataType* dataitems;
			int count;

			int activeCount = size - 1;
			int *actives = new int[activeCount];
			for (int i = 0; i < activeCount; ++i) {
				actives[i] = 1;
			}

			while (done == 0) {
				MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_world, &hasMessage, &stat);

				if (hasMessage) {
					
					MPI_Get_count(&stat, MPI_CHAR, &input_size);

					int target = stat.MPI_SOURCE;
					int tag = stat.MPI_TAG;


					if (stat.MPI_TAG == 1) {
						printf("worker %d done\n", stat.MPI_SOURCE);
						int s;
						MPI_Recv(&s, 1, MPI_CHAR, stat.MPI_SOURCE, stat.MPI_TAG, comm_world, &stat2);

						if (actives[stat.MPI_SOURCE - 1] == 1) {
							actives[stat.MPI_SOURCE - 1] = 0;
							activeCount--;
						}
						if (activeCount == 0) {
							done = 1;
						}
						continue;
					}


					if (mbuffer.canPushMPI()) {
						printf("receiving %d bytes from source %d, tag %d\n", input_size, stat.MPI_SOURCE, stat.MPI_TAG);

						input = malloc(input_size);
						in = std::make_pair(input_size, input);


						MPI_Request *req = new MPI_Request[1];
						MPI_Irecv(input, input_size, MPI_CHAR, stat.MPI_SOURCE, stat.MPI_TAG, comm_world, req);
	//					int completed;
	//					MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
	//					MPI_Recv(input, input_size, MPI_CHAR, target, tag, comm_world, MPI_STATUS_IGNORE);
//						if (completed == 0) {

						status = mbuffer.pushMPI(req, in);

						printf("MPIRecvDataBuffer size %d, %d, current status is %d \n", mbuffer.getMPIBufferSize(), mbuffer.getBufferSize(), status);
//						} else {
//							printf("already completed \n");
//							if (mbuffer.canPush()) {
//								mbuffer.push(in);
//							}
//						}
					}
					if (mbuffer.isFull()) {
						if (mbuffer.canPopMPI()) {
							count = mbuffer.popMPI(dataitems);
							for (int i = 0; i < count; ++i) {
								printf("MPI receiv buffer completed: count %d, remain %d, outdata %d %p \n", count, mbuffer.getMPIBufferSize(), dataitems[i].first, dataitems[i].second);
							}
							if (dataitems != NULL) delete [] dataitems;
						}

						while (mbuffer.canPop()) {
							status = mbuffer.pop(out);
							printf("mpi recv buffer testing: buffer pop status %d, size %d, outdata %d %p\n", status, mbuffer.getBufferSize(), out.first, out.second);
							free(out.second);
						}


					}


				}
			}


			if (mbuffer.canPopMPI()) {
				count = mbuffer.popMPI(dataitems);
				for (int i = 0; i < count; ++i) {
					printf("MPI receiv buffer completed: count %d, remain %d, outdata %d %p \n", count, mbuffer.getMPIBufferSize(), dataitems[i].first, dataitems[i].second);
				}
				if (dataitems != NULL) delete [] dataitems;
			}

			while (mbuffer.canPop()) {
				status = mbuffer.pop(out);
				printf("mpi recv buffer testing: buffer pop status %d, size %d, outdata %d %p\n", status, mbuffer.getBufferSize(), out.first, out.second);
				free(out.second);
			}
			printf("buffer size = %d, %d\n", mbuffer.getBufferSize(), mbuffer.getMPIBufferSize());

		} else {

			MPISendDataBuffer mbuffer(4);

			for (int i = 0; i < 4; ++i) {
				void* out2 = malloc(output_size);
				memcpy(out2, output, output_size);

				in = std::make_pair(output_size, out2);
				status = mbuffer.push(in);
//				printf("mpi send buffer testing: iter %d, buffer push status %d, size %d, outdata %d %p\n", i, status, mbuffer.getBufferSize(), in.first, in.second);
			}

			// pop out and push into MPI one...;
			while (mbuffer.canPop()) {
				status = mbuffer.pop(out);
				printf("mpi send buffer testing: buffer pop status %d, size %d, outdata %d %p\n", status, mbuffer.getBufferSize(), out.first, out.second);

				if (mbuffer.canPushMPI()) {
					MPI_Request *req = new MPI_Request[1];
					MPI_Isend(out.second, out.first, MPI_CHAR, 0, 0, comm_world, req);

//					int test = 0;
//					while (test == 0) {
//						MPI_Test(&req, &test, MPI_STATUS_IGNORE);
//					}
//					printf("completed\n");
					status = mbuffer.pushMPI(req, out);
				} else {
					mbuffer.push(out);
					printf("unable to push MPI?\n");
				}
			}

			DataBuffer::DataType* dataitems = NULL;
			int count = 0;
			while (mbuffer.canPopMPI()) {
			
				count = mbuffer.popMPI(dataitems);
				for (int i = 0; i < count; ++i) {
					printf("MPI send buffer completed: count %d of %d, remain %d, outdata %d %p \n", i+1, count, mbuffer.getMPIBufferSize(), dataitems[i].first, dataitems[i].second);
					free(dataitems[i].second);
				}
				delete [] dataitems;
				dataitems = NULL;
			}

			char done = 1;
			MPI_Send(&done, 1, MPI_CHAR, 0, 1, comm_world);

		}
    }
//	dest3 = new cci::rt::adios::CVImage(output_size, output);
//	printf("deserialized image name %s, filename %s, data size %d\n\n", dest3->getImageName(dummy, dummy2), dest3->getSourceFileName(dummy, dummy2), dest3->getMetadata().info.data_size);


#endif


	// clean up
	delete src;

	cci::rt::adios::CVImage::freeSerializedData(output);

	im.release();


	MPI_Finalize();
	return 0;


}
