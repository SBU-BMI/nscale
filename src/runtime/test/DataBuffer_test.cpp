/*
 * CVImagetest.cpp
 *
 *  Created on: Jul 9, 2012
 *      Author: tcpan
 */

#include <string>
#include <vector>
#include "FileUtils.h"
#include "DataBuffer.h"
#include "MPISendDataBuffer.h"
#include "MPIRecvDataBuffer.h"
#include <cstdlib>
#include <unistd.h>
#include "Debug.h"


using namespace std;
using namespace cci::rt;

int main (int argc, char **argv){

	// init MPI
	int threading_provided;
	int err  = MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &threading_provided);

	int rank = MPI_UNDEFINED;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int size = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int output_size = 4096*4096*4;
	void *output = 0;
	void *dummy = malloc(rank * 1000);

	///// create buffer with capacity of 4
	DataBuffer buffer(4);

	// add in
	int status;
	DataBuffer::DataType in;
	for (int i = 0; i < 5; ++i) {
		output = malloc(output_size);
		memset(output, rank * 10 + i, output_size);
		cci::common::Debug::print("buffer testing: output %p in.second %p\n", output, in.second);

		in = std::make_pair(output_size, output);
		cci::common::Debug::print("buffer testing 2: output %p in.second %p\n", output, in.second);
		if (buffer.canPush()) {
			status = buffer.push(in);
			cci::common::Debug::print("buffer testing: iter %d, buffer push status %d, size %ld, outdata %d %p %d\n", i, status, buffer.debugBufferSize(), in.first, in.second, ((char*)in.second)[0]);
		} else {
			cci::common::Debug::print("Buffer full!  data discarded\n");
			free(output);
		}
	}

	// pop out;
	DataBuffer::DataType out;
	for (int i = 0; i < 5; ++i) {
		if (buffer.canPop()) {
			status = buffer.pop(out);
			cci::common::Debug::print("buffer testing: iter %d, buffer pop status %d, size %ld, outdata %d %p %d\n", i, status, buffer.debugBufferSize(), out.first, out.second, ((char*)out.second)[0]);
			free(out.second);
			out.second = NULL;
		}
	}

	free(dummy);

	printf("TESTING MPI!!!!!\n");


#if defined(WITH_MPI)
	if (size > 1)  {

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

			int activeCount = size-1;
			int *actives = new int[size];
			for (int i = 0; i < size; ++i) {
				actives[i] = 1;
			}
			actives[0] = 0;

			while (done == 0) {
				MPI_Iprobe(MPI_ANY_SOURCE, 0, comm_world, &hasMessage, &stat);

				if (hasMessage) {
					
					MPI_Get_count(&stat, MPI_CHAR, &input_size);

					int target = stat.MPI_SOURCE;

					if (input_size == 0) {
						cci::common::Debug::print("worker %d done\n", stat.MPI_SOURCE);
						char s;
						MPI_Recv(&s, 0, MPI_CHAR, stat.MPI_SOURCE, 0, comm_world, &stat2);

						if (actives[stat.MPI_SOURCE] == 1) {
							actives[stat.MPI_SOURCE] = 0;
							activeCount--;
						}
						if (activeCount == 0) {
							done = 1;
						}
						continue;
					}


					if (mbuffer.canTransmit()) {
						cci::common::Debug::print("receiving %d bytes from source %d, tag %d\n", input_size, stat.MPI_SOURCE, stat.MPI_TAG);

						status = mbuffer.transmit(stat.MPI_SOURCE, stat.MPI_TAG, MPI_CHAR, comm_world, input_size);


						cci::common::Debug::print("MPIRecvDataBuffer size %ld, current status is %d \n", mbuffer.debugBufferSize(), status);

					}
					if (mbuffer.isFull()) {

						while (mbuffer.canPop()) {
							status = mbuffer.pop(out);
							cci::common::Debug::print("mpi recv buffer testing: buffer pop status %d, size %ld, outdata %d %p\n", status, mbuffer.debugBufferSize(), out.first, out.second);
							free(out.second);
							out.second = NULL;
						}


					}


				}
			}



			while (mbuffer.canPop()) {
				status = mbuffer.pop(out);
				cci::common::Debug::print("mpi recv buffer testing: buffer pop status %d, size %ld, outdata %d %p\n", status, mbuffer.debugBufferSize(), out.first, out.second);
				free(out.second);
				out.second = NULL;
			}
			printf("buffer size = %ld\n", mbuffer.debugBufferSize());

		} else {

			MPISendDataBuffer mbuffer(4);

			for (int i = 0; i < 4; ++i) {
				output = malloc(output_size);
				memset(output, i, output_size);

				in = std::make_pair(output_size, output);
				status = mbuffer.push(in);
//				printf("mpi send buffer testing: iter %d, buffer push status %d, size %d, outdata %d %p\n", i, status, mbuffer.getBufferSize(), in.first, in.second);
			}
			cci::common::Debug::print("MPI send worker set up done.\n");

			// pop out and push into MPI one...;
			while (mbuffer.canTransmit()) {
//				printf("%d sending data to manager\n", rank);
				mbuffer.transmit(0, 0, MPI_CHAR, comm_world, -1);
			}
			cci::common::Debug::print("MPI send worker transmit staged.\n");

			mbuffer.stop();
			DataBuffer::DataType* dataitems = NULL;
			int count = 0;
			while (!mbuffer.isFinished()) ;
			
			cci::common::Debug::print("MPI send buffer completed: count %d, remain %ld \n", mbuffer.debug_complete_count, mbuffer.debugBufferSize());

			char done = 1;
			MPI_Send(&done, 0, MPI_CHAR, 0, 0, comm_world);

		}
    }
//	dest3 = new cci::rt::adios::CVImage(output_size, output);
//	printf("deserialized image name %s, filename %s, data size %d\n\n", dest3->getImageName(dummy, dummy2), dest3->getSourceFileName(dummy, dummy2), dest3->getMetadata().info.data_size);

	}
#endif



	MPI_Finalize();
	return 0;


}
