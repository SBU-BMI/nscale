/*
 * test.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
#include <iostream>
#include <string.h>
#include "mpi.h"
#include <cstdlib>
#include <string>
#include "pthread.h"
#include "SCIOUtilsLogger.h"

#include <unistd.h>

int main (int argc, char **argv) {
	long long t1, t2, t3, t4;

	// parse inputs
	if (argc < 6) 
		printf("syntax:  %s io_node_count io_group_size output_count io_buffer_size data_size\n", argv[0]);
	int io_node_count = atoi(argv[1]);
	int io_group_size = atoi(argv[2]);
	int output_num = atoi(argv[3]);
	int buffer_size = atoi(argv[4]);
	int data_size = atoi(argv[5]);
	data_size = data_size * data_size * 4;

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

    if (size == 1) {
    	printf("num procs needs to be at least 2.\n");
    	return -1;
    }

	bool isIO = rank < io_node_count;
	int groupid = rank / io_group_size;
	MPI_Comm comm;
        MPI_Comm_split(comm_world, (isIO ? 2 + groupid : 0), rank, &comm);


    printf("comm-world:  %s: %d. %d of %d\n", hostname.c_str(), (isIO ? 2 + groupid : 0), rank, size);


	srand(rank);


// initialize the logger 
        cciutils::SCIOLogger *logger = new cciutils::SCIOLogger(rank, hostname, 0);
	cciutils::SCIOLogSession *logsession = logger->getSession((isIO ? "io" : "compute"));


	char notes[21];

	void *data;
	int node_id;
// set up the manager/worker
	if (isIO) {
		t1 = ::cciutils::event::timestampInUS();

		int hasMessage;
		MPI_Status mstatus;
		int recv_size = 0;
		int recv_count = 0;
		int iter = 0;
		char fn[256];
		unsigned char* dummy = NULL;
		FILE *fid;
		int max_count = 0;
		
	int callCount = 0;
	
		// initialize buffer
		data = malloc(data_size * buffer_size);
		memset(data, 0, data_size * buffer_size);

		int sender_count = size - io_node_count;
		t2 = ::cciutils::event::timestampInUS();
		logsession->log(cciutils::event(0, std::string("setup"), t1, t2, std::string(), ::cciutils::event::NETWORK_WAIT));

		t1 = ::cciutils::event::timestampInUS();
		
		// probe to get the data
		while (sender_count > 0 || max_count > 0) {

		        MPI_Iprobe(MPI_ANY_SOURCE, 1, comm_world, &hasMessage, &mstatus);
        		if (hasMessage) {
//			printf("%d, %d senders, %d max IO previously", rank, sender_count, max_count);
			if (callCount % 200 == 0) printf("\n%d ", rank);
	                	node_id = mstatus.MPI_SOURCE;
        	                MPI_Get_count(&mstatus, MPI_CHAR, &recv_size);
//printf(".\n");	
				if (recv_size == 0) {
					// done for that worker
//					printf("%d ended by %d \n", rank, node_id);
					printf("e");
					MPI_Recv(dummy, 0, MPI_CHAR, node_id, 1, comm_world, MPI_STATUS_IGNORE);

					--sender_count;
						t2 = ::cciutils::event::timestampInUS();
			memset(notes, 0, 21);
		sprintf(notes, "%d", node_id);
					logsession->log(cciutils::event(0, std::string("worker end"), t1, t2, std::string(notes), ::cciutils::event::NETWORK_IO));
						
				} else {
					// data.  put into buffer
//					printf("%d receiving from %d\n", rank, node_id);
					printf("r");
					MPI_Recv((unsigned char*)data + recv_count * data_size, recv_size, MPI_CHAR, node_id, 1, comm_world, MPI_STATUS_IGNORE);
					++recv_count;
	
					t2 = ::cciutils::event::timestampInUS();
			memset(notes, 0, 21);
		sprintf(notes, "%d", recv_size);
					
					logsession->log(cciutils::event(0, std::string("recv"), t1, t2, std::string(notes), ::cciutils::event::NETWORK_IO));
				}
			
				t1 = ::cciutils::event::timestampInUS();
				callCount++;
			}

			// call this with or without messages.  some nodes may not get a message and we want to make sure all processes call Allreduce the same number of times.
					// and write out if needed.
					// can't just call this here - sender_count may be 0 for another process
					//t3 = ::cciutils::event::timestampInUS();
				//	printf("%d allreduce\n", rank);
					MPI_Allreduce(&recv_count, &max_count, 1, MPI_INT, MPI_MAX, comm);
					//t4 = ::cciutils::event::timestampInUS();
					//logsession->log(cciutils::event(0, std::string("check to write"), t3, t4, std::string(), ::cciutils::event::NETWORK_WAIT));

					if (max_count >= buffer_size ||
						(sender_count == 0 && recv_count > 0)) {
						// simulate coordinated write	
					printf("%d", iter);


						for (int j = 0; j < recv_count; ++j) {
							t3 = ::cciutils::event::timestampInUS();
							memset(fn, 0, 256);
							sprintf(fn, "%d.%d.%d.raw", rank, iter, j);
							// 	printf("%d writing\n", rank);
							printf("w");
				                        fid = fopen(fn, "wb");
				                        if (!fid) {
	        	                		        printf("ERROR: can't open %s to write\n", fn);
		        		                } else {
                				                fwrite((unsigned char*)data + j * data_size, 1, data_size, fid);
			                	                fclose(fid);
	                        			}
							t4 = ::cciutils::event::timestampInUS();
		memset(notes, 0, 21);
		sprintf(notes, "%d", data_size);
	
							logsession->log(cciutils::event(0, std::string("write"), t3, t4, std::string(notes), ::cciutils::event::FILE_O));

						}						
						++iter;
						recv_count = 0;
						memset(data, 0, data_size * buffer_size);
				t1 = ::cciutils::event::timestampInUS();
					}


		}
	} else {
		data = malloc(data_size);

		// send fixed number of data items.

		printf("%d ", rank);
		for (int i = 0; i < output_num; ++i) {
			t1 = ::cciutils::event::timestampInUS();
			sleep(1);
			t2 = ::cciutils::event::timestampInUS();
			logsession->log(cciutils::event(0, std::string("compute"), t1, t2, std::string(), ::cciutils::event::COMPUTE));

			t1 = ::cciutils::event::timestampInUS();
			memset(data, i, data_size);
			node_id = rand() % io_node_count;
			//			printf("%d sending to %d\n", rank, node_id);
			printf("s");
			MPI_Send(data, data_size, MPI_CHAR, node_id, 1, comm_world);
			t2 = ::cciutils::event::timestampInUS();
			memset(notes, 0, 21);
		sprintf(notes, "%d", node_id);
			logsession->log(cciutils::event(0, std::string("send"), t1, t2, std::string(notes), ::cciutils::event::NETWORK_IO));


		}
			
		printf("\n%d ", rank);
		// send zero length data to everyone to mark end.
		t1 = ::cciutils::event::timestampInUS();
                MPI_Request *reqs = new MPI_Request[io_node_count];
		for (int i = 0; i < io_node_count; ++i) {
			//printf("%d ending for %d\n", rank, i);
			printf("t");
               		MPI_Isend(data, 0, MPI_CHAR, i, 1, comm_world, &(reqs[i]));
		}
		MPI_Waitall(io_node_count, reqs, MPI_STATUSES_IGNORE);
		delete [] reqs;			
		t2 = ::cciutils::event::timestampInUS();
		logsession->log(cciutils::event(0, std::string("worker finish"), t1, t2, std::string(), ::cciutils::event::NETWORK_WAIT));


	}
// clean up
	printf("\n");
	free(data);

        logger->writeCollectively(std::string("mpi_datasize_test"), rank, 0, comm_world);

	
	delete logger;

	MPI_Comm_free(&comm);
	MPI_Finalize();
}
