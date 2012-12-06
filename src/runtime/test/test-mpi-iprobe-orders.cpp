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

#include <unistd.h>

int main (int argc, char **argv) {
	long long t1, t2, t3, t4;

	// parse inputs
	if (argc < 5) 
		printf("syntax:  %s io_node_count output_count data_size barrierOn\n", argv[0]);
	int io_node_count = atoi(argv[1]);
	int output_num = atoi(argv[2]);
	int data_size = atoi(argv[3]);
	bool barrierOn = strcmp(argv[4], "off") == 0 ? false : true;
	if (barrierOn) printf("MPI Barrier on.\n");
	else printf("MPI Barrier off.\n");

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
	MPI_Comm comm;
        MPI_Comm_split(comm_world, (isIO ? 1 : 0), rank, &comm);


    printf("comm-world:  %s: %d. %d of %d\n", hostname.c_str(), (isIO ? 1 : 0), rank, size);


	srand(rank);




	char notes[21];

	void *data;
	int node_id;
// set up the manager/worker
	if (isIO) {

		int hasMessage;
		MPI_Status mstatus;
		int recv_size = 0;
		int recv_count = 0;
		int iter = 0;
		char fn[256];
		unsigned char* dummy = NULL;
		FILE *fid;
		int callCount;		
	
		// initialize buffer
		data = malloc(data_size);
		memset(data, 0, data_size);

		int sender_count = size - io_node_count;
		int sender_id;

if (barrierOn)	MPI_Barrier(comm_world);
		// probe to get the data
		printf("[%d] ", rank);
		while (sender_count > 0) {
			sender_id = rand() % (size - io_node_count) + io_node_count;			
		        MPI_Iprobe(sender_id, 1, comm_world, &hasMessage, &mstatus);
        		if (hasMessage) {

				if (callCount % 200 == 0) printf("\n[%d] ", rank);

				callCount++;
	                	node_id = mstatus.MPI_SOURCE;
        	                MPI_Get_count(&mstatus, MPI_CHAR, &recv_size);
				if (recv_size == 0) {
					// done for that worker
				printf("(%d),", node_id); 
					MPI_Recv(dummy, 0, MPI_CHAR, node_id, 1, comm_world, MPI_STATUS_IGNORE);

					--sender_count;
						
				} else {
					// data.  put into buffer
				printf("%d,", node_id); 
					MPI_Recv(data, recv_size, MPI_CHAR, node_id, 1, comm_world, MPI_STATUS_IGNORE); 
					++recv_count;
	
				}
			
			}

		}
		printf("\n");
	} else {
		data = malloc(data_size);

		// send fixed number of data items.

		printf("[%d] ", rank);
	        MPI_Request *reqs = new MPI_Request[output_num];
		for (int i = 0; i < output_num; ++i) {
			sleep(1);

			node_id = rand() % io_node_count;
			printf("%d,", node_id);
			//MPI_Send(data, data_size, MPI_CHAR, node_id, 1, comm_world);
			MPI_Isend(data, data_size, MPI_CHAR, node_id, 1, comm_world, &(reqs[i]));

		}
if (barrierOn)	MPI_Barrier(comm_world);

		MPI_Waitall(output_num, reqs, MPI_STATUSES_IGNORE);
		delete [] reqs;			
	
                reqs = new MPI_Request[io_node_count];
		for (int i = 0; i < io_node_count; ++i) {
			printf("(%d),", i);
               		MPI_Isend(data, 0, MPI_CHAR, i, 1, comm_world, &(reqs[i]));
			
		}
		printf("\n");
		MPI_Waitall(io_node_count, reqs, MPI_STATUSES_IGNORE);
		delete [] reqs;			
	}
// clean up
	free(data);

	MPI_Comm_free(&comm);
	MPI_Finalize();
}
