/*
 * Assign_test.cpp
 *
 *  Created on: Jun 26, 2012
 *      Author: tcpan
 */


#include "Save.h"
#include "mpi.h"
#include <vector>


int main (int argc, char **argv){
	int threading_provided = 0;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &threading_provided);

	MPI_Comm comm_world;
	//comm_world = MPI_COMM_NULL;
	comm_world = MPI_COMM_WORLD;

	int size=0, rank=0;
	MPI_Comm_size(comm_world, &size);
	MPI_Comm_rank(comm_world, &rank);

	std::vector<cci::rt::Communicator_I *> handlers;



	// then within IO group, split to subgroups, for adios.
	int group_size = 12;
	int group_interleave = 4;
	int g2 = -1;

	int io_root = 0;
	if (group_size == 1) {
		g2 = rank;
	} else if (group_size < 1) {
		if (rank == io_root) g2 = 3;
		else g2 = 4;
	} else {
		if (rank == io_root) g2 = 0;
		else {
			if (group_interleave > 1) {
				int blockid = rank / (group_size * group_interleave);
				g2 = blockid * group_interleave + rank % group_interleave;
			} else {
				g2 = rank / group_size;
			}
			++g2;
		}
	}

	cci::rt::DataBuffer *rbuf = new cci::rt::DataBuffer(10);
	cci::rt::Action_I *save = new cci::rt::Save(&comm_world, g2, rbuf, NULL, NULL);
	handlers.push_back(save);

	int j = 0;
	int count = sizeof(int);
	void *data = NULL;
	int *temp = NULL;
	cci::rt::DataBuffer::DataType dstr;
	int stat;

	int result = cci::rt::Communicator_I::READY, oresult = cci::rt::Communicator_I::READY;
	while (!handlers.empty() ) {


		if (j >= 10 & j < 30) {
			data = malloc(count);
			temp = (int*) data;
			temp[0] = j;
			dstr = std::make_pair(count, data);
			stat = save->getInputBuffer()->push(dstr);
			if (stat == cci::rt::DataBuffer::STOP ||
					stat == cci::rt::DataBuffer::FULL ||
					stat == cci::rt::DataBuffer::BAD_DATA) {
				printf("input added at iteration j %d: %d. data ptr %p, pair ptr %p\n", j, temp[0], data, dstr.second);
				printf("WARNING:  data was not inserted because stat is %d.  delete data\n", stat);
				if (data != NULL) {
					free(data);
					data = NULL;
				}
			}
//			free(data);

			data = malloc(count);
			temp = (int*) data;
			temp[0] = j * 2;
			dstr = std::make_pair(count, data);
			stat = save->getInputBuffer()->push(dstr);
			if (stat == cci::rt::DataBuffer::STOP ||
					stat == cci::rt::DataBuffer::FULL ||
					stat == cci::rt::DataBuffer::BAD_DATA) {
				printf("input added at iteration j %d: %d. data ptr %p, pair ptr %p\n", j, temp[0], data, dstr.second);
				printf("WARNING:  data was not inserted because stat is %d.  delete data\n", stat);
				if (data != NULL) {
					free(data);
					data = NULL;
				}
			}
//			free(data);
		} else if (j == 60)
			save->getInputBuffer()->stop();

		// j < 10: ready and waiting
		// j >= 10, < 30:  ready and input coming, fast
		// j >= 30, j < 40:  ready, has input, but no new ones
		// j == 40:  mark input as done.
		// j > 40, <= 50:  done, has input still
		// j >50:  done, no more input.

		for (std::vector<cci::rt::Communicator_I *>::iterator iter = handlers.begin();
				iter != handlers.end(); ) {

			result = (*iter)->run();
			if (result == cci::rt::Communicator_I::DONE || result == cci::rt::Communicator_I::ERROR) {
				printf("no output at iter j %d .  DONE or error state %d\n", j, result);
				delete (*iter);

				iter = handlers.erase(iter);
			} else if (result == cci::rt::Communicator_I::READY ) {
				printf("output generated at iteration j %d: %d\n", j, result);
				++iter;
			} else {
				printf("no output at iter j %d .  wait state %d\n", j, result);
				++iter;
			}
		}
		++j;
	}

	MPI_Finalize();

}


