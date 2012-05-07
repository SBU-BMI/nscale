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
#include "adios.h"
#include "adios_read.h"
#include "adios_internals.h"

#include <algorithm>

int main (int argc, char **argv) {

	// init MPI
	int ierr = MPI_Init(&argc, &argv);

	std::string hostname;
    char * temp = (char*)malloc(256);
    gethostname(temp, 255);
    hostname.assign(temp);
    free(temp);

	int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	printf("comm-world:  %s: %d of %d\n", hostname.c_str(), rank, size);

    // init sub communicator
	// to allow freeing the communicator, we split instead of creating a new one
	// MPI_Comm_free does not support free MPI_COMM_NULL (at master).
	// create new group from old group
	int managerid = 0;

	MPI_Comm comm_worker = MPI_COMM_WORLD;
	MPI_Comm_split(MPI_COMM_WORLD, (rank == managerid ? 1 : 0), rank, &comm_worker);
	if (comm_worker == MPI_COMM_NULL) {
		printf("MPI_COMM comm_worker is MPI_COMM_NULL for node %d\n", rank);
	}
	
	int worker_rank = rank;
	int worker_size = size;
	if (rank != managerid) {
		MPI_Comm_size(comm_worker, &worker_size);
	    MPI_Comm_rank(comm_worker, &worker_rank);
	} else {
		worker_rank = -1;
	}


	MPI_Barrier(MPI_COMM_WORLD);
	printf("comm-worker: world %d: worker %d of %d\n", rank, worker_rank, worker_size);



	// now do an adios test with global comm.
//	int err = adios_init("adios_xml/test-adios2.xml");
//	if (err != 1) printf("ERROR: adios error code on init: %d \n", err);
//
//	const char * fn = "test-adios2.world.bp";
//	int fnlen = strlen(fn);
//	char * filename = (char*)malloc(fnlen + 1);
//	memset(filename, 0, fnlen + 1);
//	strncpy(filename, fn, fnlen);
//	int64_t adios_handle;
//	err = adios_open(&adios_handle, "source", filename, "w", &MPI_COMM_WORLD);
//	if (err != 0) printf("ERROR: adios error code on open: %d \n", err);
//	free(filename);
//
//	uint64_t total_size;
//	err = adios_group_size (adios_handle, 4, &total_size);
//	if (err != 0) printf("ERROR: adios error code on groupsize: %d \n", err);
//	printf("adios total size = %lu\n", total_size);
//
//	err = adios_write (adios_handle, "x", &rank);
//	if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
//
//	err = adios_close(adios_handle);
//	if (err != 0) printf("ERROR: adios error code on close: %d \n", err);
//
//	MPI_Barrier(MPI_COMM_WORLD);
//	err = adios_finalize(rank);


	// now do another test with comm_worker only
	int err = adios_init("adios_xml/test-adios.xml");
	if (err != 1) printf("ERROR: adios error code on init: %d \n", err);

	if (worker_rank >= 0) {

		const char * fn2 = "test-adios.worker.source.bp";
		int fnlen = strlen(fn2);
		char *filename = (char*)malloc(fnlen + 1);
		memset(filename, 0, fnlen + 1);
		strncpy(filename, fn2, fnlen);
		// test to see if ADIOS can open multiple groups in same file

		int64_t adios_handle;
		err = adios_open(&adios_handle, "source", filename, "w", &comm_worker);
		if (err != 0) printf("ERROR: adios error code on open: %d \n", err);

//		printf("adios handle %ld\n", adios_handle);




		// nodes can write different amount of data.
		uint64_t total_size;

		if (worker_rank == 0) {
			err = adios_group_size (adios_handle, 4, &total_size);
			if (err != 0) printf("ERROR: adios error code on groupsize: %d \n", err);
//			printf("adios total size = %lu\n", total_size);

			err = adios_write (adios_handle, "x", &worker_rank);
			printf("rank %d source test 1 x %d\n", worker_rank, worker_rank);
			if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
		} else {
			err = adios_group_size (adios_handle, 0, &total_size);
			if (err != 0) printf("ERROR: adios error code on groupsize: %d \n", err);
//			printf("adios total size = %lu\n", total_size);
			printf("rank %d source test 1 x %s\n", worker_rank, "null");
		}

		err = adios_close(adios_handle);
		if (err != 0) printf("ERROR: adios error code on close: %d \n", err);
		MPI_Barrier(comm_worker);

		/** test read back.
		 *
		 */
		if (worker_rank == 0) printf("READING source\n");
		ADIOS_FILE *f = adios_fopen(filename, comm_worker);
		if (f == NULL) {
			printf("can't open file %s\n", adios_errmsg());
			return -1;
		}
		ADIOS_GROUP * g = adios_gopen(f, "source");
		if (g == NULL) {
			printf("can't open group %s\n", adios_errmsg());
			return -1;
		}

		// READ A REGULAR ARRAY
		ADIOS_VARINFO * v = adios_inq_var(g, "x");
		if (v == NULL) {
			printf("can't inq var %s\n", adios_errmsg());
		}
		printf("scalar data: x = %d\n", *((int *)(v->value)));

		adios_free_varinfo(v);
		 adios_gclose (g);
		 adios_fclose (f);

		 MPI_Barrier(comm_worker);
			free(filename);


		fn2 = "test-adios.worker.source.bp";
		fnlen = strlen(fn2);
		filename = (char*)malloc(fnlen + 1);
		memset(filename, 0, fnlen + 1);
		strncpy(filename, fn2, fnlen);

		// append works fine.
		err = adios_open(&adios_handle, "source", filename, "a", &comm_worker);
		if (err != 0) printf("ERROR: adios error code on open: %d \n", err);

		//printf("adios handle %ld\n", adios_handle);


		// test to see if each node can handle different number of writes: NO
		if (worker_rank == 0) {
			err = adios_group_size (adios_handle, 4, &total_size);
			if (err != 0) printf("ERROR: adios error code on groupsize: %d \n", err);
			//printf("adios total size = %lu\n", total_size);

			err = adios_write (adios_handle, "x", &worker_rank);
			if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
		} else {
			err = adios_group_size (adios_handle, 0, &total_size);
			if (err != 0) printf("ERROR: adios error code on groupsize: %d \n", err);
			//printf("adios total size = %lu\n", total_size);
		}

		err = adios_close(adios_handle);
		if (err != 0) printf("ERROR: adios error code on close: %d \n", err);
		MPI_Barrier(comm_worker);
		/** test read back.
		 *
		 */
		if (worker_rank == 0) printf("READING source \n");
		f = adios_fopen(filename, comm_worker);
		if (f == NULL) {
			printf("can't open file %s\n", adios_errmsg());
			return -1;
		}
//		printf("number of groups is %d\n", f->groups_count);

		g = adios_gopen(f, "source");
		if (g == NULL) {
			printf("can't open group %s\n", adios_errmsg());
			return -1;
		}

		// READ A REGULAR ARRAY
		v = adios_inq_var(g, "x");
//		printf("group id is = %d\n", g->grpid);
		printf("scalar data: x = %d\n", *((int *)(v->value)));
		adios_free_varinfo(v);
		 adios_gclose (g);
		 adios_fclose (f);

		 MPI_Barrier(comm_worker);


		free(filename);


		fn2 = "test-adios.worker.source2.bp";
		fnlen = strlen(fn2);
		filename = (char*)malloc(fnlen + 1);
		memset(filename, 0, fnlen + 1);
		strncpy(filename, fn2, fnlen);


		// different amount of data in global array
		err = adios_open(&adios_handle, "source2", filename, "w", &comm_worker);
		if (err != 0) printf("ERROR: adios error code on open: %d \n", err);


		int size = worker_rank + 1;
		// get the size info from other workers
		int offset = 0, total=0;
		MPI_Scan(&size, &offset, 1, MPI_INT, MPI_SUM, comm_worker);
		offset -= size;
		MPI_Allreduce(&size, &total, 1, MPI_INT, MPI_SUM, comm_worker);
		int *values = (int*)malloc(size * sizeof(int));
		for (int i = 0; i < size; ++i) {
			values[i] = rank;
		}
//		printf("size %d, total %d, offset %d\n", size, total, offset);


		err = adios_group_size (adios_handle, 4 * 3 + size * 4, &total_size);
		if (err != 0) printf("ERROR: adios error code on groupsize: %d \n", err);
	//	printf("adios total size = %lu\n", total_size);

		err = adios_write (adios_handle, "size", &size);
		if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
		err = adios_write (adios_handle, "offset", &offset);
		if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
		err = adios_write (adios_handle, "total", &total);
		if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
		err = adios_write (adios_handle, "values", values);
		if (err != 0) printf("ERROR: adios error code on write: %d \n", err);

		err = adios_close(adios_handle);
		if (err != 0) printf("ERROR: adios error code on close: %d \n", err);
		MPI_Barrier(comm_worker);

		free(values);

		if (worker_rank == 0) printf("READING source 2\n");
		f = adios_fopen(filename, comm_worker);
		if (f == NULL) {
			printf("can't open file %s\n", adios_errmsg());
			return -1;
		}
//		printf("number of groups is %d\n", f->groups_count);

		g = adios_gopen(f, "source2");
		if (g == NULL) {
			printf("can't open group %s\n", adios_errmsg());
			return -1;
		}

		// READ A REGULAR ARRAY
		v = adios_inq_var(g, "values");

		/* use fewer redaers to read the global array back */
		uint64_t slice_size = v->dims[0] / worker_size;  // each read at least these many
		uint64_t remainder = v->dims[0] % worker_size;  // what's left

		uint64_t start, count, bytes_read = 0;

		if ( worker_rank < remainder ) {
			start = (slice_size + 1) * worker_rank;
			count = slice_size + 1;
		} else {
			start = slice_size * worker_rank + remainder;
			count = slice_size;
		}



		void * ival = malloc(count  * sizeof(int));
		if (ival == NULL) {
			fprintf(stderr, "malloc failed. \n");
			return -1;
		}

		bytes_read = adios_read_var(g, "values", &start, &count, ival);


		printf ("worker rank %d: [%lld:%lld]\n\t", worker_rank, start, count);
		for (int i = 0; i < count; i++) {
			printf ("%d ", *((int *)ival + i));
		 }
		printf("\n");

		free(ival);

		adios_free_varinfo(v);
		 adios_gclose (g);
		 adios_fclose (f);
			MPI_Barrier(comm_worker);

		free(filename);


		fn2 = "test-adios.worker.source3.bp";
		fnlen = strlen(fn2);
		filename = (char*)malloc(fnlen + 1);
		memset(filename, 0, fnlen + 1);
		strncpy(filename, fn2, fnlen);


		// different amount of data in global array at different timesteps
		for (int it = 0; it < 5; ++it) {
			err = adios_open(&adios_handle, "source3", filename, "a", &comm_worker);
			if (err != 0) printf("ERROR: adios error code on open: %d \n", err);

//			printf("time index adios handle %ld\n", adios_handle);

			size = (worker_rank + 1)* (it + 1);
			offset = 0;
			total=0;
			MPI_Scan(&size, &offset, 1, MPI_INT, MPI_SUM, comm_worker);
			offset -= size;
			MPI_Allreduce(&size, &total, 1, MPI_INT, MPI_SUM, comm_worker);
			values = (int*)malloc(size * sizeof(int));
			for (int i = 0; i < size; ++i) {
				values[i] = rank + it;
			}
	//		printf("size %d, total %d, offset %d\n", size, total, offset);

			err = adios_group_size (adios_handle, 4 * 3 + size * 4, &total_size);
			if (err != 0) printf("ERROR: adios error code on groupsize: %d \n", err);
		//	printf("adios total size = %lu\n", total_size);

			err = adios_write (adios_handle, "size", &size);
			if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
			err = adios_write (adios_handle, "offset", &offset);
			if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
			err = adios_write (adios_handle, "total", &total);
			if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
			err = adios_write (adios_handle, "values", values);
			if (err != 0) printf("ERROR: adios error code on write: %d \n", err);

			err = adios_close(adios_handle);
			if (err != 0) printf("ERROR: adios error code on close: %d \n", err);
			MPI_Barrier(comm_worker);

			free(values);
		}

		if (worker_rank == 0) printf("READING source 3\n");
		f = adios_fopen(filename, comm_worker);
		if (f == NULL) {
			printf("can't open file %s\n", adios_errmsg());
			return -1;
		}
//		printf("number of groups is %d\n", f->groups_count);

		g = adios_gopen(f, "source3");
		if (g == NULL) {
			printf("can't open group %s\n", adios_errmsg());
			return -1;
		}
		// READ A REGULAR ARRAY
		v = adios_inq_var(g, "values");

		// there are only 2 dims.
		int timedim = v->timedim;
		// timedim is either 0 or last
		int datadim = 0;
		if (timedim == 0) datadim = 1;
		else datadim = 0;

		/* use fewer redaers to read the global array back */
		slice_size = v->dims[datadim] / worker_size;  // each read at least these many
		remainder = v->dims[datadim] % worker_size;  // what's left

		if ( worker_rank < remainder ) {
			start = (slice_size + 1) * worker_rank;
			count = slice_size + 1;
		} else {
			start = slice_size * worker_rank + remainder;
			count = slice_size;
		}

		ival = malloc(count * v->dims[timedim] * sizeof(int));
		if (ival == NULL) {
			fprintf(stderr, "malloc failed. \n");
			return -1;
		}
		uint64_t start2d[2], count2d[2];
		start2d[datadim] = start;
		start2d[timedim] = 0;
		count2d[datadim] = count;
		count2d[timedim] = v->dims[timedim];

		bytes_read = adios_read_var(g, "values", start2d, count2d, ival);


		printf ("worker rank %d: [%lld:%lld]\n\t", worker_rank, start, count);
		for (int it = 0; it < v->dims[datadim]; ++it) {
			printf("\ttime %d: \n", it);
			for (int i = 0; i < count; i++) {
				printf ("%d ", *((int *)ival + it * count + i));
			}
			printf("\n");
		}
		free(ival);

		adios_free_varinfo(v);
		 adios_gclose (g);
		 adios_fclose (f);
			MPI_Barrier(comm_worker);


		free(filename);


		fn2 = "test-adios.worker.source4.bp";
		fnlen = strlen(fn2);
		filename = (char*)malloc(fnlen + 1);
		memset(filename, 0, fnlen + 1);
		strncpy(filename, fn2, fnlen);



		printf("testing source4.\n");
		// multiple variables in same global array specification.
		// also, multiple iterations to grow the global array size....
		uint64_t adios_groupsize, adios_totalsize;

		int ttotal = 0;
		err = adios_open(&adios_handle, "source4", filename, "w", &comm_worker);
		if (err != 0) printf("ERROR: adios error code on open: %d \n", err);

//		printf("adios handle %ld\n", adios_handle);
		struct adios_file_struct * fd = (struct adios_file_struct *) adios_handle;
		struct adios_group_struct *gd = (struct adios_group_struct *) fd->group;
		printf("TIME INDEX %u \n", gd->time_index);

//		if (fd == NULL) {
//			printf("FD is null\n");
//		} else {
//			gd = (struct adios_group_struct *) fd->group;
//			if (gd == NULL) {
//				printf("GD is null\n");
//			} else {
//				struct adios_var_struct *vd = (struct adios_var_struct *) gd->vars;
//				if (vd == NULL) {
//					printf("VD is null\n");
//				} else {
//
//					struct adios_var_struct *v1d = vd;
//					while (v1d != NULL) {
//						if (v1d->name != NULL && strcmp(v1d->name, "values") == 0) break;
//						v1d = v1d->next;
//					}
//					//v1d = adios_find_var_by_name (vd, blah, gd->all_unique_var_names);
//					if (v1d != NULL) {
//						struct adios_dimension_struct * ds = v1d->dimensions;
//						if (ds == NULL) {
//							printf("DS is null \n");
//						} else {
//							while (ds != NULL) {
//								printf("TIME INDEX ? %u \n", ds->dimension.time_index ? ds->dimension.id: 0);
//								printf("TIME INDEX ? %u \n", ds->global_dimension.time_index ? ds->global_dimension.id: 0);
//								printf("TIME INDEX ? %u \n", ds->local_offset.time_index ? ds->local_offset.id: 0);
//								ds = ds->next;
//							}
//						}
//					} else {
//						printf("NOT MATCHING VAR NAME %s \n", "values");
//					}
//				}
//			}
//		}


		size = worker_rank + 1;
		// get the size info from other workers
		offset = 0;
		MPI_Scan(&size, &offset, 1, MPI_INT, MPI_SUM, comm_worker);
		int step_total = 0;
		MPI_Allreduce(&offset, &step_total, 1, MPI_INT, MPI_MAX, comm_worker);
		//printf("step total %d offset %d size %d\n", step_total, offset, size);
		offset = ttotal + offset - size;
		ttotal += step_total;
		printf("total %d offset %d size %d\n", ttotal, offset, size);
		total = 20;

		values = (int*)malloc(size * sizeof(int));
		for (int i = 0; i < size; ++i) {
			values[i] = rank;
		}
//		printf("size %d, total %d, offset %d\n", size, total, offset);
		int *values2 = values;

#include "gwrite_source4.ch"
//		printf("TIME INDEX before close %u \n", gd->time_index);

		err = adios_close(adios_handle);
		if (err != 0) printf("ERROR: adios error code on close: %d \n", err);
		free(values);

		err = adios_open(&adios_handle, "source4", filename, "a", &comm_worker);
		if (err != 0) printf("ERROR: adios error code on open: %d \n", err);

		fd = (struct adios_file_struct *) adios_handle;
		gd = (struct adios_group_struct *) fd->group;
		printf("TIME INDEX %u \n", gd->time_index);


//		printf("adios handle %ld\n", adios_handle);

		size = worker_rank + 2;
		// get the size info from other workers
		offset = 0;
		MPI_Scan(&size, &offset, 1, MPI_INT, MPI_SUM, comm_worker);
		step_total = 0;
		MPI_Allreduce(&offset, &step_total, 1, MPI_INT, MPI_MAX, comm_worker);
		//printf("step total %d offset %d size %d\n", step_total, offset, size);

		offset = ttotal + offset - size;
		ttotal += step_total;
		printf("total %d offset %d size %d\n", ttotal, offset, size);

		values = (int*)malloc(size * sizeof(int));
		for (int i = 0; i < size; ++i) {
			values[i] = rank + i;
		}
//		printf("size %d, total %d, offset %d\n", size, total, offset);
		values2 = values;

#include "gwrite_source4.ch"
//		printf("TIME INDEX 2 before close %u \n", gd->time_index);
               printf("rank %d, group id %u, membercount %u, offset %lu, timeindex %u, proc id %u\n", worker_rank, gd->id, gd->member_count, gd->group_offset, gd->time_index, gd->process_id);
                printf("rank %d, file datasize %lu, writesizebytes %lu, pgstart %lu, baseoffset %lu, offset %lu, bytewritten %lu, bufsize %lu\n", worker_rank, fd->data_size, fd->write_size_bytes, fd->pg_start_in_file, fd->base_offset, fd->offset, fd->bytes_written, fd->buffer_size);
  
		gd->time_index = 1;
		printf("TIME INDEX modified %u \n", gd->time_index);
		err = adios_close(adios_handle);
		if (err != 0) printf("ERROR: adios error code on close: %d \n", err);
		free(values);


		MPI_Barrier(comm_worker);

		MPI_Comm comm2;
		MPI_Comm_split(comm_worker, (worker_rank > 0? 1 : 0), worker_rank, &comm2);
		int rank2, size2;
		
		if (worker_rank == 0) printf("READING source 4\n");
		if (worker_rank > 0) {

		MPI_Comm_rank(comm2, &rank2);
		MPI_Comm_size(comm2, &size2);

		f = adios_fopen(filename, comm2);
		if (f == NULL) {
			printf("can't open file %s\n", adios_errmsg());
			return -1;
		}
//		printf("number of groups is %d\n", f->groups_count);
		g = adios_gopen(f, "source4");
		if (g == NULL) {
			printf("can't open group %s\n", adios_errmsg());
			return -1;
		}

		// READ A REGULAR ARRAY
		v = adios_inq_var(g, "values");

		printf("Size is : %d\n", v->dims[0]);

		/* use fewer redaers to read the global array back */
		//slice_size = v->dims[0] / worker_size;  // each read at least these many
		//remainder = v->dims[0] % worker_size;  // what's left

		slice_size = 15 / size2;  // each read at least these many
		remainder = 15 % size2;  // what's left

		if ( worker_rank < remainder ) {
			start = (slice_size + 1) * rank2;
			count = slice_size + 1;
		} else {
			start = slice_size * rank2 + remainder;
			count = slice_size;
		}

		ival = malloc(count  * sizeof(int));
		if (ival == NULL) {
			fprintf(stderr, "malloc failed. \n");
			return -1;
		}

		bytes_read = adios_read_var(g, "values", &start, &count, ival);

		printf ("worker rank %d: [%lld:%lld] values \n\t", rank2, start, count);
		for (int i = 0; i < count; i++) {
			printf ("%d ", *((int *)ival + i));
		 }
		printf("\n");

		free(ival);
		adios_free_varinfo(v);

		// READ A REGULAR ARRAY
		v = adios_inq_var(g, "values2");

		/* use fewer redaers to read the global array back */
//		slice_size = v->dims[0] / worker_size;  // each read at least these many
//		remainder = v->dims[0] % worker_size;  // what's left
		slice_size = 15 / size2;  // each read at least these many
		remainder = 15 % size2;  // what's left

		if ( worker_rank < remainder ) {
			start = (slice_size + 1) * rank2;
			count = slice_size + 1;
		} else {
			start = slice_size * rank2 + remainder;
			count = slice_size;
		}

		ival = malloc(count  * sizeof(int));
		if (ival == NULL) {
			fprintf(stderr, "malloc failed. \n");
			return -1;
		}

		bytes_read = adios_read_var(g, "values2", &start, &count, ival);

		printf ("worker rank %d: [%lld:%lld] values 2\n\t", rank2, start, count);
		for (int i = 0; i < count; i++) {
			printf ("%d ", *((int *)ival + i));
		 }
		printf("\n");

		free(ival);
		adios_free_varinfo(v);


		adios_gclose (g);
		 adios_fclose (f);
}
		MPI_Barrier(comm2);
		MPI_Comm_free(&comm2);
			MPI_Barrier(comm_worker);

		free(filename);


		fn2 = "test-adios.worker.source5.bp";
		fnlen = strlen(fn2);
		filename = (char*)malloc(fnlen + 1);
		memset(filename, 0, fnlen + 1);
		strncpy(filename, fn2, fnlen);



	//	printf("Source 5 testing \n");
		// string in global array
		err = adios_open(&adios_handle, "source5", filename, "w", &comm_worker);
		if (err != 0) printf("ERROR: adios error code on open: %d \n", err);

		//printf("adios handle %ld\n", adios_handle);

		// get the size info from other workers
		int count5 = worker_rank + 2;
		offset = 0, total=0;
		MPI_Scan(&count5, &offset, 1, MPI_INT, MPI_SUM, comm_worker);
		offset -= count5;
		MPI_Allreduce(&count5, &total, 1, MPI_INT, MPI_SUM, comm_worker);
		int maxstrlen = strlen("mercedes");
		printf("maxstrlen = %d\n", maxstrlen);


		char * strval = (char*) malloc(maxstrlen * count5);
		memset(strval, 0, maxstrlen * count5);
		for (int i = 0; i < count5; ++i) {
			strncpy(strval + i * maxstrlen, "mercedes", std::min(maxstrlen - 1, i + 1));
			printf("%s\n", strval + i*maxstrlen);
		}

		err = adios_group_size (adios_handle, 4 * 4 + maxstrlen * count5, &total_size);
		if (err != 0) printf("ERROR: adios error code on groupsize: %d \n", err);
		printf("adios total size = %lu\n", total_size);

		err = adios_write (adios_handle, "count", &count5);
		if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
		err = adios_write (adios_handle, "offset", &offset);
		if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
		err = adios_write (adios_handle, "total", &total);
		if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
		err = adios_write (adios_handle, "maxstrlen", &maxstrlen);
		if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
		err = adios_write (adios_handle, "strval", strval);
		if (err != 0) printf("ERROR: adios error code on write: %d \n", err);



		err = adios_close(adios_handle);
		if (err != 0) printf("ERROR: adios error code on close: %d \n", err);
		MPI_Barrier(comm_worker);

		free(strval);

		if (worker_rank == 0) printf("READING source 4\n");
		f = adios_fopen(filename, comm_worker);
		if (f == NULL) {
			printf("can't open file %s\n", adios_errmsg());
			return -1;
		}
//		printf("number of groups is %d\n", f->groups_count);
		g = adios_gopen(f, "source5");
		if (g == NULL) {
			printf("can't open group %s\n", adios_errmsg());
			return -1;
		}

		// READ A REGULAR ARRAY
		v = adios_inq_var(g, "strval");

		/* use fewer redaers to read the global array back */
		slice_size = v->dims[0] / worker_size;  // each read at least these many
		remainder = v->dims[0] % worker_size;  // what's left

		if ( worker_rank < remainder ) {
			start = (slice_size + 1) * worker_rank;
			count = slice_size + 1;
		} else {
			start = slice_size * worker_rank + remainder;
			count = slice_size;
		}

		ival = malloc(count  * v->dims[1] );
		if (ival == NULL) {
			fprintf(stderr, "malloc failed. \n");
			return -1;
		}
		start2d[0] = start;
		start2d[1] = 0;
		count2d[0] = count;
		count2d[1] = v->dims[1];

		bytes_read = adios_read_var(g, "strval", start2d, count2d, ival);

		printf ("worker rank %d: [%lld:%lld] values \n", worker_rank, start, count);
		char str[v->dims[1] + 1];
		for (int i = 0; i < count; i++) {
			strncpy(str, (char *)ival + i * v->dims[1], v->dims[1]);
			printf ("\t%s\n", str);
		 }

		free(ival);
		adios_free_varinfo(v);

		adios_gclose (g);
		 adios_fclose (f);


		free(filename);
		MPI_Barrier(comm_worker);

	}

	if (err != 0) printf("ERROR: adios error code on close: %d \n", err);


	printf("finalize adios!\n");
	err = adios_finalize(rank);

	MPI_Barrier(comm_worker);
	MPI_Comm_free(&comm_worker);


	printf("finalize mpi!\n");
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize();
}

