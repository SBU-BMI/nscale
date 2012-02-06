/*
 * h5-utils.cpp
 *
 *  Created on: Dec 9, 2011
 *      Author: tcpan
 */

#include "hdf5.h"


void createExtensibleDataset(hid_t &file_id, int rank, hsize_t *maxdims, hsize_t *chunkdims, hid_t &type, char *datasetname) {
	// create feature dataspace with unlimited dimensions
	hid_t space = H5Screate_simple(rank, chunkdims, maxdims);
	// now create  the creation property list and set chunk size
	hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
	herr_t status = H5Pset_chunk(dcpl, rank, chunkdims);
	// nowcreate the dataset
	hid_t dset = H5Dcreate(file_id, datasetname, type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);

	// clean up now.
	status = H5Pclose(dcpl);
	status = H5Dclose(dset);
	status = H5Sclose(space);
}

void extendAndWrite(hid_t &file_id, char *datasetname, int rank, hsize_t *addDims, hid_t &type, const void *data, bool first) {
	hid_t dset = H5Dopen(file_id, datasetname, H5P_DEFAULT);

	// get the current size
	hsize_t *currSize = new hsize_t[rank];
	herr_t status;
	hid_t space ;
	if (first) {
		for (int i = 1; i < rank; ++i) {
			currSize[i] = addDims[i];
		}
		currSize[0] = 0;
	} else {
		space = H5Dget_space(dset);
		H5Sget_simple_extent_dims(space, currSize, NULL);
		status = H5Sclose(space);
	}

	hsize_t *size = new hsize_t[rank];
	for (int i = 1; i < rank; ++i) {
		size[i] = currSize[i];
	}
	size[0] = currSize[0] + addDims[0];
	// extend it
	status = H5Dset_extent(dset, size);

	// get the offset
	hsize_t *offset = new hsize_t[rank];
	for (int i = 1; i < rank; ++i) {
		offset[i] = 0;
	}
	offset[0] = currSize[0];

	// get the new dataspace.
	space = H5Dget_space(dset);
	// get the hyperslab
	status = H5Sselect_hyperslab(space, H5S_SELECT_SET, offset, NULL, addDims, NULL);

	// get dataspace
	hid_t dataspace = H5Screate_simple(rank, addDims, NULL);

	// write the data
	status = H5Dwrite(dset, type, dataspace, space, H5P_DEFAULT, data);
	status = H5Sclose(dataspace);
	status = H5Sclose(space);
	status = H5Dclose(dset);

	delete [] currSize;
	delete [] size;
	delete [] offset;

}
