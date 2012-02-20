/*
 * h5utils.h
 *
 *  Created on: Dec 9, 2011
 *      Author: tcpan
 */

#ifndef H5UTILS_H_
#define H5UTILS_H_

#include "hdf5.h"

//extern int _debug = 1;


// version 0.2
#define NS_H5_VER_ATTR "ns_h5_version"
#define CHUNK_SIZE_ROWS 1000


inline herr_t createVarStringType(hid_t &strType) {
	strType = H5Tcopy(H5T_C_S1);
	return H5Tset_size(strType, H5T_VARIABLE);
}


void createExtensibleDataset(hid_t &file_id, int rank, hsize_t *maxdims, hsize_t *chunkdims, hid_t &type, char *datasetname);
void extendAndWrite(hid_t &file_id, char *datasetname, int rank, hsize_t *addDims, hid_t &type, const void *data, bool first);


#endif /* H5UTILS_H_ */
