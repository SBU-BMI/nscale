/*
 * h5utils.h
 *
 *  Created on: Dec 9, 2011
 *      Author: tcpan
 */

#ifndef H5UTILS_H_
#define H5UTILS_H_

#include "hdf5.h"


inline hid_t createVarStringType(herr_t &status) {
	hid_t strType = H5Tcopy(H5T_C_S1);
	status = H5Tset_size(strType, H5T_VARIABLE);
	return strType;
}



#endif /* H5UTILS_H_ */
