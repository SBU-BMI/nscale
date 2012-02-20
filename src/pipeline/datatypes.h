/*
 * datatypes.h
 *
 *  Created on: Dec 9, 2011
 *      Author: tcpan
 */

#ifndef DATATYPES_H_
#define DATATYPES_H_

#include "hdf5.h"
#include "h5utils.h"

// version 0.1
char NS_FEATURE_SET_01[] = "/data";
char NS_NU_INFO_SET_01[] = "/metadata";
char NS_IMG_TILE_ATTR[] = "image_tile";
char NS_MASK_TILE_ATTR[] = "mask_tile";
char NS_SUM_ATTR[] = "sums";
char NS_SUM_SQUARE_ATTR[] = "sum_squares";

// version 0.2
char NS_FEATURE_SET[] = "/features";
char NS_NU_INFO_SET[] = "/nu-info";
char NS_TILE_SUM_SET[] = "/tile-sum";
char NS_TILE_INFO_SET[] = "/tile-info";
char NS_IMAGE_SUM_SET[] = "/image-sum";
char NS_IMAGE_INFO_SET[] = "/image-info";

char NS_TILE_X_ATTR[] = "tile_x";
char NS_TILE_Y_ATTR[] = "tile_y";
char NS_IMG_NAME_ATTR[] = "image_name";
char NS_MEAN_ATTR[] = "mean";
char NS_STDEV_ATTR[] = "stdev";
char NS_COUNT_ATTR[] = "count";
char NS_NUM_BAD_VALUES_ATTR[] = "num_bad_values";
char NS_FULL_MEAN_ATTR[] = "full_mean";
char NS_FULL_STDEV_ATTR[] = "full_stdev";
char NS_SAMPLE_RATE_ATTR[] = "sample_rate";

char NS_FILE_CONTENT_TYPE[] = "feature_content_type";


typedef struct {
	char *img_tile_name;
	char *mask_tile_name;
	char *feature_tile_name;
	int tile_x;
	int tile_y;
} tile_info_t;


inline hid_t createTileInfoMemtype() {
	hid_t varstr_t;
	herr_t status = createVarStringType(varstr_t);

    hid_t memtype = H5Tcreate (H5T_COMPOUND, sizeof (tile_info_t));
    status = H5Tinsert (memtype, "img_tile_name",
                HOFFSET (tile_info_t, img_tile_name), varstr_t);
    status = H5Tinsert (memtype, "mask_tile_name",
                HOFFSET (tile_info_t, mask_tile_name), varstr_t);
    status = H5Tinsert (memtype, "feature_tile_name",
                HOFFSET (tile_info_t, feature_tile_name), varstr_t);
    status = H5Tinsert (memtype, "tile_x",
                HOFFSET (tile_info_t, tile_x), H5T_NATIVE_INT);
    status = H5Tinsert (memtype, "tile_y",
                HOFFSET (tile_info_t, tile_y), H5T_NATIVE_INT);
	status = H5Tclose(varstr_t);
	return memtype;
}

inline hid_t createTileInfoFiletype() {
	hid_t varstr_t;
	herr_t status = createVarStringType(varstr_t);

    hid_t memtype = H5Tcreate (H5T_COMPOUND, 3 * sizeof(hvl_t) + 2 * 4);
    status = H5Tinsert (memtype, "img_tile_name", 0, varstr_t);
    status = H5Tinsert (memtype, "mask_tile_name", sizeof(hvl_t), varstr_t);
    status = H5Tinsert (memtype, "feature_tile_name", sizeof(hvl_t) * 2, varstr_t);
    status = H5Tinsert (memtype, "tile_x", sizeof(hvl_t) * 3, H5T_STD_I32LE);
    status = H5Tinsert (memtype, "tile_y", sizeof(hvl_t) * 3 + 4, H5T_STD_I32LE);
	status = H5Tclose(varstr_t);
	return memtype;
}



typedef struct {
	char *img_name;
	char *feature_name;
} image_info_t;


inline hid_t createImageInfoMemtype() {
	hid_t varstr_t;
	herr_t status = createVarStringType(varstr_t);

    hid_t memtype = H5Tcreate (H5T_COMPOUND, sizeof (image_info_t));
    status = H5Tinsert (memtype, "img_name",
                HOFFSET (image_info_t, img_name), varstr_t);
    status = H5Tinsert (memtype, "feature_name",
                HOFFSET (image_info_t, feature_name), varstr_t);
	status = H5Tclose(varstr_t);
	return memtype;
}

inline hid_t createImageInfoFiletype() {
	hid_t varstr_t;
	herr_t status = createVarStringType(varstr_t);

    hid_t memtype = H5Tcreate (H5T_COMPOUND, 2 * sizeof(hvl_t));
    status = H5Tinsert (memtype, "img_name", 0, varstr_t);
    status = H5Tinsert (memtype, "feature_name", sizeof(hvl_t), varstr_t);
	status = H5Tclose(varstr_t);
	return memtype;
}

typedef struct {
	unsigned long nu_count;
	char *feature_name;
} feature_info_t;

inline hid_t createImageInfoFeatureMemtype() {
	hid_t varstr_t;
	herr_t status = createVarStringType(varstr_t);

    hid_t memtype = H5Tcreate(H5T_COMPOUND, sizeof(feature_info_t));
    status = H5Tinsert (memtype, "feature_name", HOFFSET(feature_info_t, feature_name), varstr_t);
	status = H5Tclose(varstr_t);
	return memtype;
}
inline hid_t createImageInfoNuCountMemtype() {
    hid_t memtype = H5Tcreate (H5T_COMPOUND, sizeof(feature_info_t));
    herr_t status = H5Tinsert (memtype, "nu_count", HOFFSET(feature_info_t, nu_count), H5T_NATIVE_ULONG);
	return memtype;
}


typedef struct {
	unsigned long nu_count;
	double nu_sum[74];
	double nu_sum_square[74];
	double nu_mean[74];
	double nu_stdev[74];
	unsigned int bad_values[74];
} image_sum_t;


inline hid_t createImageSumMemtype() {
	// create array type
	hsize_t dim[1] = {74};
	hid_t double_array_t = H5Tarray_create(H5T_NATIVE_DOUBLE, 1, dim);
	hid_t uint_array_t = H5Tarray_create(H5T_NATIVE_UINT, 1, dim);

	hid_t type = H5Tcreate(H5T_COMPOUND, sizeof(image_sum_t));
	herr_t status = H5Tinsert(type, "nu_count", HOFFSET(image_sum_t, nu_count), H5T_NATIVE_ULONG);
	status = H5Tinsert(type, "nu_sum", HOFFSET(image_sum_t, nu_sum), double_array_t);
	status = H5Tinsert(type, "nu_sum_square", HOFFSET(image_sum_t, nu_sum_square), double_array_t);
	status = H5Tinsert(type, "nu_mean", HOFFSET(image_sum_t, nu_mean), double_array_t);
	status = H5Tinsert(type, "nu_stdev", HOFFSET(image_sum_t, nu_stdev), double_array_t);
	status = H5Tinsert(type, "num_bad_values", HOFFSET(image_sum_t, bad_values), uint_array_t);
	status = H5Tclose(double_array_t);
	status = H5Tclose(uint_array_t);
	return type;
}

inline hid_t createImageSumFiletype() {
	hsize_t dim[1] = {74};
	hid_t double_array_t = H5Tarray_create(H5T_IEEE_F64LE, 1, dim);
	hid_t uint_array_t = H5Tarray_create(H5T_STD_U32LE, 1, dim);

	hid_t type = H5Tcreate(H5T_COMPOUND, 8 + 8 * 74 * 4 + 4 * 74);
    herr_t status = H5Tinsert(type, "nu_count", 0, H5T_STD_U64LE);
    status = H5Tinsert(type, "nu_sum", 8, double_array_t);
    status = H5Tinsert(type, "nu_sum_square", 8 + 8 * 74, double_array_t);
    status = H5Tinsert(type, "nu_mean", 8 + 8 * 74 * 2, double_array_t);
    status = H5Tinsert(type, "nu_stdev", 8 + 8 * 74 * 3, double_array_t);
    status = H5Tinsert(type, "num_bad_values", 8 + 8 * 74 * 4, uint_array_t);
    status = H5Tclose(double_array_t);
	status = H5Tclose(uint_array_t);
	return type;
}




typedef struct {
	unsigned long nu_count;
	double nu_sum[74];
	double nu_sum_square[74];
	unsigned int bad_values[74];
} nu_sum_t;


inline hid_t createTileSumMemtype() {
	// create array type
	hsize_t dim[1] = {74};
	hid_t double_array_t = H5Tarray_create(H5T_NATIVE_DOUBLE, 1, dim);
	hid_t uint_array_t = H5Tarray_create(H5T_NATIVE_UINT, 1, dim);

	hid_t type = H5Tcreate(H5T_COMPOUND, sizeof(nu_sum_t));
	herr_t status = H5Tinsert(type, "nu_count", HOFFSET(nu_sum_t, nu_count), H5T_NATIVE_ULONG);
	status = H5Tinsert(type, "nu_sum", HOFFSET(nu_sum_t, nu_sum), double_array_t);
	status = H5Tinsert(type, "nu_sum_square", HOFFSET(nu_sum_t, nu_sum_square), double_array_t);
	status = H5Tinsert(type, "num_bad_values", HOFFSET(nu_sum_t, bad_values), uint_array_t);
	status = H5Tclose(double_array_t);
	status = H5Tclose(uint_array_t);
	return type;
}

inline hid_t createTileSumFiletype() {
	hsize_t dim[1] = {74};
	hid_t double_array_t = H5Tarray_create(H5T_IEEE_F64LE, 1, dim);
	hid_t uint_array_t = H5Tarray_create(H5T_STD_U32LE, 1, dim);

	hid_t type = H5Tcreate(H5T_COMPOUND, 8 + 8 * 74 * 2 + 4 * 74);
    herr_t status = H5Tinsert(type, "nu_count", 0, H5T_STD_U64LE);
    status = H5Tinsert(type, "nu_sum", 8, double_array_t);
    status = H5Tinsert(type, "nu_sum_square", 8 + 8 * 74, double_array_t);
    status = H5Tinsert(type, "num_bad_values", 8 + 8 * 74 * 2, uint_array_t);
    status = H5Tclose(double_array_t);
	status = H5Tclose(uint_array_t);
	return type;
}


typedef struct {
	unsigned int tile_id;
	float bbox_x;
	float bbox_y;
	float bbox_w;
	float bbox_h;
	float centroid_x;
	float centroid_y;
} nu_info_t;


inline hid_t createNuInfoMemtype() {

	hid_t nu_info_type = H5Tcreate(H5T_COMPOUND, sizeof(nu_info_t));
    herr_t status = H5Tinsert(nu_info_type, "tile_id", HOFFSET(nu_info_t, tile_id), H5T_NATIVE_UINT);
    status = H5Tinsert(nu_info_type, "bbox_x", HOFFSET(nu_info_t, bbox_x), H5T_NATIVE_FLOAT);
    status = H5Tinsert(nu_info_type, "bbox_y", HOFFSET(nu_info_t, bbox_y), H5T_NATIVE_FLOAT);
    status = H5Tinsert(nu_info_type, "bbox_w", HOFFSET(nu_info_t, bbox_w), H5T_NATIVE_FLOAT);
    status = H5Tinsert(nu_info_type, "bbox_h", HOFFSET(nu_info_t, bbox_h), H5T_NATIVE_FLOAT);
    status = H5Tinsert(nu_info_type, "centroid_x", HOFFSET(nu_info_t, centroid_x), H5T_NATIVE_FLOAT);
    status = H5Tinsert(nu_info_type, "centroid_y", HOFFSET(nu_info_t, centroid_y), H5T_NATIVE_FLOAT);
	return nu_info_type;
}

inline hid_t createNuInfoFiletype() {
        hid_t file_nu_type = H5Tcreate(H5T_COMPOUND, 28);
        herr_t status = H5Tinsert(file_nu_type, "tile_id", 0, H5T_STD_U32LE);
        status = H5Tinsert(file_nu_type, "bbox_x", 4, H5T_IEEE_F32LE);
        status = H5Tinsert(file_nu_type, "bbox_y", 8, H5T_IEEE_F32LE);
        status = H5Tinsert(file_nu_type, "bbox_w", 12, H5T_IEEE_F32LE);
        status = H5Tinsert(file_nu_type, "bbox_h", 16, H5T_IEEE_F32LE);
        status = H5Tinsert(file_nu_type, "centroid_x", 20, H5T_IEEE_F32LE);
        status = H5Tinsert(file_nu_type, "centroid_y", 24, H5T_IEEE_F32LE);
	return file_nu_type;
}



#endif /* DATATYPES_H_ */

