#ifndef IMAGE_REGION_NUCLEI_DATA_IO_H
#define IMAGE_REGION_NUCLEI_DATA_IO_H
#include "ImageRegionNucleiData.h"

int savePolygonMask(char *outFile, PolygonList *contours, int compCount, int rows, int cols);
int writeCSVFile(char *outFile, ShapeFeatureList& shapeList, TextureFeatureList* textureList, 
				int* bbox, int compCount, int locX, int locY);
int writeTSVFile(char *outFile, ImageRegionNucleiData& nucleiData);
int updateFCSFileHeaderTextSegment(FILE *outfile, int dataOffset, int64_t dataLen, 
									int compCount, int numFeatures, char *featureNames[]);
int initFCSFileHeaderTextSegment(FILE *outfile, int numFeatures, char *featureNames[]);
int64_t writeFCSDataSegment(FILE *outfile, ShapeFeatureList& shapeList, TextureFeatureList* textureList, 
							int compCount, int *featureIdx, int numFeatures);

#endif 
