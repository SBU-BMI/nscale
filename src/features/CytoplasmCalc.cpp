/*
 * ObjFeatures.cpp
 *
 *  Created on: Mar 13, 2012
 *      Author: gteodor
 */

#include "CytoplasmCalc.h"

#include <cv.h>
#include <highgui.h>

using namespace cv;


namespace nscale{


CvRect getCytoplasmBounds(CvSize tileSize, CvRect bounding_box,  int delta){
        CvRect cytoplasmBounds;
//      cout << "Cytobounding box = x -> "<<bounding_box.x << " y= " << bounding_box.y<< " width = "<< bounding_box.width << " height = "<<bounding_box.height<<endl;

        // limit box from moving outside x limits
        cytoplasmBounds.x = max(0, (bounding_box.x - delta));

        // calc actually change in X
        int leftMoveX = bounding_box.x - cytoplasmBounds.x;

        // limit box from move outside y limits
        cytoplasmBounds.y = max(0, (bounding_box.y - delta));

        // calc actually change in Y
        int upMoveY = bounding_box.y - cytoplasmBounds.y;


        if(bounding_box.x + bounding_box.width + delta > tileSize.width){
                cytoplasmBounds.width = tileSize.width - bounding_box.x + leftMoveX;
        }else{
                cytoplasmBounds.width = bounding_box.width + delta + leftMoveX;
        }

        if(bounding_box.y + bounding_box.height + delta > tileSize.height){
                cytoplasmBounds.height = tileSize.height - bounding_box.y + upMoveY;
        }else{
                cytoplasmBounds.height = bounding_box.height + delta + upMoveY;
        }

        return cytoplasmBounds;
}


int* CytoplasmCalc::calcCytoplasm(const int* boundingBoxesInfo, int compCount, const cv::Mat& labeledMask) {
//	| offsetToData | x | y | width | height |
	int* cytoplasmBoundingBoxes = NULL;

	// dilation size of the cytoplasm
	int delta = 8;
	
	if(compCount > 0){
		// this refers to metadata containing info about bounding boxes location
		int cytoplasmDataSize = 5 * sizeof(int) * compCount;

		// first alloc space to store only the "metadata"
		cytoplasmBoundingBoxes = (int*) malloc(cytoplasmDataSize);

		// Calculate the size of the cytoplasm masks
		for(int i = 0; i < compCount; i++){
			int label = boundingBoxesInfo[i];
			int minx = boundingBoxesInfo[compCount+i];
			int maxx = boundingBoxesInfo[compCount*2+i];
			int miny = boundingBoxesInfo[compCount*3+i];
			int maxy = boundingBoxesInfo[compCount*4+i];

			CvRect nucleusBB;
			nucleusBB.x = minx;
			nucleusBB.y = miny;
			nucleusBB.width = maxx-minx+1;
			nucleusBB.height = maxy-miny+1;
			CvRect cytoplasmBB = getCytoplasmBounds(labeledMask.size(), nucleusBB, 8);

			cytoplasmBoundingBoxes[i*5] = cytoplasmDataSize;
			cytoplasmBoundingBoxes[i*5 + 1] = cytoplasmBB.x;
			cytoplasmBoundingBoxes[i*5 + 2] = cytoplasmBB.y;
			cytoplasmBoundingBoxes[i*5 + 3] = cytoplasmBB.width;
			cytoplasmBoundingBoxes[i*5 + 4] = cytoplasmBB.height;
			
			if(i == 0 || i == 1){
				cout << "NucleusBB-> x="<<nucleusBB.x << " y=" << nucleusBB.y<< " width="<< nucleusBB.width << " height="<< nucleusBB.height<<endl;
				cout << "CytoBB-> x="<<cytoplasmBB.x << " y=" << cytoplasmBB.y<< " width="<< cytoplasmBB.width << " height="<< cytoplasmBB.height<<endl;
				cout << "DataSize="<< cytoplasmDataSize << endl;
			}
			cytoplasmDataSize+=(cytoplasmBB.width*cytoplasmBB.height*sizeof(char));

		}

		int *aux = (int*) malloc(cytoplasmDataSize);
		memcpy(aux, cytoplasmBoundingBoxes, sizeof(int)*5*compCount);
		free(cytoplasmBoundingBoxes);
		cytoplasmBoundingBoxes = aux;

		// Realloc to store cytoplasm masks as well
	//	cytoplasmBoundingBoxes = (int*) realloc(cytoplasmBoundingBoxes, cytoplasmDataSize);

		cout << "cytoplasmDataSize="<< cytoplasmDataSize << " compCount="<< compCount<<endl;

		for(int i = 0; i < compCount; i++){
			// Get nucleus info
			int label = boundingBoxesInfo[i];
			int minx = boundingBoxesInfo[compCount+i];
			int maxx = boundingBoxesInfo[compCount*2+i];
			int miny = boundingBoxesInfo[compCount*3+i];
			int maxy = boundingBoxesInfo[compCount*4+i];


			int offset = cytoplasmBoundingBoxes[i*5];
			// Points to address where cytoplasm masks supposed to be stored: root address + dataOffset
			char *dataAddress = ((char*)(cytoplasmBoundingBoxes))+offset;

			// Create a Mat header point to the data we allocate
			cv::Mat cytoMask(cytoplasmBoundingBoxes[i*5 + 4], cytoplasmBoundingBoxes[i*5 + 3], CV_8UC1, dataAddress );

			// Set cytoplasm mask to zero
			cytoMask = cv::Scalar(0);

			// For how many pixels is the nucleus shifted inside the cytoplasm
			//  offsetColumn = nuclueus.x - cytoplasm.x
			int offsetColumn = minx - cytoplasmBoundingBoxes[i * 5 + 1];

			//  offsetRow = nuclueus.y - cytoplasm.y
			int offsetRow = miny - cytoplasmBoundingBoxes[i*5+ 2];

			int nucleusHeight = maxy - miny + 1;
			int nucleusWidth = maxx - minx + 1;
			int area = 0;

			// Copy nucleus to cytoplasm mask, shifting it				
			for(int y = 0; y < nucleusHeight; y++){
				const int *labeledMaskPtr = labeledMask.ptr<int>(y+miny);
				char* cytoLinePtr = cytoMask.ptr<char>(y+offsetRow);
				for(int x = 0; x < nucleusWidth; x++){
					if(labeledMaskPtr[x+minx] == label){
						cytoLinePtr[x+offsetColumn] = 255;
						area++;
					}	
				} 
			}
//			if(i==100){
//				imwrite("mask.tif", cytoMask);
// 				namedWindow( "Gray image", CV_WINDOW_AUTOSIZE );
// 				imshow( "Gray image", cytoMask );
//				waitKey(0);
//				cout << "Area="<<area<<endl;
//
//			}			
	                unsigned char disk_r8_17_17[289] = {
        	                0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,
                	        0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,
                        	0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
	                        0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
        	                0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
                	        0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
                        	0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
	                        0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
        	                1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                	        0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
                        	0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
	                        0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
        	                0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
                	        0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
                        	0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
	                        0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,
        	                0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0
                	};
			
			std::vector<unsigned char> disk17vec(disk_r8_17_17, disk_r8_17_17+289);
			Mat disk17(disk17vec);
			disk17 = disk17.reshape(1, 17);

			dilate( cytoMask, cytoMask, disk17 );

//			if(i==0){
//				imshow( "Dilation", cytoMask );
//				waitKey(0);
//			}

			// Remove nucleus area from the cytoplasm mask, shifting it				
			for(int y = 0; y < nucleusHeight; y++){
				const int *labeledMaskPtr = labeledMask.ptr<int>(y+miny);
				char* cytoLinePtr = cytoMask.ptr<char>(y+offsetRow);
				for(int x = 0; x < nucleusWidth; x++){
					if(labeledMaskPtr[x+minx] == label){
						cytoLinePtr[x+offsetColumn] = 0;
						area++;
					}	
				} 
			}
//			if(i==100){
//				imwrite("cytoMask.tif", cytoMask);
//				imshow( "CytoFinal", cytoMask );
//				waitKey(0);
//			}
		}
	}

	return cytoplasmBoundingBoxes;
}

}

