#include <stdio.h> 
#include <stdlib.h>
#include <dirent.h> 
#include <string> 
#include <vector>
#include <iostream>
#include <tiff.h>
#include <tiffio.h>

#define TILE_SIZE 512

typedef unsigned char uint8;
typedef unsigned int uint32;

uint8* TIFFreadRGB(const char* filename, uint32 & width, uint32 & height) {
    TIFF* in = TIFFOpen(filename, "r");
	
    if (in) {
        size_t npixels;
        uint8* raster;

        TIFFGetField(in, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(in, TIFFTAG_IMAGELENGTH, &height);

        npixels = width * height;

        raster = (uint8*) malloc(3 * npixels);

        if (raster != NULL ) {	//memory was allocated by system
			for (size_t i = 0; i < TIFFNumberOfStrips(in);  i++) {
				TIFFReadEncodedStrip(in, i, raster+i*TIFFStripSize(in), (tsize_t) -1);
			}

			return(raster);
        }
		else {
			printf("Could not allocate sufficient memory for read.  Read aborted.\n"); 

			TIFFClose(in);
			return(NULL);
		}
    }
	else {
		return(NULL);
	}
}

int TIFFwriteTiledRGB(uint8* image, const char* filename, uint32 width, uint32 height) {
	TIFF* out = TIFFOpen(filename, "w");

	TIFFSetField (out, TIFFTAG_IMAGEWIDTH, width);  // set the width of the image
	TIFFSetField(out, TIFFTAG_IMAGELENGTH, height);    // set the height of the image
	TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 3);   // set number of channels per pixel
	TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 8);    // set the size of the channels
	TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);    // set the origin of the image.
	TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
	TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
	TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
	TIFFSetField(out, TIFFTAG_FILLORDER, FILLORDER_MSB2LSB);
	TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, 1); //entire image is one strip, alternatively strip smaller to improve previewer performance

	for (uint32 i = 0; i < height;  i++) {
		TIFFWriteEncodedStrip(out, i, image + 3*i*width, 3*width);
	}

	TIFFClose(out);
}

void GetTiles(const char* path, std::vector<std::string> & files, std::vector<unsigned long> &x, std::vector<unsigned long> &y) {
//Given the path 'path', GetTiles builds a list of .tif files in the folder 'path' and extracts the x,y coordinates
//from their filename.

	DIR* directory;
	struct dirent* dir;

	//open directory
	directory = opendir (path);
	
	//iterate through files in directory to identify .tifs.
	while (dir = readdir (directory)) {
		
		//check if read is file
		if(dir->d_type == DT_REG) {
			
			//capture filename
			std::string file = dir->d_name;
			
			//check filename length
			if(file.length() > 4) {
				
				//get extension
				std::string extension = file.substr(file.rfind(".", std::string::npos),
											file.length()-file.rfind(".", std::string::npos));
				
				//determine if .tif extension
				if(extension.compare(".tif") || extension.compare(".tiff") ||
					extension.compare(".TIF") || extension.compare(".TIFF")) {
					
					//get coordinates
					size_t px, py, pext;
					pext = file.rfind(".", std::string::npos);
					py = file.rfind(".", pext-1);
					px = file.rfind(".", py-1);
										
					//check if coordinates found
					if((pext != std::string::npos) && (px != std::string::npos) && (py != std::string::npos)) {
						
						//push onto vector
						files.push_back(file);
					
						//extract x, y
						std::string xs = file.substr(px+1, py-1);
						std::string ys = file.substr(py+1, pext-1);
						x.push_back((unsigned long)atoi(xs.c_str()));
						y.push_back((unsigned long)atoi(ys.c_str()));
						
					}
				}
			}
		}
	}
	
	//close directory structure
	closedir(directory);
	
}

void GetDimensions(std::vector<unsigned long> &x, std::vector<unsigned long> &y, 
					uint32 &MinX, uint32 &MaxX, uint32 &MinY, uint32 &MaxY) {
//examines outputs 'x', 'y' of 'GetTiles' to determine bounding box for tiles

	//initialize Min, Max
	for(int i = 0; i < x.size(); i++) {
		if(i == 0) {
			MinX = x[0];
			MaxX = x[0];
			MinY = y[0];
			MaxY = y[0];		
		}
		else{
			if(x[i] < MinX)
				MinX = x[i];
			if(x[i] > MaxX)
				MaxX = x[i];
			if(y[i] < MinY)
				MinY = y[i];
			if(y[i] > MaxY)
				MaxY = y[i];
		}
	}
}

uint32 Search(std::vector<unsigned long> x, std::vector<unsigned long> y, uint32 tx, uint32 ty) {
//searches paired vectors x,y for entry (tx, ty)

	//search
	for(uint32 i=0; i < x.size(); i++){
		if(x[i] == tx && y[i] == ty)
			return(i);
	}

	//not found
	return((uint32)x.size()+1);
	
}


int main (int argc, char *argv[]) {

	//variables
	uint8* tile;
	uint32 Width, Height, tWidth, tHeight, MinX, MinY, MaxX, MaxY;
	std::vector<std::string> files;
	std::vector<unsigned long> x;
	std::vector<unsigned long> y;
	std::string path;
	
	//capture path
	path.assign((const char*)argv[1]);
	
	//get list of tiles
	GetTiles(path.c_str(), files, x, y);
	
	//get min, max coordinates, tilesize
	GetDimensions(x, y, MinX, MaxX, MinY, MaxY);	
	
	//check if tiles were found
	if(x.size() > 0) {
		
		//get tile size
		std::string full = path + files[0];
		tile = TIFFreadRGB(full.c_str(), Width, Height);
		if(tile == NULL) {
			printf("Error: Could not read tile %s\n", full.c_str());
			return(0);
		}
		
		
		//DEBUG
		printf("output image is %d x %d\n", MaxX-MinX+Width, MaxY-MinY+Height);
		
		
		
		//create output tiff
		TIFF* out = TIFFOpen(argv[2], "w");
		
		if(out) {
			//allocate tile-let
			uint8* small = (uint8*)malloc(3 * TILE_SIZE * TILE_SIZE);
			
			//set parameters
			TIFFSetField (out, TIFFTAG_IMAGEWIDTH, MaxX-MinX+Width);  // set the width of the image
			TIFFSetField(out, TIFFTAG_IMAGELENGTH, MaxY-MinY+Height);    // set the height of the image			
			TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 3);   // set number of channels per pixel
			TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 8);    // set the size of the channels
			TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);    // set the origin of the image.
			TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
			TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
			TIFFSetField(out, TIFFTAG_FILLORDER, FILLORDER_MSB2LSB);
			TIFFSetField(out, TIFFTAG_TILEWIDTH, TILE_SIZE); //tiled image
			TIFFSetField(out, TIFFTAG_TILELENGTH, TILE_SIZE); //tiled image

			//set compression
			if(strcmp(argv[3], "-lzw")) {
				TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
				TIFFSetField(out, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
			}
			else if(strcmp(argv[3], "-jpg") || strcmp(argv[3], "-jpeg")) {
				TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_JPEG);
			}
			else { //LZW by default
				TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
				TIFFSetField(out, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
			}
				
			//generate background tile
			uint8 * background = (uint8*)malloc(3*Width*Height);
			memset(background, 0, 3*Width*Height);
			
			//read in input tiles
			for(uint32 ty = MinY; ty < MaxY+Height; ty+=Height) { //for each tile in output
				for(uint32 tx = MinX; tx < MaxX+Width; tx+=Width) {
		
					//check if tile exists in files
					uint32 Hit = Search(x, y, tx, ty);
					if(Hit < x.size()) {
					
						//build filename
						std::string full = path + files[Hit];
						
						//DEBUG
						printf("%s, %d x %d\n", full.c_str(), tx, ty);
			
						//read in tile
						tile = TIFFreadRGB(full.c_str(), tWidth, tHeight);
						if(tile == NULL) {
							printf("Error: Could not read tile %s\n", full.c_str());
							return(0);
						}
	
						//check for dimension consistency
						if(tWidth != Width || tHeight != Height) {
							printf("Error: Tile dimensions do not match other tiles %s, %d x %d versus %d x %d\n", 
									full.c_str(), tWidth, tHeight, Width, Height);
						}					
					
						//write out output tiles
						for(uint32 i=0; i < Height; i += TILE_SIZE) {
							for(uint32 j=0; j < Width; j += TILE_SIZE) {
							
								//calculate tile
								uint32 index = TIFFComputeTile(out, j + tx - MinX, i + ty - MinY, 0, 0);
								
								//copy to small
								for(uint32 k=0; k < TILE_SIZE; k++){
									memcpy(small + 3 * k * TILE_SIZE, tile + 3 * ((i + k) * Width + j), 3 * TILE_SIZE);
								}
								
								//write encoded tile
								TIFFWriteEncodedTile(out, index, small, 3 * TILE_SIZE * TILE_SIZE);
								
							}
						}
						
						//free tile
						if(tile != NULL)
							free(tile);
						
					}
					else { //write background tile
													
						//DEBUG
						printf("Background, %d x %d\n", tx, ty);
						
						for(uint32 i=0; i < Height; i += TILE_SIZE) {
							for(uint32 j=0; j < Width; j += TILE_SIZE) {
								
								//calculate tile
								uint32 index = TIFFComputeTile(out, j + tx - MinX, i + ty - MinY, 0, 0);
														
								//copy to small
								for(uint32 k=0; k < TILE_SIZE; k++){
									memcpy(small + 3 * k * TILE_SIZE, background + 3 * ((i + k) * Width + j), 3 * TILE_SIZE);
								}
								
								//write encoded tile
								TIFFWriteEncodedTile(out, index, small, 3 * TILE_SIZE * TILE_SIZE);
								
							}
						}
					}
				}
			}
		
			//write page
			TIFFWriteDirectory(out);
		
			//close tif
			TIFFClose(out);	
			
			//free memory
			free(small);
			free(background);
		
		}
		else {
			printf("Error: Could not create output file %s.\n", argv[2]);
		}
	}

	return 1;
}
