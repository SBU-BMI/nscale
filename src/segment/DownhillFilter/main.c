// *******************************************************************
//  Copyright ©2003, Vision Systems Group, Dublin City University
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//  See www.eeng.dcu.ie/~vsl/vsgLicense.html for details of this
//  software's License.
//
//  You should have received a copy of the License along with this
//  program; if not, email vsg@eeng.dcu.ie for details.
//
// *******************************************************************
// 
//  Author:            Kevin Robinson
//  Date:              02-Sep-2003
//
//  File Description:  main.c
//    A sample program to illustrate the use of the downhill filter
//
// *******************************************************************
#include "VSGCommon.h"

//
// Possible seed positions for the spiral and loop images:
//   Spiral: (245,9)
//   Loop:   (112,93)
//
void main (int argc, char *argv[])
{
  int sizex;
  int sizey;
  int seedx;
  int seedy;
  int *mask;
  int *marker;

  if (argc != 5)                                            // If the command line looks wrong say so and quit
  {
    printf("\n\nUsage:\n");
    printf("  %s infile outfile seedx seedy\n\n",argv[0]);
    printf("    infile  - the input file, must be a PGM\n");
    printf("    outfile - the file to save the result to\n");
    printf("    seedx   - x coordinate of the seed point\n");
    printf("    seedy   - y coordinate of the seed point\n\n");
    printf("Eg:-\n\n");
    printf("  %s Loop.pgm out.pgm 112 93\n\n",argv[0]);
    return;
  }
  VSG_GetPGMSize(argv[1],&sizex,&sizey);                    // Read the input image size from the PGM file
  mask = (int*)malloc(sizex*sizey*sizeof(int));             // Allocate the space required for the mask image
  VSG_ReadPGM(argv[1],mask);                                // Read the mask image from the file specified
  seedx = atoi(argv[3]);                                    // Read the seed X coordinate from the command line
  seedy = atoi(argv[4]);                                    // Read the seed Y coordinate from the command line
  marker = (int*)calloc(sizex*sizey,sizeof(int));           // calloc initialises the space with zeros
  marker[seedx+seedy*sizex]=mask[seedx+seedy*sizex];        // Set the seed pixel in the marker
  VSG_Downhill1(mask,marker,sizex,sizey);                   // Run the downhill filter on the loaded data
  VSG_WritePGM(argv[2],sizex,sizey,marker);                 // Save the results to the specified output file
  free(mask);
  free(marker);
}
