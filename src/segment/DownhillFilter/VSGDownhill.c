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
//  File Description:  VSGDownhill.c
//    Two versions of the downhill filter
//
// *******************************************************************
#include "VSGCommon.h"

// Type 1: each pixel in the marker image must be equal to
// either the equivelent pixel in the mask or to zero.
//
// mask   - the conditioning mask used for reconstruction
// marker - contains initial markers, the reconstruction is performed in situ
// width  - width of the mask and marker (which must be the same)
// height - height of the mask and marker (which must be the same)
//
// The reconstruction is performed directly in marker
//
// A speed increase in the order of 30%-40% can be achieved
// by unwrapping and optimising the two for loops which examine
// the current pixels 8-neighbourhood
void VSG_Downhill1(const int *mask, int *marker, int width, int height)
{
  int ix,iy,ox,oy,offset;
  int currentQ,currentPixel;
  int pixelsPerImage=width*height;
  int maxVal=0;
  int *istart,*iarray;

  for (offset = pixelsPerImage-1; offset >= 0; offset--)    // Find the maximum in the marker image
    if (marker[offset] > maxVal)
      maxVal = marker[offset];
  istart = (int*)malloc((maxVal+1)*sizeof(int));            // Allocate an array of list heads
  iarray = (int*)malloc(pixelsPerImage*sizeof(int));        // Allocate space for all the lists
  memset(istart,0xfe,(maxVal+1)*sizeof(int));               // 0xfe will mark the tail of a list
  memset(iarray,0xff,pixelsPerImage*sizeof(int));           // 0xff will mark unfinalised pixels
  for (offset = pixelsPerImage-1; offset >= 0; offset--) {  // Add seed pixels to their lists
    if (marker[offset] != 0) {
      iarray[offset] = istart[marker[offset]];
      istart[marker[offset]] = offset;
    }
  }
  for (currentQ = maxVal; currentQ > 0; currentQ--) {       // Process each list from high to low
    currentPixel = istart[currentQ];
    while (currentPixel != 0xfefefefe) {                    // While the current list is not empty
      istart[currentQ] = iarray[currentPixel];
      ix = currentPixel%width;
      iy = currentPixel/width;
      for (oy = iy-1; oy <= iy+1; oy++) {                   // Examine the pixels 8-neighbourhood
        for (ox = ix-1; ox <= ix+1; ox++) {
          if (ox >= 0 && oy >= 0 && ox < width && oy < height &&
              iarray[offset = ox+oy*width] == 0xffffffff) {
            marker[offset] = mask[offset]>currentQ?currentQ:mask[offset];
            iarray[offset] = istart[marker[offset]];
            istart[marker[offset]] = offset;
          }
        }
      }
      currentPixel = istart[currentQ];
    }
  }
  free(istart);
  free(iarray);
}

// Type 2: each pixel in the marker image must be less than
// or equal to the equivelent pixel in the mask.
//
// mask   - the conditioning mask used for reconstruction
// marker - contains initial markers, the reconstruction is performed in situ
// width  - width of the mask and marker (which must be the same)
// height - height of the mask and marker (which must be the same)
//
// The reconstruction is performed directly in marker
void VSG_Downhill2(const int *mask, int *marker, int width, int height)
{
  int ix,iy,ox,oy,offset;
  int currentQ,currentP;
  int pixPerImg=width*height;
  int val1,val2,maxVal=0;
  int *istart,*irev,*ifwd;

  for (offset = pixPerImg-1; offset >= 0; offset--)
    if (marker[offset] > maxVal)
      maxVal = marker[offset];
  istart = (int*)malloc((maxVal+pixPerImg*2)*sizeof(int));
  irev = istart+maxVal;
  ifwd = irev+pixPerImg;
  for (offset = -maxVal; offset < 0; offset++)
    irev[offset] = offset;
  for (offset = pixPerImg-1; offset >= 0; offset--) {
    if (marker[offset] > 0) {
      val1 = -marker[offset];
      irev[offset] = val1;
      ifwd[offset] = irev[val1];
      irev[val1] = offset;
      if (ifwd[offset] >= 0)
        irev[ifwd[offset]] = offset;
    }
  }
  for (currentQ = -maxVal; currentQ < 0; currentQ++) {
    currentP = irev[currentQ];
    while (currentP >= 0) {
      irev[currentQ] = ifwd[currentP];
      irev[currentP] = currentQ;
      ix = currentP%width;
      iy = currentP/width;
      for (oy = iy-1; oy <= iy+1; oy++) {
        for (ox = ix-1; ox <= ix+1; ox++) {
          if (ox >= 0 && oy >= 0 && ox < width && oy < height) {
            offset = ox+oy*width;
            val1 = marker[offset];
            val2 = marker[currentP]<mask[offset]?marker[currentP]:mask[offset];
            if (val1 < val2) {
              if (val1 != 0) {
                ifwd[irev[offset]] = ifwd[offset];
                if (ifwd[offset] >= 0)
                  irev[ifwd[offset]] = irev[offset];
              }
              marker[offset] = val2;
              irev[offset] = -val2;
              ifwd[offset] = irev[-val2];
              irev[-val2] = offset;
              if (ifwd[offset] >= 0)
                irev[ifwd[offset]] = offset;
            }
          }
        }
      }
      currentP = irev[currentQ];
    }
  }
  free(istart);
}

