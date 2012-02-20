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
//  File Description:  VSGFileImage.h
//    Utility function prototypes for reading and writing PGM images
//
// *******************************************************************
#ifndef __VSG_FILE_IMAGE_H__
#define __VSG_FILE_IMAGE_H__

void VSG_GetPGMSize (const char *filename, int *width, int *height);
void VSG_ReadPGM    (const char *filename, int *buffer);
void VSG_WritePGM   (const char *filename, int width, int height, int *buffer);

#endif
