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
//  File Description:  VSGDownhill.h
//    Function prototypes for two versions of the downhill filter
//
// *******************************************************************
#ifndef __VSG_DOWNHILL_H__
#define __VSG_DOWNHILL_H__

void VSG_Downhill1(const int *mask, int *marker, int width, int height);
void VSG_Downhill2(const int *mask, int *marker, int width, int height);

#endif
