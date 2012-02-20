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
//  File Description:  VSGFileImage.c
//    Utility functions for reading and writing PGM images
//
// *******************************************************************
#include "VSGCommon.h"

int VSG_GetNextValue(FILE *fp);

void VSG_GetPGMSize(const char *filename, int *width, int *height)
{
  FILE *fp;
  char ch;
  int w, h;

  *width = -1;
  *height = -1;
  fp = fopen(filename, "rb");
  ch = (char)fgetc(fp);
  if (ch != 'p' && ch != 'P')
  {
    printf("Error encountered...");
    fclose(fp);
    return;
  }
  ch = (char)fgetc(fp);
  if (ch != '5')
  {
    printf("Error encountered...");
    fclose(fp);
    return;
  }
  w = VSG_GetNextValue(fp);
  if (w == -1)
  {
    printf("Error encountered...");
    fclose(fp);
    return;
  }
  h = VSG_GetNextValue(fp);
  if (h == -1)
  {
    printf("Error encountered...");
    fclose(fp);
    return;
  }
  fclose(fp);
  *width = w;
  *height = h;
}

void VSG_ReadPGM(const char *filename, int *buffer)
{
  FILE *fp;
  char ch;
  int w;
  int h;
  int num;
  int i;
  unsigned char *data;
  
  fp = fopen(filename, "rb");
  ch = (char)fgetc(fp);
  if (ch != 'p' && ch != 'P')
  {
    printf("Error encountered...");
    fclose(fp);
    return;
  }
  ch = (char)fgetc(fp);
  if (ch != '5')
  {
    printf("Error encountered...");
    fclose(fp);
    return;
  }
  w = VSG_GetNextValue(fp);
  if (w == -1)
  {
    printf("Error encountered...");
    fclose(fp);
    return;
  }
  h = VSG_GetNextValue(fp);
  if (h == -1)
  {
    printf("Error encountered...");
    fclose(fp);
    return;
  }
  num = w * h;
  if (VSG_GetNextValue(fp) == -1)
  {
    printf("Error encountered...");
    fclose(fp);
    return;
  }
  data = (unsigned char *)malloc(num * sizeof(unsigned char));
  fread(data, num, 1, fp);
  fclose(fp);
  for (i = 0; i < num; i++)
  {
    *(buffer + i) = *(data + i);
  }
  free(data);
  return;
}

void VSG_WritePGM(const char *filename, int width, int height, int *buffer)
{
  FILE *fp;
  int num = width * height;
  int i;
  unsigned char *data;

  fp = fopen(filename, "wb");
  fprintf(fp, "P5\n%d %d\n255\n", width, height);
  data = (unsigned char *)malloc(num * sizeof(unsigned char));
  for (i = 0; i < num; i++)
  {
    *(data + i) = (unsigned char)*(buffer + i);
  }
  fwrite(data, num, 1, fp);
  fclose(fp);
  free(data);
  return;
}

int VSG_GetNextValue(FILE *fp)
{
  int finished = 0;
  char ch;
  char str[16];
  int index = 0;

  //
  // Skip comments and whitespace
  //

  ch = (char)fgetc(fp);
  while (!finished)
  {
    if (ch == 0x23)                                         // 0x23 == '#' character - indicates comment
    {
      while (ch != 0x0a && ch != 0x0d)                      // 0x0a == LF or 0x0d == CR, indicats end of comment line
      {
        ch = (char)fgetc(fp);
      }
    }
    while (ch == 0x09 || ch == 0x0a || ch == 0x0d || ch == 0x20)
    {                                                       // 0x09 == TAB, 0x20 == SPACE
      ch = (char)fgetc(fp);
    }
    if (ch != 0x23 && ch != 0x09 && ch != 0x0a && ch != 0x0d && ch != 0x20)
      finished = 1;
  }                                                         // End while not finished
  if (ch < 0x30 || ch > 0x39)                               // If you don't now have a decimal digit it's an error
    return (-1);
  str[index++] = ch;
  while ((ch = (char)fgetc(fp)) >= 0x30 && ch <= 0x39 && index < 15)
    str[index++] = ch;
  if (index == 15)
    return (-1);
  str[index] = 0;
  return (atoi (str));
}

