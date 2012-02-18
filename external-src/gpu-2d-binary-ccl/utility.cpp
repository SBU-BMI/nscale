#include "math.h"
#include <stdlib.h>
#include <stdio.h>
#include "utility.h"

#include "stdlib.h"
#define MAXSLINK     256*20   /* 0 --- 255  */
#define MAXLINK      1024*40  /* 0 --- 1023 */
#define MAXSUBBLOCK  1023*20  /* 1 --- 1023 */
#define MAXBLOCK     255*80   /* 1 --- 255  */
#define MAXGREYLEVEL 255*10   /* 0 --- 255  */

#define NEIGHBORDATATYPE unsigned short
#define TmpAreaDATATYPE unsigned short

int createMatrix(LABELDATATYPE*data,int row,int col,int posibility){
	for (int i=0;i<row;i++){
		for (int j=0;j<col;j++){
			if ( (i==row-1)||(j==col-1)||(i==0)||(j==0)){//保证边界为0
				data[i*col+j]=0;
			}else{
				data[i*col+j]=( (rand() % 100)>posibility?0:1);
			}
		}
	}
	return 0;
}

int dumpMatrix1(LABELDATATYPE *data,int rowLen,int colLen)
{
	printf ("%2d: ",rowLen);
	for (int i=0; i<rowLen; i++)
	{
		printf("%3d:",i);
	}
	printf("\n");
	for (int i=0;i<colLen;i++)
	{
		printf ("%2d: ",i);
		for (int j=0;j<rowLen;j++)
		{
			printf ("%3d,",data[i*rowLen+j]);
		}
		printf("\n");
	}
	printf ("======================================================================\n");
	return 0;
};
int dumpMatrix2(LABELDATATYPE *data,int rowLen,int colLen)
{
	for (int i=0;i<colLen;i++){
		printf ("%2d: ",i);
		for (int j=0;j<rowLen;j++){
			int rt=data[i*rowLen+j];
			if (rt!=0){
				printf ("%3d ",rt);
			}else {
				printf ("    ");
			}
		}
		printf("\n");
	}
	printf ("======================================================================\n");
	return 0;
};

int dumpStructureMatrix(LineStream *picLine,int *picRow,int rowNum)
{
	int addr=0;
	for (int i=0;i<=rowNum;i++)
	{
		printf ("%2d ROW, EntryNum:%4d : ",i, picRow[i]);
		for (int j=0;j<picRow[i];j++)
		{
			//LineStream rt = picLine[addr];
			printf("(%d| %d,%d) ",picLine[addr].root, picLine[addr].SP, picLine[addr].EP);
			addr++;
		}
		printf("\n");
	}
	printf ("======================================================================\n");
	return 0;
};
