/*
 * ComponentLabel.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */

#include "ComponentLabel.h"

namespace nscale {

ComponentLabel::ComponentLabel() {
	// TODO Auto-generated constructor stub

}

ComponentLabel::~ComponentLabel() {
	// TODO Auto-generated destructor stub
}

}



#include "stdafx.h"
#include "cvaux.h"
#include "highgui.h"




struct region_props
{
int value;
int area;
float x;
float y;
} rp[50];

int no_of_labels[50];
int eqMat[50][50];
int cnt=0;

int check_label(int value)
{
int i;
for(i=0;i {
if(no_of_labels[i]==value)
return 0;
}
if(i==cnt)
return 1;
}

void add_to_labels(int value)
{
int success_flag;
success_flag= check_label(value);
if(success_flag==1)
{
no_of_labels[cnt]=value;
cnt++;
}
}

void equivalance_matrix(int row_no, int col_no)
{
eqMat[row_no][col_no]=1;
eqMat[col_no][row_no]=1;
}
void regionprops(IplImage* img)
{

int i,j;
for(i=0;i<50;i++)
{
for(j=0;j<50;j++)
{
eqMat[i][j]=0;
}
}

//IplImage *img = cvLoadImage("I:/test.bmp",0);
int h1=img->height,w1=img->width;
unsigned long ul;
float varf;

float var,var1,var2,var3,var4;
//unsigned long ul;
int x;
int label=0;
///////////////////////////////////////////////////////////////////////
//fill the first row, first column, last row, last column by 0
for(i=0;i {
cvSetReal2D(img,0,i,0);
cvSetReal2D(img,(h1-1),i,0);
}

for(i=0;i

{
cvSetReal2D(img,i,0,0);
cvSetReal2D(img,i,(w1-1),0);
}
///////////////////////////////////////////////////////////////////////
for(i=0;i {
for(j=0;j {
varf=(float)cvGetReal2D(img,i,j);
ul=(unsigned long)varf;
printf(" %d", ul);
}
printf("\n");
}

for(i=1;i {
for(j=1;j {
var=(float)cvGetReal2D(img,i,j);
ul=(unsigned long)var;
x=ul;
if(i==2)
printf("\n i=%d j=%d",i,j);
if(x!=0)
{
var1=(float)cvGetReal2D(img,i,j-1);
var2=(float)cvGetReal2D(img,i-1,j-1);
var3=(float)cvGetReal2D(img,i-1,j);
var4=(float)cvGetReal2D(img,i-1,j+1);
int cnt=0;
if(var1!=0)
{
cnt++;
//cvSetReal2D(img,i,j,var1);
}
if(var2!=0)
{
cnt++;
//cvSetReal2D(img,i,j,var2);
}
if(var3!=0)
{
cnt++;
//cvSetReal2D(img,i,j,var3);
}
if(var4!=0)
{
cnt++;
//cvSetReal2D(img,i,j,var4);
}
if(cnt==0)
{
//central pixel non zero & neighbours zero
label++;
cvSetReal2D(img,i,j,label);
}
if(cnt==1)
{
// one neighbour is non zero
if(var1!=0)
{
if(var1!=255)
cvSetReal2D(img,i,j,var1);
else
{
//
}
}
else
{
if(var2!=0)
{
cvSetReal2D(img,i,j,var2);
}
else
{
if(var3!=0)
{
cvSetReal2D(img,i,j,var3);
}
else
{
if(var4!=0)
{
cvSetReal2D(img,i,j,var4);
}
}
}
}
}
if(cnt>=2)
{
// more than two elements are non zero fill the equivalance matrix.
unsigned long middle_pixel;
middle_pixel=-1;
if(var1!=0)
{
cvSetReal2D(img,i,j,var1);
middle_pixel=(unsigned long)var1;
}
else
{
if(var2!=0)
{
cvSetReal2D(img,i,j,var2);
middle_pixel=var2;
}
else
{
if(var3!=0)
{
cvSetReal2D(img,i,j,var3);
middle_pixel=var3;
}
else
{
if(var4!=0)
{
cvSetReal2D(img,i,j,var4);
middle_pixel=var4;
}
}
}
}

// do entry in equivalance matrix.
if(middle_pixel!=var1 && var1!=0 && var1!=255)
{
equivalance_matrix(middle_pixel,(unsigned long)var1);
}
if(middle_pixel!=var2 && var2!=0 && var2!=255)
{
equivalance_matrix(middle_pixel,(unsigned long)var2);
}
if(middle_pixel!=var3 && var3!=0 && var3!=255)
{
equivalance_matrix(middle_pixel,(unsigned long)var3);
}
if(middle_pixel!=var4 && var4!=0 && var4!=255)
{
equivalance_matrix(middle_pixel,(unsigned long)var4);
}
}


}
}
}

printf("\n\n\n\n\n");
for(i=0;i<10;i++)
{
for(j=0;j<10;j++)
{
printf(" %d", eqMat[i][j] );
}
printf("\n");
}



int k;

for(j=0;j<50;j++)
{
for(i=0;i<50;i++)
{
if(eqMat[i][j]==1)
{
for(k=1;k<50;k++)
{
eqMat[i][k]=eqMat[i][k];
eqMat[i][k]=eqMat[j][k];
}
}
if(i==j) // set digonal to 1
eqMat[i][j]=1;
}
}







printf("\n\n\n\n\n");
for(i=0;i<10;i++)
{
for(j=0;j<10;j++)
{
printf(" %d", eqMat[i][j] );
}
printf("\n");
}

printf("\nB4 replacement \n\n\n\n");
for(i=0;i {
for(j=0;j {
varf=(float)cvGetReal2D(img,i,j);
ul=(unsigned long)varf;
printf(" %d", ul);
}
printf("\n");
}

int rw, col;

for(i=5;i>=0;i--)
{
for(j=5;j>=0;j--)
{
if(eqMat[i][j]==1)
{
if(i {
for(rw=0;rw {
for(col=0;col {
//find j in image & replace it with i
var=(float) cvGetReal2D(img,rw,col);
ul=(unsigned long) var;
if(var==j)
{
cvSetReal2D(img,rw,col,i);
}

}
}
}
else
{
if(i>j)
{
for(rw=0;rw {
for(col=0;col {
//find i in image & replace it with j
var=(float) cvGetReal2D(img,rw,col);
ul=(unsigned long) var;
if(var==i)
{
cvSetReal2D(img,rw,col,j);
}

}
}
}
}
}
}
}

printf("\n\n\n\n\n");
for(i=0;i {
for(j=0;j {
varf=(float)cvGetReal2D(img,i,j);
ul=(unsigned long)varf;
printf(" %d", ul);
}
printf("\n");
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
// region props
//////////////////////////////////////////////////////////////////////////////////////////////////////


for(i=0;i {
for(j=0;j {
varf=(float)cvGetReal2D(img,i,j);
ul=(unsigned long)varf;
if(ul!=0)
{
add_to_labels(ul);
}
}
}

printf("\n\n\n\n\n");

printf("\n No of labels are =%d", cnt);

for(i=0;i{
printf(" %d",no_of_labels[i]);
}

// find out the count for each of the label...
int struct_index=0,l,m;

for(i=0;i{
for(l=1;l {
for(m=1;m {
varf=(float)cvGetReal2D(img,l,m);
ul=(unsigned long)varf;
if(ul==no_of_labels[i])
{
rp[struct_index].area ++;
rp[struct_index].value=ul;
rp[struct_index].x+= l;
rp[struct_index].y+= m;
}
}
}
struct_index ++;
}


printf("\n\n\n\n\n");

for(i=0;i{
printf("\n label is=%d, Area is=%d Centroid is (x=%f,y=%f) ",rp[i].value,rp[i].area,(rp[i].x/rp[i].area),(rp[i].y/rp[i].area));
}


}
