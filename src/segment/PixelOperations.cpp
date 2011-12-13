/*
 * ScanlineOperations.cpp
 *
 *  Created on: Aug 2, 2011
 *      Author: tcpan
 */

#include "PixelOperations.h"
#include <limits>


namespace nscale {

using namespace cv;

template <typename T>
Mat PixelOperations::invert(const Mat& img) {
	// write the raw image
	CV_Assert(img.channels() == 1);

	if (std::numeric_limits<T>::is_integer) {

		if (std::numeric_limits<T>::is_signed) {
			Mat output;
			bitwise_not(img, output);
			return output + 1;
		} else {
			// unsigned int
			return std::numeric_limits<T>::max() - img;
		}

	} else {
		// floating point type
		return -img;
	}


}


Mat PixelOperations::bgr2gray(const ::cv::Mat& img){
	int imageChannels = img.channels();
	assert(imageChannels == 3);
	assert(img.type() == CV_8UC3);

	Mat gray = Mat(img.size(), CV_8UC1);

	// Same constants as used by Matlab
	double r_const = 0.298936021293776;
	double g_const = 0.587043074451121;
	double b_const = 0.114020904255103;

	int nr = img.rows, nc = img.cols;

	for(int i=0; i<nr; i++){
		const uchar* data_in = img.ptr<uchar>(i);
		uchar* data_out = gray.ptr<uchar>(i);
		for(int j=0; j<nc; j++){
			uchar b = data_in[j * imageChannels];
			uchar g = data_in[j * imageChannels + 1];
			uchar r = data_in[j * imageChannels + 2];
			double grayPixelValue = r_const * (double)r + g_const * (double)g + b_const * (double)b;
			data_out[j] = cciutils::double2uchar(grayPixelValue);
		}
	}
	return gray;
}
void PixelOperations::ColorDeconv( const Mat& image, const Mat& M, const Mat& b, Mat& H, Mat& E, bool BGR2RGB){

	long t1 = cciutils::ClockGetTime();
	//initialize normalized stain deconvolution matrix
	Mat normal_M;

	M.copyTo(normal_M);

	//stain normalization
	double col_Norm;
	for(int i=0; i< M.cols; i++){
		col_Norm = norm(M.col(i));
		if(  col_Norm > (double) 0 ){
			normal_M.col(i) = M.col(i) / col_Norm;
		}
	}
	//showMat(normal_M, "normal_M--stain normalization: ");

	//find last column of the normalized stain deconvolution matrix
	int last_col_index = M.cols-1;
	Mat last_col = normal_M.col(last_col_index); //or//Mat last_col = normal_M(Range::all(), Range(2,3));
	//showMat( last_col, "last column " );

	//normalize the stain deconvolution matrix again
	if(norm(last_col) == (double) 0){
		for(int i=0; i< normal_M.rows; i++){

			if( norm(normal_M.row(i)) > 1 ){
				normal_M.at<double>(i,last_col_index) = 0;
			}
			else{
				normal_M.at<double>(i,last_col_index) =  sqrt( 1 - pow(norm(normal_M.row(i)), 2) );
			}
		}
		normal_M.col(last_col_index) = normal_M.col(last_col_index) / norm(normal_M.col(last_col_index));
	}
	//showMat(normal_M, "normal_M");

	//take the inverse of the normalized stain deconvolution matrix
	Mat Q = normal_M.inv();
	if (b.size().height != 1){
		printf("b has to be a row vector \n");
		exit(-1);
	}

	//select rows in Q with a true marker in b
	Mat T(1,Q.cols,Q.type());
	for (int i=0; i<b.size().width; i++){
		if( b.at<char>(0,i) == 1 )
			T.push_back(Q.row(i));
	}
	Q = T.rowRange(Range(1,T.rows));

	long t2 = cciutils::ClockGetTime();

	cout << "	Before normalized = "<< t2-t1 <<endl;

    	//normalized image
	int nr = image.rows, nc = image.cols;
	Mat dn = Mat::zeros(nr, nc, CV_64FC3);
	if (image.channels() == 1){
		for(int i=0; i<nr; i++){
			const uchar* data_in = image.ptr<uchar>(i);
			double* data_out = dn.ptr<double>(i);
			for(int j=0; j<nc; j++){
				data_out[j] = -(255.0*log(((double)(data_in[j])+1.0)/255.0))/log(255.0);
			}
		}
	}else if(image.channels() == 3){
		int imageChannels = image.channels();
		vector<double> precomp_res;
		for(int i=0; i < 256; i++){
			double temp = -(255.0*log(((double)i +1.0)/255.0))/log(255.0);
			precomp_res.push_back(temp);
		}

		for(int i=0; i<nr; i++){
			const uchar* data_in = image.ptr<uchar>(i);
			double* data_out = dn.ptr<double>(i);
				for(int j=0; j<nc; j++){
					for(int k=0; k<imageChannels; k++){
//						data_out[j*imageChannels+k] = -(255.0*log(((double)data_in[j*imageChannels+k] +1.0)/255.0))/log(255.0);

						data_out[j*imageChannels+k] = precomp_res[data_in[j*imageChannels+k]];
					}
				}
		}
	}

	long t1loop = cciutils::ClockGetTime();
	cout << "	After first loop = "<< t1loop - t2 <<endl;

	//channel deconvolution
	Mat cn = Mat::zeros(nr, nc, CV_64FC2);
	int dn_channels = dn.channels();
	int cn_channels = cn.channels();

	double *Q_ptr = NULL;
	if(Q.isContinuous()){
		Q_ptr = Q.ptr<double>(0);
	}else{
		Q_ptr = (double *)malloc(sizeof(double) * Q.rows * Q.cols);
		assert(Q_ptr != NULL);

		for(int i = 0; i < Q.rows; i++){
			for(int j = 0; j < Q.cols; j++){
				Q_ptr[i * Q.cols + j ] = Q.at<double>(i,j);
			}
		}
	}


	for(int i=0; i<nr; i++){
		const double *dn_ptr = dn.ptr<double>(i);
		double *cn_ptr = cn.ptr<double>(i);
		for(int j=0; j<nc; j++){
			if (dn.channels() == 3){
				for(int k=0; k<dn_channels; k++){
					for(int Q_i=0; Q_i<Q.rows; Q_i++)
						if( BGR2RGB ){
							cn_ptr[j * cn_channels + Q_i] += Q_ptr[Q_i * Q.cols + k]  * dn_ptr[ j * dn_channels + dn_channels-1-k];
						}else{
							cn_ptr[j * cn_channels + Q_i] += Q_ptr[Q_i * Q.cols + k]  * dn_ptr[ j * dn_channels + k];
						}
				}
			}
			else{
				printf("Image must have 3 channels for color deconvolution \n");
				exit(-1);
			}
		}
	}

	if(!Q.isContinuous()){
		free(Q_ptr);
	}
	long t2loop = cciutils::ClockGetTime();
	cout << "	After 2 loop = "<< t2loop - t1loop <<endl;

	//denormalized H and E channels
	double temp;
	for(int i=0; i<nr; i++){
		uchar *E_ptr = E.ptr<uchar>(i);
		uchar *H_ptr = H.ptr<uchar>(i);
		const double *cn_ptr = cn.ptr<double>(i);
		double log255div255 = log(255.0)/255.0;

		for(int j=0; j<nc; j++){
			temp = exp(-(cn_ptr[j * cn_channels]-255.0)*log255div255);
			H_ptr[j] = cciutils::double2uchar(temp);

			temp = exp(-(cn_ptr[j * cn_channels + 1]-255.0)*log255div255);

			E_ptr[j] = cciutils::double2uchar(temp);
		}
	}

	long t3 = cciutils::ClockGetTime();
	cout << "	Rest = "<< t3-t2loop<<endl;
}

template Mat PixelOperations::invert<unsigned char>(const Mat&);
template Mat PixelOperations::invert<float>(const Mat&);

}


