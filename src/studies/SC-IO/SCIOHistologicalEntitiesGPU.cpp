/*
 * RedBloodCell.cpp
 *
 *  Created on: Jul 1, 2011
 *      Author: tcpan
 */

#include "SCIOHistologicalEntities.h"
#include <iostream>
#include "MorphologicOperations.h"
#include "PixelOperations.h"
#include "utils.h"
#include "gpu_utils.h"
#include "opencv2/gpu/gpu.hpp"
#include "NeighborOperations.h"

//#define HAVE_CUDA

#if defined (HAVE_CUDA)
#include "ccl_uf.cuh"
#include "opencv2/gpu/stream_accessor.hpp"
#endif

namespace nscale {

using namespace cv;

namespace gpu {

using namespace cv::gpu;




#if !defined (HAVE_CUDA)
GpuMat SCIOHistologicalEntities::getRBC(const std::vector<GpuMat>& bgr, Stream& stream,
		::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) { throw_nogpu(); }
GpuMat SCIOHistologicalEntities::getBackground(const std::vector<GpuMat>& g_bgr, Stream& stream,
				::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) { throw_nogpu(); }
int SCIOHistologicalEntities::segmentNuclei(const GpuMat& g_img, GpuMat& g_output,
		int &compcount, int *&g_bbox,  cv::gpu::Stream *str,
		::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) { throw_nogpu(); }
int SCIOHistologicalEntities::segmentNuclei(const Mat& img, Mat& output,
		int &compcount, int *&bbox, cv::gpu::Stream *str,
		::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) { throw_nogpu(); }
int SCIOHistologicalEntities::segmentNuclei(const std::string& input, const std::string& output,
		int &compcount, int *&bbox, cv::gpu::Stream *str,
		::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) { throw_nogpu(); }


// the following are specific to the task based implementation for HPDC paper.  The pipeline is refactoring into this form so we're maintaining one set of code.
int plFindNucleusCandidates(const cv::gpu::GpuMat& img, cv::gpu::GpuMat& nuclei,
		::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) { throw_nogpu(); }  // S1
int plSeparateNuclei(const cv::gpu::GpuMat& img, const cv::gpu::GpuMat& seg_open, cv::gpu::GpuMat& seg_nonoverlap,
		::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) { throw_nogpu(); }// A4



#else

GpuMat SCIOHistologicalEntities::getRBC(const std::vector<GpuMat>& bgr, Stream& stream,
		::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) {
	CV_Assert(bgr.size() >= 3);
	/*
	%T1=2.5; T2=2;
    T1=5; T2=4;

	imR2G = double(r)./(double(g)+eps);
    bw1 = imR2G > T1;
    bw2 = imR2G > T2;
    ind = find(bw1);
    if ~isempty(ind)
        [rows, cols]=ind2sub(size(imR2G),ind);
        rbc = bwselect(bw2,cols,rows,8) & (double(r)./(double(b)+eps)>1);
    else
        rbc = zeros(size(imR2G));
    end
	 */
//	float T1 = 5.0;
//	float T2 = 4.0;
	double T1 = 5.0;
	double T2 = 4.0;
	Size s = bgr[0].size();
//	int newType = CV_32FC1;
	int newType = CV_64FC1;
	GpuMat bd, gd, rd;
//	stream.enqueueConvert(bgr[0], bd, newType, 1.0, ::std::numeric_limits<float>::epsilon());
//	stream.enqueueConvert(bgr[1], gd, newType, 1.0, ::std::numeric_limits<float>::epsilon());
	stream.enqueueConvert(bgr[0], bd, newType, 1.0, ::std::numeric_limits<double>::epsilon());
	stream.enqueueConvert(bgr[1], gd, newType, 1.0, ::std::numeric_limits<double>::epsilon());
//	stream.enqueueConvert(bgr[0], bd, newType);
//	::cv::gpu::add(bd, Scalar(::std::numeric_limits<double>::epsilon()), stream);
//	stream.enqueueConvert(bgr[1], gd, newType);
//	::cv::gpu::add(gd, Scalar(::std::numeric_limits<double>::epsilon()), stream);
	stream.enqueueConvert(bgr[2], rd, newType);
	stream.waitForCompletion();
	if (iresHandler) iresHandler->saveIntermediate(rd, 103);
	if (iresHandler) iresHandler->saveIntermediate(gd, 104);
	if (iresHandler) iresHandler->saveIntermediate(bd, 105);

	GpuMat imR2G, imR2B;
//	divide(rd, gd, imR2G, stream);
//	divide(rd, bd, imR2B, stream);
	imR2G = ::nscale::gpu::PixelOperations::divide<double>(rd, gd, stream);
	imR2B = ::nscale::gpu::PixelOperations::divide<double>(rd, bd, stream);

	stream.waitForCompletion();
	rd.release();
	gd.release();
	bd.release();
//	GpuMat bw3 = PixelOperations::threshold<float>(imR2B, 1.0, std::numeric_limits<float>::max(), stream);
//	GpuMat bw1 = PixelOperations::threshold<float>(imR2G, T1, std::numeric_limits<float>::max(), stream);
//	GpuMat bw2 = PixelOperations::threshold<float>(imR2G, T2, std::numeric_limits<float>::max(), stream);
	GpuMat bw3 = PixelOperations::threshold<double>(imR2B, 1.0, false, std::numeric_limits<double>::max(), true, stream);
	GpuMat bw1 = PixelOperations::threshold<double>(imR2G, T1, false, std::numeric_limits<double>::max(), true, stream);
	GpuMat bw2 = PixelOperations::threshold<double>(imR2G, T2, false, std::numeric_limits<double>::max(), true, stream);
	stream.waitForCompletion();
	if (iresHandler) iresHandler->saveIntermediate(imR2G, 101);
	if (iresHandler) iresHandler->saveIntermediate(imR2B, 102);
	if (iresHandler) iresHandler->saveIntermediate(bw1, 107);
	if (iresHandler) iresHandler->saveIntermediate(bw2, 108);
	if (iresHandler) iresHandler->saveIntermediate(bw3, 109);

	imR2G.release();
	imR2B.release();


	GpuMat rbc(s, CV_8UC1);
	stream.enqueueMemSet(rbc, Scalar(0));


	if (countNonZero(bw1) > 0) {
		GpuMat temp = ::nscale::gpu::bwselect<unsigned char>(bw2, bw1, 8, stream);

//	strange.  have to copy it else bitwise_and gets misaligned address error (see it in debugger)
		GpuMat temp2(temp.size(), temp.type());
		stream.enqueueCopy(temp, temp2);
		temp.release();

		bitwise_and(temp2, bw3, rbc, GpuMat(), stream);
	    stream.waitForCompletion();
		temp2.release();
	}
	bw1.release();
	bw2.release();
	bw3.release();

	return rbc;
}

GpuMat SCIOHistologicalEntities::getBackground(const std::vector<GpuMat>& g_bgr, Stream& stream,
		::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) {
	/*
	* this part to decide if the tile is background or foreground
	THR = 0.9;
    grayI = rgb2gray(I);
    area_bg = length(find(I(:,:,1)>=220&I(:,:,2)>=220&I(:,:,3)>=220));
    ratio = area_bg/numel(grayI);
    if ratio >= THR
        return;
    end
	 */

	GpuMat b1, g1, r1;
	GpuMat g_bg(g_bgr[0].size(), CV_8U);
	stream.enqueueMemSet(g_bg, Scalar(0));
	unsigned char max = std::numeric_limits<unsigned char>::max();
	threshold(g_bgr[0], b1, 220, max, THRESH_BINARY, stream);
	threshold(g_bgr[1], g1, 220, max, THRESH_BINARY, stream);
	threshold(g_bgr[2], r1, 220, max, THRESH_BINARY, stream);
	bitwise_and(b1, r1, g_bg, g1, stream);
	stream.waitForCompletion();

	b1.release();
	g1.release();
	r1.release();

	return g_bg;
}



// S1
int SCIOHistologicalEntities::plFindNucleusCandidates(const GpuMat& g_img, GpuMat& g_seg_norbc,
		Stream& stream,
		::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) {

	std::vector<GpuMat> g_bgr;
	split(g_img, g_bgr, stream);
	stream.waitForCompletion();
	if (logsession) logsession->logTimeSinceLastLog("toRGB");

	GpuMat g_bg = ::nscale::gpu::SCIOHistologicalEntities::getBackground(g_bgr, stream, logsession, iresHandler);
	stream.waitForCompletion();
	if (iresHandler) iresHandler->saveIntermediate(g_bg, 1);

	int bgArea = countNonZero(g_bg);
	g_bg.release();

	float ratio = (float)bgArea / (float)(g_img.size().area());
	//std::cout << " background size: " << bgArea << " ratio: " << ratio << std::endl;
	if (logsession) logsession->log("backgroundRatio", ratio);
	if (ratio >= 0.99) {
		//std::cout << "background.  next." << std::endl;
		if (logsession) logsession->logTimeSinceLastLog("background");
		//if (logsession) logsession->endSession();
		g_bgr[0].release();
		g_bgr[1].release();
		g_bgr[2].release();
		return ::nscale::SCIOHistologicalEntities::BACKGROUND;
	} else if (ratio >= 0.9) {
		//std::cout << "background.  next." << std::endl;
		if (logsession) logsession->logTimeSinceLastLog("background likely");
		//if (logsession) logsession->endSession();
		g_bgr[0].release();
		g_bgr[1].release();
		g_bgr[2].release();

		return ::nscale::SCIOHistologicalEntities::BACKGROUND_LIKELY;
	}
	if (logsession) logsession->logTimeSinceLastLog("background");

	GpuMat g_rbc = ::nscale::gpu::SCIOHistologicalEntities::getRBC(g_bgr, stream, logsession, iresHandler);
	stream.waitForCompletion();
	if (iresHandler) iresHandler->saveIntermediate(g_rbc, 2);

	if (logsession) logsession->logTimeSinceLastLog("RBC");
	int rbcPixelCount = countNonZero(g_rbc);
	if (logsession) logsession->log("RBCPixCount", rbcPixelCount);

	GpuMat g_r(g_bgr[2]);
	g_bgr[0].release();
	g_bgr[1].release();
	g_bgr[2].release();



//	Mat rbc(g_rbc.size(), g_rbc.type());
//	stream.enqueueDownload(g_rbc, rbc);
//	stream.waitForCompletion();
//  g_rbc.release();
//	if (logsession) logsession->logTimeSinceLastLog("cpuCopyRBC");
//	imwrite("test/out-rbc.pbm", rbc);

	/*
	rc = 255 - r;
    rc_open = imopen(rc, strel('disk',10));
    rc_recon = imreconstruct(rc_open,rc);
    diffIm = rc-rc_recon;
	 */

	GpuMat g_rc = ::nscale::gpu::PixelOperations::invert<unsigned char>(g_r, stream);
	stream.waitForCompletion();
	g_r.release();
	if (logsession) logsession->logTimeSinceLastLog("invert");

	GpuMat g_rc_open(g_rc.size(), g_rc.type());
	//Mat disk19 = getStructuringElement(MORPH_ELLIPSE, Size(19,19));
	// structuring element is not the same between matlab and opencv.  using the one from matlab explicitly....
	// (for 4, 6, and 8 connected, they are approximations).
	unsigned char disk19raw[361] = {
			0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
			0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
			0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
			0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
			0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
			0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
			0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0};
	std::vector<unsigned char> disk19vec(disk19raw, disk19raw+361);
	Mat disk19(disk19vec);
	disk19 = disk19.reshape(1, 19);
//	imwrite("test/out-rcopen-strel.pbm", disk19);
	// filter doesnot check borders.  so need to create border.
	g_rc_open = ::nscale::gpu::morphOpen<unsigned char>(g_rc, disk19, stream);

//	GpuMat rc_border;
//	copyMakeBorder(g_rc, rc_border, 9,9,9,9, Scalar(0), stream);
////	GpuMat rc_roi(rc_border, Range(9, 9+ g_rc.size().height), Range(9, 9+g_rc.size().width));
////	morphologyEx(rc_roi, g_rc_open, MORPH_OPEN, disk19, Point(-1, -1), 1, stream);
//	GpuMat rc_roi(rc_border.size(), rc_border.type());
//	morphologyEx(rc_border, rc_roi, MORPH_OPEN, disk19, Point(-1, -1), 1, stream);
//	g_rc_open = rc_roi(Rect(9, 9, g_rc.cols, g_rc.rows));
//	stream.waitForCompletion();
//	rc_border.release();
//	rc_roi.release();
	if (iresHandler) iresHandler->saveIntermediate(g_rc_open, 3);

	if (logsession) logsession->logTimeSinceLastLog("open19");
//	imwrite("test/out-rcopen.ppm", rc_open);


	unsigned int iter;
	GpuMat g_rc_recon = ::nscale::gpu::imreconstructQueueSpeedup<unsigned char>(g_rc_open, g_rc, 8, 1, stream);
	stream.waitForCompletion();
//	std::cout << "\tIterations: " << iter << std::endl;
	if (iresHandler) iresHandler->saveIntermediate(g_rc_recon, 4);


	GpuMat g_diffIm;
	subtract(g_rc, g_rc_recon, g_diffIm, stream);
	stream.waitForCompletion();
	if (iresHandler) iresHandler->saveIntermediate(g_diffIm, 5);


	if (logsession) logsession->logTimeSinceLastLog("reconToNuclei");
	//if (logsession) logsession->log("rc_reconIter", iter);
	int rc_openPixelCount = countNonZero(g_rc_open);
	if (logsession) logsession->log("rc_openPixCount", rc_openPixelCount);
	g_rc_open.release();
	g_rc.release();
	g_rc_recon.release();

	Mat diffIm(g_diffIm.size(), g_diffIm.type());
	stream.enqueueDownload(g_diffIm, diffIm);
	stream.waitForCompletion();
	if (logsession) logsession->logTimeSinceLastLog("downloadReconToNuclei");

/*
    G1=80; G2=45; % default settings
    %G1=80; G2=30;  % 2nd run

    bw1 = imfill(diffIm>G1,'holes');
 *
 */
	unsigned char G1 = 80;
	GpuMat g_diffIm2;
	threshold(g_diffIm, g_diffIm2, G1, std::numeric_limits<unsigned char>::max(), THRESH_BINARY, stream);
	stream.waitForCompletion();
	if (iresHandler) iresHandler->saveIntermediate(g_diffIm2, 6);

	if (logsession) logsession->logTimeSinceLastLog("threshold1");


	GpuMat g_bw1 = ::nscale::gpu::imfillHoles<unsigned char>(g_diffIm2, true, 4, stream);
	stream.waitForCompletion();
	if (logsession) logsession->logTimeSinceLastLog("fillHoles1");
	if (iresHandler) iresHandler->saveIntermediate(g_bw1, 7);

	g_diffIm2.release();
	stream.waitForCompletion();
//	if (logsession) logsession->logTimeSinceLastLog("GPU done");
	
	
	
	//	// TODO: change back
//	return ::nscale::SCIOHistologicalEntities::SUCCESS;
/*
 *     %CHANGE
    [L] = bwlabel(bw1, 8);
    stats = regionprops(L, 'Area');
    areas = [stats.Area];

    %CHANGE
    ind = find(areas>10 & areas<1000);
    bw1 = ismember(L,ind);
    bw2 = diffIm>G2;
    ind = find(bw1);

    if isempty(ind)
        return;
    end
 *
 */
	int compcount2;
//#if defined (USE_UF_CCL)
	GpuMat g_bw1_t = ::nscale::gpu::bwareaopen2(g_bw1, false, true, 11, 1000, 8, compcount2, stream);
	stream.waitForCompletion();
	if (iresHandler) iresHandler->saveIntermediate(g_bw1_t, 8);
printf(" gpu compcount 11-1000 = %d\n", compcount2);
//#else
//	Mat bw1(g_bw1.size(), g_bw1.type());
//	stream.enqueueDownload(g_bw1, bw1);
//	stream.waitForCompletion();
//
//	bw1 = ::nscale::bwareaopen<unsigned char>(bw1, 11, 1000, 8, compcount);
//	if (iresHandler) iresHandler->saveIntermediate(bw1, 8);
//
//	GpuMat g_bw1_t(bw1.size(), bw1.type());
//	stream.enqueueUpload(bw1, g_bw1_t);
//	stream.waitForCompletion();
//	bw1.release();
//#endif
	if (logsession) logsession->logTimeSinceLastLog("areaThreshold1");
	g_bw1.release();
	if (compcount2 == 0) {
		g_diffIm.release();
		g_rbc.release();
		printf("no candidates left: compcount2 = %d\n", compcount2);
		return ::nscale::SCIOHistologicalEntities::NO_CANDIDATES_LEFT;
	}


	unsigned char G2 = 45;
	GpuMat g_bw2;
	threshold(g_diffIm, g_bw2, G2, std::numeric_limits<unsigned char>::max(), THRESH_BINARY, stream);
	stream.waitForCompletion();
	g_diffIm.release();
	if (iresHandler) iresHandler->saveIntermediate(g_bw2, 9);

		/*
	 *
    [rows,cols] = ind2sub(size(diffIm),ind);
    seg_norbc = bwselect(bw2,cols,rows,8) & ~rbc;
    seg_nohole = imfill(seg_norbc,'holes');
    seg_open = imopen(seg_nohole,strel('disk',1));
	 *
	 */

	GpuMat g_seg_norbc2 = ::nscale::gpu::bwselect<unsigned char>(g_bw2, g_bw1_t, 8, stream);
	stream.waitForCompletion();
	g_bw1_t.release();
	g_bw2.release();
	if (iresHandler) iresHandler->saveIntermediate(g_seg_norbc2, 10);

	g_seg_norbc = ::cv::gpu::createContinuous(g_seg_norbc2.size(), g_seg_norbc2.type());
	GpuMat temp(g_rbc.size(), g_rbc.type());
	bitwise_not(g_rbc, temp, GpuMat(), stream);
	bitwise_and(g_seg_norbc2, temp, g_seg_norbc, GpuMat(), stream);

	stream.waitForCompletion();
	g_rbc.release();
	temp.release();
	g_seg_norbc2.release();
	if (iresHandler) iresHandler->saveIntermediate(g_seg_norbc, 11);

	//	imwrite("test/out-nucleicandidatesnorbc.ppm", seg_norbc);
	if (logsession) logsession->logTimeSinceLastLog("blobsGt45");

	return ::nscale::SCIOHistologicalEntities::CONTINUE;
}

// A4
int SCIOHistologicalEntities::plSeparateNuclei(const GpuMat& g_img, GpuMat& g_seg_open, GpuMat& g_seg_nonoverlap, Stream& stream,
		::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) {

	/*
	 *
	seg_big = imdilate(bwareaopen(seg_open,30),strel('disk',1));
	 */
	// bwareaopen is done as a area threshold.
	int compcount2;
//#if defined (USE_UF_CCL)
	GpuMat g_seg_big_t = ::nscale::gpu::bwareaopen2(g_seg_open, false, true, 30, std::numeric_limits<int>::max(), 8, compcount2, stream);
	stream.waitForCompletion();
	if (iresHandler) iresHandler->saveIntermediate(g_seg_big_t, 14);
printf(" gpu compcount 30-1000 = %d\n", compcount2);
//#else
//	Mat seg_open(g_seg_open.size(), g_seg_open.type());
//	stream.enqueueDownload(g_seg_open, seg_open);
//	stream.waitForCompletion();
//
//	Mat seg_big_t = ::nscale::bwareaopen<unsigned char>(seg_open, 30, std::numeric_limits<int>::max(), 8, compcount);
//	if (iresHandler) iresHandler->saveIntermediate(seg_big_t, 14);
//	seg_open.release();
//
//	GpuMat g_seg_big_t(seg_big_t.size(), seg_big_t.type());
//	stream.enqueueUpload(seg_big_t, g_seg_big_t);
//	stream.waitForCompletion();
//	seg_big_t.release();
//#endif
	if (logsession) logsession->logTimeSinceLastLog("30To1000");
	// a 3x3 mat with a cross
	unsigned char disk3raw[9] = {
			0, 1, 0,
			1, 1, 1,
			0, 1, 0};
	std::vector<unsigned char> disk3vec(disk3raw, disk3raw+9);
	Mat disk3(disk3vec);
	// can't use morphologyEx.  the erode phase is not creating a border even though the method signature makes it appear that way.
	// because of this, and the fact that erode and dilate need different border values, have to do the erode and dilate myself.
	//	morphologyEx(seg_nohole, seg_open, CV_MOP_OPEN, disk3, Point(1,1)); //, Point(-1, -1), 1, BORDER_REFLECT);
	disk3 = disk3.reshape(1, 3);

	GpuMat g_seg_big = ::nscale::gpu::morphDilate<unsigned char>(g_seg_big_t, disk3, stream);
//	GpuMat g_twm;
//	copyMakeBorder(g_seg_big_t, g_twm, 1, 2, 1, 2, Scalar(std::numeric_limits<unsigned char>::min()), stream);
//	GpuMat g_t_big= ::cv::gpu::createContinuous(g_twm.size(), g_twm.type());
//	stream.enqueueMemSet(g_t_big, Scalar(0));
//	dilate(g_twm, g_t_big, disk3, Point(-1, -1), 1, stream);
//	GpuMat g_seg_big(g_seg_big_t.size(), g_t_big.type());
//	stream.enqueueCopy(g_t_big(Rect(1, 1, g_seg_big_t.cols, g_seg_big_t.rows)), g_seg_big);
//	stream.waitForCompletion();
//
//	g_t_big.release();
//	g_seg_big_t.release();
//	g_twm.release();

	if (iresHandler) iresHandler->saveIntermediate(g_seg_big, 15);

	//	imwrite("test/out-nucleicandidatesbig.ppm", seg_big);
	if (logsession) logsession->logTimeSinceLastLog("dilate");

	/*
	 *
		distance = -bwdist(~seg_big);
		distance(~seg_big) = -Inf;
		distance2 = imhmin(distance, 1);
		 *
	 *
	 */
	// distance transform:  matlab code is doing this:
	// invert the image so nuclei candidates are holes
	// compute the distance (distance of nuclei pixels to background)
	// negate the distance.  so now background is still 0, but nuclei pixels have negative distances
	// set background to -inf

	// really just want the distance map.  CV computes distance to 0.
	// background is 0 in output.
	// then invert to create basins
	Mat dist(g_seg_big.size(), CV_32FC1);
	Mat seg_big(g_seg_big.size(), g_seg_big.type());
	stream.enqueueDownload(g_seg_big, seg_big);
	stream.waitForCompletion();

	// opencv: compute the distance to nearest zero
	// matlab: compute the distance to the nearest non-zero
	distanceTransform(seg_big, dist, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	seg_big.release();

	GpuMat g_dist(dist.size(), dist.type());
	stream.enqueueUpload(dist, g_dist);
	stream.waitForCompletion();
	dist.release();
	

//	double mmin, mmax;
//	minMaxLoc(g_dist, &mmin, &mmax);
	if (iresHandler) iresHandler->saveIntermediate(g_dist, 16);

	// invert and shift (make sure it's still positive)
	GpuMat g_distneg = ::nscale::gpu::PixelOperations::invert<float>(g_dist, stream);
	stream.waitForCompletion();
	g_dist.release();

	// then set the background to -inf and do imhmin
	//Mat distance = Mat::zeros(dist.size(), dist.type());
	// appears to work better with -inf as background
	GpuMat g_distance(g_distneg.size(), g_distneg.type());
	stream.enqueueMemSet(g_distance, -std::numeric_limits<float>::max());
	//dist.copyTo(distance, seg_big);
	bitwise_and(g_distneg, g_distneg, g_distance, g_seg_big, stream);
	stream.waitForCompletion();
	g_distneg.release();
//	cciutils::cv::imwriteRaw("test/out-distance", distance);
	if (logsession) logsession->logTimeSinceLastLog("distTransform");
	if (iresHandler) iresHandler->saveIntermediate(g_distance,17);

	// then do imhmin. (prevents small regions inside bigger regions)
	GpuMat g_distance2 = ::nscale::gpu::imhmin<float>(g_distance, 1.0f, 8, stream);
	stream.waitForCompletion();
	g_distance.release();
	if (logsession) logsession->logTimeSinceLastLog("imhmin");
	if (iresHandler) iresHandler->saveIntermediate(g_distance2, 18);

//cciutils::cv::imwriteRaw("test/out-distanceimhmin", distance2);


	/*
	 *
		seg_big(watershed(distance2)==0) = 0;
		seg_nonoverlap = seg_big;
     *
	 */

	GpuMat g_watermask = ::nscale::gpu::watershedDW(g_seg_big, g_distance2, -1, 8, stream);
	stream.waitForCompletion();
	g_distance2.release();


	// watershed in openCV requires labels.  input foreground > 0, 0 is background
	// critical to use just the nuclei and not the whole image - else get a ring surrounding the regions.
	//Mat watermask = ::nscale::watershed2(nuclei, distance2, 8);
//	cciutils::cv::imwriteRaw("test/out-watershed", watermask);
	if (logsession) logsession->logTimeSinceLastLog("watershed");
	if (iresHandler) iresHandler->saveIntermediate(g_watermask, 20);


	//// MASK APPROACH
//	// background is -1.
	g_seg_nonoverlap = ::cv::gpu::createContinuous(g_seg_big.size(), g_seg_big.type());
	stream.enqueueMemSet(g_seg_nonoverlap, Scalar(0));
	//	CPU: seg_big.copyTo(seg_nonoverlap, (watermask >= 0));
	GpuMat g_wmask = ::nscale::gpu::PixelOperations::threshold<int>(g_watermask, 0, true, std::numeric_limits<int>::max(), true, stream);
	bitwise_and(g_wmask, g_seg_big, g_seg_nonoverlap, GpuMat(), stream);
	stream.waitForCompletion();
	g_wmask.release();
	if (logsession) logsession->logTimeSinceLastLog("water to mask");
	if (iresHandler) iresHandler->saveIntermediate(g_seg_nonoverlap, 21);



	// LABEL Approach - cpu version does not support erode
	// GPU watershed has -1 as background.  so don't need to replace?
//	g_seg_nonoverlap = ::nscale::gpu::PixelOperations::replace<int>(g_watermask, (int)0, (int)-1, stream);
//	g_seg_nonoverlap = g_watermask;

	g_watermask.release();
	g_seg_big.release();


	return ::nscale::SCIOHistologicalEntities::CONTINUE;
}


int SCIOHistologicalEntities::segmentNuclei(const std::string& in, const std::string& out,
		int &compcount, int *&bbox,  cv::gpu::Stream *str,
		::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) {
	Mat input = imread(in);
	if (!input.data) return ::nscale::SCIOHistologicalEntities::INVALID_IMAGE;

	Mat output;

	int status = ::nscale::gpu::SCIOHistologicalEntities::segmentNuclei(input, output, compcount, bbox, str, logsession, iresHandler);
	input.release();

	if (status == ::nscale::SCIOHistologicalEntities::SUCCESS)
		imwrite(out, output);

	output.release();

	return status;
}



int SCIOHistologicalEntities::segmentNuclei(const Mat& img, Mat& output,
		int &compcount, int *&bbox,  cv::gpu::Stream *str,
		::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) {
	if (!img.data) return ::nscale::SCIOHistologicalEntities::INVALID_IMAGE;

	Stream *str1 = str;
	if (str1 == NULL) {
		str1 = new Stream();
	}
	Stream &stream = *str1;

	GpuMat g_img;
	stream.enqueueUpload(img, g_img);
	stream.waitForCompletion();
	GpuMat g_output;
	int *g_bbox;

	int status = ::nscale::gpu::SCIOHistologicalEntities::segmentNuclei(g_img, g_output, compcount, g_bbox, str1, logsession, iresHandler);

	output = cv::Mat::zeros(g_output.size(), g_output.type());
	stream.enqueueDownload(g_output, output);
	stream.waitForCompletion();
	if (logsession) logsession->logTimeSinceLastLog("uploaded image");
	g_output.release();
	g_img.release();

	bbox = (int *)malloc(sizeof(int) * compcount * 5);
	cudaMemcpy(bbox, g_bbox, sizeof(int) * compcount * 5, cudaMemcpyDeviceToHost);\
	cudaFree(g_bbox);
        printf(" bbox: %d, %d, %d, %d, %d\n", bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]);

	return status;


}

int SCIOHistologicalEntities::segmentNuclei(const GpuMat& g_img, GpuMat& g_output,
		int &compcount, int *&g_bbox,  cv::gpu::Stream *str,
		::cciutils::SCIOLogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) {


	// image in BGR format
	if (!g_img.data) return ::nscale::SCIOHistologicalEntities::INVALID_IMAGE;


	Stream *str1 = str;
	if (str1 == NULL) {
		str1 = new Stream();
	}
	Stream &stream = *str1;

	if (logsession) logsession->logT0("start");
	if (iresHandler) iresHandler->saveIntermediate(g_img, 0);

	GpuMat g_seg_norbc(g_img.size(), CV_8U);
	int findCandidateResult = ::nscale::gpu::SCIOHistologicalEntities::plFindNucleusCandidates(g_img, g_seg_norbc, stream, logsession, iresHandler);
	if (findCandidateResult != ::nscale::SCIOHistologicalEntities::CONTINUE) {
		return findCandidateResult;
	}

	GpuMat g_seg_nohole = ::nscale::gpu::imfillHoles<unsigned char>(g_seg_norbc, true, 4, stream);
	stream.waitForCompletion();
	g_seg_norbc.release();
	if (logsession) logsession->logTimeSinceLastLog("fillHoles2");
	if (iresHandler) iresHandler->saveIntermediate(g_seg_nohole, 12);


	// a 3x3 mat with a cross
	unsigned char disk3raw[9] = {
			0, 1, 0,
			1, 1, 1,
			0, 1, 0};
	std::vector<unsigned char> disk3vec(disk3raw, disk3raw+9);
	Mat disk3(disk3vec);
	// can't use morphologyEx.  the erode phase is not creating a border even though the method signature makes it appear that way.
	// because of this, and the fact that erode and dilate need different border values, have to do the erode and dilate myself.
	//	morphologyEx(seg_nohole, seg_open, CV_MOP_OPEN, disk3, Point(1,1)); //, Point(-1, -1), 1, BORDER_REFLECT);
	disk3 = disk3.reshape(1, 3);

	GpuMat g_seg_open = ::nscale::gpu::morphOpen<unsigned char>(g_seg_nohole, disk3, stream);
//	//	imwrite("test/out-rcopen-strel.pbm", disk19);
//	// filter doesnot check borders.  so need to create border.
//	GpuMat g_t_seg_nohole;
//	copyMakeBorder(g_seg_nohole, g_t_seg_nohole, 1,1,1,1, Scalar(std::numeric_limits<unsigned char>::max()), stream);
//	GpuMat g_t_seg_erode(g_t_seg_nohole.size(), g_t_seg_nohole.type());
//	erode(g_t_seg_nohole, g_t_seg_erode, disk3, Point(-1,-1), 1, stream);
//	GpuMat g_seg_erode = g_t_seg_erode(Rect(1, 1, g_seg_nohole.cols, g_seg_nohole.rows));
//	GpuMat g_t_seg_erode2;
//	copyMakeBorder(g_seg_erode, g_t_seg_erode2, 1,1,1,1, Scalar(std::numeric_limits<unsigned char>::min()), stream);
//	GpuMat g_t_seg_open(g_t_seg_erode2.size(), g_t_seg_erode2.type());
//	dilate(g_t_seg_erode2, g_t_seg_open, disk3, Point(-1,-1), 1, stream);
//	GpuMat g_seg_open = g_t_seg_open(Rect(1, 1, g_seg_nohole.cols, g_seg_nohole.rows));
//	stream.waitForCompletion();
//	g_t_seg_open.release();
//	g_t_seg_erode2.release();
//	g_seg_erode.release();
//	g_t_seg_erode.release();
//	g_t_seg_nohole.release();
	g_seg_nohole.release();

//	imwrite("test/out-nucleicandidatesopened.ppm", seg_open);
	if (logsession) logsession->logTimeSinceLastLog("openBlobs");
	if (iresHandler) iresHandler->saveIntermediate(g_seg_open, 13);

	GpuMat g_seg_nonoverlap(g_img.size(), CV_8U);
	
	int sepResult = ::nscale::gpu::SCIOHistologicalEntities::plSeparateNuclei(g_img, g_seg_open, g_seg_nonoverlap, stream, logsession, iresHandler);
	g_seg_open.release();
	if (sepResult != ::nscale::SCIOHistologicalEntities::CONTINUE) {
		return sepResult;
	}

	// erode by 1 - need to be fixed  - erode does not check borders.
	// ERODE IS ONLY NEEDED BY CPU Watershed,

	/*
     %CHANGE
    [L] = bwlabel(seg_nonoverlap, 4);
    stats = regionprops(L, 'Area');
    areas = [stats.Area];

    %CHANGE
    ind = find(areas>20 & areas<1000);

    if isempty(ind)
        return;
    end
    seg = ismember(L,ind);
	 *
	 */
	int compcount2;
//#if defined (USE_UF_CCL)
	// MASK approach
	GpuMat g_seg = ::nscale::gpu::bwareaopen2(g_seg_nonoverlap, false, true, 21, 1000, 4, compcount2, stream);
	// LABEL approach - cpu does not support 32bit int erode.
//	GpuMat g_seg = ::nscale::gpu::bwareaopen2(g_seg_nonoverlap, true, false, 21, 1000, 4, compcount2, stream);
	stream.waitForCompletion();
	if (iresHandler) iresHandler->saveIntermediate(g_seg, 23);
printf("  gpu components 21-1000 = %d\n", compcount2);	
//#else
//	Mat seg_nonoverlap(g_seg_nonoverlap.size(), g_seg_nonoverlap.type());
//	stream.enqueueDownload(g_seg_nonoverlap, seg_nonoverlap);
//	stream.waitForCompletion();
//
//	Mat seg = ::nscale::bwareaopen<unsigned char>(seg_nonoverlap, 21, 1000, 4, compcount);
//	if (iresHandler) iresHandler->saveIntermediate(seg, 23);
//	seg_nonoverlap.release();
////	imwrite("test/out-seg.ppm", seg);
//
//	GpuMat g_seg(seg.size(), seg.type());
//	stream.enqueueUpload(seg, g_seg);
//	stream.waitForCompletion();
//	seg.release();
//#endif
	if (logsession) logsession->logTimeSinceLastLog("20To1000");
	if (compcount2 == 0) {
		printf("no candidates left: compcount2 = %d\n", compcount2);

		return ::nscale::SCIOHistologicalEntities::NO_CANDIDATES_LEFT;
	}
	g_seg_nonoverlap.release();
	
 /*
	 *     %CHANGE
    %[L, num] = bwlabel(seg,8);

    %CHANGE
    [L,num] = bwlabel(imfill(seg, 'holes'),4);
	 *
	 */
	// don't worry about bwlabel.
	// MASK approach
	GpuMat g_final = ::nscale::gpu::imfillHoles<unsigned char>(g_seg, true, 8, stream);
	// LABEL approach - cpu version does not support erode
//	g_output = ::nscale::gpu::imfillHoles<int>(g_seg, false, 8, stream);
	stream.waitForCompletion();
	if (logsession) logsession->logTimeSinceLastLog("fillHolesLast");
	g_seg.release();
	
	// MASK approach
	g_output = nscale::gpu::bwlabel(g_final, 8, true, stream);
	g_final.release();

	g_bbox = ::nscale::gpu::boundingBox2(g_output.cols, g_output.rows, (int*)g_output.data, 0, compcount, cv::gpu::StreamAccessor::getStream(stream));
	stream.waitForCompletion();
	
	printf("  gpu number of bounding boxes = %d\n", compcount);
 
//	imwrite("test/out-nuclei.ppm", seg);

//	if (logsession) logsession->endSession();

	return ::nscale::SCIOHistologicalEntities::SUCCESS;

}

#endif
}}
