#ifndef	COLOR_DECONVOLUTION
#define	COLOR_DECONVOLUTION

void ColorDeconv( const Mat& image, const Mat& M, const Mat& b, Mat& H, Mat& E, bool BGR2RGB=true);
void ColorDeconvOptimized( const Mat& image, const Mat& M, const Mat& b, Mat& H, Mat& E, bool BGR2RGB=true);

void ColorDeconvOptimizedFloat( const Mat& image, const Mat& M, const Mat& b, Mat& H, Mat& E, bool BGR2RGB=true);


#endif
