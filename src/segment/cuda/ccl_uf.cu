/**
 * adapted from https://github.com/victormatheus/CCL-GPU.
 * removed CPU code
 * also made it 4 and 8 connected capable.
 * assume data is already on GPU.
 */

#include "textures.cuh"
#include "ccl_uf.cuh"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/unique.h>

namespace nscale {
namespace gpu {


#define UF_BLOCK_SIZE_X 32
#define UF_BLOCK_SIZE_Y 16


//CUDA

#define START_TIME_T cudaEventRecord(startt,0)
#define STOP_TIME_T  cudaEventRecord(stopt,0 ); \
                   cudaEventSynchronize(stopt); \
                   cudaEventElapsedTime( &ett, startt, stopt )

__device__ int find(int* buf, int x) {
    while (x != buf[x]) {
      x = buf[x];
    }
    return x;
}

__device__ void findAndUnion(int* buf, int g1, int g2) {
    bool done;    
    do {

      g1 = find(buf, g1);
      g2 = find(buf, g2);    
 
      // it should hold that g1 == buf[g1] and g2 == buf[g2] now
    
      if (g1 < g2) {
          int old = atomicMin(&buf[g2], g1);
          done = (old == g2);
          g2 = old;
      } else if (g2 < g1) {
          int old = atomicMin(&buf[g1], g2);
          done = (old == g1);
          g1 = old;
      } else {
          done = true;
      }

    } while(!done);
}

__global__ void UF_local(int* label, int w, int h, int connectivity) {    
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int global_index = x+y*w;
    int block_index = UF_BLOCK_SIZE_X * threadIdx.y + threadIdx.x;

    __shared__ int s_buffer[UF_BLOCK_SIZE_X * UF_BLOCK_SIZE_Y];
    __shared__ unsigned char s_img[UF_BLOCK_SIZE_X * UF_BLOCK_SIZE_Y];

    bool in_limits = x < w && y < h;

    s_buffer[block_index] = block_index;
    s_img[block_index] = in_limits? tex2D(imgtex, x, y) : 0xFF;
    __syncthreads();

    unsigned char v = s_img[block_index];

    if (in_limits && threadIdx.x>0 && s_img[block_index-1] == v) {
        findAndUnion(s_buffer, block_index, block_index - 1);
    }

    __syncthreads();

    if (in_limits && threadIdx.y>0 && s_img[block_index-UF_BLOCK_SIZE_X] == v) {
        findAndUnion(s_buffer, block_index, block_index - UF_BLOCK_SIZE_X);
    }

    __syncthreads();

    if (connectivity == 8) {
    	if (in_limits && (threadIdx.x>0 && threadIdx.y>0) && s_img[block_index-UF_BLOCK_SIZE_X-1] == v) {
        	findAndUnion(s_buffer, block_index, block_index - UF_BLOCK_SIZE_X - 1);
    	}	

    	__syncthreads();

    	if (in_limits && (threadIdx.y>0 && threadIdx.x<UF_BLOCK_SIZE_X-1) && s_img[block_index-UF_BLOCK_SIZE_X+1] == v) {
        	findAndUnion(s_buffer, block_index, block_index - UF_BLOCK_SIZE_X + 1);
    	}

    	__syncthreads();
    }
    if (in_limits) {
    int f = find(s_buffer, block_index);
    int fx = f % UF_BLOCK_SIZE_X;
    int fy = f / UF_BLOCK_SIZE_X;
    label[global_index] = (blockIdx.y*UF_BLOCK_SIZE_Y + fy)*w +
                            (blockIdx.x*blockDim.x + fx);
    }

}

__global__ void UF_global(int* label, int w, int h, int connectivity) {    
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int global_index = x+y*w;
    
    bool in_limits = x < w && y < h;
    unsigned char v = (in_limits? tex2D(imgtex, x, y) : 0xFF);
 
    if (in_limits && y>0 && threadIdx.y==0 && tex2D(imgtex, x, y-1) == v) {
        findAndUnion(label, global_index, global_index - w);
    }

    if (in_limits && x>0 && threadIdx.x==0 && tex2D(imgtex, x-1, y) == v) {
        findAndUnion(label, global_index, global_index - 1);
    }
// TONY:  this algorithm chunks the image, do local UF, then use only the first row or first column
//  to merge the chunks. (above 2 lines).  now we also need to do diagonals.
// upper left diagonal needs to be updated for the left and top lines
    if (connectivity == 8) {
    	if (in_limits && y>0 && x>0 && (threadIdx.y==0 || threadIdx.x==0) && tex2D(imgtex, x-1, y-1) == v) {
        	findAndUnion(label, global_index, global_index - w - 1);
    	}

// upper right diagonal needs to be updated for the top and right lines.
    	if (in_limits && x<w-1 && y>0 && (threadIdx.y==0 || threadIdx.x==UF_BLOCK_SIZE_X-1) && tex2D(imgtex, x+1, y-1) == v) {
        	findAndUnion(label, global_index, global_index - w + 1);
    	}
    }
}


__global__ void UF_final(int* label, int w, int h, int bgval) {    
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int global_index = x+y*w;
    
    bool in_limits = x < w && y < h;

    if (in_limits) {
        label[global_index] = (tex2D(imgtex, x, y) == 0 ? bgval : find(label, global_index));
    }
}

// img on GPU already.  also label should be allocated on GPU as well.
void CCL(unsigned char* img, int w, int h, int* d_label, int bgval, int connectivity, cudaStream_t stream) {

    cudaError_t err;

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("before startERROR: %s\n", cudaGetErrorString(err));
        return;
    }

//    cudaEvent_t startt,stopt;
//    cudaEventCreate( &startt );
//    cudaEventCreate( &stopt );
//    float ett;


    //START_TIME_T;
    cudaChannelFormatDesc uchardesc = 
        cudaCreateChannelDesc<unsigned char>();
    cudaBindTexture2D(0, imgtex, img, uchardesc, w, h, w * sizeof(unsigned char));
    //STOP_TIME_T;
	//printf("   uf bind texture: %f\n", ett);

    dim3 block (UF_BLOCK_SIZE_X, UF_BLOCK_SIZE_Y);
    dim3 grid ((w+UF_BLOCK_SIZE_X-1)/UF_BLOCK_SIZE_X,
               (h+UF_BLOCK_SIZE_Y-1)/UF_BLOCK_SIZE_Y);
 
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("startERROR: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaThreadSetCacheConfig(cudaFuncCachePreferShared);

    //START_TIME_T;
    UF_local <<<grid, block, 0, stream>>>(d_label, w, h, connectivity);
    //STOP_TIME_T;
	//printf("   uf_local: %f\n", ett);

    cudaThreadSetCacheConfig(cudaFuncCachePreferL1);

	//START_TIME_T;
        UF_global <<<grid, block, 0, stream>>>(d_label, w, h, connectivity);
    //STOP_TIME_T;
	//printf("   uf_global: %f\n", ett);

	//START_TIME_T;
        UF_final <<<grid, block, 0, stream>>>(d_label, w, h, bgval);
    //STOP_TIME_T;
	//printf("   uf_final: %f\n", ett);
 
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("endERROR: %s\n", cudaGetErrorString(err));
        return;
    }
}


// this functor returns true if the argument is odd, and false otherwise
//template <typename TN, typename TO>
//struct TupleElementGreater : public thrust::unary_function<TN, TO>
//{
//    __host__ __device__
//    TO operator()(TN x)
//    {
//        return thrust::get<0>(x) > thrust::get<1>(x) ? (TO)1 : (TO)0;
//    }
//};
/*
template<typename T>
struct IsGreater : public thrust::binary_function<T, T, T>
{
	__host__ __device__
	T operator()(T x, T xm1) {
		return x>xm1 ? (T)1 : (T)0;
	}
};
*/
__global__ void relabel_flatten (int* label, int* roots, int w, int h, int bgval) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int global_index = x+y*w;

    bool in_limits = x < w && y < h;

    if (in_limits) {
        int target = label[global_index];
		if (target == bgval) label[global_index] = 0;
		else if (roots[global_index] == 0)
        	label[global_index] = label[target];
    }
}


template <typename T>
struct greaterThanK
{
	T t;

	greaterThanK(T thresh) : t(thresh) {}

	__host__ __device__
	bool operator()(T x) {
		return x > t;
	}

};

template <typename T>
struct equalsK
{
	T t;

	equalsK(T thresh) : t(thresh) {}

	__host__ __device__
	bool operator()(T x) {
		return x == t;
	}
};


// relabelling.  this relies on bg being smaller in value than all foreground labels.
int relabel(int w, int h, int* d_label, int bgval, cudaStream_t stream) {
	// do this by:  sort by value, find segment, scan, sort by index
/*	typedef typename thrust::tuple<int, int> LabelCompareType;

	thrust::counting_iterator<int> idx(0);
	thrust::device_vector<int> ids(idx, idx+w*h);
	thrust::device_ptr<int> label(d_label);
	thrust::device_vector<int> oldlabel(label, label+w*h);

	// sort by label value
	thrust::stable_sort_by_key(oldlabel.begin(), oldlabel.end(), ids.begin());

	// now use a permutation iterator (ids is now permuted)
	// find where values changes.  do this with a zip iterator
	thrust::device_vector<int> newLabels(w*h, 0);
	IsGreater<int> diffOp;
	thrust::adjacent_difference(oldlabel.begin(), oldlabel.end(), newLabels.begin(), diffOp);
	thrust::fill_n(newLabels.begin(), 1, 0);
	thrust::plus<int> binaryOp;
	thrust::inclusive_scan(newLabels.begin(), newLabels.end(), newLabels.begin(), binaryOp);

//	thrust::transform_inclusive_scan(
//		// input iterator start
//		thrust::make_zip_iterator(
//			// combine a shifted version of the iterators
//			thrust::make_tuple(
//				oldlabel.begin() + 1,
//				oldlabel.begin()
//				)
//			),
//		// input iterator end
//		thrust::make_zip_iterator(
//			// combine a shifted version of the iterators
//			thrust::make_tuple(
//				oldlabel.end(),
//				oldlabel.end()-1
//				)
//			),
//		newLabels.begin() + 1,
//		TupleElementGreater<LabelCompareType, int>(),
//		binaryOp
//		);

	int j = newLabels[w*h-1];
	// now that we have the sequential labels, we can permute back.
	thrust::stable_sort_by_key(ids.begin(), ids.end(), newLabels.begin());
	thrust::copy(newLabels.begin(), newLabels.end(), label);

	// cleanup.
	ids.clear();
	oldlabel.clear();
	newLabels.clear();

	return j;
*/

// smarter way of doing this:  (way faster)
//   transform to find root (label[i] == i)  (element wise)
//   find the number of components via scan (global reduction type)
//   next change roots to the scan id using root map as stencil (element wise)
// 	finally run another flatten operation.	(global random access)
//    cudaEvent_t startt,stopt;
//    cudaEventCreate( &startt );
//    cudaEventCreate( &stopt );
//    float ett;

    cudaError_t err;


	//START_TIME_T;
	thrust::counting_iterator<int> idx(0);
	//	STOP_TIME_T;
	//printf("   uf relabel idx alloc %f\n", ett);
 //START_TIME_T;
	thrust::device_ptr<int> label(d_label);

//    thrust::device_vector<int> uniquelabels(w*h);
//    thrust::device_vector<int> labelcopy(label, label+w*h);
//    thrust::sort(labelcopy.begin(), labelcopy.end());
//    thrust::device_vector<int>::iterator end = thrust::unique_copy(labelcopy.begin(), labelcopy.end(), uniquelabels.begin());
//    thrust::host_vector<int> hlabels(uniquelabels.begin(), end);
//     printf("input unique sorted labels: ");
//     for (int i = 0; i < hlabels.size(); ++i) {
//     	printf("%d, ", hlabels[i]);
//     }
//     printf("\n");

	//	STOP_TIME_T;
	//printf("   uf relabel label alloc: %f\n", ett);
 //START_TIME_T;
	thrust::device_vector<int> roots(w*h);
	//	STOP_TIME_T;
	//printf("   uf relabel root alloc: %f\n", ett);
 //START_TIME_T;
	thrust::device_vector<int> newlabel(w*h);
	//STOP_TIME_T;
	//printf("   uf relabel newlabel alloc: %f\n", ett);
 
	// get loc of roots
	//TART_TIME_T;

	// in debug mode, this has problem...  - zeros all the way through...
	thrust::transform(label, label+w*h, idx, roots.begin(), thrust::equal_to<int>());
	//	STOP_TIME_T;
	//printf("   uf relabel find roots: %f\n", ett);

//    int rootcount = thrust::count_if(roots.begin(), roots.end(), equalsK<int>(1));
//    printf("root count = %d\n", rootcount);

 //START_TIME_T;
	thrust::inclusive_scan(roots.begin(), roots.end(), newlabel.begin());
	//	STOP_TIME_T;
	//printf("   uf relabel scan: %f\n", ett);
 //START_TIME_T;

//    thrust::device_vector<int>::iterator end0 = thrust::unique_copy(newlabel.begin(), newlabel.end(), uniquelabels.begin());
//    thrust::host_vector<int> hnewlabels(uniquelabels.begin(), end0);
//     printf("input labels: ");
//     for (int i = 0; i < hnewlabels.size(); ++i) {
//     	printf("%d, ", hnewlabels[i]);
//     }
//     printf("\n");
//    int count = thrust::distance(uniquelabels.begin(), end0);
//    printf("label sorted and count = %d\n", count);
//    hnewlabels.clear();


	int count = newlabel[w*h-1];
//	printf("new labels direct access count = %d \n", count);
//	thrust::host_vector<int> countvec(newlabel.end()-1, newlabel.end());
//	printf("new label copied to host = %d\n", countvec[0]);

		//STOP_TIME_T;
	//printf("   uf relabel get count: %f\n", ett);
 //START_TIME_T;
	thrust::transform_if(label, label+w*h, newlabel.begin(), roots.begin(), 
		 label,
		 thrust::project2nd<int, int>(),
	 	 greaterThanK<int>(0));
	//STOP_TIME_T;
	//printf("   uf relabel reset root: %f\n", ett);
 // now flatten

	dim3 block (UF_BLOCK_SIZE_X, UF_BLOCK_SIZE_Y);
    	dim3 grid ((w+UF_BLOCK_SIZE_X-1)/UF_BLOCK_SIZE_X,
               (h+UF_BLOCK_SIZE_Y-1)/UF_BLOCK_SIZE_Y);
 
	//START_TIME_T;
    relabel_flatten <<<grid, block, 0, stream>>>(d_label,
		thrust::raw_pointer_cast(roots.data()), w, h, bgval);
	//STOP_TIME_T;
	//printf("   uf relabel flatten: %f\n", ett);
 
	roots.clear();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("endERROR: %s\n", cudaGetErrorString(err));
        return -1;
    }

//    thrust::device_vector<int>::iterator end3 = thrust::unique_copy(newlabel.begin(), newlabel.end(), uniquelabels.begin());
//    thrust::host_vector<int> hlabels3(uniquelabels.begin(), end3);
//    printf("unique new labels: ");
//    for (int i = 0; i < hlabels3.size(); ++i) {
//    	printf("%d, ", hlabels3[i]);
//    }
//    printf("\n");
//    count = thrust::distance(uniquelabels.begin(), end3);
//    printf("uniquelabels new count = %d\n", count);

//    thrust::device_vector<int> labelcopy2(label, label+w*h);
//    thrust::sort(labelcopy2.begin(), labelcopy2.end());
//    thrust::device_vector<int>::iterator end2 = thrust::unique_copy(labelcopy2.begin(), labelcopy2.end(), uniquelabels.begin());
//    thrust::host_vector<int> hlabels2(uniquelabels.begin(), end);
//     printf("unique sorted labels: ");
//     for (int i = 0; i < hlabels2.size(); ++i) {
//     	printf("%d, ", hlabels2[i]);
//     }
//     printf("\n");
//    count = thrust::distance(uniquelabels.begin(), end2);
//    printf("label sorted and count = %d\n", count);

//    uniquelabels.clear();
//    hlabels.clear();
//    labelcopy.clear();
//    hlabels2.clear();
//    labelcopy2.clear();
//    hlabels3.clear();

    newlabel.clear();


    return count;	
}
template<typename TT, typename T>
struct apply_threshold
{
	const T mn, mx, bg;
	T *l;

  __host__ __device__
  apply_threshold(T *label, T bgval, T minSize, T maxSize) :
	 l(label), bg(bgval), mn(minSize), mx(maxSize) {}

  __host__ __device__
  void operator()(TT x)
  {
	T area = thrust::get<1>(x);
	T label = thrust::get<0>(x);
	if (label == bg) // ignore this label.
		return;
	if (area < mn || area >= mx) {
		l[label] = bg;
	}
  }
};
template<typename TT, typename T>
struct check_threshold
{
	const T mn, mx, bg;

  __host__ __device__
  check_threshold(T bgval, T minSize, T maxSize) :
	 bg(bgval), mn(minSize), mx(maxSize) {}

  __host__ __device__
  bool operator()(TT x)
  {
	T area = thrust::get<1>(x);
	T label = thrust::get<0>(x);
	return (label != bg && area >= mn && area < mx);
  }
};


template<typename TT, typename T>
struct is_root
{
	__host__ __device__
	T operator()(TT x) {
		return (thrust::get<0>(x) == thrust::get<1>(x)) ? (T)1 : (T)0;
	}
};

__global__ void area_flatten (int* label, int w, int h, int bgval) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int global_index = x+y*w;

    bool in_limits = x < w && y < h;
    int target = label[global_index];
	

    if (in_limits) {
	if (target != bgval) label[global_index] = label[target];
    }
};

template <typename T>
struct notEqualToK
{
	T t;

	notEqualToK(T thresh) : t(thresh) {}

	__host__ __device__
	bool operator()(T x) {
		return x != t;
	}

};


// area thresholding.
int areaThreshold(int w, int h, int* d_label, int bgval, int minSize, int maxSize, cudaStream_t stream) {
	// do this by:  
	///  segmented reduce (global op)
	///  sort (global)
	///  then segmented reduce again (global)
	///  then selectively update the roots (using the label to area map) (label wise, global memory access)
	///  then do a flattening to propagate changes.
	// the first three steps compute area per label.  we do it as 2 reduces because sort first would cost about 42 ms without reduce.  doing it this way costs around 45 ms total, including teh last reduce.
//    cudaEvent_t startt,stopt;
//    cudaEventCreate( &startt );
//    cudaEventCreate( &stopt );
//    float ett;

    cudaError_t err;

	
// 	START_TIME_T;
	thrust::counting_iterator<int> idx(0);
	thrust::device_ptr<int> label(d_label);
	thrust::device_vector<int> ol(w*h);
//		STOP_TIME_T;
//	printf("   uf area label alloc: %f\n", ett);
 

//	STOP_TIME_T;
//	printf("   uf area test alloc 1: %f\n", ett);

	// pre sort the id based on label
//	START_TIME_T;
	thrust::device_vector<int>::iterator end = thrust::copy_if(label,
			label+w*h,
			ol.begin(),
			notEqualToK<int>(bgval));
	int fgcount = thrust::distance(ol.begin(), end);

	// testing
//		START_TIME_T;
	thrust::device_vector<int> area(fgcount, 1);
	thrust::pair<thrust::device_vector<int>::iterator,
	 	thrust::device_vector<int>::iterator> newend, newend2;
	typedef thrust::tuple<int, int> XY;
	thrust::device_vector<int> nl(fgcount);
	thrust::device_vector<int> newarea(fgcount, 0);

//		STOP_TIME_T;
//	printf("   uf area test alloc 1: %f\n", ett);

	// first do segmented reduction. this is obviously faster when 
	// there is more background and larger objects
//	START_TIME_T;
	newend = thrust::reduce_by_key(ol.begin(), ol.begin()+fgcount, area.begin(), nl.begin(), newarea.begin());
//		STOP_TIME_T;
//	printf("   uf area test reduce 1: %f\n", ett);

	area.clear();
	ol.clear();

	// then we sort by key.  faster when fewer, larger objects
//	START_TIME_T;
	thrust::sort_by_key(nl.begin(), newend.first, newarea.begin());
//		STOP_TIME_T;
//	printf("   uf area test sort 1: %f\n", ett);

//	START_TIME_T;
	thrust::device_vector<int> nna(newend.second - newarea.begin(), 0);
	thrust::device_vector<int> nnl(newend.first - nl.begin());
//		STOP_TIME_T;
//	printf("   uf area test alloc 2: %f\n", ett);

	// then we reduce again. to get the areas to label mapping
//	START_TIME_T;
	newend2 = thrust::reduce_by_key(nl.begin(), newend.first, newarea.begin(), nnl.begin(), nna.begin());
//	STOP_TIME_T;
//	printf("   uf area test reduce 2: %f\n", ett);

	newarea.clear();
	nl.clear();

	// now iterate over all the labels that are left, and update the roots
//	START_TIME_T;
	typedef typename thrust::tuple<int, int> LabelCompareType;
	thrust::for_each(thrust::make_zip_iterator(
			thrust::make_tuple(nnl.begin(), nna.begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple(newend2.first, newend2.second)),
		apply_threshold<LabelCompareType, int>(d_label, bgval, minSize, maxSize));
//	STOP_TIME_T;
//	printf("   uf area test update roots: %f\n", ett);
	int j = thrust::count_if(thrust::make_zip_iterator(
			thrust::make_tuple(nnl.begin(), nna.begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple(newend2.first, newend2.second)),
		check_threshold<LabelCompareType, int>(bgval, minSize, maxSize));


	nna.clear();
	nnl.clear();

	// then propagate the roots.
	dim3 block (UF_BLOCK_SIZE_X, UF_BLOCK_SIZE_Y);
    	dim3 grid ((w+UF_BLOCK_SIZE_X-1)/UF_BLOCK_SIZE_X,
               (h+UF_BLOCK_SIZE_Y-1)/UF_BLOCK_SIZE_Y);

//	START_TIME_T;
    area_flatten <<<grid, block, 0, stream>>>(d_label, w, h, bgval);
//	STOP_TIME_T;
//	printf("   uf area flatten: %f\n", ett);

//	printf(" inside cu areathreshold: compcount = %d\n", j);



    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("endERROR: %s\n", cudaGetErrorString(err));
        return -1;
    }

	return j;
// original thought was to sort, then scan to get area, then scan to get max area for each label, then reset any element that has lower area.  this did not work right.

}

// area thresholding.
int areaThreshold2(int w, int h, int* d_label, int bgval, int minSize, int maxSize, cudaStream_t stream) {
	// do this by:
	///  segmented reduce (global op)
	///  sort (global)
	///  then segmented reduce again (global)
	///  then selectively update the roots (using the label to area map) (label wise, global memory access)
	///  then do a flattening to propagate changes.
	// the first three steps compute area per label.  we do it as 2 reduces because sort first would cost about 42 ms without reduce.  doing it this way costs around 45 ms total, including teh last reduce.
//    cudaEvent_t startt,stopt;
//    cudaEventCreate( &startt );
//    cudaEventCreate( &stopt );
//    float ett;

    cudaError_t err;


// 	START_TIME_T;
	thrust::counting_iterator<int> idx(0);
	thrust::device_ptr<int> label(d_label);
//		STOP_TIME_T;
//	printf("   uf area label alloc: %f\n", ett);

	thrust::device_vector<int> ol(w*h);

//	STOP_TIME_T;
//	printf("   uf area test alloc 1: %f\n", ett);

	// pre sort the id based on label
//	START_TIME_T;
	thrust::device_vector<int>::iterator end = thrust::copy_if(label,
			label+w*h,
			ol.begin(),
			notEqualToK<int>(bgval));
	int fgcount = thrust::distance(ol.begin(), end);

	// then we sort by key.  faster when fewer, larger objects
//	START_TIME_T;
	thrust::sort(ol.begin(), end);
//		STOP_TIME_T;
//	printf("   uf area test sort 1: %f\n", ett);


	// testing
//		START_TIME_T;
	thrust::device_vector<int> area(fgcount, 1);
	thrust::pair<thrust::device_vector<int>::iterator,
	 	thrust::device_vector<int>::iterator> newend, newend2;
	typedef thrust::tuple<int, int> XY;
	thrust::device_vector<int> nl(fgcount);
	thrust::device_vector<int> newarea(fgcount, 0);

//		STOP_TIME_T;
//	printf("   uf area test alloc 1: %f\n", ett);

	// first do segmented reduction. this is obviously faster when
	// there is more background and larger objects
//	START_TIME_T;
	newend = thrust::reduce_by_key(ol.begin(), end, area.begin(), nl.begin(), newarea.begin());
//		STOP_TIME_T;
//	printf("   uf area test reduce 1: %f\n", ett);

	area.clear();
	ol.clear();



	// now iterate over all the labels that are left, and update the roots
//	START_TIME_T;
	typedef typename thrust::tuple<int, int> LabelCompareType;
	thrust::for_each(thrust::make_zip_iterator(
			thrust::make_tuple(nl.begin(), newarea.begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple(newend.first, newend.second)),
		apply_threshold<LabelCompareType, int>(d_label, bgval, minSize, maxSize));
//	STOP_TIME_T;
//	printf("   uf area test update roots: %f\n", ett);
	int j = thrust::count_if(thrust::make_zip_iterator(
			thrust::make_tuple(nl.begin(), newarea.begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple(newend.first, newend.second)),
		check_threshold<LabelCompareType, int>(bgval, minSize, maxSize));


	newarea.clear();
	nl.clear();

	// then propagate the roots.
	dim3 block (UF_BLOCK_SIZE_X, UF_BLOCK_SIZE_Y);
    	dim3 grid ((w+UF_BLOCK_SIZE_X-1)/UF_BLOCK_SIZE_X,
               (h+UF_BLOCK_SIZE_Y-1)/UF_BLOCK_SIZE_Y);
 
//	START_TIME_T;
        area_flatten <<<grid, block, 0, stream>>>(d_label, w, h, bgval);
//	STOP_TIME_T;
//	printf("   uf area flatten: %f\n", ett);
 

//	printf(" inside cu areathreshold: compcount = %d\n", j);



    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("endERROR: %s\n", cudaGetErrorString(err));
        return -1;
    }

	return j;
// original thought was to sort, then scan to get area, then scan to get max area for each label, then reset any element that has lower area.  this did not work right.

}



template <typename TI, typename TO, typename T>
struct computeXY : public thrust::unary_function<TI, TO>
{
	T w;

	computeXY(T width) : w(width) {}

	__host__ __device__
	TO operator()(TI li) {
		return thrust::make_tuple<T, T, T>(thrust::get<0>(li),
				thrust::get<1>(li) % w, thrust::get<1>(li) / w);
	}

};
template <typename T, typename TO>
struct computeXY2 : public thrust::unary_function<T, TO>
{
	T w;

	computeXY2(T width) : w(width) {}

	__host__ __device__
	TO operator()(T id) {
		return thrust::make_tuple<T, T>(id % w, id / w);
	}

};


template <typename T, typename TT>
struct computeBounds : public thrust::binary_function<TT, TT, TT>
{
	thrust::maximum<T> mx;
	thrust::minimum<T> mn;

	__host__ __device__
	TT operator()(TT a, TT b) {
		return thrust::make_tuple<T, T, T, T>(
			mn(thrust::get<0>(a), thrust::get<0>(b)),
			mx(thrust::get<1>(a), thrust::get<1>(b)),
			mn(thrust::get<2>(a), thrust::get<2>(b)),
			mx(thrust::get<3>(a), thrust::get<3>(b)));
	}

};


// bounding box finding.
// output is a n x 5 array, first n is label, next n is minx, next n maxx, and the miny, maxy.  this is allocated here.
int* boundingBox(const int w, const int h, int* d_label, int bgval, int &compcount, cudaStream_t stream) {
	// do this by:
	///  segmented reduce (global op)
	///  sort (global)
	///  then segmented reduce again (global)
	// the first three steps compute bounding box per label.  we do it as 2 reduces because sort first would cost about 42 ms without reduce.  doing it this way costs around 45 ms total, including teh last reduce.
//    cudaEvent_t startt,stopt;
//    cudaEventCreate( &startt );
//    cudaEventCreate( &stopt );
//    float ett;

    cudaError_t err;


// 	START_TIME_T;
	thrust::counting_iterator<int> idx(0);
	thrust::device_ptr<int> lab(d_label);

	thrust::device_vector<int> label(w*h);
	thrust::device_vector<int> x(w*h);
	thrust::device_vector<int> y(w*h);
	typedef thrust::tuple<thrust::device_ptr<int>,
		thrust::device_ptr<int>,
		thrust::device_ptr<int> > IteratorTuple3;
	typedef thrust::zip_iterator<IteratorTuple3> ZipIterator3;
	typedef thrust::tuple<int, int> LI;
	typedef thrust::tuple<int, int, int> LXY;

	// precompute x and y
//	START_TIME_T;
//	thrust::transform(idx, idx+w*h, thrust::make_zip_iterator(
//			thrust::make_tuple(x.begin(), y.begin())), computeXY<int, XY>(w));
	// remove background
	ZipIterator3 start = thrust::make_zip_iterator(thrust::make_tuple(label.data(), x.data(), y.data()));
	ZipIterator3 end = thrust::copy_if(thrust::make_transform_iterator(
				thrust::make_zip_iterator(thrust::make_tuple(lab, idx)), computeXY<LI, LXY, int>(w)),
			thrust::make_transform_iterator(
				thrust::make_zip_iterator(thrust::make_tuple(lab+w*h, idx+w*h)), computeXY<LI, LXY, int>(w)),
			lab,
			start,
			notEqualToK<int>(bgval));

	int non_bg_count = thrust::distance(start, end);
	thrust::device_vector<int> nl(non_bg_count);
	thrust::device_vector<int> minx(non_bg_count, std::numeric_limits<int>::max());
	thrust::device_vector<int> maxx(non_bg_count, std::numeric_limits<int>::min());
	thrust::device_vector<int> miny(non_bg_count, std::numeric_limits<int>::max());
	thrust::device_vector<int> maxy(non_bg_count, std::numeric_limits<int>::min());
// have to set the iterator to use device ptr.  else type mismatch later.
	typedef thrust::tuple<thrust::device_ptr<int>,
		thrust::device_ptr<int>,
		thrust::device_ptr<int>,
		thrust::device_ptr<int> > IteratorTuple;
	typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
	typedef thrust::tuple<int, int, int, int> MinMaxXY;
	thrust::pair<thrust::device_vector<int>::iterator,
	 	ZipIterator> newend, newend2;
//	STOP_TIME_T;
//	printf("   uf area test alloc 1: %f\n", ett);


	// have to set the iterator to use device ptr.  else type mismatch later.
	ZipIterator minmax = thrust::make_zip_iterator(thrust::make_tuple(
			minx.data(), maxx.data(), miny.data(), maxy.data()));
//	STOP_TIME_T;
//	printf("   init: %f\n", ett);


	// first do segmented reduction. this is obviously faster when
	// there is more background and larger objects
//	START_TIME_T;
	newend = thrust::reduce_by_key(label.begin(), label.begin() + non_bg_count,
			thrust::make_zip_iterator(thrust::make_tuple(
					x.begin(), x.begin(), y.begin(), y.begin())),
			nl.begin(),
			minmax,
			thrust::equal_to<int>(), computeBounds<int, MinMaxXY>());
//		STOP_TIME_T;
//	printf("   uf area test reduce 1: %f\n", ett);

	label.clear();
	x.clear();
	y.clear();

	// then we sort by key.  faster when fewer, larger objects
//	START_TIME_T;
	thrust::sort_by_key(nl.begin(), newend.first,
			minmax);
//	STOP_TIME_T;
//	printf("   uf area test sort 1: %f\n", ett);

//	START_TIME_T;
	int tempcount = thrust::distance(nl.begin(), newend.first);
	int *tbox;
	cudaMalloc(&tbox, tempcount * 5 * sizeof(int));
	thrust::device_ptr<int> nnl(tbox);
	thrust::device_ptr<int> tmnx(tbox + tempcount);
	thrust::device_ptr<int> tmxx(tbox + 2*tempcount);
	thrust::device_ptr<int> tmny(tbox + 3*tempcount);
	thrust::device_ptr<int> tmxy(tbox + 4*tempcount);
	ZipIterator minmax2 = thrust::make_zip_iterator(thrust::make_tuple(
			tmnx, tmxx, tmny, tmxy));
//	STOP_TIME_T;
//	printf("   uf area test alloc 2: %f\n", ett);

	// then we reduce again. to get the areas to label mapping
//	START_TIME_T;
	newend2 = thrust::reduce_by_key(nl.begin(), newend.first,
			minmax,
			nnl,
			minmax2,
			thrust::equal_to<int>(), computeBounds<int, MinMaxXY>());
//	STOP_TIME_T;
//	printf("   uf area test reduce 2: %f\n", ett);

	minx.clear();
	maxx.clear();
	miny.clear();
	maxy.clear();
	nl.clear();

	compcount = thrust::distance(minmax2, newend2.second);
//	printf(" inside cu: compcount = %d\n", compcount);

	int *bbox;
	cudaMalloc(&bbox, compcount * 5 * sizeof(int));
	cudaMemcpy(bbox, tbox, compcount * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(bbox+compcount, tbox+tempcount, compcount * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(bbox+2*compcount, tbox+2*tempcount, compcount * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(bbox+3*compcount, tbox+3*tempcount, compcount * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(bbox+4*compcount, tbox+4*tempcount, compcount * sizeof(int), cudaMemcpyDeviceToDevice);

	cudaFree(tbox);


    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("endERROR: %s\n", cudaGetErrorString(err));
        compcount = -1;
	return NULL;
    }

	return bbox;

}
// bounding box finding.
int* boundingBox2(const int w, const int h, int* d_label, int bgval, int &compcount, cudaStream_t stream) {
	// do this by:
	///  segmented reduce (global op)
	///  sort (global)
	///  then segmented reduce again (global)
	// the first three steps compute bounding box per label.  we do it as 2 reduces because sort first would cost about 42 ms without reduce.  doing it this way costs around 45 ms total, including teh last reduce.
//    cudaEvent_t startt,stopt;
//    cudaEventCreate( &startt );
//    cudaEventCreate( &stopt );
//    float ett;

    cudaError_t err;


// 	START_TIME_T;
	thrust::counting_iterator<int> idx(0);
	thrust::device_ptr<int> lab(d_label);

	thrust::device_vector<int> label(w*h);
	thrust::device_vector<int> sortedIdx(w*h);
	typedef thrust::tuple<int, int> XY;
	typedef thrust::tuple<thrust::device_ptr<int>,
		thrust::device_ptr<int> > IteratorTuple2;
	typedef thrust::zip_iterator<IteratorTuple2> ZipIterator2;

//	STOP_TIME_T;
//	printf("   uf area test alloc 1: %f\n", ett);

	// pre sort the id based on label
//	START_TIME_T;
	ZipIterator2 start = thrust::make_zip_iterator(thrust::make_tuple(label.data(), sortedIdx.data()));
	ZipIterator2 end = thrust::copy_if(thrust::make_zip_iterator(thrust::make_tuple(lab, idx)),
			thrust::make_zip_iterator(thrust::make_tuple(lab+w*h, idx+w*h)),
			lab,
			start,
			notEqualToK<int>(bgval));
	int fgcount = thrust::distance(start, end);
	thrust::sort_by_key(label.begin(), label.begin() + fgcount, sortedIdx.begin());
//	STOP_TIME_T;
//	printf("   sort: %f\n", ett);

	// precompute x and y
//	START_TIME_T;
	thrust::device_vector<int> x(fgcount);
	thrust::device_vector<int> y(fgcount);
	thrust::transform(sortedIdx.begin(), sortedIdx.begin() + fgcount, thrust::make_zip_iterator(
			thrust::make_tuple(x.begin(), y.begin())), computeXY2<int, XY>(w));
//	STOP_TIME_T;
//	printf("   init: %f\n", ett);


//	START_TIME_T;
        int *tbox;
        cudaMalloc(&tbox, fgcount * 5 * sizeof(int));
        thrust::device_ptr<int> nnl(tbox);
        thrust::device_ptr<int> tmnx(tbox + fgcount);
        thrust::device_ptr<int> tmxx(tbox + 2*fgcount);
        thrust::device_ptr<int> tmny(tbox + 3*fgcount);
        thrust::device_ptr<int> tmxy(tbox + 4*fgcount);

	typedef thrust::tuple<thrust::device_ptr<int>,
		thrust::device_ptr<int>,
		thrust::device_ptr<int>,
		thrust::device_ptr<int> > IteratorTuple;
	typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
	ZipIterator minmax = thrust::make_zip_iterator(thrust::make_tuple(
			tmnx, tmxx, tmny, tmxy));
	typedef thrust::tuple<int, int, int, int> MinMaxXY;
	thrust::pair<thrust::device_vector<int>::iterator,
	 	ZipIterator> newend;
//	STOP_TIME_T;
//	printf("   uf area test alloc 2: %f\n", ett);

	// now do reduce, all 4 at the same time?
	// then we reduce again. to get the areas to label mapping
//	START_TIME_T;
	newend = thrust::reduce_by_key(label.begin(), label.begin()+fgcount,
			thrust::make_zip_iterator(thrust::make_tuple(
				x.begin(), x.begin(), y.begin(), y.begin())),
			nnl,
			minmax,
			thrust::equal_to<int>(), computeBounds<int, MinMaxXY>());
//	STOP_TIME_T;
//	printf("   uf area test reduce 2: %f\n", ett);


	label.clear();
	x.clear();
	y.clear();
    sortedIdx.clear();

	compcount = thrust::distance(minmax, newend.second);
//	printf(" inside cu: compcount = %d\n", compcount);

	int *bbox;
	cudaMalloc(&bbox, compcount * 5 * sizeof(int));
	cudaMemcpy(bbox, tbox, compcount * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(bbox+compcount, tbox+fgcount, compcount * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(bbox+2*compcount, tbox+2*fgcount, compcount * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(bbox+3*compcount, tbox+3*fgcount, compcount * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(bbox+4*compcount, tbox+4*fgcount, compcount * sizeof(int), cudaMemcpyDeviceToDevice);

	cudaFree(tbox);
//printf("bbox adderss inside cu %p\n", bbox);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("endERROR: %s\n", cudaGetErrorString(err));
        compcount = -1;
        return NULL;
    }

	return bbox;

}


}}
