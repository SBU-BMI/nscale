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
#include <thrust/adjacent_difference.h>
#include <thrust/scan.h>
#include <thrust/count.h>

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
    cudaEvent_t startt,stopt;
    cudaEventCreate( &startt );
    cudaEventCreate( &stopt );
    float ett;

    cudaError_t err;

    START_TIME_T;
    cudaChannelFormatDesc uchardesc = 
        cudaCreateChannelDesc<unsigned char>();
    cudaBindTexture2D(0, imgtex, img, uchardesc, w, h, w * sizeof(unsigned char));
    STOP_TIME_T;
	printf("   uf bind texture: %f\n", ett);

    dim3 block (UF_BLOCK_SIZE_X, UF_BLOCK_SIZE_Y);
    dim3 grid ((w+UF_BLOCK_SIZE_X-1)/UF_BLOCK_SIZE_X,
               (h+UF_BLOCK_SIZE_Y-1)/UF_BLOCK_SIZE_Y);
 
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("startERROR: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaThreadSetCacheConfig(cudaFuncCachePreferShared);

    START_TIME_T;
    UF_local <<<grid, block, 0, stream>>>(d_label, w, h, connectivity);
    STOP_TIME_T;
	printf("   uf_local: %f\n", ett);

    cudaThreadSetCacheConfig(cudaFuncCachePreferL1);

	START_TIME_T;
        UF_global <<<grid, block, 0, stream>>>(d_label, w, h, connectivity);
    STOP_TIME_T;
	printf("   uf_global: %f\n", ett);

	START_TIME_T;
        UF_final <<<grid, block, 0, stream>>>(d_label, w, h, bgval);
    STOP_TIME_T;
	printf("   uf_final: %f\n", ett);
 
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
    int target = label[global_index];

    if (in_limits) {
	if (target == bgval) label[global_index] = 0;
	else if (roots[global_index] == 0)
        	label[global_index] = label[target];
    }
}

// relabelling.  this relies on bg being smaller in value than all foreground labels.
int relabel(int w, int h, int* d_label, int bgval, cudaStream_t stream) {
	// do this by:  sort by value, find segment, scan, sort by index
/*	typedef typename thrust::tuple<int, int> LabelCompareType;

	thrust::counting_iterator<int> idx;
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
    cudaEvent_t startt,stopt;
    cudaEventCreate( &startt );
    cudaEventCreate( &stopt );
    float ett;

    cudaError_t err;


	START_TIME_T;
	thrust::counting_iterator<int> idx;
		STOP_TIME_T;
	printf("   uf relabel idx alloc %f\n", ett);
 START_TIME_T;
	thrust::device_ptr<int> label(d_label);
		STOP_TIME_T;
	printf("   uf relabel label alloc: %f\n", ett);
 START_TIME_T;
	thrust::device_vector<int> roots(w*h, 0);
		STOP_TIME_T;
	printf("   uf relabel root alloc: %f\n", ett);
 START_TIME_T;
	thrust::device_vector<int> newlabel(w*h, 0);
	STOP_TIME_T;
	printf("   uf relabel newlabel alloc: %f\n", ett);
 
	// get loc of roots
	START_TIME_T;
	thrust::transform(label, label+w*h, idx, roots.begin(), thrust::equal_to<int>());
		STOP_TIME_T;
	printf("   uf relabel find roots: %f\n", ett);
 START_TIME_T;
	thrust::inclusive_scan(roots.begin(), roots.end(), newlabel.begin());
		STOP_TIME_T;
	printf("   uf relabel scan: %f\n", ett);
 START_TIME_T;
	int count = newlabel[w*h-1];
		STOP_TIME_T;
	printf("   uf relabel get count: %f\n", ett);
 START_TIME_T;
	thrust::transform_if(label, label+w*h, newlabel.begin(), roots.begin(), 
		 label,
		 thrust::project2nd<int, int>(),
	 	 thrust::identity<int>());
	STOP_TIME_T;
	printf("   uf relabel reset root: %f\n", ett);
 // now flatten

	dim3 block (UF_BLOCK_SIZE_X, UF_BLOCK_SIZE_Y);
    	dim3 grid ((w+UF_BLOCK_SIZE_X-1)/UF_BLOCK_SIZE_X,
               (h+UF_BLOCK_SIZE_Y-1)/UF_BLOCK_SIZE_Y);
 
	START_TIME_T;
        relabel_flatten <<<grid, block, 0, stream>>>(d_label, 
		thrust::raw_pointer_cast(roots.data()), w, h, bgval);
	STOP_TIME_T;
	printf("   uf relabel flatten: %f\n", ett);
 
	roots.clear();
	newlabel.clear();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("endERROR: %s\n", cudaGetErrorString(err));
        return -1;
    }

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


// relabelling.  .
int areaThreshold(int w, int h, int* d_label, int bgval, int minSize, int maxSize, cudaStream_t stream) {
	// do this by:  
	///  segmented reduce (global op)
	///  sort (global)
	///  then segmented reduce again (global)
	///  then selectively update the roots (using the label to area map) (label wise, global memory access)
	///  then do a flattening to propagate changes.
	// the first three steps compute area per label.  we do it as 2 reduces because sort first would cost about 42 ms without reduce.  doing it this way costs around 45 ms total, including teh last reduce.
    cudaEvent_t startt,stopt;
    cudaEventCreate( &startt );
    cudaEventCreate( &stopt );
    float ett;

    cudaError_t err;

	
 	START_TIME_T;
	thrust::counting_iterator<int> idx;
	thrust::device_ptr<int> label(d_label);
		STOP_TIME_T;
	printf("   uf area label alloc: %f\n", ett);
 
	// testing
		START_TIME_T;
	thrust::device_vector<int> area(w*h, 1);
	thrust::device_vector<int> ol(label, label+w*h);
	thrust::device_vector<int> newarea(w*h, 0);
	thrust::device_vector<int> nl(w*h, 0);
	thrust::pair<thrust::device_vector<int>::iterator,
	 	thrust::device_vector<int>::iterator> newend, newend2;
		STOP_TIME_T;
	printf("   uf area test alloc 1: %f\n", ett);

	// first do segmented reduction. this is obviously faster when 
	// there is more background and larger objects
	START_TIME_T;
	newend = thrust::reduce_by_key(ol.begin(), ol.end(), area.begin(), nl.begin(), newarea.begin());
		STOP_TIME_T;
	printf("   uf area test reduce 1: %f\n", ett);

	area.clear();
	ol.clear();

	// then we sort by key.  faster when fewer, larger objects
	START_TIME_T;
	thrust::sort_by_key(nl.begin(), newend.first, newarea.begin());
		STOP_TIME_T;
	printf("   uf area test sort 1: %f\n", ett);

	START_TIME_T;
	thrust::device_vector<int> nna(newend.second - newarea.begin(), 0);
	thrust::device_vector<int> nnl(newend.first - nl.begin(), 0);
		STOP_TIME_T;
	printf("   uf area test alloc 2: %f\n", ett);

	// then we reduce again. to get the areas to label mapping
	START_TIME_T;
	newend2 = thrust::reduce_by_key(nl.begin(), newend.first, newarea.begin(), nnl.begin(), nna.begin());
	STOP_TIME_T;
	printf("   uf area test reduce 2: %f\n", ett);

	newarea.clear();
	nl.clear();

	// now iterate over all the labels that are left, and update the roots
	START_TIME_T;
	typedef typename thrust::tuple<int, int> LabelCompareType;
	thrust::for_each(thrust::make_zip_iterator(
		thrust::make_tuple(
			nnl.begin(), nna.begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple(newend2.first, newend2.second)),
		apply_threshold<LabelCompareType, int>(d_label, bgval, minSize, maxSize));
	STOP_TIME_T;
	printf("   uf area test update roots: %f\n", ett);


	nna.clear();
	nnl.clear();

	// then propagate the roots.
	dim3 block (UF_BLOCK_SIZE_X, UF_BLOCK_SIZE_Y);
    	dim3 grid ((w+UF_BLOCK_SIZE_X-1)/UF_BLOCK_SIZE_X,
               (h+UF_BLOCK_SIZE_Y-1)/UF_BLOCK_SIZE_Y);
 
	START_TIME_T;
        area_flatten <<<grid, block, 0, stream>>>(d_label, w, h, bgval);
	STOP_TIME_T;
	printf("   uf area flatten: %f\n", ett);
 
	// get the count
	START_TIME_T;
	int j =	thrust::count_if(thrust::make_zip_iterator(
		thrust::make_tuple(
			label, idx)),
		thrust::make_zip_iterator(
			thrust::make_tuple(label+w*h, idx+w*h)),
		is_root<LabelCompareType, int>());
	STOP_TIME_T;
	printf("   uf area count updated roots: %f\n", ett);

		


    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("endERROR: %s\n", cudaGetErrorString(err));
        return -1;
    }

	return j;
// original thought was to sort, then scan to get area, then scan to get max area for each label, then reset any element that has lower area.  this did not work right.

}


}}
