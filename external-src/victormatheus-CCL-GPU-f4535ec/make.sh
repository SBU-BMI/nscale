/usr/local/cuda/bin/nvcc -O3 -DTHREADED -arch sm_12 ccl_test.cu -o bin/ccl_test
#/usr/local/cuda/bin/nvcc -O3 -Xptxas -v -arch sm_13 --maxrregcount=20 distance_transform2.cu -o bin/distance_transform2
