/*common_kernels.cl*/

typedef struct Buffer2D
{
	int dimx,dimy;
	__global double * data;
};

/** Sum of two buffers of doubles**/
__kernel void sum(	__global write_only double * result,
					__global read_only double * arg1,
					__global read_only double * arg2)
{
	size_t i = get_global_id(0);
	result[i]= arg1[i]+arg2[i];
}

/** Subtraction of two buffers of doubles**/
__kernel void sub(	__global write_only double * result,
					__global read_only double * arg1,
					__global read_only double * arg2)
{
	size_t i = get_global_id(0);
	result[i]= arg1[i]-arg2[i];
}

/** Zero-initialization **/
__kernel void zero_memory(__global write_only double * result)
{
	size_t i = get_global_id(0);
	result[i] = 0;
}
