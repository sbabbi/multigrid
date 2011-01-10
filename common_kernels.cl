/*common_kernels.cl*/

/*#pragma OPENCL EXTENSION cl_khr_fp64: enable*/

/** Sum of two buffers of floats**/
__kernel void sum(	__global write_only float * result,
					__global read_only float * arg1,
					__global read_only float * arg2)
{
	size_t i = get_global_id(0);
	result[i]= arg1[i]+arg2[i];
}

/** Subtraction of two buffers of floats**/
__kernel void sub(	__global write_only float * result,
					__global read_only float * arg1,
					__global read_only float * arg2)
{
	size_t i = get_global_id(0);
	result[i]= arg1[i]-arg2[i];
}

/** Zero-initialization **/
__kernel void zero_memory(__global write_only float * result)
{
	size_t i = get_global_id(0);
	result[i] = 0;
}
