/*common_kernels.cl*/

#include "real.cl"

/** Zero-initialization **/
__kernel void zero_memory(__global write_only real * result)
{
	size_t i = get_global_id(0);
	result[i] = 0;
}

/** Parallel L2-Norm **/
__kernel void L2Norm(__global read_only real * input,
					 __global write_only real * output)
{
	int outPos = get_global_id(0);

	output[outPos] = input[outPos]*input[outPos];
}

/** Parallel LInf-Norm **/
__kernel void LInfNorm(__global read_only real * input,
					 __global write_only real * output,
					 int inputSize,
					 int chunks)
{
	int outPos = get_global_id(0);
	real res = 0.0;

	int base = outPos*chunks;
	int end = min(inputSize,base+chunks);
	for ( ;base < end;++base)
		res = max(res,fabs(input[base]));

	output[outPos] = res;
}

/** Parallel SumAll **/
__kernel void SumAll(__global read_only real * input,
					 __global write_only real * output,
					 int inputSize,
					 int chunks)
{
	int outPos = get_global_id(0);
	real res = 0.0;

	int base = outPos*chunks;
	int end = min(inputSize,base+chunks);
	for ( ;base < end;++base)
		res+= input[base];

	output[outPos] = res;
}

/** Parallel mult by constant **/
__kernel void Mult(__global read_only real * input,
				   __global write_only real * output,
					 float m)
{
	int outPos = get_global_id(0);

	output[outPos] = input[outPos]*m;
}
