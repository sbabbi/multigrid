
#ifdef USE_DOUBLE

#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define real double
#define real2 double2
#define real4 double4
#define real8 double8
#define real16 double16
#else
#define real float
#define real2 float2
#define real4 float4
#define real8 float8
#define real16 float16

#endif //USE_DOUBLE