
#include "real.cl"

/* Border management **/
typedef struct tagCell {
	real2 _normals; //MUST BE Manhattan-Normalized
}Cell;

#define CELL_INSIDE 0x1
#define CELL_OUTSIDE 0x2
#define CELL_DIRICHLET 0x4
#define CELL_NEUMANN 0x8

inline int getCellType(const __global read_only Cell * c)
{
	if ( c->_normals.x == 0.0 && c->_normals.y == 0.0) return CELL_INSIDE;
	if ( isnan(c->_normals.x) && isnan(c->_normals.y)) return CELL_OUTSIDE;
	if ( isnan(c->_normals.x)) return CELL_DIRICHLET;
	return CELL_NEUMANN;
}

inline int isBorder(const __global read_only Cell * c)
{
	int ans = getCellType(c);
	if (ans == CELL_DIRICHLET || ans == CELL_NEUMANN) return 1;
	return 0;
}

/*** Dumped jacobi iteratation ***/
inline real jacobi_iteration(int base,
							 int xsize,
							 global read_only real * src,
							 global read_only real * func)
{
	return  0.25 * (
		src[base +1] +
		src[base -1] +
		src[base + xsize] +
		src[base - xsize] -
		func[base] );
}

/*** Residuals ***/
inline real residual(int base,
					 int xsize,
					 global read_only real * src,
					 global read_only real * func)
{
	return src[base +1] + src[base -1] + src[base + xsize] + src[base -xsize] - 4* src[base] - func[base];
}

/*** Do a red-black gauss seidel iteration ***/
void do_rbgauss(global read_only Cell * domain,
			global real * dest,
			global read_only real * func,
			real w,
			int base,
			int sizex)
{
	real val;
	switch (getCellType(domain+base) )
	{
		case CELL_INSIDE:
			val = jacobi_iteration(base,sizex,dest,func);
			dest[base] = val * w + (1.0-w)* dest[base];
			break;
		case CELL_OUTSIDE:
			break;
		case CELL_DIRICHLET:
			dest[base] = func[base];
			break;
		case CELL_NEUMANN:
			break;
	}
}

/*** iteration_kernel
	* This kernel computes an iteration of the red-black gauss seidel method.

	* domain is a pointer to a bidimensional array of Cell data, which represents the description of the domains and borders
	* dest is the bidimensional input and output array
	* func is the target function (on the borders too!)
	* w is the relaxation parameter

	* Notice that the size of domain,dest,src and func MUST be equal to (int2)(get_global_size(0),get_global_size(1))
*/
__kernel void iteration_kernel(global read_only Cell * domain,
							global real * dest,
							global read_only real * func,
							real w)
{
	int base = get_global_id(0) + get_global_size(0)*get_global_id(1);
	int sizex = get_global_size(0);

	if ( (get_global_id(0)+get_global_id(1)) % 2 == 1)
		do_rbgauss(domain,dest,func,w,base,sizex);

	barrier(CLK_GLOBAL_MEM_FENCE);

	if ( (get_global_id(0)+get_global_id(1)) % 2 == 0)
		do_rbgauss(domain,dest,func,w,base,sizex);
}

/*** residual_kernel
	* This kernel computes the residuals of the current solution

	* domain,dest,src,func,size as in iteration_kernel
*/
__kernel void residual_kernel(global read_only Cell * domain,
								global write_only real * dest,
								global read_only real * src,
								global read_only real * func)
{
	int base = get_global_id(0) + get_global_size(0)*get_global_id(1);
	int sizex = get_global_size(0);

	switch (getCellType(domain+base) )
	{
		case CELL_INSIDE:
			dest[base] = residual(base,sizex,src,func);
			break;
		case CELL_OUTSIDE:
			break;
		case CELL_DIRICHLET:
			dest[base] = src[base]-func[base];
			break;
		case CELL_NEUMANN:
			break;
	}
}

real four_point_stencil_reduction(global read_only Cell * domain,
						global read_only real * input,
						int base,
						int sizex)
{
	real4 in = (real4)(input[base],input[base+1],input[base+sizex],input[base+sizex+1]);
	real4 wg = (real4)(
		isBorder(domain+base),
		isBorder(domain+base+1),
		isBorder(domain+base+sizex),
		isBorder(domain+base+sizex+1));

	real den = wg.x+wg.y+wg.z+wg.w;

	if (den == 0) return 0.25*(in.x+in.y+in.z+in.w);
	return dot(wg,in)/den;
}

/*** reduction_kernel
	* This kernel computes a Full-weighting reduction of the function passed in src

	* domain as in iteration_kernel. domain size MUST be equal to "size"
	* dest is the output 2d-array, whose size MUST be equal to (int2)(get_global_size(0),get_global_size(1))
	* src is the input function
	* size is the size of the INPUT function

	* Notice that the size of dest MUST be equal to (int2)(get_global_size(0),get_global_size(1))
	* Also, the destination size MUST be half of the src size
*/
__kernel void reduction_kernel(global read_only Cell * domain,
								global write_only real * dest,
								global read_only real * src,
								int2 size)
{
	int destbase = get_global_id(0)+get_global_size(0)*get_global_id(1);
	int sourcebase = get_global_id(0)*2 + size.x*get_global_id(1)*2;

	dest[destbase] = four_point_stencil_reduction(domain,src,sourcebase,size.x);
}

/*** residual_correct_kernel
	* This kernel prolongate the error err, and use the prolongation of err to correct the solution in input

	* domain,dest,input are bidimension arrays of size (int2)(get_global_size(0),get_global_size(1))
	* err is a bidimensional array

	* Notice that err size must be HALF of dest size.
***/
__kernel void residual_correct_kernel(global read_only Cell * domain,
										global write_only real * dest,
										global read_only real * input,
										global read_only real * err)
{
	int destbase = get_global_id(0)+get_global_size(0)*get_global_id(1);

	real val = err[ (get_global_id(0)/2) + (get_global_size(0)/2) * (get_global_id(1)/2) ];

	dest [destbase] = input[destbase] - val*4;
}

/***TODO: Better interpolation ***/
__kernel void prolongation_kernel(global read_only real2 * border,
									global write_only real * dest,
									global read_only real * input)
{
	int destbase = get_global_id(0)+get_global_size(0)*get_global_id(1);

	real val = input[ (get_global_id(0)/2) + (get_global_size(0)/2) * (get_global_id(1)/2) ];

	dest [destbase] = val;
}
