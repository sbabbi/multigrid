
#include "real.cl"

/* Border management **/
typedef struct tagCell {
	real2 _normals; //MUST BE L2 normalized
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
	if (getCellType(c) & ( CELL_DIRICHLET | CELL_NEUMANN)) return 1;
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
	return func[base] - (src[base +1] + src[base -1] + src[base + xsize] + src[base -xsize] - 4* src[base]);
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
							real w,
							int2 size,
							int odd)
{
	int base = 2*get_global_id(0) + (odd+get_global_id(1))%2 + size.x*get_global_id(1);

	if (2*get_global_id(0) + (odd+get_global_id(1))%2 >= size.x) return;

	do_rbgauss(domain,dest,func,w,base,size.x);
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
			dest[base] = func[base]-src[base];
			break;
		case CELL_NEUMANN:
			break;
	}
}

/*** reduction_kernel
	* This kernel computes a Full-weighting reduction of the function passed in src

	* domain as in iteration_kernel. domain size MUST be equal to "size"
	* dest is the output 2d-array, whose size MUST be equal to (int2)(get_global_size(0),get_global_size(1))
	* src is the input function
	* size is the size of the INPUT function

	* Notice that the size of dest MUST be equal to (int2)(get_global_size(0),get_global_size(1))
	* Also, the destination size MUST be half +1 of the src size
*/
__kernel void reduction_kernel(global read_only Cell * domain,
								global write_only real * dest,
								global read_only real * src,
								int2 size)
{
	int destbase = get_global_id(0)+get_global_size(0)*get_global_id(1);
	int sourcebase = get_global_id(0)*2 + size.x*(get_global_id(1)*2);

	if (isBorder(domain+sourcebase) || isBorder(domain+sourcebase+1) ||
		isBorder(domain+sourcebase+size.x) || isBorder(domain+sourcebase+size.x+1))
	{
		dest[destbase] = src[sourcebase];
	}
	else
		dest[destbase] =
			1.0/16.0 * ( src[sourcebase-size.x-1] + src[sourcebase-size.x+1] + src[sourcebase+size.x-1] + src[sourcebase+size.x+1]) +
			1.0/8.0 * ( src[sourcebase-size.x] + src[sourcebase+1] + src[sourcebase-1] + src[sourcebase+size.x]) +
			1.0/4.0 * (src[sourcebase]);
}

/*** residual_correct_kernel
	* This kernel prolongate the error err, and use the prolongation of err to correct the solution in input

	* domain,dest,input are bidimension arrays of size (int2)(get_global_size(0),get_global_size(1))
	* err is a bidimensional array

	* Notice that err size must be HALF +1 of dest size.
***/
__kernel void residual_correct_kernel(global read_only Cell * domain,
										global write_only real * dest,
										global read_only real * input,
										global read_only real * err)
{
	int destbase = get_global_id(0)+get_global_size(0)*get_global_id(1);
	int2 pos = (int2)(get_global_id(0)/2,get_global_id(1)/2);
	int2 size = (int2)(get_global_size(0)/2 +1,get_global_size(1)/2+1);
	int sourcebase = pos.x +pos.y*size.x;

	real val;
	if (isBorder(domain+destbase))
		val = err[sourcebase];
	else
	{
		real u = 0.5*(get_global_id(0)%2);
		real w = 0.5*(get_global_id(1)%2);

		real4 n = (real4)( (1.0-u)*(1.0-w), (1.0-w)*u, (1.0-u)*w,w*u);

		val = dot ((real4)( err[sourcebase],
						err[sourcebase+1],
						err[sourcebase+size.x],
						err[sourcebase+size.x+1]),n);
	}

	dest [destbase] = input[destbase] + val*4;
}

/***TODO: Better interpolation ***/
__kernel void prolongation_kernel(global read_only Cell * domain,
									global write_only real * dest,
									global read_only real * input)
{
	int destbase = get_global_id(0)+get_global_size(0)*get_global_id(1);
	int2 pos = (int2)(get_global_id(0)/2,get_global_id(1)/2);
	int2 size = (int2)(get_global_size(0)/2 +1,get_global_size(1)/2+1);

	real val;
	if (isBorder(domain+destbase))
		val = input[ pos.x + size.x * pos.y ];
	else
	{
		if (get_global_id(0)%2 == 0 && get_global_id(1) % 2 == 0)
			val = input[pos.x + size.x*pos.y];
		else if (get_global_id(0) % 2 == 0 && get_global_id(1) % 2 == 1)
			val = 0.5*(input[pos.x + size.x*pos.y]+input[pos.x + size.x*(pos.y+1)]);
		else if (get_global_id(0) % 2 == 1 && get_global_id(1) % 2 == 0)
			val = 0.5*(input[pos.x + size.x*pos.y]+input[pos.x +1 + size.x*pos.y]);
		else
			val = 0.25*(input[pos.x + size.x*pos.y] + input[pos.x +size.x*(pos.y+1)] +
				input[pos.x+1+size.x*(pos.y+1)]+input[pos.x+1+size.x*pos.y]);
	}

	dest[destbase] = val;
}

__kernel void zero_out(global read_only Cell * domain,
						global real * input)
{
	int base = get_global_id(0)+get_global_size(0)*get_global_id(1);
	if (getCellType(domain+base) == CELL_OUTSIDE)
		input[base] = 0;
}
