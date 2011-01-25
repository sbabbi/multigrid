
#include "real.cl"

/* Border management **/
typedef struct tagCell {
	real4 _normals; //MUST BE L2 normalized. 4th parameter ignored
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
							 int xysize,
							 global read_only real * src,
							 global read_only real * func)
{
	return  (1.0/6.0) * (
		src[base +1] +
		src[base -1] +
		src[base + xsize] +
		src[base - xsize] +
		src[base + xysize] +
		src[base - xysize] -
		func[base] );
}

/*** Residuals ***/
inline real residual(int base,
					 int xsize,
					 int xysize,
					 global read_only real * src,
					 global read_only real * func)
{
	return func[base] - (src[base +1] + src[base -1] + src[base + xsize] + src[base -xsize] + src[base + xysize] + src[base -xysize] - 6* src[base]);
}

/*** Do a red-black gauss seidel iteration ***/
void do_rbgauss(global read_only Cell * domain,
			global real * dest,
			global read_only real * func,
			real w,
			int base,
			int sizex,
			int sizexy)
{
	real val;
	switch (getCellType(domain+base) )
	{
		case CELL_INSIDE:
			val = jacobi_iteration(base,sizex,sizexy,dest,func);
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
							int4 size,
							int odd1,
							int odd2)
{
	int base = 2*get_global_id(0) + (odd1+get_global_id(1))%2 + size.x*get_global_id(1) +
			size.y*size.x*(2*get_global_id(2)+odd2);

	if (2*get_global_id(0) + (odd1+get_global_id(1))%2 >= size.x ||
			2*get_global_id(2)+odd2 >= size.z) return;

	do_rbgauss(domain,dest,func,w,base,size.x,size.x*size.y);
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
	int base = get_global_id(0) + get_global_size(0)*get_global_id(1)+get_global_id(2)*get_global_size(0)*get_global_size(1);
	int sizex = get_global_size(0);
	int sizexy = get_global_size(0)*get_global_size(1);

	switch (getCellType(domain+base) )
	{
		case CELL_INSIDE:
			dest[base] = residual(base,sizex,sizexy,src,func);
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
								int4 size)
{
	int destbase = get_global_id(0)+get_global_size(0)*get_global_id(1)+get_global_size(0)*get_global_size(1)*get_global_id(2);
	int sourcebase = get_global_id(0)*2 + size.x*(get_global_id(1)*2) + size.x*size.y*(get_global_id(2)*2);

	if (isBorder(domain+sourcebase) || isBorder(domain+sourcebase+1) ||
		isBorder(domain+sourcebase+size.x) || isBorder(domain+sourcebase+size.x+1) ||
		isBorder(domain+sourcebase+size.x*size.y) || isBorder(domain+sourcebase+size.x*size.y+1) ||
		isBorder(domain+sourcebase+size.x*size.y+size.x) || isBorder(domain+sourcebase+size.x*size.y+1+size.x))
	{
		dest[destbase] = src[sourcebase];
	}
	else
		dest[destbase] =
			1.0/64.0 * ( src[sourcebase+size.x*size.y+size.x+1] +
						src[sourcebase+size.x*size.y+size.x-1] +
						src[sourcebase+size.x*size.y-size.x+1] +
						src[sourcebase+size.x*size.y-size.x-1] +
						src[sourcebase-size.x*size.y+size.x+1] +
						src[sourcebase-size.x*size.y+size.x-1] +
						src[sourcebase-size.x*size.y-size.x+1] +
						src[sourcebase-size.x*size.y-size.x-1]) +
			1.0/32.0 * (src[sourcebase+size.x+1] +
						src[sourcebase+size.x-1] +
						src[sourcebase-size.x+1] +
						src[sourcebase-size.x-1] +
						src[sourcebase-size.x*size.y+size.x] +
						src[sourcebase-size.x*size.y-size.x] +
						src[sourcebase-size.x*size.y+1] +
						src[sourcebase-size.x*size.y-1] +
						src[sourcebase+size.x*size.y+size.x] +
						src[sourcebase+size.x*size.y-size.x] +
						src[sourcebase+size.x*size.y+1] +
						src[sourcebase+size.x*size.y-1] ) +
			1.0/16.0 * ( src[sourcebase+size.x] +
						src[sourcebase-size.x] +
						src[sourcebase+1] +
						src[sourcebase-1] +
						src[sourcebase+size.x*size.y] +
						src[sourcebase-size.x*size.y]) +
			1.0/8.0 * (src[sourcebase]);
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
	int destbase = get_global_id(0)+get_global_size(0)*get_global_id(1)+get_global_size(0)*get_global_size(1)*get_global_id(2);
	int4 pos = (int4)(get_global_id(0)/2,get_global_id(1)/2,get_global_id(2)/2,1);
	int4 size = (int4)(get_global_size(0)/2 +1,get_global_size(1)/2+1,get_global_size(2)/2+1,1);
	int sourcebase = pos.x +pos.y*size.x + pos.z*size.x*size.y;

	real val;
	if (isBorder(domain+destbase))
		val = err[sourcebase];
	else
	{
		real u = 0.5*(get_global_id(0)%2);
		real w = 0.5*(get_global_id(1)%2);
		real v = 0.5*(get_global_id(2)%2);

		real4 n1 = (real4)( (1.0-u)*(1.0-w)*(1.0-v), (1.0-w)*u*(1.0-v), (1.0-u)*w*(1.0-v),w*u*(1.0-v));
		real4 n2 = (real4)( (1.0-u)*(1.0-w)*v, (1.0-w)*u*v, (1.0-u)*w*v,w*u*v);

		val = dot ((real4)( err[sourcebase],
						err[sourcebase+1],
						err[sourcebase+size.x],
						err[sourcebase+size.x+1]),n1) +
			dot ((real4)( err[sourcebase+size.x*size.y],
						err[sourcebase+1+size.x*size.y],
						err[sourcebase+size.x+size.x*size.y],
						err[sourcebase+size.x+1+size.x*size.y]),n2);
	}

	dest [destbase] = input[destbase] + val*4;
}

/***TODO: Better interpolation ***/
__kernel void prolongation_kernel(global read_only Cell * domain,
									global write_only real * dest,
									global read_only real * input)
{
	int destbase = get_global_id(0)+get_global_size(0)*get_global_id(1)+get_global_size(0)*get_global_size(1)*get_global_id(2);
	int4 pos = (int4)(get_global_id(0)/2,get_global_id(1)/2,get_global_id(2)/2,1);
	int4 size = (int4)(get_global_size(0)/2 +1,get_global_size(1)/2+1,get_global_size(2)/2+1,1);
	int sourcebase = pos.x +pos.y*size.x + pos.z*size.x*size.y;

	real val;
	if (isBorder(domain+destbase))
		val = input[sourcebase];
	else
	{
		real u = 0.5*(get_global_id(0)%2);
		real w = 0.5*(get_global_id(1)%2);
		real v = 0.5*(get_global_id(2)%2);

		real4 n1 = (real4)( (1.0-u)*(1.0-w)*(1.0-v), (1.0-w)*u*(1.0-v), (1.0-u)*w*(1.0-v),w*u*(1.0-v));
		real4 n2 = (real4)( (1.0-u)*(1.0-w)*v, (1.0-w)*u*v, (1.0-u)*w*v,w*u*v);

		val = dot ((real4)( input[sourcebase],
						input[sourcebase+1],
						input[sourcebase+size.x],
						input[sourcebase+size.x+1]),n1) +
			dot ((real4)( input[sourcebase+size.x*size.y],
						input[sourcebase+1+size.x*size.y],
						input[sourcebase+size.x+size.x*size.y],
						input[sourcebase+size.x+1+size.x*size.y]),n2);
	}
	dest[destbase] = val;
}

__kernel void zero_out(global read_only Cell * domain,
						global write_only real * input)
{
	int base = get_global_id(0)+get_global_size(0)*get_global_id(1)+get_global_size(0)*get_global_size(1)*get_global_id(2);
	if (getCellType(domain+base) == CELL_OUTSIDE)
		input[base] = 0;
}
