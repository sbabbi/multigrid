
#include "real.cl"

/* Border management **/
typedef struct tagCell {
	real2 _normals; //MUST BE Manhattan-Normalized
}Cell;

#define CELL_INSIDE 0
#define CELL_OUTSIDE 1
#define CELL_DIRICHLET 2
#define CELL_NEUMANN 3

inline int getCellType(const __global read_only Cell * c)
{
	if ( c._normals == (real2)(0.0,0.0)) return CELL_INSIDE;
	if ( c._normals[0] == NAN && c._normals[1] == NAN) return CELL_OUTSIDE;
	if ( c._normals[0] == NAN) return CELL_DIRICHLET;
	return CELL_NEUMANN;
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


/*** iteration_kernel
	* This kernel computes an iteration of the weighted-jacobi method.

	* domain is a pointer to a bidimensional array of Cell data, which represents the description of the domains and borders
	* dest is the bidimensional output array
	* src is the current solution, on which we should perform the iteration
	* func is the target function (on the borders too!)
	* w is the omega parameter for the dumped jacobi iteration

	* Notice that the size of domain,dest,src and func MUST be equal to (int2)(get_global_size(0),get_global_size(1))
*/
__kernel void iteration_kernel(global read_only Cell * domain,
							global write_only real * dest,
							global read_only real * src,
							global read_only real * func,
							real w)
{
	int base = get_global_id(0) + get_global_size(0)*get_global_id(1);
	int sizex = get_global_size(0);

	real val;
	switch (getCellType(domain+base) )
	{
	case CELL_INSIDE:
		val = jacobi_iteration(base,sizex,src,func);
		dest[base] = val * w + (1.0-w)* src[base];
		break;
	case CELL_OUTSIDE
		break;
	case CELL_DIRICHLET:
		dest[base] = src[base];
		break;
	case CELL_NEUMANN:
		break;
	}
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
		case CELL_OUTSIZE:
			break;
		case CELL_DIRICHLET:
			dest[base] = src[base]-func[base];
			break;
		case CELL_NEUMANN:
			break;
	}
}

const __global read_only real4 RED_STENCIL[3] = {
	(1.0/16.0, 1.0/8.0,1.0/16.0,0),
	(1.0/8.0, 1.0/4.0,1.0/8.0,0),
	(1.0/16.0, 1.0/8.0,1.0/16.0,0)
}

/*** reduction_kernel
	* This kernel computes a Full-weighting reduction of the function passed in src

	* domain as in iteration_kernel. domain size MUST be equal to (int2)(get_global_size(0),get_global_size(1))
	* dest is the output 2d-array, whose size MUST be equal to (int2)(get_global_size(0),get_global_size(1))
	* src is the input function
	* size is the size of the INPUT function

	* Notice that the size of dest MUST be equal to (int2)(get_global_size(0),get_global_size(1))
	* Also, the destination size MUST be half of the src size
*/
__kernel void reduction_kernel(global read_only Cell * domain
								global write_only real * dest,
								global read_only real * src,
								int2 size)
{
	int destbase = get_global_id(0)+get_global_size(0)*get_global_id(1);
	int destsizex = get_global_id(0);

	int srcbase = 2*get_global_id(0)+ size.x*2*get_global_id(1);
	int srcsizex = size.x;

	switch (getCellType(domain+destbase) )
	{
	case CELL_INSIDE:
		dest[destbase] =
			dot (RED_STENCIL[0],(real4)( src[srcbase-srcsizex-1],src[srcbase-srcsizex],src[srcbase-srcsizex+1],0) ) +
			dot (RED_STENCIL[1],(real4)( src[srcbase-1],src[srcbase],src[srcbase+1],0) ) +
			dot (RED_STENCIL[0],(real4)( src[srcbase+srcsizex-1],src[srcbase+srcsizex],src[srcbase+srcsizex+1],0) );
		break;
	case CELL_OUTSIDE:
		break;
	case CELL_DIRICHLET:
		break;
	case CELL_NEUMANN:
		break;
	}
}

/*** residual_correct_kernel
	* This kernel prolongate the error err, and use the prolongation of err to correct the solution in input

	* domain,dest,input are bidimension arrays of size (int2)(get_global_size(0),get_global_size(1))
	* err is a bidimensional array

	* Notice that err size must be HALF of dest size.
***/
__kernel void residual_correct_kernel(global read_only Cell * domain
										global write_only real * dest,
										global read_only real * input,
										global read_only real * err)
{
	int destbase = get_global_id(0)+get_global_size(0)*get_global_id(1);
	int destsizex = get_global_id(0);

	int errsizex = get_global_size(0)/2;
	int errbase = get_global_id(0)/2 + errsizex*(get_global_id(1)/2);
	
	real2 p = (real2)( 0.5 - (get_global_id(0)%2)*(1.0/(2*get_global_id(0))),
					0.5 - (get_global_id(1)%2)*(1.0/(2*get_global_id(1))) );

	real2 errCoord = (real2)(get_global_id(0)+0.5,get_global_id(1)+0.5)*p;
	real2 w = errCoord-floor(errCoord);

	real val = err[errbase] * (1.0-w.x)*(1.0-w.y)+
				err[errbase+1] * (w.x)*(1.0-w.y)+
				err[errbase+errsizex] * (1.0-w.x)*w.y +
				err[errbase+errsizex+1] * w.x*w.y;


	dest [destbase] = src[destbase] - val*4;
}

/***TODO: Better interpolation ***/
__kernel void prolongation_kernel(global read_only real2 * border,
								  global write_only real * dest,
									  global read_only real * res,
										int2 size)
{
	int2 _X = (int2)(get_global_id(0),get_global_id(1));
	int2 _resSize =  (size-(int2)(1,1))/2 + (int2)(1,1);

	real val;

	if ( _X.x % 2 == 0 && _X.y % 2 == 0)
		val = res [_X.x/2 + _resSize.x * _X.y/2];
	else if (_X.x % 2 == 1 && _X.y % 2 == 0)
		val = 0.5* (res[_X.x/2 + _resSize.x * (_X.y/2)] + res[_X.x/2 + 1 + _resSize.x * (_X.y/2)]);
	else if (_X.x % 2 == 0 && _X.y % 2 == 1)
		val = 0.5* (res[_X.x/2 + _resSize.x * (_X.y/2)] + res[_X.x/2 + _resSize.x * (_X.y/2+1)]);
	else
		val = 0.25*(res[_X.x/2 + _resSize.x * (_X.y/2)] +
					res[_X.x/2 + _resSize.x * (_X.y/2+1)] +
					res[_X.x/2 + 1 + _resSize.x * (_X.y/2)] +
					res[_X.x/2 + 1 + _resSize.x * (_X.y/2+1)]);


	dest [ _X.x + size.x * _X.y] = val;
}
