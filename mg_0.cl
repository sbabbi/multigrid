/*** Dumped jacobi ker ***/

#define real float

inline real getval(int2 _X,int2 size,global read_only real * src,global read_only real * func)
{
	return  0.25f * (
		src[_X.x + size.x * _X.y + 1] +
		src[_X.x + size.x * _X.y - 1] +
		src[_X.x + size.x * (_X.y +1)] +
		src[_X.x + size.x * (_X.y -1)] -
		func[_X.x + size.x * _X.y] );
}

__kernel void iteration_kernel(global write_only real * dest,
							global read_only real * src,
							global read_only real * func,
							int2 size,
							real w)
{
	int2 _X = (int2)(get_global_id(0),get_global_id(1));

	real val = getval(_X,size,src,func);
	dest [ _X.x + size.x * _X.y] = val * w + (1.0-w)* src[_X.x + size.x * _X.y];
}

__kernel void iteration_kernel_border(global write_only real * dest,
									  global read_only real * src,
									  global read_only real * func,
									  int2 size,
									  real w)
{
	int2 _X = (int2)(get_global_id(0),get_global_id(1));

	/** Implement dirichlet condition **/
	dest [_X.x +size.x* _X.y] = func [_X.x + size.x* _X.y];
}

__kernel void residual_kernel(global write_only real * dest,
								global read_only real * src,
								global read_only real * func,
								int2 size)
{
	int2 _X = (int2)(get_global_id(0),get_global_id(1));

	real val =
		src[_X.x + size.x * _X.y + 1] +
		src[_X.x + size.x * _X.y - 1] +
		src[_X.x + size.x * (_X.y +1)] +
		src[_X.x + size.x * (_X.y -1)] -
		4 * src[_X.x + size.x * _X.y] -
		func[_X.x + size.x * _X.y];

	dest [ _X.x + size.x * _X.y] = val;
}

__kernel void residual_kernel_border(global write_only real * dest,
								global read_only real * src,
								global read_only real * func,
								int2 size)
{
	int2 _X = (int2)(get_global_id(0),get_global_id(1));

	real val = src[_X.x + size.x * _X.y] - func[_X.x + size.x * _X.y];

	dest [ _X.x + size.x * _X.y] = val;
}

/**Full-weighting**/
__kernel void reduction_kernel(global write_only real * dest,
								global read_only real *src,
							   int2 size)
{
	int2 _X = (int2)(get_global_id(0),get_global_id(1));
	int2 _srcSize =  (size-(int2)(1,1))*2 + (int2)(1,1);

	dest [ _X.x + size.x * _X.y] = (
		4 * src [ 2 * _X.x + 2 * _srcSize.x * _X.y] +
		2 * src [ 2 * _X.x + 1 + 2 * _srcSize.x * _X.y] +
		2 * src [ 2 * _X.x - 1 + 2 * _srcSize.x * _X.y] +
		2 * src [ 2 * _X.x + _srcSize.x * (2*_X.y+1)] +
		2 * src [ 2 * _X.x + _srcSize.x * (2*_X.y-1)] +
		src [ 2 * _X.x+1 + _srcSize.x * (2*_X.y-1)] +
		src [ 2 * _X.x-1 + _srcSize.x * (2*_X.y+1)] +
		src [ 2 * _X.x+1 + _srcSize.x * (2*_X.y+1)] +
		src [ 2 * _X.x-1 + _srcSize.x * (2*_X.y-1)])/16.0;
}

/** Half weighting
__kernel void reduction_kernel(global write_only real * dest,
								global read_only real *src,
							   int2 size)
{
	int2 _X = (int2)(get_global_id(0),get_global_id(1));
	int2 _srcSize =  (size-(int2)(1,1))*2 + (int2)(1,1);

	dest [ _X.x + size.x * _X.y] = (
		4 * src [ 2 * _X.x + 2 * _srcSize.x * _X.y] +
		src [ 2 * _X.x + 1 + 2 * _srcSize.x * _X.y] +
		src [ 2 * _X.x - 1 + 2 * _srcSize.x * _X.y] +
		src [ 2 * _X.x + _srcSize.x * (2*_X.y+1)] +
		src [ 2 * _X.x + _srcSize.x * (2*_X.y-1)] )/8.0;
}**/

__kernel void reduction_kernel_border(global write_only real * dest,
								global read_only real *src,
							   int2 size)
{
	int2 _X = (int2)(get_global_id(0),get_global_id(1));
	int2 _srcSize =  (size-(int2)(1,1))*2 + (int2)(1,1);

	dest [ _X.x + size.x * _X.y] = src [ 2 * _X.x + 2 * _srcSize.x * _X.y];
}

__kernel void residual_correct_kernel(global write_only real * dest,
										global read_only real *src,
										global read_only real * res,
										int2 size)
{
	int2 _X = (int2)(get_global_id(0),get_global_id(1));
	int2 _resSize =  (size-(int2)(1,1))/2 + (int2)(1,1);

	real val;

	if ( _X.x % 2 == 0 && _X.y % 2 == 0)
		val = res [_X.x/2 + _resSize.x * _X.y/2];
	else if (_X.x % 2 == 1 && _X.y % 2 == 0)
		val = 0.5* (res[_X.x/2 + _resSize.x * _X.y/2] + res[_X.x/2 + 1 + _resSize.x * _X.y/2]);
	else if (_X.x % 2 == 0 && _X.y % 2 == 1)
		val = 0.5* (res[_X.x/2 + _resSize.x * (_X.y/2)] + res[_X.x/2 + _resSize.x * (_X.y/2+1)]);
	else
		val = 0.25*(res[_X.x/2 + _resSize.x * (_X.y/2)] +
					res[_X.x/2 + _resSize.x * (_X.y/2+1)] +
					res[_X.x/2 + 1 + _resSize.x * (_X.y/2)] +
					res[_X.x/2 + 1 + _resSize.x * (_X.y/2+1)]);


	dest [ _X.x + size.x * _X.y] = src[_X.x + size.x * _X.y] - val*4;
}

/***TODO: Better interpolation ***/
__kernel void prolongation_kernel(global write_only real * dest,
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
