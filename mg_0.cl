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

/** Border test kernel **/

typedef struct
{
	union {float2 normal;int child[4];};
	bool leaf;
}Node;

inline int child(int x,int y,int xdimp2,int ydimp2)
{
	return (x >= (1 << (xdimp2-1))) + 2*( (y >= (1 << (ydimp2-1))) );
}

inline int rebase(int coord,int dimp2)
{
	return coord % ( 1 << (dimp2));
}

inline int log2_int(int c)
{
	for (int i=31;i >= 0;--i)
		if (c & (1 << i)) return i+ (c % (1 << i) != 0);
	return 0;
}

int go_base(int x,int y,int xdimp2,int ydimp2,int destxdimp2,int destydimp2,__global read_only Node * tree)
{
	int base = 0;
	for (int j=0;j < 10000;++j)
	{
		if (xdimp2 <= destxdimp2 || ydimp2 <= destydimp2) return base;

		int next = child(x,y,max(xdimp2,ydimp2),max(xdimp2,ydimp2));

		if (tree[base].child[next] == -1) return -1;

		int new_base_x = xdimp2 < ydimp2 ? xdimp2 : xdimp2-1;
		int new_base_y = ydimp2 < xdimp2 ? ydimp2 : ydimp2-1;
		x = rebase(x,new_base_x);
		y = rebase(y,new_base_y);
		xdimp2 = new_base_x;
		ydimp2 = new_base_y;
		base = tree[base].child[next];
	}
	return -1;
}

bool find_node(int x, int y, int xdimp2, int ydimp2, __private write_only float2* out,__global read_only Node * tree,int base)
{
	if (base == -1) return 0;
	for (int j=0;j < 10000;++j)
	{
		if (tree[base].leaf)
		{
			/*assert(x == 0 && y == 0 && xdimp2 == 0 && ydimp2 == 0);*/
			*out = tree[base].normal;
			return 1;
		}
		int next = child(x,y,max(xdimp2,ydimp2),max(xdimp2,ydimp2));

		if (tree[base].child[next] == -1) return 0;

		int new_base_x = xdimp2 < ydimp2 ? xdimp2 : xdimp2-1;
		int new_base_y = ydimp2 < xdimp2 ? ydimp2 : ydimp2-1;
		x = rebase(x,new_base_x);
		y = rebase(y,new_base_y);
		xdimp2 = new_base_x;
		ydimp2 = new_base_y;
		base = tree[base].child[next];
	}
	return 0;
}

__kernel void test_border(global write_only real * dest,
						  global read_only Node * bord,
						  int2 size)
{
	__local int base;

	int2 _X = (int2)(get_global_id(0),get_global_id(1));

	int p2sizex = log2_int(size.x);
	int p2sizey = log2_int(size.y);

	if (get_local_id(0) == 0 && get_local_id(1) == 0)
	{
		base = go_base(_X.x,_X.y,p2sizex,p2sizey,log2_int(get_local_size(0)),log2_int(get_local_size(1)),bord);
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	float2 ans;

	dest [ _X.x + size.x * _X.y] = find_node(_X.x % get_local_size(0),_X.y % get_local_size(1),log2_int(get_local_size(0)),log2_int(get_local_size(1)),&ans,bord,base);
}
