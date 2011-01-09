/*gauss_seidel.cl*/

typedef struct Buffer2D
{
	int dimx,dimy;
	__global double * data;
};

inline int make_addr(int2 v,int2 s)
{
	return v.x+v.y*s.x;
}

/** Bidimensional gauss seidel **/
__kernel void gauss_seidel(	__global Buffer2D sol,
					__global read_only Buffer2D func)
{
	int2 pos = (int2)(get_global_id(0),get_global_id(1));
	const int2 size = (int2)(get_global_size(0),get_global_size(1));

	int top = make_addr( pos+(int2)(0,-1),size);
	int left = make_addr( pos+(int2)(-1,0),size);
	int bot = make_addr( pos+(int2)(0,1),size);
	int right = make_addr( pos+(int2)(1,0),size);
	
	sol[ make_addr(pos,size)] = 0.25 * ( sol[make_addr(top,size)]+
										sol[make_addr(left,size)]+
										sol[make_addr(bot,size)]+
										sol[make_addr(right,size)]-
										func[make_addr(pos,size)] );
}
