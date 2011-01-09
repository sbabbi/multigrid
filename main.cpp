#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>

#include "clcontextloader.h"
#include "buffer.h"

using namespace std;

class GaussSeidel
{
public:
	GaussSeidel(cl::Kernel & GSKer);

	cl::Event iterate(Buffer2D & sol,const Buffer2D & func,int times = 1);

private:
	cl::Kernel & m_kernel;
	cl::CommandQueue m_queue;
};

GaussSeidel::GaussSeidel(cl::Kernel& GSKer) :
	m_kernel(GSKer),
	m_queue( CLContextLoader::getContext(),CLContextLoader::getDevice())
{
}

cl::Event GaussSeidel::iterate(Buffer2D& sol, const Buffer2D& func, int times)
{
	assert( sol.width() == func.width()&& sol.height() == func.height());
	m_kernel.setArg(0,sol());
	m_kernel.setArg(1,func());
	for (int i=0;i < times-1;++i)
		m_queue.enqueueNDRangeKernel(m_kernel,
								   cl::NDRange(1,1),
								   cl::NDRange( sol.width()-1,sol.height()-1),
								   cl::NullRange);

	cl::Event ans;
	m_queue.enqueueNDRangeKernel(m_kernel,
								 cl::NDRange(0,0),
								 cl::NDRange( sol.width()-1,sol.height()-1),
								 cl::NullRange,
								 0,
								 &ans);
	return ans;
}

std::ostream& operator<<(std::ostream & os,const boost::numeric::ublas::matrix<double> & m)
{
	for (int i=0;i < m.size1();++i)
	{
		for (int j=0;j < m.size2();++j)
			os << m(i,j) << " ";
		os << std::endl;
	}
	return os;
}

int main()
{
	using namespace boost::numeric::ublas;
	/*try*/ {
		const int dimx = 32, dimy = 32;
		matrix<double> test(dimx,dimy);
		for (int i=0;i < dimx;++i) for (int j=0;j < dimy;++j)
			if (i == 0 || j == 0 || i == dimx-1 || j == dimy-1)
				test(i,j) = 1;
			else
				test(i,j) = 0;

		Buffer2D sol (dimx,dimy,&test.data()[0]);
		Buffer2D f = Buffer2D::empty(dimx,dimy);

		cl::Program GsProg = CLContextLoader::loadProgram("gauss_seidel.cl");
		cl::Kernel GsKer = cl::Kernel(GsProg,"gauss_seidel");

		GaussSeidel s (GsKer);
		s.iterate(sol,f,1).wait();
		cout << sol << endl;
	}
/*	catch(std::exception & r)
	{
		std::cout << r.what() << std::endl;
	}
	catch(...)
	{
		std::cout << "Unknown error" << std::endl;
	}*/
	return 0;
}