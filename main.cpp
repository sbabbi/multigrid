#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>

#include "clcontextloader.h"
#include "buffer.h"
#include "multigridsolver0.h"

using namespace std;

std::ostream& operator<<(std::ostream & os,const boost::numeric::ublas::matrix<float> & m)
{
	for (int i=0;i < m.size1();++i)
	{
		for (int j=0;j < m.size2();++j)
			os << m(i,j) << " ";
		os << std::endl;
	}
	return os;
}

class RectangularBorderHandler : public BorderHandler
{
public:
	void compute(cl::CommandQueue & queue,cl::Kernel & innerKer,cl::Kernel & borderKer,int dimx,int dimy) const
	{
		//Top and bottom border
		queue.enqueueNDRangeKernel(borderKer,cl::NDRange(0,0),cl::NDRange(dimx-1,1),cl::NDRange(1,1));
		queue.enqueueNDRangeKernel(borderKer,cl::NDRange(1,dimy-1),cl::NDRange(dimx-1,1),cl::NDRange(1,1));

		//Left and right borders
		queue.enqueueNDRangeKernel(borderKer,cl::NDRange(0,1),cl::NDRange(1,dimy-1),cl::NDRange(1,1));
		queue.enqueueNDRangeKernel(borderKer,cl::NDRange(dimx-1,0),cl::NDRange(1,dimy-1),cl::NDRange(1,1));

		//Enqueue inner ker
		queue.enqueueNDRangeKernel(innerKer,cl::NDRange(1,1),cl::NDRange( (dimx-2),(dimy-2)),cl::NDRange(1,1));
		queue.enqueueBarrier();
	}
};

class FunctionTest
{
public:
	FunctionTest( float(*p)(float,float),
				  float(*bord)(float,float),
				  float(*sol)(float,float) = 0) : m_pFunc(p),
												m_pBord(bord),
												m_pSol(sol)
	{}

	Buffer2D makeBuffer(int dimx,int dimy);
	double L2Error(Buffer2D & ans);
	double L2Norm(Buffer2D & ans);
	double LInfNorm(Buffer2D & ans);
	double LInfError(Buffer2D & ans);

	boost::numeric::ublas::matrix<float> solution(int dimx,int dimy);

private:
	float (*m_pFunc)(float,float);
	float (*m_pBord)(float,float);
	float (*m_pSol)(float,float);
};

Buffer2D FunctionTest::makeBuffer(int dimx, int dimy)
{
	using namespace boost::numeric::ublas;

	float dx = 1.0f/(dimx-1);
	float dy = 1.0f/(dimy-1);

	if (dx != dy) cout << "Warning: dx!=dy" << endl;

	matrix<float> buf(dimy,dimx);
	for (int i=0;i < dimx;++i) for (int j=0;j < dimy;++j)
		if (i == 0 || j == 0 || i == dimx-1 || j == dimy-1)
			buf(j,i) = m_pBord( (float)(i)/(dimx-1), (float)(j)/(dimy-1) );
		else
			buf(j,i) = m_pFunc( (float)(i)/(dimx-1), (float)(j)/(dimy-1) )*dx*dx;

	return Buffer2D(dimx,dimy,&buf.data()[0]);
}

double FunctionTest::L2Error(Buffer2D& ans)
{
	using namespace boost::numeric::ublas;

	if (!m_pSol) throw std::runtime_error("Can not compute L2Error without a known solution");

	double l2err = 0;
	int dimx = ans.width();
	int dimy = ans.height();
	matrix<float> res = ans;

	for (int i=0;i < dimx;++i) for (int j=0;j < dimy;++j)
	{
		double val = m_pSol( (float)(i)/(dimx-1), (float)(j)/(dimy-1) );
		double err = val - res(j,i);
		l2err += (err*err);
	}
	return sqrt(l2err);
}

double FunctionTest::L2Norm(Buffer2D& ans)
{
	using namespace boost::numeric::ublas;

	double l2err = 0;
	int dimx = ans.width();
	int dimy = ans.height();
	matrix<float> res = ans;

	for (int i=0;i < dimx;++i) for (int j=0;j < dimy;++j)
	{
		double val = res(j,i);
		l2err += (val*val);
	}
	return sqrt(l2err);
}

double FunctionTest::LInfNorm(Buffer2D& ans)
{
	using namespace boost::numeric::ublas;

	double linferr = 0;
	int dimx = ans.width();
	int dimy = ans.height();
	matrix<float> res = ans;

	for (int i=0;i < dimx;++i) for (int j=0;j < dimy;++j)
	{
		double val = res(j,i);
		linferr = max(linferr,fabs(val));
	}
	return linferr;
}

boost::numeric::ublas::matrix<float> FunctionTest::solution(int dimx,int dimy)
{
	using namespace boost::numeric::ublas;

	matrix<float> res(dimy,dimx);
	for (int i=0;i < dimx;++i) for (int j=0;j < dimy;++j)
		res(j,i) = m_pSol( (float)(i)/(dimx-1), (float)(j)/(dimy-1) );
	return res;
}

double FunctionTest::LInfError(Buffer2D& ans)
{
	using namespace boost::numeric::ublas;

	if (!m_pSol) throw std::runtime_error("Can not compute L2Error without a known solution");

	double linferr = 0;
	int dimx = ans.width();
	int dimy = ans.height();
	matrix<float> res = ans;

	for (int i=0;i < dimx;++i) for (int j=0;j < dimy;++j)
	{
		double val = m_pSol( (float)(i)/(dimx-1), (float)(j)/(dimy-1) );
		double err = fabs(val - res(j,i));
		linferr = max(linferr,err);
	}
	return sqrt(linferr);
}

float ones(float,float) {return 1;}
float zeros(float,float) {return 0;}
float prettyFunc1(float x,float y)
{
	return -2*( (1-6*x*x)*y*y*(1-y*y)+
				(1-6*y*y)*x*x*(1-x*x));
}
float prettyFunc1Sol(float x,float y)
{
	return (x*x-x*x*x*x)*(y*y*y*y-y*y);
}

enum SolverMode {Fmg = 0,Smooth = 1,Multigrid = 2};

Buffer2D solve(MultigridSolver0 & s,SolverMode m,const Buffer2D & f,int a1,int a2,int v,float omega = 1.0)
{
	int dimx = f.width();
	int dimy = f.height();

	switch (m)
	{
	case Fmg:
	{
		cout << "FMG\t\t";
		clock_t init = clock();
		Buffer2D res = s.fmg(f,omega,a1,a2,v);
		s.wait();
		clock_t end = clock();
		cout << ((double)(end-init))/CLOCKS_PER_SEC << "\t\t";
		return res;
	}
	case Smooth:
	{
		cout << "Smoother\t";
		Buffer2D tmp = Buffer2D::empty(dimx,dimy);
		Buffer2D sol = Buffer2D::empty(dimx,dimy);
		clock_t init = clock();
		s.smoother_iterate(sol,tmp,f,omega,a1);
		s.wait();
		clock_t end = clock();
		cout << ((double)(end-init))/CLOCKS_PER_SEC << "\t\t";
		return sol;
	}
	case Multigrid:
	{
		cout << "Multigrid\t";
		Buffer2D tmp = Buffer2D::empty(dimx,dimy);
		clock_t init = clock();
		Buffer2D res = s.iterate(tmp,f,omega,a1,a2,v);
		s.wait();
		clock_t end = clock();
		cout << ((double)(end-init))/CLOCKS_PER_SEC << "\t\t";
		return res;
	}
	}
	throw std::runtime_error("Wtf??");
}

int main(int argc,char ** argv)
{
	std::cout << std::fixed << std::setprecision(7);

	int args[5] = {9,9,5,5,1};
	float omega = 1.0;

	if (argc < 8) { cout << "Need more args" << endl; return 1;}

	SolverMode m = (SolverMode)atoi(argv[1]);
	for (int i=0;i < 5;++i) args[i] = atoi(argv[2+i]);
	omega = atof(argv[7]);

	try {
		const int dimx = args[0], dimy = args[1];
		FunctionTest testFunction(prettyFunc1,zeros,prettyFunc1Sol);
// 		FunctionTest testFunction(zeros,ones,ones);

		Buffer2D f = testFunction.makeBuffer(dimx,dimy);

		cout << "Solver\t\tTime\t\t\tL2Err\t\t\tLInfErr\t\t\tL2Res\t\t\tLinfRes\t\t\t" << endl;
		RectangularBorderHandler borderHandler;
		MultigridSolver0 s ("mg_0.cl",borderHandler);

		Buffer2D sol = solve(s,m,f,args[2],args[3],args[4],omega);

		try {
			cout << testFunction.L2Error(sol) << "\t\t";
			cout << testFunction.LInfError(sol) << "\t\t";

			Buffer2D res (dimx,dimy);
			s.compute_residuals(res,sol,f);
			s.wait();
			cout << testFunction.L2Norm(res) << "\t\t";
			cout << testFunction.LInfNorm(res) << endl;

// 			cout << res << endl;
// 			cout << sol << endl;
// 			cout << "Correct solution is: " << testFunction.solution(dimx,dimy) << endl;
		}
		catch (std::runtime_error) {}
	}
	catch(cl::Error & r)
	{
		std::cout << "Cl error in " << r.what() << " code: " << r.err() << std::endl;
	}
	catch(std::exception & r)
	{
		std::cout << r.what() << std::endl;
	}
	catch(...)
	{
		std::cout << "Unknown error" << std::endl;
	}
	return 0;
}