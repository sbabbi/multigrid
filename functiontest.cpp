/*
    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#include "functiontest.h"

double L2Norm(Buffer2D& val)
{
	cl_float res;
	cl::Buffer result ( CLContextLoader::getContext(),CL_MEM_USE_HOST_PTR|CL_MEM_READ_WRITE,sizeof(cl_float),&res);

	cl::Kernel & l2Ker = CLContextLoader::getL2NormKer();

	l2Ker.setArg(0,val());
	l2Ker.setArg(1,val.width()*val.height());
	l2Ker.setArg(2,result());

	cl::Event ev;
	CLContextLoader::getQueue().enqueueTask(l2Ker,0,&ev);
	ev.wait();
	return res;
}

double LInfNorm(Buffer2D& val)
{
	cl_float res;
	cl::Buffer result ( CLContextLoader::getContext(),CL_MEM_USE_HOST_PTR|CL_MEM_READ_WRITE,sizeof(cl_float),&res);

	cl::Kernel & lInfKer = CLContextLoader::getLInfNormKer();

	lInfKer.setArg(0,val());
	lInfKer.setArg(1,val.width()*val.height());
	lInfKer.setArg(2,result());

	cl::Event ev;
	CLContextLoader::getQueue().enqueueTask(lInfKer,0,&ev);
	ev.wait();
	return res;
}

Buffer2D FunctionTest::makeBuffer(int dimx, int dimy,float dh)
{
	using namespace boost::numeric::ublas;


	matrix<float> buf(dimy,dimx);
	for (int i=0;i < dimx;++i) for (int j=0;j < dimy;++j)
		if (i == 0 || j == 0 || i == dimx-1 || j == dimy-1)
			buf(j,i) = m_pBord( (float)(i)/(dimx-1), (float)(j)/(dimy-1) );
		else
			buf(j,i) = m_pFunc( (float)(i)/(dimx-1), (float)(j)/(dimy-1) )*dh*dh;

		return Buffer2D(dimx,dimy,&buf.data()[0]);
}

boost::numeric::ublas::matrix<float> FunctionTest::solution(int dimx,int dimy,float dh)
{
	using namespace boost::numeric::ublas;

	matrix<float> res(dimy,dimx);
	for (int i=0;i < dimx;++i) for (int j=0;j < dimy;++j)
		res(j,i) = m_pSol( (float)(i)/(dimx-1), (float)(j)/(dimy-1) );
	return res;
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
		linferr = std::max(linferr,err);
	}
	return sqrt(linferr);
}
