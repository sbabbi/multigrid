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

#include "multigridsolver2D.h"
#include "auxiliary.h"
#include <iostream>

using namespace std;

std::ostream& operator<<(std::ostream & os,const BidimArray< ::real> & m);

MultigridSolver2D::MultigridSolver2D(const char* filename, BorderHandler2D& handl) :
	m_theProgram ( CLContextLoader::loadProgram(filename)),
	m_iterationKernel(m_theProgram,"iteration_kernel"),
	m_residualKernel(m_theProgram,"residual_kernel"),
	m_reductionKernel(m_theProgram,"reduction_kernel"),
	m_residualCorrectKernel(m_theProgram,"residual_correct_kernel"),
	m_prolongationKernel(m_theProgram,"prolongation_kernel"),
	m_zeroOutKernel(m_theProgram,"zero_out"),
	m_queue( CLContextLoader::getContext(),CLContextLoader::getDevice()),
	m_Handl(handl),
	m_debugPrintResiduals(false)
{
}

Buffer2D MultigridSolver2D::iterate(Buffer2D & in,
								   const Buffer2D& func,
								   real omega,
								   int a1,
								   int a2,
								   int v)
{
	assert( in.width() == func.width()&& in.height() == func.height());

	smoother_iterate(in,func,omega,a1);
	if (in.width() > 3 && in.height() > 3)
	{
		Buffer2D tmp (in.width(),in.height());
		Buffer2D residuals ( (in.width()+1)/2,(in.height()+1)/2);
		Buffer2D i (residuals.width(),residuals.height());

		for (int k=0;k <v;++k)
		{
			zero_mem(i);
			//Compute residuals
			compute_residuals(tmp,in,func);

			if (m_debugPrintResiduals)
			{
				::real ans = L2Norm(tmp,m_queue);
				cout << "DEBUG, printing residuals on grid: " << in.width() <<"x" << in.height() << " k= " <<k << " BEFORE correction res= "<< ans << endl;
			}

			//Restrict residuals
			restrict(residuals,tmp);

			//Solve residuals
			iterate(i,residuals,omega,a1,a2,v);

			//Prolungate residuals
			correct_residual(tmp,in,i);

			if (m_debugPrintResiduals)
			{
				Buffer2D deb(in.width(),in.height());
				compute_residuals(deb,tmp,func);
				::real ans = L2Norm(deb,m_queue);
				cout << "DEBUG, printing residuals on grid: " << in.width() <<"x" << in.height() << " k= " <<k << " AFTER correction res= "<< ans << endl;
			}

			std::swap(in,tmp);
		}
	}

	//Post smooth
	smoother_iterate(in,func,omega,a2);

	return in;
}

Buffer2D MultigridSolver2D::fmg(const Buffer2D& func,
								   real omega,
								   int a1,
								   int a2,
								   int v)
{
	if (!(func.width() > 3 && func.height() > 3))
	{
		//Solve with func
		Buffer2D x0 (func.width(),func.height());
		zero_mem(x0);

		return iterate(x0,func,omega,a1,a2,v);
	}

	Buffer2D red_func ( (func.width()+1)/2,(func.height()+1)/2);
	restrict(red_func,func);

	Buffer2D iGuess = fmg(red_func,omega,a1,a2,v);

	m_queue.enqueueBarrier();

	Buffer2D x0 (func.width(),func.height());
	prolongate(x0,iGuess);

	return iterate(x0,func,omega,a1,a2,v);
}

void MultigridSolver2D::smoother_iterate(Buffer2D& res, const Buffer2D& func, real omega, int a1)
{
	m_Handl.setarg(0,m_iterationKernel,res.width(),res.height());
	m_iterationKernel.setArg(1,res());
	m_iterationKernel.setArg(2,func());
	m_iterationKernel.setArg(3,omega);

	cl_int2 dim = {res.width(),res.height()};
	m_iterationKernel.setArg(4, dim);

	cl::NDRange dims ( dim.s[0]/2 + dim.s[0]%2,dim.s[1]);
	cl::NDRange wsDim = getBestWorkspaceDim(dims);

	for (int i=0;i < a1;++i)
	{
		m_iterationKernel.setArg(5,0);
		m_queue.enqueueNDRangeKernel(m_iterationKernel,cl::NDRange(0,0),dims,wsDim);

		m_iterationKernel.setArg(5,1);
		m_queue.enqueueNDRangeKernel(m_iterationKernel,cl::NDRange(0,0),dims,wsDim);
	}
}

void MultigridSolver2D::compute_residuals(Buffer2D& res, const Buffer2D& input, const Buffer2D& func)
{
	m_residualKernel.setArg(1,res());
	m_residualKernel.setArg(2,input());
	m_residualKernel.setArg(3,func());

	m_Handl.setarg(0,m_residualKernel, res.width(),res.height());
	m_queue.enqueueNDRangeKernel(m_residualKernel,cl::NDRange(0,0),cl::NDRange(res.width(),res.height()),
																getBestWorkspaceDim(cl::NDRange(res.width(),res.height())));
}

void MultigridSolver2D::restrict(Buffer2D& res,const Buffer2D& input)
{
// 	assert(res.width() == input.width()/2);
// 	assert(res.height() == input.height()/2);

	m_reductionKernel.setArg(1,res());
	m_reductionKernel.setArg(2,input());
	m_reductionKernel.setArg(3,input.size());

	m_Handl.setarg(0,m_reductionKernel,input.width(),input.height());
	m_queue.enqueueNDRangeKernel(m_reductionKernel,cl::NDRange(0,0),cl::NDRange(res.width(),res.height()),
																				getBestWorkspaceDim(cl::NDRange(res.width(),res.height())));
}

void MultigridSolver2D::correct_residual(Buffer2D& res,const Buffer2D& input, Buffer2D& residual)
{
// 	assert(input.width()/2 == residual.width());
// 	assert(input.height()/2 == residual.height());

	m_residualCorrectKernel.setArg(1,res());
	m_residualCorrectKernel.setArg(2,input());
	m_residualCorrectKernel.setArg(3,residual());

	m_Handl.setarg(0,m_residualCorrectKernel,res.width(),res.height());
	m_queue.enqueueNDRangeKernel(m_residualCorrectKernel,cl::NDRange(0,0),cl::NDRange(res.width(),res.height()),
																					  getBestWorkspaceDim(cl::NDRange(res.width(),res.height())));
}

void MultigridSolver2D::prolongate(Buffer2D& res,const Buffer2D& input)
{
// 	assert(res.width()/2 == input.width());
// 	assert(res.height()/2 ==input.height());

	m_prolongationKernel.setArg(1,res());
	m_prolongationKernel.setArg(2,input());

	m_Handl.setarg(0,m_prolongationKernel,res.width(),res.height());
	m_queue.enqueueNDRangeKernel(m_prolongationKernel,cl::NDRange(0,0),cl::NDRange(res.width(),res.height()),
																				   getBestWorkspaceDim(cl::NDRange(res.width(),res.height())));
}

void MultigridSolver2D::zero_mem(Buffer2D& res)
{
	CLContextLoader::getZeroMemKer().setArg(0,res());
	m_queue.enqueueNDRangeKernel(CLContextLoader::getZeroMemKer(),cl::NDRange(0),cl::NDRange(res.width()*res.height()),cl::NDRange(1));
}

void MultigridSolver2D::zero_out(Buffer2D& res)
{
	m_Handl.setarg(0,m_zeroOutKernel,res.width(),res.height());
	m_zeroOutKernel.setArg(1,res());
	m_queue.enqueueNDRangeKernel(m_zeroOutKernel,cl::NDRange(0,0),cl::NDRange(res.width(),res.height()),
								 getBestWorkspaceDim(cl::NDRange(res.width(),res.height())));
}
