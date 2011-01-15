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

#include "multigridsolver0.h"
#include <iostream>
using namespace std;


MultigridSolver0::MultigridSolver0(const char* filename, const BorderHandler& handl) :
	m_theProgram ( CLContextLoader::loadProgram(filename)),
	m_iterationKernel(m_theProgram,"iteration_kernel"),
	m_residualKernel(m_theProgram,"residual_kernel"),
	m_reductionKernel(m_theProgram,"reduction_kernel"),
	m_residualCorrectKernel(m_theProgram,"residual_correct_kernel"),
	m_prolongationKernel(m_theProgram,"prolongation_kernel"),
	m_queue( CLContextLoader::getContext(),CLContextLoader::getDevice(),CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE),
	m_Handl(handl)
{
}

Buffer2D MultigridSolver0::iterate(Buffer2D & in,
								   const Buffer2D& func,
								   float omega,
								   int a1,
								   int a2,
								   int v)
{
	assert( in.width() == func.width()&& in.height() == func.height());

	Buffer2D tmp (in.width(),in.height());

	smoother_iterate(in,tmp,func,omega,a1);
	if (in.width() > 3 && in.height() > 3)
	{
		Buffer2D residuals ( in.width()/2,in.height()/2);
		Buffer2D i (residuals.width(),residuals.height());
		for (int k=0;k <v;++k)
		{
			zero_mem(i);
			//Compute residuals
			compute_residuals(tmp,in,func);

			//Restrict residuals
			restrict(residuals,tmp);

			//Solve residuals
			iterate(i,residuals,omega,a1,a2,v);

			//Prolungate residuals
			correct_residual(tmp,in,i);

			std::swap(in,tmp);
		}
	}

	//Post smooth
	smoother_iterate(in,tmp,func,omega,a2);

	return in;
}

Buffer2D MultigridSolver0::fmg(const Buffer2D& func,
								   float omega,
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

	Buffer2D red_func (func.width()/2,func.height()/2);
	restrict(red_func,func);

	Buffer2D iGuess = fmg(red_func,omega,a1,a2,v);

	Buffer2D x0 (func.width(),func.height());
	prolongate(x0,iGuess);

	return iterate(x0,func,omega,a1,a2,v);
}

void MultigridSolver0::smoother_iterate(Buffer2D& res, Buffer2D& auxiliary, const Buffer2D& func, float omega, int a1)
{
	m_iterationKernel.setArg(3,func());
	m_iterationKernel.setArg(4,omega);

	for (int i=0;i < a1;++i)
	{
		m_iterationKernel.setArg(1,auxiliary());
		m_iterationKernel.setArg(2,res());


		m_queue.enqueueBarrier();
		m_Handl.compute(m_queue,m_iterationKernel,res.width(),res.height());

		 std::swap(auxiliary,res);
	}
}

void MultigridSolver0::compute_residuals(Buffer2D& res, const Buffer2D& input, const Buffer2D& func)
{
	m_residualKernel.setArg(1,res());
	m_residualKernel.setArg(2,input());
	m_residualKernel.setArg(3,func());

	m_queue.enqueueBarrier();
	m_Handl.compute(m_queue,m_residualKernel,res.width(),res.height());
}

void MultigridSolver0::restrict(Buffer2D& res,const Buffer2D& input)
{
	assert(res.width() == input.width()/2);
	assert(res.height() == input.height()/2);

	m_reductionKernel.setArg(1,res());
	m_reductionKernel.setArg(2,input());
	m_reductionKernel.setArg(2,input.size());

	m_queue.enqueueBarrier();
	m_Handl.compute(m_queue,m_reductionKernel,res.width(),res.height());
}

void MultigridSolver0::correct_residual(Buffer2D& res,const Buffer2D& input, Buffer2D& residual)
{
	assert(input.width()/2 == residual.width());
	assert(input.height()/2 == residual.height());

	m_residualCorrectKernel.setArg(1,res());
	m_residualCorrectKernel.setArg(2,input());
	m_residualCorrectKernel.setArg(3,residual());

	m_queue.enqueueBarrier();
	m_Handl.compute(m_queue,m_residualCorrectKernel,res.width(),res.height());
}

void MultigridSolver0::prolongate(Buffer2D& res,const Buffer2D& input)
{
	assert(res.width() == (input.width()-1)*2+1);
	assert(res.height() == (input.height()-1)*2+1);

	m_prolongationKernel.setArg(0,res());
	m_prolongationKernel.setArg(1,input());
	m_prolongationKernel.setArg(2,res.size());

	m_queue.enqueueBarrier();
	m_Handl.compute(m_queue,m_prolongationKernel,res.width(),res.height());
}

void MultigridSolver0::zero_mem(Buffer2D& res)
{
	CLContextLoader::getZeroMemKer().setArg(0,res());
	m_queue.enqueueBarrier();
	m_queue.enqueueNDRangeKernel(CLContextLoader::getZeroMemKer(),cl::NDRange(0),cl::NDRange(res.width()*res.height()),cl::NDRange(1));
}
