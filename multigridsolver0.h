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

#ifndef MULTIGRIDSOLVER0_H
#define MULTIGRIDSOLVER0_H

#include "buffer.h"

class BorderHandler
{
public:
	virtual void compute(cl::CommandQueue & queue,cl::Kernel & innerKer,cl::Kernel & borderKer,int dimx,int dimy) = 0;
private:
};

class MultigridSolver0
{
public:
	MultigridSolver0(const char * filename,BorderHandler & handl);

	Buffer2D iterate(Buffer2D & in,
					const Buffer2D & func,
					 float omega = 2.0/3.0,
					 int a1 = 4,
					 int a2 = 4,
					 int v = 1);

	Buffer2D fmg(const Buffer2D & func,
			float omega = 2.0/3.0,
			int a1 = 4,
			int a2 = 4,
			int v = 1);

	void wait() { m_queue.finish();}

	void smoother_iterate(Buffer2D& res, Buffer2D & auxiliary, const Buffer2D& func, float omega, int a1);
	void compute_residuals(Buffer2D& res,const Buffer2D & input, const Buffer2D& func);
	void restrict(Buffer2D& res,const Buffer2D & input);
	void correct_residual(Buffer2D & res,const Buffer2D & input,Buffer2D & residual);
	void prolongate(Buffer2D & res,const Buffer2D & input);
	void zero_mem(Buffer2D & res);

private:
	cl::Program m_theProgram;

	cl::Kernel m_iterationKernel;
	cl::Kernel m_iterationKernelBorder;

	cl::Kernel m_residualKernel;
	cl::Kernel m_residualKernelBorder;

	cl::Kernel m_reductionKernel;
	cl::Kernel m_reductionKernelBorder;

	cl::Kernel m_residualCorrectKernel;
	cl::Kernel m_residualCorrectKernelBorder;

	cl::Kernel m_prolongationKernel;

	cl::CommandQueue m_queue;

	BorderHandler & m_Handl;
};

#endif // MULTIGRIDSOLVER0_H
