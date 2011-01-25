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

#ifndef MULTIGRIDSOLVER1_H
#define MULTIGRIDSOLVER1_H

#include "buffer.h"

class BorderHandler3D
{
public:
	enum CellType { CellInner,CellOuter,CellDirichlet,CellNeumann };

	virtual void setarg(int arg,cl::Kernel & ker,int dimx,int dimy,int dimz) = 0;

	virtual CellType cellType(int x,int y,int z,int dimx,int dimy,int dimz) const = 0;

private:
};

class MultigridSolver3D
{
public:
	MultigridSolver3D(const char * filename,BorderHandler3D & handl);

	Buffer3D iterate(Buffer3D & in,
					const Buffer3D & func,
					 real omega = 2.0/3.0,
					 int a1 = 4,
					 int a2 = 4,
					 int v = 1);

	Buffer3D fmg(const Buffer3D & func,
			real omega = 2.0/3.0,
			int a1 = 4,
			int a2 = 4,
			int v = 1);

	void wait() { m_queue.finish();}

	cl::CommandQueue & queue() {return m_queue;}

	void smoother_iterate(Buffer3D& res, const Buffer3D& func, real omega, int a1);
	void compute_residuals(Buffer3D& res,const Buffer3D & input, const Buffer3D& func);
	void restrict(Buffer3D& res,const Buffer3D & input);
	void correct_residual(Buffer3D & res,const Buffer3D & input,Buffer3D & residual);
	void prolongate(Buffer3D & res,const Buffer3D & input);
	void zero_mem(Buffer3D & res);
	void zero_out(Buffer3D & res);

private:
	cl::Program m_theProgram;

	cl::Kernel m_iterationKernel;
	cl::Kernel m_residualKernel;
	cl::Kernel m_reductionKernel;
	cl::Kernel m_residualCorrectKernel;
	cl::Kernel m_prolongationKernel;
	cl::Kernel m_zeroOutKernel;
	cl::Buffer m_emptyBuf;

	cl::CommandQueue m_queue;

	BorderHandler3D & m_Handl;
};

#endif // MULTIGRIDSOLVER1_H
