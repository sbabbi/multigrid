/*
    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

*/

#include "functionhandler.h"
#include <cmath>
#include <stdexcept>

Buffer2D FunctionHandler::discretize_func(int dimx, int dimy, real dh, const BorderHandler& bord)
{
	BidimArray<real> buf (dimx,dimy);

	for (int i=0;i < dimx;++i) for (int j=0;j < dimy;++j)
		switch (bord.cellType(i,j,dimx,dimy))
		{
		case BorderHandler::CellInner:
			buf(i,j)= m_pFunc( (real)(i)/(dimx-1), (real)(j)/(dimy-1) )*dh*dh;
			break;
		case BorderHandler::CellOuter:
			break;
		case BorderHandler::CellDirichlet:
			buf(i,j) = m_pBord( (real)(i)/(dimx-1), (real)(j)/(dimy-1) );
			break;
		case BorderHandler::CellNeumann:
			buf(i,j) = m_pBord( (real)(i)/(dimx-1), (real)(j)/(dimy-1) )*dh;
			break;
		}
	return Buffer2D(dimx,dimy,buf.data(),Buffer2D::ReadOnly);
}

Buffer2D FunctionHandler::discretize_sol(int dimx, int dimy, real dh, const BorderHandler& bord)
{
	BidimArray<real> buf (dimx,dimy);

	for (int i=0;i < dimx;++i) for (int j=0;j < dimy;++j)
		switch (bord.cellType(i,j,dimx,dimy))
		{
			case BorderHandler::CellOuter:
				buf(i,j) = 0;
				break;
			default:
				buf(i,j) = m_pSol( (real)(i)/(dimx-1), (real)(j)/(dimy-1) );
				break;
		}
	return Buffer2D(dimx,dimy,buf.data(),Buffer2D::ReadOnly);
}

real FunctionHandler::L2Error(Buffer2D& ans,cl::CommandQueue & q)
{
	if (!m_pSol) throw std::runtime_error("Can not compute L2Error without a known solution");

	real l2err = 0;
	int dimx = ans.width();
	int dimy = ans.height();
	BidimArray<real> res = ans.read(q);

	for (int i=0;i < dimx;++i) for (int j=0;j < dimy;++j)
	{
		real val = m_pSol( (real)(i)/(dimx-1), (real)(j)/(dimy-1) );
		real err = val - res(i,j);
		l2err += (err*err);
	}
	return sqrt(l2err);
}

real FunctionHandler::LInfError(Buffer2D& ans,cl::CommandQueue & q)
{
	if (!m_pSol) throw std::runtime_error("Can not compute L2Error without a known solution");

	real linferr = 0;
	int dimx = ans.width();
	int dimy = ans.height();
	BidimArray<real>  res = ans.read(q);

	for (int i=0;i < dimx;++i) for (int j=0;j < dimy;++j)
	{
		real val = m_pSol( (real)(i)/(dimx-1), (real)(j)/(dimy-1) );
		real err = fabs(val - res(i,j));
		linferr = std::max(linferr,err);
	}
	return linferr;
}