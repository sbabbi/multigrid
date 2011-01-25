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

Buffer2D FunctionHandler2D::discretize_func(int dimx, int dimy, real dh, const BorderHandler2D& bord)
{
	BidimArray<real> buf (dimx,dimy);

	for (int i=0;i < dimx;++i) for (int j=0;j < dimy;++j)
		switch (bord.cellType(i,j,dimx,dimy))
		{
		case BorderHandler2D::CellInner:
			buf(i,j)= m_pFunc( (real)(i)/(dimx-1), (real)(j)/(dimy-1) )*dh*dh;
			break;
		case BorderHandler2D::CellOuter:
			break;
		case BorderHandler2D::CellDirichlet:
			buf(i,j) = m_pBord( (real)(i)/(dimx-1), (real)(j)/(dimy-1) );
			break;
		case BorderHandler2D::CellNeumann:
			buf(i,j) = m_pBord( (real)(i)/(dimx-1), (real)(j)/(dimy-1) )*dh;
			break;
		}
	return Buffer2D(dimx,dimy,buf.data(),Buffer2D::ReadOnly);
}

Buffer2D FunctionHandler2D::discretize_sol(int dimx, int dimy, real dh, const BorderHandler2D& bord)
{
	BidimArray<real> buf (dimx,dimy);

	for (int i=0;i < dimx;++i) for (int j=0;j < dimy;++j)
		switch (bord.cellType(i,j,dimx,dimy))
		{
			case BorderHandler2D::CellOuter:
				buf(i,j) = 0;
				break;
			default:
				buf(i,j) = m_pSol( (real)(i)/(dimx-1), (real)(j)/(dimy-1) );
				break;
		}
	return Buffer2D(dimx,dimy,buf.data(),Buffer2D::ReadOnly);
}

Buffer3D FunctionHandler3D::discretize_func(int dimx, int dimy,int dimz, real dh, const BorderHandler3D & bord)
{
	TridimArray<real> buf (dimx,dimy,dimz);

	for (int i=0;i < dimx;++i) for (int j=0;j < dimy;++j) for (int k=0;k < dimz;++k)
		switch (bord.cellType(i,j,k,dimx,dimy,dimz))
		{
			case BorderHandler3D::CellInner:
				buf(i,j,k)= m_pFunc( (real)(i)/(dimx-1), (real)(j)/(dimy-1),(real)(k)/(dimz-1) )*dh*dh;
				break;
			case BorderHandler3D::CellOuter:
				break;
			case BorderHandler3D::CellDirichlet:
				buf(i,j,k) = m_pBord( (real)(i)/(dimx-1), (real)(j)/(dimy-1),(real)(k)/(dimz-1) );
				break;
			case BorderHandler3D::CellNeumann:
				buf(i,j,k) = m_pBord( (real)(i)/(dimx-1), (real)(j)/(dimy-1),(real)(k)/(dimz-1) )*dh;
				break;
		}
	return Buffer3D(dimx,dimy,dimz,buf.data(),Buffer3D::ReadOnly);
}

Buffer3D FunctionHandler3D::discretize_sol(int dimx, int dimy,int dimz, real dh, const BorderHandler3D& bord)
{
	TridimArray<real> buf (dimx,dimy,dimz);

	for (int i=0;i < dimx;++i) for (int j=0;j < dimy;++j) for (int k=0;k < dimz;++k)
		switch (bord.cellType(i,j,k,dimx,dimy,dimz))
		{
			case BorderHandler3D::CellOuter:
				buf(i,j,k) = 0;
				break;
			default:
				buf(i,j,k) = m_pSol( (real)(i)/(dimx-1), (real)(j)/(dimy-1),(real)(k)/(dimz-1) );
				break;
		}
	return Buffer3D(dimx,dimy,dimz,buf.data(),Buffer3D::ReadOnly);
}
