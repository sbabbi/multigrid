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

#include "rectangularborderhandler.h"
#include "auxiliary.h"
#include <limits>

typedef struct tagCell {
	real2 _normals; //MUST BE Manhattan-Normalized
}Cell;

BorderHandler::CellType RectangularBorderHandler::cellType(int x, int y, int dimx, int dimy) const
{
	if (x == 0 || y == 0 || x == dimx-1 || y == dimy-1) return CellDirichlet;
	if (x < 0 || y < 0 || x >= dimx || y >= dimy) return CellOuter;
	return CellInner;
}


void RectangularBorderHandler::setarg(int arg,cl::Kernel& ker, int dimx, int dimy)
{
	if (m_bufferMap.find(std::make_pair(dimx,dimy)) == m_bufferMap.end())
		genBuffer(dimx,dimy);

	ker.setArg(arg,m_bufferMap.at(std::make_pair(dimx,dimy)) ());
}

void RectangularBorderHandler::genBuffer(int dimx, int dimy)
{
	BidimArray<Cell> buf (dimx,dimy);

	for (int i=0;i < dimx;++i) for (int j=0;j < dimy;++j)
	{
		real2 val = {0,0};
		buf(i,j)._normals = val;
	}

	Cell c;
	real2 val = {std::numeric_limits<real>::quiet_NaN(),1};
	c._normals = val;
	for (int i=0;i < dimx;++i)
		buf(i,0) = buf(i,dimy-1) = c;
	for (int j=0;j < dimy;++j)
		buf(0,j) = buf(dimx-1,j) = c;

	m_bufferMap.insert( std::make_pair (std::make_pair(dimx,dimy),
								cl::Buffer(CLContextLoader::getContext(),
										   CL_MEM_COPY_HOST_PTR|CL_MEM_READ_ONLY,
										   sizeof(Cell)*dimx*dimy,buf.data())));
}
