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

typedef struct tagCell {
	real2 _normals; //MUST BE Manhattan-Normalized
}Cell;

BorderHandler::CellType RectangularBorderHandler::cellType(int x, int y, int dimx, int dimy) const
{
	if (x == 0 || y == 0 || x == dimx-1 || y == dimy-1) return CellDirichlet;
	if (x < 0 || y < 0 || x >= dimx || y >= dimy) return CellOuter;
	return CellInner;
}


void RectangularBorderHandler::compute(cl::CommandQueue& queue, cl::Kernel& ker, int dimx, int dimy,int bord_dimx,int bord_dimy)
{
	if (m_bufferMap.find(std::make_pair(bord_dimx,bord_dimy)) == m_bufferMap.end())
		genBuffer(bord_dimx,bord_dimy);

	ker.setArg(0,m_bufferMap.at(std::make_pair(bord_dimx,bord_dimy)) ());

	cl::NDRange wsDim = getBestWorkspaceDim(cl::NDRange(dimx,dimy));

	queue.enqueueNDRangeKernel(ker,cl::NDRange(0,0),cl::NDRange(dimx,dimy),wsDim);
}

void RectangularBorderHandler::genBuffer(int dimx, int dimy)
{
	boost::multi_array<Cell,2> buf (boost::extents[dimy][dimx]);

	for (int i=0;i < dimx;++i) for (int j=0;j < dimy;++j)
	{
		real2 val = {0,0};
		buf[j][i]._normals = val;
	}

	Cell c;
	real2 val = {std::numeric_limits<real>::quiet_NaN(),1};
	c._normals = val;
	for (int i=0;i < dimx;++i)
		buf[0][i] = buf[dimy-1][i] = c;
	for (int j=0;j < dimy;++j)
		buf[j][0] = buf[j][dimx-1] = c;

	m_bufferMap.insert( std::make_pair (std::make_pair(dimx,dimy),
								cl::Buffer(CLContextLoader::getContext(),
										   CL_MEM_COPY_HOST_PTR|CL_MEM_READ_ONLY,
										   sizeof(Cell)*dimx*dimy,buf.data())));
}
