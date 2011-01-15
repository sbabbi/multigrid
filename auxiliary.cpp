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

#include "auxiliary.h"

cl::NDRange getBestWorkspaceDim(cl::NDRange wsDim)
{
	static std::vector<size_t> MaxDims = CLContextLoader::getDevice().getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

	switch (wsDim.dimensions())
	{
		case 1:
			return cl::NDRange( std::min(wsDim[0],MaxDims[0]) );
		case 2:
			return cl::NDRange( std::min(wsDim[0],MaxDims[0]),std::min(wsDim[1],MaxDims[1]));
		case 3:
			return cl::NDRange( std::min(wsDim[0],MaxDims[0]),std::min(wsDim[1],MaxDims[1]),std::min(wsDim[2],MaxDims[2]));
		default:
			throw std::runtime_error("Wrong dimensions in getBestWorkspaceDim");
	}
}

Buffer2D perform2DReduction(Buffer2D& in, cl::Kernel & ker,cl::CommandQueue & q,int xtill,int ytill )
{
	/*** Reduction kernel arguments ***/
	/* 	__global read_only real * input,
		__global write_only real * output,
		int inputSize,
		int chunks */

	if (in.width() <= xtill && in.height() <= ytill)
		return in;

	Buffer2D aux ( std::max(xtill,in.height()/2),std::max(ytill,in.width()/2));
	ker.setArg(0,in());
	ker.setArg(1,aux());
	ker.setArg(2,in.size());
	ker.setArg(3,4);

	q.enqueueNDRangeKernel(ker,cl::NDRange(0,0),cl::NDRange(aux.width(),aux.height()),
													 getBestWorkspaceDim(cl::NDRange(aux.width(),aux.height())));
	q.enqueueBarrier();

	return perform2DReduction(aux,ker,q,xtill,ytill);
}
