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

	static int totMax = CLContextLoader::getDevice().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

	std::vector<size_t> dims (wsDim.dimensions());

	std::transform(static_cast<const size_t*>(wsDim),static_cast<const size_t*>(wsDim)+wsDim.dimensions(),
				MaxDims.begin(),dims.begin(),std::min<size_t>);


	int prod  = 1;
	int cnt = 0;

	for (int i=0;i < dims.size();++i) prod*=dims[i];

	while (prod > totMax)
	{
		dims[ (cnt++)%dims.size()]/=2;
		prod /=2 ;
	}

	switch (dims.size())
	{
	case 1: return cl::NDRange(dims[0]);
	case 2: return cl::NDRange(dims[0],dims[1]);
	case 3: return cl::NDRange(dims[0],dims[1],dims[2]);
	}
	return cl::NullRange;
}

cl::Buffer performReduction(cl::Buffer & in,cl::Kernel & ker,cl::CommandQueue & q,int size)
{
	if (size == 1) return in;

	int newsize = std::max(1,size/4);
	cl::Buffer tmp (CLContextLoader::getContext(),CL_MEM_READ_WRITE,sizeof(real)*newsize);

	ker.setArg(0,in());
	ker.setArg(1,tmp());
	ker.setArg(2,size);
	ker.setArg(3,4);

	q.enqueueNDRangeKernel(ker,cl::NDRange(0),cl::NDRange(newsize),
												 getBestWorkspaceDim(cl::NDRange(newsize)));
	q.enqueueBarrier();

	return performReduction(tmp,ker,q,newsize);
}

real L2Norm(Buffer2D & in,cl::CommandQueue & q)
{
	cl::Buffer ans (CLContextLoader::getContext(),CL_MEM_READ_WRITE,sizeof(real)*in.width()*in.height());

	CLContextLoader::getRedL2NormKer().setArg(0,in());
	CLContextLoader::getRedL2NormKer().setArg(1,ans());

	q.enqueueNDRangeKernel(CLContextLoader::getRedL2NormKer(),
					cl::NDRange(0),
					cl::NDRange(in.width()*in.height()),
					getBestWorkspaceDim(cl::NDRange(in.width()*in.height())));

	q.enqueueBarrier();

	ans = performReduction(ans,CLContextLoader::getRedSumAllKer(),q,in.width()*in.height());

	real res;
	q.enqueueReadBuffer(ans,true,0,sizeof(real),&res);
	return sqrt(res);
}
