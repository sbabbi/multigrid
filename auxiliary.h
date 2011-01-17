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

#ifndef AUXILIARY_H
#define AUXILIARY_H

#include "buffer.h"

cl::Buffer performReduction(const cl::Buffer & in,cl::Kernel & ker,cl::CommandQueue & q,int size);
cl::NDRange getBestWorkspaceDim(cl::NDRange wsDim);

real L2Norm(const Buffer2D & in,cl::CommandQueue & q);

inline real LInfNorm(const Buffer2D & in,cl::CommandQueue & q)
{
	cl::Buffer buf = performReduction(in.data(),
							CLContextLoader::getRedLInfKer(),
							q,
							in.width()*in.height());

	real ans;
	q.enqueueReadBuffer(buf,true,0,sizeof(real),&ans);
	return ans;
}

inline real Average(Buffer2D & in,cl::CommandQueue & q)
{
	cl::Buffer buf = performReduction(in.data(),
							CLContextLoader::getRedSumAllKer(),
							q,
							in.width()*in.height());

	real ans;
	q.enqueueReadBuffer(buf,true,0,sizeof(real),&ans);
	return ans/ (in.width()*in.height());;
}

Buffer2D fromBitmap(const char * filename);
void toBitmap(const Buffer2D & in,cl::CommandQueue & q,const char * filename);

#endif //AUXILIARY_H
