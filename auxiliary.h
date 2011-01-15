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

Buffer2D perform2DReduction(Buffer2D & in,cl::Kernel & ker,cl::CommandQueue & q,int xtill = 1,int ytill = 1);
cl::NDRange getBestWorkspaceDim(cl::NDRange wsDim);

inline real L2Norm(Buffer2D & in,cl::CommandQueue & q)
{
	return perform2DReduction(in,CLContextLoader::getRedL2NormKer(),q).read(q)[0][0];
}

inline real LInfNorm(Buffer2D & in,cl::CommandQueue & q)
{
	return perform2DReduction(in,CLContextLoader::getRedLInfKer(),q).read(q)[0][0];
}

inline real Average(Buffer2D & in,cl::CommandQueue & q)
{
	return perform2DReduction(in,CLContextLoader::getRedSumAllKer(),q).read(q)[0][0] / (in.width()*in.height());
}

#endif //AUXILIARY_H
