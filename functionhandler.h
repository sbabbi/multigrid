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

#ifndef FUNCTIONHANDLER_H
#define FUNCTIONHANDLER_H

#include "multigridsolver0.h"

typedef real (*Function2D)(real,real);

class FunctionHandler
{
public:
	FunctionHandler( Function2D f,
				  Function2D borders,
				  Function2D solution = 0) : m_pFunc(f),
												m_pBord(borders),
												m_pSol(solution)
	{}

	Buffer2D discretize(int dimx,int dimy,real dh,const BorderHandler & bord);
	real L2Error(Buffer2D& ans, cl::CommandQueue& q);
	real LInfError(Buffer2D & ans, cl::CommandQueue& q);

	boost::multi_array<real,2> solution(int dimx,int dimy);

private:
	Function2D m_pFunc;
	Function2D m_pBord;
	Function2D m_pSol;
};

#endif // FUNCTIONHANDLER_H
