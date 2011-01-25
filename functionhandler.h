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

#include "multigridsolver2D.h"
#include "multigridsolver3D.h"

typedef real (*Function2D)(real,real);

class FunctionHandler2D
{
public:
	FunctionHandler2D( Function2D f,
				  Function2D borders,
				  Function2D solution = 0) : m_pFunc(f),
												m_pBord(borders),
												m_pSol(solution)
	{}

	Buffer2D discretize_func(int dimx,int dimy,real dh,const BorderHandler2D & bord);
	Buffer2D discretize_sol(int dimx,int dimy,real dh,const BorderHandler2D & bord);

	BidimArray<real> solution(int dimx,int dimy);

	bool hasSol() const {return m_pSol;}

private:
	Function2D m_pFunc;
	Function2D m_pBord;
	Function2D m_pSol;
};

typedef real (*Function3D)(real,real,real);

class FunctionHandler3D
{
public:
	FunctionHandler3D( Function3D f,
				  Function3D borders,
				  Function3D solution = 0) : m_pFunc(f),
												m_pBord(borders),
												m_pSol(solution)
	{}

	Buffer3D discretize_func(int dimx,int dimy,int dimz,real dh,const BorderHandler3D & bord);
	Buffer3D discretize_sol(int dimx,int dimy,int dimz,real dh,const BorderHandler3D & bord);

	BidimArray<real> solution(int dimx,int dimy,int dimz);

	bool hasSol() const {return m_pSol;}

private:
	Function3D m_pFunc;
	Function3D m_pBord;
	Function3D m_pSol;
};

#endif // FUNCTIONHANDLER_H
