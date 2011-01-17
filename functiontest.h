/*
    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef FUNCTIONTEST_H
#define FUNCTIONTEST_H

#include "buffer.h"

double L2Norm(Buffer2D & val);
double LInfNorm(Buffer2D & val);

class FunctionTest
{
public:
	FunctionTest( float(*p)(float,float),
				  float(*bord)(float,float),
				  float(*sol)(float,float) = 0) : m_pFunc(p),
												m_pBord(bord),
												m_pSol(sol)
	{}

	Buffer2D makeBuffer(int dimx,int dimy,float dh);
	double L2Error(Buffer2D & ans);
	double LInfError(Buffer2D & ans);

	boost::numeric::ublas::matrix<float> solution(int dimx,int dimy,float dh);

private:
	float (*m_pFunc)(float,float);
	float (*m_pBord)(float,float);
	float (*m_pSol)(float,float);
};

#endif // FUNCTIONTEST_H
