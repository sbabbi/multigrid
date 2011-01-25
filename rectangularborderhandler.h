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

#ifndef RECTANGULARBORDERHANDLER_H
#define RECTANGULARBORDERHANDLER_H

#include "multigridsolver2D.h"
#include "multigridsolver3D.h"
#include <map>

class RectangularBorderHandler : public BorderHandler2D
{
public:
	virtual void setarg(int arg, cl::Kernel& ker, int dimx, int dimy);
	virtual CellType cellType(int x,int y,int dimx,int dimy) const;

private:
	void genBuffer(int dimx,int dimy);
	std::map<std::pair<int,int>,cl::Buffer> m_bufferMap;
};

class ParallelepipedalBorderHandler : public BorderHandler3D
{
public:
	virtual void setarg(int arg, cl::Kernel& ker, int dimx, int dimy,int dimz);
	virtual CellType cellType(int x,int y,int z,int dimx,int dimy,int dimz) const;

private:
	void genBuffer(int dimx,int dimy,int dimz);

	struct tri {
		int x,y,z;
		tri(int _x,int _y,int _z) : x(_x),y(_y),z(_z) {}

		bool operator<(const tri & r) const{
			if (x != r.x) return x < r.x;
			if (y != r.y) return y < r.y;
			return z < r.z;
		}
	};
	std::map<tri,cl::Buffer> m_bufferMap;
};

#endif // RECTANGULARBORDERHANDLER_H
