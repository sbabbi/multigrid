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

#include "multigridsolver0.h"
#include <map>

class RectangularBorderHandler : public BorderHandler
{
public:
	virtual void setarg(int arg, cl::Kernel& ker, int dimx, int dimy);
	virtual CellType cellType(int x,int y,int dimx,int dimy) const;

private:
	void genBuffer(int dimx,int dimy);
	std::map<std::pair<int,int>,cl::Buffer> m_bufferMap;
};

#endif // RECTANGULARBORDERHANDLER_H
