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

#include "multi_array.h"
#include "clcontextloader.h"

template<class _T>
BidimArray<_T>& BidimArray<_T>::operator=(const BidimArray& r)
{
	if (m_dimx*m_dimy != r.m_dimx*r.m_dimy)
	{
		delete []_data;
		_data = reinterpret_cast<_T*>(new char [sizeof (_T)*r.m_dimx*r.m_dimy]);
		std::uninitialized_copy(r._data,r._data+r.m_dimx*r.m_dimy,_data);
	}
	else
		std::copy(r._data,r._data+r.m_dimx*r.m_dimy,_data);
	m_dimx = r.m_dimx;
	m_dimy = r.m_dimy;
	return *this;
}

template<class _T>
BidimArray<_T>::BidimArray(const BidimArray & r) : m_dimx(r.m_dimx),m_dimy(r.m_dimy),_data(
	reinterpret_cast<_T*>(new char [sizeof(_T)*m_dimx*m_dimy]))
{
	std::uninitialized_copy(r._data,r._data+m_dimx*m_dimy,_data);
}

template<class _T>
TridimArray<_T>& TridimArray<_T>::operator=(const TridimArray& r)
{
	if (m_dimx*m_dimy*m_dimz != r.m_dimx*r.m_dimy*r.m_dimz)
	{
		delete []_data;
		_data = reinterpret_cast<_T*>(new char [sizeof (_T)*r.m_dimx*r.m_dimy*m_dimz]);
		std::uninitialized_copy(r._data,r._data+r.m_dimx*r.m_dimy*m_dimz,_data);
	}
	else
		std::copy(r._data,r._data+r.m_dimx*r.m_dimy*r.m_dimz,_data);
	m_dimx = r.m_dimx;
	m_dimy = r.m_dimy;
	m_dimz = r.m_dimz;
	return *this;
}

template<class _T>
TridimArray<_T>::TridimArray(const TridimArray & r) : m_dimx(r.m_dimx),m_dimy(r.m_dimy),m_dimz(r.m_dimz),
	_data(reinterpret_cast<_T*>(new char [sizeof(_T)*m_dimx*m_dimy*m_dimz]))
{
	std::uninitialized_copy(r._data,r._data+m_dimx*m_dimy*m_dimz,_data);
}

template class BidimArray<real>;
template class TridimArray<real>;
template class BidimArray<char>;
template class TridimArray<char>;
