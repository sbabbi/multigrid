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

#ifndef MULTI_ARRAY_H
#define MULTI_ARRAY_H

#include <algorithm>
#include <cstddef>

template<class _T>
class BidimArray
{
public:
	BidimArray(int w,int h) : m_dimx(w),m_dimy(h),_data(new _T[w*h]) {}
	~BidimArray() {delete [] _data;}

	_T * data() {return _data;}
	const _T * data() const {return _data;}

	_T& operator()(int x,int y) {return _data [x + y*m_dimx];}
	const _T& operator()(int x,int y) const {return _data[x + y*m_dimx];}

	BidimArray(const BidimArray & r);

	BidimArray& operator=(const BidimArray & r);

	void swap(BidimArray & r)
	{
		std::swap(m_dimx,r.m_dimx);
		std::swap(m_dimy,r.m_dimy);
		std::swap(_data,r._data);
	}

	int width() const {return m_dimx;}
	int height() const {return m_dimy;}

private:
	int m_dimx;
	int m_dimy;
	_T * _data;
};

namespace std
{
	template<class _T>
	void swap(BidimArray<_T> & a,BidimArray<_T> & b) {a.swap(b);}
}

template<class _T>
class TridimArray
{
public:
	TridimArray(int w,int h,int d) : m_dimx(w),m_dimy(h),m_dimz(d),_data(new _T[w*h*d]) {}
	~TridimArray() {delete [] _data;}

	_T * data() {return _data;}
	const _T * data() const {return _data;}

	_T& operator()(int x,int y,int z) {return _data [x + y*m_dimx + z*m_dimx*m_dimy];}
	const _T& operator()(int x,int y,int z) const {return _data[x + y*m_dimx + z*m_dimx*m_dimy];}

	TridimArray(const TridimArray& r);

	TridimArray& operator=(const TridimArray& r);

	void swap(TridimArray & r)
	{
		std::swap(m_dimx,r.m_dimx);
		std::swap(m_dimy,r.m_dimy);
		std::swap(_data,r._data);
	}

	int width() const {return m_dimx;}
	int height() const {return m_dimy;}
	int depth() const {return m_dimz;}

private:
	int m_dimx;
	int m_dimy;
	int m_dimz;
	_T * _data;
};

namespace std
{
	template<class _T>
	void swap(TridimArray<_T> & a,TridimArray<_T> & b) {a.swap(b);}
}

#endif // MULTI_ARRAY_H
