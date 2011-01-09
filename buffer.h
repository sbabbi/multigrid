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

#ifndef BUFFER_H
#define BUFFER_H

#include "clcontextloader.h"
#include <boost/numeric/ublas/matrix.hpp>

class Buffer2D {
public:
	enum WriteMode {
		WriteOnly = CL_MEM_WRITE_ONLY,
		ReadOnly = CL_MEM_READ_ONLY,
		ReadWrite = CL_MEM_READ_WRITE
	};
	
	Buffer2D(int w,int h,WriteMode m = ReadWrite) : m_dimx(w),m_dimy(h),
		m_data(CLContextLoader::getContext(),
			   m,sizeof(double)*m_dimx*m_dimy) {}

	Buffer2D(int w,int h,double * f,WriteMode m = ReadWrite) : m_dimx(w),m_dimy(h),
		m_data(CLContextLoader::getContext(),
			   m | CL_MEM_COPY_HOST_PTR,sizeof(double)*m_dimx*m_dimy,f) {}

	static Buffer2D empty(int w,int h)
	{
		Buffer2D res(w,h);
		CLContextLoader::getZeroMemKer().setArg(0,res.m_data());

		cl::Event ev;
		CLContextLoader::getQueue().enqueueNDRangeKernel(CLContextLoader::getZeroMemKer(),
														 cl::NDRange(0),
														 cl::NDRange(w*h),
														 cl::NullRange,
														 0,
														 &ev);
		ev.wait();
		return res;
	}

	operator boost::numeric::ublas::matrix<double>() const
	{
		boost::numeric::ublas::matrix<double> ans (m_dimx,m_dimy);

		CLContextLoader::getQueue().enqueueBarrier();
		CLContextLoader::getQueue().enqueueReadBuffer(m_data,true,0,sizeof(double)*m_dimx*m_dimy, &ans.data()[0]);
		return ans;
	}

	struct buffer_2d {
		int dimx;
		int dimy;
		cl_mem data;
	};

	buffer_2d operator()() const {
		buffer_2d ans;
		ans.dimx = m_dimx;
		ans.dimy = m_dimy;
		ans.data = m_data();
		return ans;
	}

	int width() const {return m_dimx;}
	int height() const {return m_dimy;}

private:
	int m_dimx;
	int m_dimy;
	cl::Buffer m_data;
};

#endif // BUFFER_H
