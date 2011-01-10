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

std::ostream& operator<<(std::ostream & os,const boost::numeric::ublas::matrix<float> & m);

class Buffer2D {
public:
	enum WriteMode {
		WriteOnly = CL_MEM_WRITE_ONLY,
		ReadOnly = CL_MEM_READ_ONLY,
		ReadWrite = CL_MEM_READ_WRITE
	};

	Buffer2D(int w,int h,WriteMode m = ReadWrite) : m_dimx(w),m_dimy(h),
		m_data(CLContextLoader::getContext(),
			   m,sizeof(float)*m_dimx*m_dimy) {}

	Buffer2D(int w,int h,float * f,WriteMode m = ReadWrite) : m_dimx(w),m_dimy(h),
		m_data(CLContextLoader::getContext(),
			   m | CL_MEM_COPY_HOST_PTR,sizeof(float)*m_dimx*m_dimy,f) {}

	Buffer2D(const Buffer2D & r) : m_dimx(r.m_dimx),m_dimy(r.m_dimy),m_data(r.m_data)
	{
	}

	Buffer2D& operator=(const Buffer2D & r)
	{
		m_dimx = r.m_dimx;
		m_dimy = r.m_dimy;
		m_data = r.m_data;
	}

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

	operator boost::numeric::ublas::matrix<float>() const
	{
		boost::numeric::ublas::matrix<float> ans (m_dimy,m_dimx);

		CLContextLoader::getQueue().enqueueBarrier();
		CLContextLoader::getQueue().enqueueReadBuffer(m_data,true,0,sizeof(float)*m_dimx*m_dimy, &ans.data()[0]);
		return ans;
	}

	struct buffer_2d {
		cl_int dimx;
		cl_int dimy;
		cl_mem data;
	};

	cl_mem operator()() const {
		return m_data();
	}

	int width() const {return m_dimx;}
	int height() const {return m_dimy;}
    cl_int2 size() const {return cl_int2 {m_dimx,m_dimy};}

private:
	int m_dimx;
	int m_dimy;
	cl::Buffer m_data;
};

#endif // BUFFER_H
