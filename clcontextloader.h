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

#ifndef CLCONTEXTLOADER_H
#define CLCONTEXTLOADER_H

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

class CLContextLoader
{
public:
	static cl::Context& getContext() {return instance().m_context;}
	static cl::CommandQueue& getQueue() {return instance().m_queue1;}
	static cl::Device & getDevice() {return instance().m_device;}
	static cl::Kernel& getZeroMemKer() {return instance().m_ZeroMemory;}
	static cl::Program loadProgram(const char * filename) {return instance().loadProgramFromFile(filename);}

protected:

	static CLContextLoader& instance() {
		static CLContextLoader inst;
		return inst;
	}

	CLContextLoader();

	cl::Program loadProgramFromFile(const char * filename);

	cl::Platform m_platform;
	cl::Device m_device;
	cl::Context m_context;
	cl::CommandQueue m_queue1;

	cl::Program m_commonKernelsProg;
	cl::Kernel m_ZeroMemory;

};

#endif // CLCONTEXTLOADER_H
