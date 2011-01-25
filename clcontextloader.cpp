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

#include "clcontextloader.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>


CLContextLoader::CLContextLoader()
{
	//Load platform
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	if (platforms.size() == 0) throw std::runtime_error("No platforms found");
	m_platform = platforms[0];

	//Load device
	std::vector<cl::Device> devices;
	m_platform.getDevices(CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU,&devices);

	if (devices.size() == 0) throw std::runtime_error("No devices found");
	m_device = devices[0];

	cl_context_properties propp [] = {
		CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(m_platform()),
		0
	};

	//Initialize the context
	m_context = cl::Context(devices,propp);

	const std::string strFilename = "common_kernels.cl";

	m_commonKernelsProg = loadProgramFromFile(strFilename.c_str());

	//Initialize the kernels
	m_ZeroMemory = cl::Kernel(m_commonKernelsProg,"zero_memory");
	m_RedSumAll = cl::Kernel(m_commonKernelsProg,"SumAll");
	m_RedL2Norm = cl::Kernel(m_commonKernelsProg,"L2Norm");
	m_MultKer = cl::Kernel(m_commonKernelsProg,"Mult");
	m_RedLInfKer = cl::Kernel(m_commonKernelsProg,"LInfNorm");
	m_DiffKer = cl::Kernel(m_commonKernelsProg,"Diff");
}

cl::Program CLContextLoader::loadProgramFromFile(const char * filename)
{
	//Load the program source file
	std::ifstream inputFile(filename);
	if (!inputFile) throw std::runtime_error( std::string("File: ") + filename + std::string(" does not exists"));

	int buffer_dim = std::distance( std::istreambuf_iterator<char>(inputFile),std::istreambuf_iterator<char>()) + 1;
	inputFile.seekg( std::ios_base::beg);

	//Load the source code
	std::string sourceCode;
	sourceCode.reserve(buffer_dim);
	std::copy(std::istreambuf_iterator<char>(inputFile),std::istreambuf_iterator<char>(), std::back_inserter(sourceCode));

	//Store the souce code in the source struct
	cl::Program::Sources inputSource(1);
	inputSource[0].first = sourceCode.c_str();
	inputSource[0].second = 0;

	cl::Program ans (m_context,inputSource);

	std::vector<cl::Device> devices (1,m_device);
	try {
		std::string options = ""; //-cl-strict-aliasing -cl-mad-enable -cl-fast-relaxed-math";

#ifdef USE_DOUBLE
		options+=" -DUSE_DOUBLE");
#endif //USE_DOUBLE
		ans.build(devices,options.c_str());
	}
	catch (cl::Error e)
	{
		if (e.err() == CL_BUILD_PROGRAM_FAILURE || e.err() == CL_INVALID_BINARY)
			throw std::runtime_error(ans.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device));
		throw;
	}
	return ans;
}
