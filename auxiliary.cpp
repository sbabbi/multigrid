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

#include "auxiliary.h"
#include "multi_array.h"
#include <cmath>
#include <fstream>
#include <map>
#include <stdexcept>
#include <set>

#ifdef _WIN32
#include <cstdint>
#endif //_WIN32

std::vector<size_t> factor(int n)
{
	std::vector<size_t> ans;
	ans.push_back(n);
	for (size_t i = n/2 ;i >= 1;--i)
		if (n % i == 0)
			ans.push_back(i);
		return ans;
}

std::vector<size_t> maximize(std::vector< std::vector<size_t> >::iterator init,
							 std::vector< std::vector<size_t> >::iterator end,
							 size_t max)
{
	if ((end-init) == 1)
	{
		for (int i=0;i < init->size();++i)
			if ( (*init)[i] <= max) return std::vector<size_t>(1,(*init)[i]);
		return std::vector<size_t> ();
	}

	for (int i=0;i < init->size();++i)
	{
		size_t val = (*init)[i];
		if (val > max) continue;

		std::vector<size_t> r = maximize(init+1,end,max/val);
		if (r.empty()) continue;

		std::vector<size_t> ans (1,val);
		std::copy(r.begin(),r.end(),std::back_inserter(ans));
		return ans;
	}
	return std::vector<size_t>();
}

cl::NDRange getBestWorkspaceDim(cl::NDRange wsDim)
{
	static std::vector<size_t> MaxDims = CLContextLoader::getDevice().getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

	static size_t totMax = CLContextLoader::getDevice().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

	typedef std::map<cl::NDRange,cl::NDRange> memo_map;
	static memo_map memoing;

	memo_map::iterator res = memoing.find(wsDim);
	if ( res != memoing.end()) return res->second;

	std::vector<std::vector<size_t> > v (wsDim.dimensions());
	for (int i=0;i < wsDim.dimensions();++i)
	{
		std::vector<size_t> s = factor(wsDim[i]);
		s.erase( std::upper_bound(s.begin(),s.end(),MaxDims[i]),s.end());
		v[i] = s;
	}

	std::vector<size_t> dims = maximize(v.begin(),v.end(),totMax);

	//OLD algorithm
	/*
	std::vector<size_t> dims (wsDim.dimensions());

	std::transform(static_cast<const size_t*>(wsDim),static_cast<const size_t*>(wsDim)+wsDim.dimensions(),
				MaxDims.begin(),dims.begin(),std::min<size_t>);


	int prod  = 1;
	int cnt = 0;

	for (size_t i=0;i < dims.size();++i) prod*=dims[i];

	while (prod > totMax)
	{
		dims[ (cnt++)%dims.size()]/=2;
		prod /=2 ;
	}
*/
	switch (dims.size())
	{
	case 1: memoing.insert(std::make_pair(wsDim,cl::NDRange(dims[0]) ) );return cl::NDRange(dims[0]);
	case 2: memoing.insert(std::make_pair(wsDim,cl::NDRange(dims[0],dims[1]) ) );return cl::NDRange(dims[0],dims[1]);
	case 3: memoing.insert(std::make_pair(wsDim,cl::NDRange(dims[0],dims[1],dims[2]) ) );return cl::NDRange(dims[0],dims[1],dims[2]);
	}
	return cl::NullRange;
}

cl::Buffer performReduction(const cl::Buffer & in,cl::Kernel & ker,cl::CommandQueue & q,int size)
{
	if (size == 1) return in;

	int newsize = std::max(1,size/4);
	cl::Buffer tmp (CLContextLoader::getContext(),CL_MEM_READ_WRITE,sizeof(real)*newsize);

	ker.setArg(0,in());
	ker.setArg(1,tmp());
	ker.setArg(2,size);
	ker.setArg(3,4);

	q.enqueueNDRangeKernel(ker,cl::NDRange(0),cl::NDRange(newsize),
												 getBestWorkspaceDim(cl::NDRange(newsize)));

	return performReduction(tmp,ker,q,newsize);
}

real L2Norm(const Buffer2D & in,cl::CommandQueue & q)
{
	cl::Buffer ans (CLContextLoader::getContext(),CL_MEM_READ_WRITE,sizeof(real)*in.width()*in.height());

	CLContextLoader::getRedL2NormKer().setArg(0,in());
	CLContextLoader::getRedL2NormKer().setArg(1,ans());

	q.enqueueNDRangeKernel(CLContextLoader::getRedL2NormKer(),
					cl::NDRange(0),
					cl::NDRange(in.width()*in.height()),
					getBestWorkspaceDim(cl::NDRange(in.width()*in.height())));

	ans = performReduction(ans,CLContextLoader::getRedSumAllKer(),q,in.width()*in.height());

	real res;
	q.enqueueReadBuffer(ans,true,0,sizeof(real),&res);
	return sqrt(res);
}

real L2Norm(const Buffer3D & in,cl::CommandQueue & q)
{
	cl::Buffer ans (CLContextLoader::getContext(),CL_MEM_READ_WRITE,sizeof(real)*in.width()*in.height()*in.depth());

	CLContextLoader::getRedL2NormKer().setArg(0,in());
	CLContextLoader::getRedL2NormKer().setArg(1,ans());

	q.enqueueNDRangeKernel(CLContextLoader::getRedL2NormKer(),
					cl::NDRange(0),
					cl::NDRange(in.width()*in.height()*in.depth()),
					getBestWorkspaceDim(cl::NDRange(in.width()*in.height()*in.depth())));

	ans = performReduction(ans,CLContextLoader::getRedSumAllKer(),q,in.width()*in.height()*in.depth());

	real res;
	q.enqueueReadBuffer(ans,true,0,sizeof(real),&res);
	return sqrt(res);
}

#ifdef _WIN32
#define PACKED
#else
#define PACKED __attribute__ ((packed))
#endif //_WIN32

#pragma pack(push, 1)

struct BitmapFileHeader
{
	uint16_t Signature;
	uint32_t Size;
	uint16_t Reserved1;
	uint16_t Reserved2;
	uint32_t BitsOffset;
} PACKED fileHeader;

struct BitmapInfoHeader
{
	uint32_t HeaderSize;
	int32_t Width;
	int32_t Height;
	uint16_t Planes;
	uint16_t BitCount;
	uint32_t Compression;
	uint32_t SizeImage;
	int32_t PelsPerMeterX;
	int32_t PelsPerMeterY;
	uint32_t ClrUsed;
	uint32_t ClrImportant;
} PACKED infoHeader;

#pragma pack(pop)

Buffer2D fromBitmap(const char* filename)
{
	std::ifstream in (filename,std::ios::binary);
	if (!in) throw std::runtime_error( std::string(filename)+ " Does not exists");

	std::streampos init = in.tellg();

	BitmapFileHeader fileHeader;
	BitmapInfoHeader infoHeader;

	assert(sizeof(BitmapFileHeader) == 14);
	assert(sizeof(BitmapInfoHeader) == 40);

	const uint16_t BitmapSignature = 19778;

	in.read( reinterpret_cast<char*>(&fileHeader),sizeof(fileHeader) );
	in.read( reinterpret_cast<char*>(&infoHeader),sizeof(infoHeader) );
	in.ignore( 1 << infoHeader.BitCount); //Ignore palette

	//unsupported
	if (infoHeader.Compression != 0) throw std::runtime_error("UNSUPPORTED: Bmp compression");
	if (infoHeader.Height < 0) throw std::runtime_error("UNSUPPORTED: Bmp negative height");

	if(in.fail() || BitmapSignature != fileHeader.Signature) throw std::runtime_error(std::string());

	if (infoHeader.BitCount != 8) throw std::runtime_error("Bitmap not monochrome");

	int w = infoHeader.Width;
	int h = infoHeader.Height;
	in.seekg(fileHeader.BitsOffset+init,std::ios::beg);

	BidimArray<char> buf (w,h);
	in.read(buf.data(),sizeof(char)*w*h);

	BidimArray<real> ans (w,h);
	for (int i=0;i < h ;++i) for (int j=0;j < w;++j)
		ans(j,i) = static_cast<real>(buf(j,i))/255.0;
	return Buffer2D(w,h,ans.data());
}

void toBitmap(const Buffer2D& in,cl::CommandQueue & q, const char* filename)
{
	std::ofstream out(filename,std::ios::binary);
	if (!out) throw std::runtime_error(std::string(filename) + " does not exists");

	BidimArray<real> ans = in.read(q);

	BitmapFileHeader fileHeader;
	BitmapInfoHeader infoHeader;

	assert(sizeof(BitmapFileHeader) == 14);
	assert(sizeof(BitmapInfoHeader) == 40);

	int dimx = in.width() + (4-in.width()%4)%4;

	fileHeader.Signature = 19778;
	fileHeader.Size = sizeof(BitmapFileHeader)+sizeof(BitmapInfoHeader)+ sizeof(char)*dimx*in.height()+256*4;
	fileHeader.Reserved1 = fileHeader.Reserved2 = 0;
	fileHeader.BitsOffset = sizeof(BitmapFileHeader)+sizeof(BitmapInfoHeader)+256*4;

	infoHeader.HeaderSize = sizeof(BitmapInfoHeader);
	infoHeader.Width = in.width();
	infoHeader.Height = in.height();
	infoHeader.Planes = 1;
	infoHeader.BitCount = 8;
	infoHeader.Compression = 0;
	infoHeader.SizeImage = dimx*in.height();
	infoHeader.PelsPerMeterX = 1;
	infoHeader.PelsPerMeterY = 1;
	infoHeader.ClrUsed = 256;
	infoHeader.ClrImportant = 256;

	out.write(reinterpret_cast<char*>(&fileHeader),sizeof(BitmapFileHeader));
	out.write(reinterpret_cast<char*>(&infoHeader),sizeof(BitmapInfoHeader));

	for (int i=0;i < 256;++i)
	{
		char b[4] = {i,i,i,i};
		out.write(b,sizeof(b));
	}

	BidimArray<char> buf (dimx,in.height());

	real l_norm = LInfNorm(in,q);


	for (int j=0;j < buf.height();++j)
	{
		for (int i=0;i < ans.width();++i)
			buf(i,j) =  255*fabs(ans(i,j))/l_norm;
		for (int i= ans.width();i < buf.width();++i)
			buf(i,j) = 0;
	}
	out.write(buf.data(), sizeof(char)*buf.height()*buf.width());
}
