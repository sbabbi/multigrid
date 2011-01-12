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

#ifndef BORDERDESCRIPTOR_H
#define BORDERDESCRIPTOR_H

#include "buffer.h"
#include <iostream>

class BorderDescriptor
{
public:
	BorderDescriptor(int dimxp2,int dimyp2) : m_dimxPow(dimxp2),m_dimyPow(dimyp2) {
		Node base;
		base.child = {-1,-1,-1,-1};
		base.leaf = false;
		m_tree.push_back(base);
	}

	static BorderDescriptor make_rectangle(int px,int py);

	void push(cl_float2 normal,int x,int y)
	{
		push_node (normal,x,y,m_dimxPow,m_dimyPow);
	}

	bool find (int x,int y,cl_float2 * out)
	{
		return find_node(x,y,m_dimxPow,m_dimyPow,out,0);
	}

	cl::Buffer operator()() const {
		return cl::Buffer(CLContextLoader::getContext(),CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
						  sizeof(Node)*m_tree.size(),
						  const_cast<Node*>(m_tree.data()));
	}

	friend std::ostream& operator<<(std::ostream & os,const BorderDescriptor &);

private:
	bool find_node(int x,int y,int xdimp2,int ydimp2,cl_float2 * out,int base = 0);
	void push_node(cl_float2 normal,int x,int y,int xdimp2,int ydimp2,int base = 0);
	void print(std::ostream & os,int node = 0,int level = 0) const;

	static int rebase(int coord,int dimp2)
	{
		return coord % ( 1 << (dimp2));
	}

	static int child(int x,int y,int xdimp2,int ydimp2)
	{
		return (x >= (1 << (xdimp2-1))) + 2*( (y >= (1 << (ydimp2-1))) );
	}

	struct Node {  union {cl_int child[4]; cl_float2 normal;}; bool leaf;};

	std::vector<Node> m_tree;
	int m_dimxPow, m_dimyPow;
};

inline std::ostream& operator<<(std::ostream & os,const BorderDescriptor & b)
{
	b.print(os);
	return os;
}

#endif // BORDERDESCRIPTOR_H
