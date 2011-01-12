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

#include "borderdescriptor.h"

bool BorderDescriptor::find_node(int x, int y, int xdimp2, int ydimp2, cl_float2* out, int base)
{
	if (m_tree[base].leaf)
	{
		assert(x == 0 && y == 0 && xdimp2 == 0 && ydimp2 == 0);
		*out = m_tree[base].normal;
		return true;
	}
	int next = child(x,y,std::max(xdimp2,ydimp2),std::max(xdimp2,ydimp2));

	if (m_tree[base].child[next] == -1) return false;

	int new_base_x = xdimp2 < ydimp2 ? xdimp2 : xdimp2-1;
	int new_base_y = ydimp2 < xdimp2 ? ydimp2 : ydimp2-1;

	return find_node(rebase(x,new_base_x),rebase(y,new_base_y),new_base_x,new_base_y,out,m_tree[base].child[next]);
}

void BorderDescriptor::push_node(cl_float2 normal, int x, int y, int xdimp2, int ydimp2, int base)
{
	if (m_tree[base].leaf)
	{
		assert(x == 0 && y == 0 && xdimp2 == 0 && ydimp2 == 0);
		m_tree[base].normal = normal;
		return;
	}

	int next = child(x,y,std::max(xdimp2,ydimp2),std::max(xdimp2,ydimp2));

	if (m_tree[base].child[next] == -1) //need to add the new node
	{
		int index = m_tree.size();
		Node n;

		if (xdimp2-1 == 0)
		{
			assert(ydimp2-1 == 0);
			n.leaf = true;
			n.normal = normal;
			m_tree[base].child[next] = index;
			m_tree.push_back(n);
		}
		else
		{
			assert(ydimp2-1 != 0);
			n.leaf = false;
			n.child = {-1,-1,-1,-1};
			m_tree[base].child[next] = index;
			m_tree.push_back(n);
			int new_base_x = xdimp2 < ydimp2 ? xdimp2 : xdimp2-1;
			int new_base_y = ydimp2 < xdimp2 ? ydimp2 : ydimp2-1;
			push_node(normal, rebase(x,new_base_x),rebase(y,new_base_y),new_base_x,new_base_y,index);
		}
	}
	else
	{
		if (m_tree[base].leaf)
			m_tree[base].normal = normal;
		else
		{
			int new_base_x = xdimp2 < ydimp2 ? xdimp2 : xdimp2-1;
			int new_base_y = ydimp2 < xdimp2 ? ydimp2 : ydimp2-1;
			push_node(normal, rebase(x,new_base_x),rebase(y,new_base_y),new_base_x,new_base_y,m_tree[base].child[next]);
		}
	}
}

void BorderDescriptor::print(std::ostream& os, int node, int level) const
{
	for (int i=0;i < level;++i) os <<'\t';
	os <<  "Node: " << node << std::endl;
	if (m_tree[node].leaf)
	{
		for (int i=0;i < level;++i) os<<'\t';
		os << "Normal: " << m_tree[node].normal.x << " " << m_tree[node].normal.y << std::endl;
	}
	else
		for (int i=0;i < 4;++i)
			if (m_tree[node].child[i] != -1)
				print(os,m_tree[node].child[i],level+1);
}

BorderDescriptor BorderDescriptor::make_rectangle(int p2x, int p2y)
{
	BorderDescriptor ans (p2x,p2y);

	cl_float2 bot = {0,-1};
	cl_float2 top = {0,1};
	cl_float2 left = {1,0};
	cl_float2 right = {-1,0};
	for (int i=0; i< (1 << p2x);++i)
		ans.push(bot,i,0);
	for (int i=0; i< (1 << p2x);++i)
		ans.push(top,i, (1 << p2y)-1);
	for (int i=0;i < (1 << p2y);++i)
		ans.push(left,0,i);
	for (int i=0;i < (1 << p2y);++i)
		ans.push(right,(1 << p2x)-1,i);
	return ans;
}
