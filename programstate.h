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

#ifndef PROGRAMSTATE_H
#define PROGRAMSTATE_H

#include "multigridsolver0.h"
#include "rectangularborderhandler.h"
#include "functionhandler.h"

class ProgramState
{
public:
	enum SolverMode {Fmg = 0,Smooth = 1,Multigrid = 2};

	ProgramState(int argc,char ** argv);

	void listenCommand();

private:

	void setdim( std::istream & params);
	void setsmoothsteps( std::istream & params);
	void setmode( std::istream & params);
	void setvcycles( std::istream & params);
	void setomega( std::istream & params);
	void state( std::istream & params);
	void print( std::istream & params);
	void quit( std::istream & params);
	void help( std::istream & params);
	void solve( std::istream & params);
	void save( std::istream & params);
	void reduce(std::istream & params);

	void helpString();

	SolverMode m_curMode;
	int m_dimx,m_dimy;
	int stepA1,stepA2,VCycles;
	real m_omega;
	bool m_bDisplaySolution,m_bDisplayResidual,m_bDisplayError;
	bool m_bProfilingMode;

	Buffer2D m_residual;
	Buffer2D m_error;
	Buffer2D m_solution;
	Buffer2D m_targetFunction;
	RectangularBorderHandler m_handler;
	MultigridSolver0 m_solver;
	FunctionHandler m_funcHandler;

	struct CommandTableEntry
	{
		const char * CmdName;
		void (ProgramState::* Func)(std::istream&);
		const char * Description;
	};

	static CommandTableEntry CommandTable [];
};

#endif // PROGRAMSTATE_H
