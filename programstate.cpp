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
#include "programstate.h"
#include <iostream>
#include <iomanip>

using namespace std;

float ones(float,float) {return 1;}
float zeros(float,float) {return 0;}

std::ostream& operator<<(std::ostream & os,const boost::multi_array<float,2> & m)
{
	for (int i=0;i < m.shape()[0];++i)
	{
		for (int j=0;j < m.shape()[1];++j)
			os << m[i][j] << " ";
		os << std::endl;
	}
	return os;
}

ProgramState::ProgramState(int argc, char** argv) :
	m_curMode(Fmg),
	m_dimx(16),
	m_dimy(16),
	stepA1(3),
	stepA2(3),
	VCycles(2),
	m_omega(1.0),
	m_bDisplaySolution(false),
	m_bDisplayResidual(false),
	m_bDisplayError(false),
	m_solver("mg_0.cl",m_handler),
	m_funcHandler(zeros,ones,ones)
{
	for (int i=0;i < argc;++i)
	{
		if (string(argv[i]) == "--solver")
		{
			++i;
			if (string(argv[i]) == "fmg")
				m_curMode = Fmg;
			else if (string(argv[i]) == "jac")
				m_curMode = Smooth;
			else if (string(argv[i]) == "mg")
				m_curMode = Multigrid;
			else
			{
				cout << argv[i] << "is not a valid solver type" << endl;
				exit(1);
			}
		}
		else if(string(argv[i]) == "--dim")
		{
			m_dimx = atoi(argv[++i]);
			m_dimy = atoi(argv[++i]);
			if (m_dimx <=0 || m_dimy >= 0)
			{
				cout << "Dimensions not valid" << endl;
				exit(1);
			}
		}
		else if(string(argv[i]) == "--smoothstep")
		{
			stepA1 = atoi(argv[++i]);
			stepA2 = atoi(argv[++i]);
			if (stepA1 < 0 || stepA2 < 0)
			{
				cout << "Can not do less than 0 smooth steps" << endl;
				exit(1);
			}
		}
		else if(string(argv[i]) == "--mgcycles")
		{
			VCycles = atoi(argv[++i]);
			if (VCycles < 0)
			{
				cout << "Can not do less than 0 V-Cycles" << endl;
				exit(1);
			}
		}
		else if(string(argv[i]) == "--displaysol")
			m_bDisplaySolution = true;
		else if(string(argv[i]) == "--displayres")
			m_bDisplayResidual = true;
		else if(string(argv[i]) == "--displayerr")
			m_bDisplayError = true;
	}
}

void ProgramState::listenCommand()
{
	while (1)
	{
		cout << ">";

		std::string cmd;
		cin >> cmd;

		if (cmd == "solve")
			solve();
		else if (cmd == "quit")
			return;
		else if (cmd == "print")
		{
			string what;
			cin >> what;
			if (cin.fail()) cout << "print what?" << endl;
			else if (what == "sol")
			{
				if (m_solution.isInitialized())
					cout << m_solution.read( m_solver.queue()) << endl;
				else
					cout << "No solution available" << endl;
			} else if (what == "res")
			{
				if (m_residual.isInitialized())
					cout << m_residual.read( m_solver.queue()) << endl;
				else
					cout << "No residuals available" << endl;
			} else if (what == "err")
			{
				if (m_error.isInitialized())
					cout << m_error.read( m_solver.queue()) << endl;
				else
					cout << "No error available" << endl;
			}
		}
		else if (cmd == "setdim")
		{
			int newdimx,newdimy;
			cin >> newdimx >> newdimy;
			if (cin.fail() || newdimx < 0 || newdimy < 0)
				cout << "Invalid dimensions" << endl;
			else
				m_dimx = newdimx,m_dimy = newdimy;
		}
	}
}

void ProgramState::solve()
{
	Buffer2D targetFunction = m_funcHandler.discretize(m_dimx,m_dimy,0.1,m_handler);
	Buffer2D emptyBuf = Buffer2D::empty(m_dimx,m_dimy,m_solver.queue());
	m_solution = Buffer2D::empty(m_dimx,m_dimy,m_solver.queue());
	m_solver.queue().enqueueBarrier();

	clock_t startTimer = clock();

	switch (m_curMode)
	{
	case Fmg:
		m_solver.fmg(targetFunction,
						m_omega,
						stepA1,
						stepA2,
						VCycles);
		break;
	case Smooth:
		m_solver.smoother_iterate(m_solution,
								  emptyBuf,
								  targetFunction,
								  m_omega,
								  stepA1);
		break;
	case Multigrid:
		m_solver.iterate(m_solution,
						 targetFunction,
						 m_omega,
						 stepA1,
						 stepA2,
						 VCycles);
		break;
		
	}
	m_solver.queue().enqueueBarrier();
	m_solver.compute_residuals(m_residual,m_solution,targetFunction);
	m_solver.wait();

	clock_t endTimer = clock();

	cout << fixed << setprecision(5);
	cout << "Time\t\t\tL2Err\t\t\tLInfErr\t\t\tL2Res\t\t\tLinfRes\t\t\t" << endl;
	cout << (double)(endTimer-startTimer)/CLOCKS_PER_SEC << "\t\t\t" <<
		m_funcHandler.L2Error(m_solution,m_solver.queue()) << "\t\t\t" <<
		m_funcHandler.LInfError(m_solution,m_solver.queue()) << "\t\t\t" <<
		L2Norm(m_residual,m_solver.queue()) << "\t\t\t" <<
		LInfNorm(m_residual,m_solver.queue()) << endl;
}
