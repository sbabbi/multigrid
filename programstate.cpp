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

float prettyFunc1(float x,float y)
{
	return -2*( (1-6*x*x)*y*y*(1-y*y)+
					(1-6*y*y)*x*x*(1-x*x));
}

float prettyFunc1Sol(float x,float y)
{
	return (x*x-x*x*x*x)*(y*y*y*y-y*y);
}

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
	m_funcHandler(prettyFunc1,zeros,prettyFunc1Sol)
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
		else if(string(argv[i]) == "-h" || string(argv[i]) == "--help")
			helpString();
		else
		{
			cout << "Unkown option " << argv[i] << endl;
			helpString();
			abort();
		}
	}
}

void ProgramState::listenCommand()
{
	while (1)
	{
		cout << fixed << setprecision(5);
		cout << ">";
		cin >> noskipws;

		string input_string;
		std::getline(cin,input_string);
		string cmd;

		istringstream input_parameters (input_string);
		input_parameters >> cmd;

		if (cmd == "solve")
			solve();
		else if (cmd == "quit")
			return;
		else if (cmd == "print")
		{
			string what;
			input_parameters >> what;
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
			} else if (what == "func")
			{
				if (m_targetFunction.isInitialized())
					cout << m_targetFunction.read( m_solver.queue()) << endl;
				else
					cout << "No function available" << endl;
			}
		}
		else if (cmd == "state")
		{
			string solverString;
			switch (m_curMode)
			{
				case Fmg:
					solverString = "FMG";
					break;
				case Smooth:
					solverString = "Jacobi Smoother";
					break;
				case Multigrid:
					solverString = "Multigrid";
					break;
			}

			cout << "Current solver: " << solverString << endl <<
				"Dimension: " << m_dimx <<"x" << m_dimy << endl <<
				"Pre smooth steps: " << stepA1 << " Post smooth steps: " << stepA2 << endl <<
				"VCycles: " << VCycles << " Omega: " << m_omega << endl;
		}
		else if (cmd == "setdim")
		{
			int newdimx,newdimy;
			input_parameters >> newdimx >> newdimy;
			if (input_parameters.fail() || newdimx < 0 || newdimy < 0)
				cout << "Invalid dimensions" << endl;
			else
				m_dimx = newdimx,m_dimy = newdimy;
		}
		else if (cmd == "setmode")
		{
			string mode;
			input_parameters >> mode;
			if (input_parameters.fail())
				cout << "Invalid mode" << endl;
			if (mode == "fmg")
				m_curMode = Fmg;
			else if (mode == "jac")
				m_curMode = Smooth;
			else if (mode == "mg")
				m_curMode = Multigrid;
			else
				cout << "Invalid mode" << endl;
		}
		else if (cmd == "setsmoothsteps")
		{
			int newa1,newa2;
			input_parameters >> newa1 >> newa2;
			if (input_parameters.fail() || newa1 < 0 || newa2 < 0)
				cout << "Invalid parameters" << endl;
			else
				stepA1 = newa1,stepA2 = newa2;
		}
		else if (cmd == "setomega")
		{
			::real omega;
			input_parameters >> omega;
			if (input_parameters.fail() || omega < 0 || omega > 2.0)
				cout << "Invalid omega" << endl;
			else
				m_omega = omega;
		}
		else
		{
			cout << "Unknown command: " << cmd << endl;
		}
	}
}

void ProgramState::solve()
{
	m_targetFunction = m_funcHandler.discretize(m_dimx,m_dimy,1.0/(m_dimx-1),m_handler);
	Buffer2D emptyBuf = Buffer2D::empty(m_dimx,m_dimy,m_solver.queue());
	m_solution = Buffer2D::empty(m_dimx,m_dimy,m_solver.queue());
	m_residual = Buffer2D(m_dimx,m_dimy);
	m_error = Buffer2D(m_dimx,m_dimy);
	m_solver.queue().enqueueBarrier();

	clock_t startTimer = clock();

	switch (m_curMode)
	{
	case Fmg:
		m_solution = m_solver.fmg(m_targetFunction,
						m_omega,
						stepA1,
						stepA2,
						VCycles);
		break;
	case Smooth:
		m_solver.smoother_iterate(m_solution,
								  m_targetFunction,
								  m_omega,
								  stepA1);
		break;
	case Multigrid:
		m_solver.iterate(m_solution,
						 m_targetFunction,
						 m_omega,
						 stepA1,
						 stepA2,
						 VCycles);
		break;

	}
	m_solver.queue().enqueueBarrier();
	m_solver.compute_residuals(m_residual,m_solution,m_targetFunction);
	m_solver.wait();

	clock_t endTimer = clock();

	cout << "Time\t\t\tL2Err\t\t\tLInfErr\t\t\tL2Res\t\t\tLinfRes\t\t\t" << endl;
	cout << (double)(endTimer-startTimer)/CLOCKS_PER_SEC << "\t\t\t" <<
		m_funcHandler.L2Error(m_solution,m_solver.queue()) << "\t\t\t" <<
		m_funcHandler.LInfError(m_solution,m_solver.queue()) << "\t\t\t" <<
		L2Norm(m_residual,m_solver.queue()) << "\t\t\t" <<
		LInfNorm(m_residual,m_solver.queue()) << endl;
}

void ProgramState::helpString()
{
	cout << "Usage: multigrid [options] " << endl;
	cout << "Valid options are: " << endl;
	cout << "\t \"--solver solvertype\": set the solver type. 'fmg' for a Fast multigrid solver, 'jac' for the Jacobi method, 'mg' for the simple multigrid method" << endl
	<< "\t \"--dim dimx dimy\": set the dimension of the grid to be used" << endl
	<< "\t \"--smoothstep a1 a2\": sets the number of steps of pre and post smoothing. For simple jacobi method only the first value is used" << endl
	<< "\t \"--mgcycles v\": sets the number of v-cycles to be used (1= V Cycle, 2 = W Cycle)" << endl
	<< "\t \"--help\" \"-h\": prints this help" << endl
	<< endl;
}
