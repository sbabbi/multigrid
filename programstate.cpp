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
#include <cmath>
#include <limits>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

#ifndef M_PI
#define M_PI 3.141592653589793238462643
#endif //M_PI

using namespace std;

ProgramState::CommandTableEntry ProgramState::CommandTable[] =
{
	{"solve",&ProgramState::solve,"solve: Solves the problem with the current parameters"},
	{"quit",&ProgramState::quit,"quit: Exits the program"},
	{"print",&ProgramState::print,"print what: Prints some function. \"print err\" prints the error function, \"print sol\" prints the solution, \"print res\" prints the residuals"},
	{"setdim",&ProgramState::setdim,"setdim dimx dimy: Sets working dimensions"},
	{"setmode",&ProgramState::setmode,"setmode mode: Sets the solving mode. \"fmg\" for FMG method, \"mg\" for default multi-grid method, \"sor\" for simple Successive-Overrelaxation"},
	{"setsmoothsteps",&ProgramState::setsmoothsteps,"setsmoothsteps a1 a2: sets the number of pre and post relaxation iteration for FMG and MG method"},
	{"setomega",&ProgramState::setomega,"setomega o: sets the omega parameter for every smoother"},
	{"state",&ProgramState::state,"state: prints the current state of the solver"},
	{"setvcycles",&ProgramState::setvcycles,"setvcycles: set the number of cycles for multigrid and FMG methods"},
	{"setiterations",&ProgramState::setiterations,"setiterations: sets the number of iterations in a mg or fmg solver"},
	{"save",&ProgramState::save,"save what filename: save (sol err res) to the output file \"filename\" "},
	{"reduce",&ProgramState::reduce,"reduce what: reduce (sol err res)"},
	{"prolongate",&ProgramState::prolongate,"prolongate what: prolongate (sol err res)"},
	{"help",&ProgramState::help,"help: lists all the commands"}

};

real ones(real,real) {return 1;}
real zeros(real,real) {return 0;}

real prettyFunc1(real x,real y)
{
	return -2*( (1-6*x*x)*y*y*(1-y*y)+
					(1-6*y*y)*x*x*(1-x*x));
}

real prettyFunc1Sol(real x,real y)
{
	return (x*x-x*x*x*x)*(y*y*y*y-y*y);
}

real prettyFunc2(real x,real y)
{
	return exp(10*x)*cos(10*y);
}

real sinfunc1(real x,real y)
{
	return -M_PI*M_PI*2* sin(M_PI*x)*sin(M_PI*y);
}

real sinfunc1sol(real x,real y)
{
	return sin(M_PI*x)*sin(M_PI*y);
}

real sinfunc2(real x,real y)
{
	return -26*26*M_PI*M_PI*sin( 26*M_PI*x) -
		50*50*M_PI*M_PI*cos(50*M_PI*y)-
		M_PI*M_PI*sin(M_PI*x);
}

real sinfunc2sol(real x,real y)
{
	return sin(M_PI*26*x)+ cos (M_PI*50*y)+ sin(M_PI*x);
}

real charge(real x,real y)
{
	if (x == 0.5 && y == 0.5) return 1;
	if (x == 0.25 && y == 0.25) return 1;
	if (x == 0.25 && y == 0.75) return 1;
	if (x == 0.75 && y == 0.25) return 1;
	if (x == 0.75 && y == 0.75) return 1;
	return 0;
}

real ones3D(real x,real y,real z) {return 1;}
real zeros3D(real x,real y,real z) {return 0;}
real triDimFuncSol1(real x,real y,real z)
{
	return exp(sqrt(2)*M_PI*x)*sin(M_PI*y)*cos(M_PI*z);
}

std::ostream& operator<<(std::ostream & os,const BidimArray<real> & m)
{
	for (int i=0;i < m.width();++i)
	{
		for (int j=0;j < m.height();++j)
			os << m(i,j) << " ";
		os << std::endl;
	}
	return os;
}

std::ostream& operator<<(std::ostream & os,const TridimArray<real> & m)
{
	for (int k=0;k < m.depth();++k)
	{
		for (int i=0;i < m.width();++i)
		{
			for (int j=0;j < m.height();++j)
				os << m(i,j,k) << " ";
			os << std::endl;
		}
		os << std::endl;
	}
	return os;
}

ProgramState::ProgramState(int argc, char** argv) :
	m_curMode(Fmg),
	m_dimx(17),
	m_dimy(17),
	stepA1(3),
	stepA2(3),
	VCycles(2),
	m_omega(1.0),
	iterations(3),
	m_bDisplaySolution(false),
	m_bDisplayResidual(false),
	m_bDisplayError(false),
	m_bProfilingMode(false),

#ifdef BIDIM
	m_solver("mg_0.cl",m_handler),
	m_funcHandler(sinfunc2,sinfunc2sol,sinfunc2sol)
#else
	m_solver("mg_1.cl",m_handler),
	m_dimz(17),
	m_funcHandler(zeros3D,triDimFuncSol1,triDimFuncSol1)
#endif //BIDIM
// 	m_funcHandler(charge,zeros)
// 	m_funcHandler(prettyFunc1,zeros,prettyFunc1Sol)
// 	m_funcHandler(sinfunc2,sinfunc2sol,sinfunc2sol)
// 	m_funcHandler(zeros,prettyFunc2,prettyFunc2)

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
#ifdef BIDIM
			m_dimx = atoi(argv[++i]);
			m_dimy = atoi(argv[++i]);
			if (m_dimx <=0 || m_dimy <= 0)
			{
				cout << "Dimensions not valid" << endl;
				exit(1);
			}
#else
			m_dimx = atoi(argv[++i]);
			m_dimy = atoi(argv[++i]);
			m_dimz = atoi(argv[++i]);
			if (m_dimx <=0 || m_dimy <= 0 || m_dimz <= 0)
			{
				cout << "Dimensions not valid" << endl;
				exit(1);
			}
#endif //BIDIM
		}
		else if(string(argv[i]) == "--smoothsteps")
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
		else if(string(argv[i]) == "--profiling")
			m_bProfilingMode = true;
		else if(string(argv[i]) == "--omega")
		{
			m_omega = atof(argv[++i]);
			if (m_omega < 0 || m_omega >= 2)
			{
				cout << "Invalid omega" << endl;
				exit(1);
			}
		}
		else if(string(argv[i]) == "--iterations")
		{
			iterations = atoi(argv[++i]);
			if (iterations < 0)
			{
				cout << "Invalid iterations" << endl;
				exit(1);
			}
		}
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
// 	cout << fixed << setprecision(5);
	cout << "Real epsilon is: " << std::numeric_limits<real>::epsilon() << endl;
	if (m_bProfilingMode)
	{
		istringstream in;
		solve(in);
		return;
	}
	while (1)
	{
		cout << ">";
		cin >> noskipws;

		string input_string;
		std::getline(cin,input_string);
		string cmd;

		istringstream input_parameters (input_string);
		input_parameters >> cmd;

		bool done = false;
		for (int i=0;i < sizeof(CommandTable)/sizeof(CommandTableEntry);++i)
			if ( cmd == CommandTable[i].CmdName)
			{
				(this->*CommandTable[i].Func) (input_parameters);
				done = true;
			}
		if (!done) cout << "Unknown command: " << cmd << endl;

		if (cmd == "quit") return;
	}
}

void ProgramState::setdim( std::istream & params)
{
	int newdimx,newdimy;

#ifdef BIDIM
	params >> newdimx >> newdimy;

	if (params.fail() || newdimx < 0 || newdimy < 0)
		cout << "Invalid dimensions" << endl;
	else
		m_dimx = newdimx,m_dimy = newdimy;
#else
	int newdimz;
	params >> newdimx >> newdimy >> newdimz;

	if (params.fail() || newdimx < 0 || newdimy < 0 || newdimz < 0)
		cout << "Invalid dimensions" << endl;
	else
		m_dimx = newdimx,m_dimy = newdimy,m_dimz = newdimz;
	#endif //BIDIM
}

void ProgramState::setsmoothsteps(istream& params)
{
	int newa1,newa2;
	params >> newa1 >> newa2;
	if (params.fail() || newa1 < 0 || newa2 < 0)
		cout << "Invalid parameters" << endl;
	else
		stepA1 = newa1,stepA2 = newa2;
}

void ProgramState::setmode(istream& params)
{
	string mode;
	params >> mode;
	if (params.fail())
		cout << "Invalid mode" << endl;
	if (mode == "fmg")
		m_curMode = Fmg;
	else if (mode == "sor")
		m_curMode = Smooth;
	else if (mode == "mg")
		m_curMode = Multigrid;
	else
		cout << "Invalid mode" << endl;
}

void ProgramState::setvcycles(istream& params)
{
	int v;
	params >> v;
	if (params.fail() || v < 0)
		cout << "Invalid number of VCycles" << endl;
	else
		VCycles = v;
}

void ProgramState::setomega( std::istream & params)
{
	::real omega;
	params >> omega;
	if (params.fail() || omega < 0 || omega > 2.0)
		cout << "Invalid omega" << endl;
	else
		m_omega = omega;
}

void ProgramState::setiterations(std::istream& params)
{
	int iters;
	params >> iters;
	if (params.fail() || iters < 0)
		cout << "Invalid iterations" << endl;
	else
		iterations = iters;
}

void ProgramState::state( std::istream & params)
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

void ProgramState::print( std::istream & params)
{
	string what;
	params >> what;
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
	else cout << "Print what?" << endl;
}

void ProgramState::quit( std::istream & params)
{
	cout << "Quitting..." << endl;
}

void ProgramState::help( std::istream & params)
{
	for (int i=0;i < sizeof(CommandTable)/sizeof(CommandTableEntry);++i)
		cout << CommandTable[i].Description << endl;
}

void ProgramState::save(istream& params)
{
	string what,filename;
	params >> what >> filename;
	if (params.fail()) cout << "Save what where?" << endl;

	Buffer * arg =0;
	if (what == "err") arg = &m_error;
	else if (what == "res") arg = &m_residual;
	else if (what == "sol") arg = &m_solution;
	else if (what == "func") arg = &m_targetFunction;
	else
	{
		cout << "Save what?" << endl;
		return;
	}

	if (!arg->isInitialized())
	{
		cout << "Data not initialized yet" << endl;
		return;
	}

	if (filename.size() > 4 && filename.substr( filename.size()-3) == "bmp")
	{
#ifdef BIDIM
		toBitmap(*arg,m_solver.queue(),filename.c_str());
#else
		cout << "Saving to bitmap not supported in 3D." << endl;
#endif //BIDIM
	}
	else
	{
		ofstream outfile(filename.c_str());
		if (!outfile) cout << "Can not open: " << filename << endl;
		else
			outfile << arg->read(m_solver.queue()) << endl;
	}
}

void ProgramState::reduce(istream& params)
{
	string what;
	params >> what;

	Buffer * p = 0;
	if (cin.fail()) cout << "reduce what?" << endl;
	else if (what == "sol")
	{
		if (m_solution.isInitialized())
			p = &m_solution;
		else
			cout << "No solution available" << endl;
	} else if (what == "res")
	{
		if (m_residual.isInitialized())
			p = &m_residual;
		else
			cout << "No residuals available" << endl;
	} else if (what == "err")
	{
		if (m_error.isInitialized())
			p = &m_error;
		else
			cout << "No error available" << endl;
	} else if (what == "func")
	{
		if (m_targetFunction.isInitialized())
			p = &m_targetFunction;
		else
			cout << "No function available" << endl;
	}
	else cout << "Reduce what?" << endl;

	if (p)
	{
#ifdef BIDIM
		Buffer s (p->width()/2+1,p->height()/2+1);
#else
		Buffer s (p->width()/2+1,p->height()/2+1,p->depth()/2+1);
#endif //BIDIM
		m_solver.restrict(s,*p);
		*p = s;
	}
}

void ProgramState::prolongate(istream& params)
{
	string what;
	params >> what;

	Buffer * p = 0;
	if (cin.fail()) cout << "prolongate what?" << endl;
	else if (what == "sol")
	{
		if (m_solution.isInitialized())
			p = &m_solution;
		else
			cout << "No solution available" << endl;
	} else if (what == "res")
	{
		if (m_residual.isInitialized())
			p = &m_residual;
		else
			cout << "No residuals available" << endl;
	} else if (what == "err")
	{
		if (m_error.isInitialized())
			p = &m_error;
		else
			cout << "No error available" << endl;
	} else if (what == "func")
	{
		if (m_targetFunction.isInitialized())
			p = &m_targetFunction;
		else
			cout << "No function available" << endl;
	}
	else cout << "Prolongate what?" << endl;

	if (p)
	{
#ifdef BIDIM
		Buffer s (p->width()*2-1,p->height()*2-1);
#else
		Buffer s (p->width()*2-1,p->height()*2-1,p->depth()*2-1);
#endif //BIDIM
		m_solver.prolongate(s,*p);
		*p = s;
	}
}

void ProgramState::solve(std::istream & is)
{
#ifdef BIDIM
	m_targetFunction = m_funcHandler.discretize_func(m_dimx,m_dimy,1.0/(m_dimx-1),m_handler);
	Buffer emptyBuf = Buffer::empty(m_dimx,m_dimy,m_solver.queue());
	m_solution = Buffer::empty(m_dimx,m_dimy,m_solver.queue());
	m_residual = Buffer(m_dimx,m_dimy);
	m_error = Buffer(m_dimx,m_dimy);
#else
	m_targetFunction = m_funcHandler.discretize_func(m_dimx,m_dimy,m_dimz,1.0/(m_dimx-1),m_handler);
	Buffer emptyBuf = Buffer::empty(m_dimx,m_dimy,m_dimz,m_solver.queue());
	m_solution = Buffer::empty(m_dimx,m_dimy,m_dimz,m_solver.queue());
	m_residual = Buffer(m_dimx,m_dimy,m_dimz);
	m_error = Buffer(m_dimx,m_dimy,m_dimz);
#endif //BIDIM

	clock_t startTimer = clock();

	switch (m_curMode)
	{
	case Fmg:
		m_solution = m_solver.fmg(m_targetFunction,
						m_omega,
						stepA1,
						stepA2,
						VCycles,
						iterations);
		break;
	case Smooth:
		m_solver.smoother_iterate(m_solution,
								  m_targetFunction,
								  m_omega,
								  stepA1);
		break;
	case Multigrid:
		m_solver.mg(m_solution,
						 m_targetFunction,
						 m_omega,
						 stepA1,
						 stepA2,
						 VCycles,
						 iterations);
		break;

	}
	m_solver.zero_out(m_solution);

	if (m_funcHandler.hasSol())
#ifdef BIDIM
		m_error = Difference(m_solution, m_funcHandler.discretize_sol(m_dimx,m_dimy,1.0/(m_dimx-1),m_handler),m_solver.queue());
#else
		m_error = Difference(m_solution, m_funcHandler.discretize_sol(m_dimx,m_dimy,m_dimz,1.0/(m_dimx-1),m_handler),m_solver.queue());
#endif //BIDIM

	m_solver.compute_residuals(m_residual,m_solution,m_targetFunction);
	m_solver.wait();

	clock_t endTimer = clock();

	cout << "Time\t\t\tL2Err\t\t\tLInfErr\t\t\tL2Res\t\t\tLinfRes\t\t\t" << endl;
	cout << (double)(endTimer-startTimer)/CLOCKS_PER_SEC << "\t\t\t" <<
		L2Norm(m_error,m_solver.queue()) << "\t\t\t" <<
		LInfNorm(m_error,m_solver.queue())<< "\t\t\t" <<
		L2Norm(m_residual,m_solver.queue()) << "\t\t\t" <<
		LInfNorm(m_residual,m_solver.queue()) << endl;
}

void ProgramState::helpString()
{
	cout << "Usage: multigrid [options] " << endl;
	cout << "Valid options are: " << endl;
	cout << "\t \"--solver solvertype\": set the solver type. 'fmg' for a Fast multigrid solver, 'jac' for the Jacobi method, 'mg' for the simple multigrid method" << endl
	<< "\t \"--dim dimx dimy\": set the dimension of the grid to be used" << endl
	<< "\t \"--smoothsteps a1 a2\": sets the number of steps of pre and post smoothing. For simple jacobi method only the first value is used" << endl
	<< "\t \"--mgcycles v\": sets the number of v-cycles to be used (1= V Cycle, 2 = W Cycle)" << endl
	<< "\t \"--omega o\": sets the omega parameter" << endl
	<< "\t \"--profiling\": profiling mode: print solution data and exits immediatly" << endl
	<< "\t \"--help\" \"-h\": prints this help" << endl
	<< endl;
}
