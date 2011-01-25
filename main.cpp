#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
#include <map>

#include "programstate.h"

using namespace std;

cl::NDRange getBestWorkspaceDim(cl::NDRange wsDim);

int main(int argc,char ** argv)
{
#if (defined (NDEBUG) || defined(_NDEBUG))
 	try
	{
		ProgramState ps (argc-1,argv+1);
		ps.listenCommand();
	}
	catch(cl::Error & r)
	{
		cout << "Cl error in " << r.what() << " code: " << r.err() << endl;
		cout << "Aborting..";
		system("PAUSE");
		return 1;
	}
	catch(std::exception & r)
	{
		cout << "Exception: "  << r.what() << endl << "Aborting...";
		system("PAUSE");
		return 1;
	}
	catch(...)
	{
		cout << "Unknown error" << endl << endl << "Aborting...";
		system("PAUSE");
		return 1;
	}

#else
	ProgramState ps (argc-1,argv+1);
	ps.listenCommand();
#endif //_NDEBUG
	return 0;
}
