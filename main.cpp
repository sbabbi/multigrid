#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
#include <map>

#include "programstate.h"

using namespace std;

int main(int argc,char ** argv)
{
	try
	{
		ProgramState ps (argc-1,argv+1);
		ps.listenCommand();
	}
	catch(cl::Error & r)
	{
		cout << "Cl error in " << r.what() << " code: " << r.err() << endl;
		cout << "Aborting..";
		return 1;
	}
	catch(std::exception & r)
	{
		cout << "Exception: "  << r.what() << endl << "Aborting...";
		return 1;
	}
	catch(...)
	{
		cout << "Unknown error" << endl << endl << "Aborting...";
		return 1;
	}

	return 0;
}
