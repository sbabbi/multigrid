#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>

#include "programstate.h"

using namespace std;

/*
float prettyFunc1(float x,float y)
{
	return -2*( (1-6*x*x)*y*y*(1-y*y)+
				(1-6*y*y)*x*x*(1-x*x));
}
float prettyFunc1Sol(float x,float y)
{
	return (x*x-x*x*x*x)*(y*y*y*y-y*y);
}
float prettyFunc2(float x,float y)
{
	return sin(x+y);
}
float OppprettyFunc2(float x,float y) {return -2 * prettyFunc2(x,y);}
*/

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
