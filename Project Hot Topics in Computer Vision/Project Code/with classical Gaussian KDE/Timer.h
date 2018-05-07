// This code is used to calculate the running time of an Algorithm

#ifndef TIMER_H
#define TIMER_H

#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;

class Timer{

	high_resolution_clock::time_point STARTING_TIME, ENDING_TIME;
	nanoseconds TIME_SPAN;

public:

	/* Prints the time in proper format */
	void PrintTime(){

		double time = TIME_SPAN.count();
		char micro = -26;

		if (time < 1000)
			cout <<  time << " ns";
		else if (time / 1000 < 1000)
			cout <<  time / 1000 << " " << micro << "s";
		else if (time / 1000000 < 1000)
			cout <<  time / 1000000 << " ms";
		else
			cout <<  time / 1000000000 << " sec.";

	}

	/* Starts the stopwatch */
	void Start(){
		STARTING_TIME = high_resolution_clock::now();
	}

	/* Ends the stopwatch */
	void End(){
		ENDING_TIME = high_resolution_clock::now();
		TIME_SPAN = duration_cast<nanoseconds>(ENDING_TIME - STARTING_TIME);		
	}	

	/* Ends the stopwatch and prints the time in some specific format */	
	void LocalEnd(){
		ENDING_TIME = high_resolution_clock::now();
		TIME_SPAN = duration_cast<nanoseconds>(ENDING_TIME - STARTING_TIME);		
		cout<<"Time: ";
		PrintTime();
	}

	/* Ends the stopwatch and prints the time in some specific format */
	void TotalEnd(){
		ENDING_TIME = high_resolution_clock::now();
		TIME_SPAN = duration_cast<nanoseconds>(ENDING_TIME - STARTING_TIME);		
		cout<<"\n => Total Time Consumed: ";
		PrintTime();
		cout<<"\n\n";
	}	
};

#endif