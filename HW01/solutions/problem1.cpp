<<<<<<< HEAD
/*
 CS 759 HW01 - Problem 1
 Prints the student ID.
 Author: Jieru Hu. Login: jhu76.
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int main()
{
	ifstream inFile("problem1.txt");
	
	if (!inFile) {
		fprintf(stderr, "There is error opening the file\n");
		return 1;
	}
	string fullId;
	getline(inFile, fullId);
	inFile.close();
	
	string printId;
	if (fullId.length() > 4){
		printId = fullId.substr(fullId.length()-4, 4);
	} else {
		printId = fullId;
	} 
	printf("Hello! I'm student %s.\n", printId.c_str());
    return 0;

}
