<<<<<<< HEAD
/*
 CS 759 HW01 - Problem 2
 Count the number of characters in the command input.
 Author: Jieru Hu. Login: jhu76.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int countChar(char *input)
{
    int count = 0;
    int i;
    for(i=0;input[i]!='\0';i++)
    {
        count++;
    }
    return count;

}

int main(int argc, char* argv[])
{
    if(argc==1){
        printf("0\n");
    }else{
        printf("%d\n",countChar(argv[1]));

    }
}

