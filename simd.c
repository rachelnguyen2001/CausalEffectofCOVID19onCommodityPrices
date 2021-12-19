#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

//for SIMD instructions
#include <immintrin.h>


//function to determine if any of the characters in inArray
//has value 128 or higher.  Should return 1 if so; 0 otherwise
int hasChar(char* inArray, int numEntries) {
	//fill in your solution here
	//(unless you filled it in inline in the tex file)
}


//this is just here in case it's useful
int main() {
	char test1[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	char test2[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }; 
}
