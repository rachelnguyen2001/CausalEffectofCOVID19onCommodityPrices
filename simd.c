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
  for (int i = 0; i < numEntries / 32; i++) {
    __m256i a = _mm256_loadu_si256((__m256i*)(inArray+32*i));
    int integerResult = _mm256_movemask_epi8(a);

    if (integerResult != 0) {
      return 1;
    }
  }

  return 0;
}


//this is just here in case it's useful
int main() {
	char test1[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	char test2[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	char *t1 = test1;
	char *t2 = test2;
	printf("%d", hasChar(t1, 32));
	printf("%d", hasChar(t2, 32));
}
