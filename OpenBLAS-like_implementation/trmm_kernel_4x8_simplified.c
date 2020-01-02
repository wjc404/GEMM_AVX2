#include "common.h"
//this file is a modification of the file "trmm_kernel_4x8.c" from OpenBLAS, with definitions of some macros omitted.
int CNAME(BLASLONG bm,BLASLONG bn,BLASLONG bk,FLOAT alpha,FLOAT* ba,FLOAT* bb,FLOAT* C,BLASLONG ldc ,BLASLONG offset){
      BLASLONG i,j,k;
      FLOAT *c_ptr,*ptrba,*ptrbb;
      BLASLONG off, temp;
#if LEFT != TRANSA
  #define BACKWARDS 1
#else
  #define BACKWARDS 0
#endif
      
#if BACKWARDS == 1
  #define INIT_set_k_and_pointers(a_copy,b_copy) \
    ptrba += off*(a_copy);\
    ptrbb += off*(b_copy);\
    temp = bk - off;
  #define SAVE_set_pointers(a_copy,b_copy) {}
#else
  #ifdef LEFT
    #define INIT_set_k_and_pointers(a_copy,b_copy) temp = off + (a_copy);
    #define SAVE_set_pointers(a_copy,b_copy) \
      temp = bk - off - (a_copy);\
      ptrba += temp * (a_copy);\
      ptrbb += temp * (b_copy);
  #else
    #define INIT_set_k_and_pointers(a_copy,b_copy) temp = off + (b_copy);
    #define SAVE_set_pointers(a_copy,b_copy) \
      temp = bk - off - (b_copy);\
      ptrba += temp * (a_copy);\
      ptrbb += temp * (b_copy);
  #endif
#endif

#ifndef LEFT
      off = -offset;
#endif
      c_ptr = C;
      for (j=0; j<bn/8; j++){
#ifdef LEFT
        off = offset;
#endif
        ptrba = ba;
        for (i=0; i<bm/4; i+=1){
          ptrbb = bb;
          INIT_set_k_and_pointers(4,8)
          INIT_m4n8
          for (k=0; k<temp; k++)  KERNEL_k1m4n8
          SAVE_m4n8 //don't forget to update c_ptr
          SAVE_set_pointers(4,8)
#ifdef LEFT
          off += 4; // number of values in A
#endif
        }//i->bm/4 loop tail
        if ( bm & 2 ){
          ptrbb = bb;
          INIT_set_k_and_pointers(2,8)
          INIT_m2n8
          for (k=0; k<temp; k++) KERNEL _k1m2n8
          SAVE_m2n8
          SAVE_set_pointers(2,8)
#ifdef LEFT
          off += 2; // number of values in A
#endif
        }
        if ( bm & 1 ){
          ptrbb = bb;
          INIT_set_k_and_pointers(1,8)
          INIT_m1n8
          for (k=0; k<temp; k++) KERNEL_k1m1n8
          SAVE_m1n8
          SAVE_set_pointers(1,8)
#ifdef LEFT
          off += 1; // number of values in A
#endif
        }
#ifndef LEFT
        off += 8;
#endif
        bb = bb + bk*8;
        c_ptr += ldc*8-bm;
      }//j -> bn/8 loop tail

      for (j=0; j<(bn&4); j+=4){
#ifdef LEFT
        off = offset;
#endif
        ptrba = ba;
        for (i=0; i<bm/4; i+=1){
          ptrbb = bb;
          INIT_set_k_and_pointers(4,4)
          INIT_m4n4
          for (k=0; k<temp; k++) KERNEL_k1m4n4
          SAVE_m4n4
          SAVE_set_pointers(4,4)
#ifdef LEFT
          off += 4; // number of values in A
#endif
        }//i -> bm/4 loop tail
        if ( bm & 2 ){
          ptrbb = bb;
          INIT_set_k_and_pointers(2,4)
          INIT_m2n4
          for (k=0; k<temp; k++) KERNEL_k1m2n4
          SAVE_m2n4
          SAVE_set_pointers(2,4)
#ifdef LEFT
          off += 2; // number of values in A
#endif
        }
        if ( bm & 1 ){
          ptrbb = bb;
          INIT_set_k_and_pointers(1,4)
          INIT_m1n4
          for (k=0; k<temp; k++) KERNEL_k1m1n4
          SAVE_m1n4
          SAVE_set_pointers(1,4)
#ifdef LEFT
          off += 1; // number of values in A
#endif
        }
#ifndef LEFT
        off += 4;
#endif
        bb = bb + bk*4;
        c_ptr += ldc*4-bm;
      }// condition j -> bn&4 tail
      for (j=0; j<(bn&2); j+=2){
#ifdef LEFT
        off = offset;
#endif
        ptrba = ba;
        for (i=0; i<bm/4; i+=1){
          ptrbb = bb;
          INIT_set_k_and_pointers(4,2)
          INIT_m4n2
          for (k=0; k<temp; k++) KERNEL_k1m4n2
          SAVE_m4n2
          SAVE_set_pointers(4,2)
#ifdef LEFT
          off += 4; // number of values in A
#endif
        }
        if ( bm & 2 ){
          ptrbb = bb;
          INIT_set_k_and_pointers(2,2)
          INIT_m2n2
          for (k=0; k<temp; k++) KERNEL_k1m2n2
          SAVE_m2n2
          SAVE_set_pointers(2,2)
#ifdef LEFT
          off += 2; // number of values in A
#endif
        }
        if ( bm & 1 ){
          ptrbb = bb;
          INIT_set_k_and_pointers(1,2)
          INIT_m1n2
          for (k=0; k<temp; k++) KERNEL_k1m1n2
          SAVE_m1n2
          SAVE_set_pointers(1,2)
#ifdef LEFT
          off += 1; // number of values in A
#endif
        }
#ifndef LEFT
        off += 2;
#endif
        bb = bb + bk*2;
        c_ptr += ldc*2-bm;
      }
      for (j=0; j<(bn&1); j+=1){
#ifdef LEFT
        off = offset;
#endif
        ptrba = ba;
        for (i=0; i<bm/4; i+=1){
          ptrbb = bb;
          INIT_set_k_and_pointers(4,1)
          INIT_m4n1
          for (k=0; k<temp; k++) KERNEL_k1m4n1
          SAVE_m4n1
          SAVE_set_pointers(4,1)
#ifdef LEFT
          off += 4; // number of values in A
#endif
        }
        if ( bm & 2 ){
          ptrbb = bb;
          INIT_set_k_and_pointers(2,1)
          INIT_m2n1
          for (k=0; k<temp; k++) KERNEL_k1m2n1
          SAVE_m2n1
          SAVE_set_pointers(2,1)
#ifdef LEFT
          off += 2; // number of values in A
#endif
        }
        if ( bm & 1 ){
          ptrbb = bb;
          INIT_set_k_and_pointers(1,1)
          INIT_m1n1
          for (k=0; k<temp; k++) KERNEL_k1m1n1
          SAVE_m1n1
          SAVE_set_pointers(1,1)
#ifdef LEFT
          off += 1; // number of values in A
#endif
        }
#ifndef LEFT
        off += 1;
#endif
        bb = bb + bk;
        c_ptr += ldc-bm;
      }
      return 0;
}
