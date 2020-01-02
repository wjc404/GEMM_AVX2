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
#if BACKWARDS == 1
          ptrba += off*4; // number of values in A
          ptrbb += off*8; // number of values in B
          temp = bk - off;
#else
  #ifdef LEFT
          temp = off + 4;
  #else
          temp = off + 8;
  #endif
#endif
          INIT_m4n8
          for (k=0; k<temp; k++)  KERNEL_k1m4n8
          SAVE_m4n8 //don't forget to update c_ptr
#if BACKWARDS == 0
          temp = bk - off;
  #ifdef LEFT
          temp -= 4;
  #else
          temp -= 8;
  #endif
          ptrba += temp*4; // number of values in A
          ptrbb += temp*8; // number of values in B
#endif
#ifdef LEFT
          off += 4; // number of values in A
#endif
        }//i->bm/4 loop tail
        if ( bm & 2 ){
          ptrbb = bb;
#if BACKWARDS == 1
          ptrba += off*2;
          ptrbb += off*8;
          temp = bk-off;
#elif defined(LEFT)
          temp = off+2;	// number of values in A
#else
          temp = off+8;	// number of values in B
#endif
          INIT_m2n8
          for (k=0; k<temp; k++) KERNEL _k1m2n8
          SAVE_m2n8
#if BACKWARDS == 0
          temp = bk - off;
  #ifdef LEFT
          temp -= 2; // number of values in A
  #else
          temp -= 8; // number of values in B
  #endif
          ptrba += temp*2;
          ptrbb += temp*8;
#endif
#ifdef LEFT
          off += 2; // number of values in A
#endif
        }
        if ( bm & 1 ){
          ptrbb = bb;
#if BACKWARDS == 1
          ptrba += off*1;
          ptrbb += off*8;
          temp = bk-off;
#elif defined(LEFT)
          temp = off+1;	// number of values in A
#else
          temp = off+8;	// number of values in B
#endif
          INIT_m1n8
          for (k=0; k<temp; k++) KERNEL_k1m1n8
          SAVE_m1n8
#if BACKWARDS == 0
          temp = bk - off;
  #ifdef LEFT
          temp -= 1; // number of values in A
  #else
          temp -= 8; // number of values in B
  #endif
          ptrba += temp*1;
          ptrbb += temp*8;
#endif
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
#if BACKWARDS == 1
          ptrba += off*4; // number of values in A
          ptrbb += off*4; // number of values in B
#endif
          INIT_m4n4
#if BACKWARDS == 1
          temp = bk - off;
#else
  #ifdef LEFT
          temp = off + 4;
  #else
          temp = off + 4;
  #endif
#endif
          for (k=0; k<temp; k++) KERNEL_k1m4n4
          SAVE_m4n4
#if BACKWARDS == 0
          temp = bk - off;
  #ifdef LEFT
          temp -= 4;
  #else
          temp -= 4;
  #endif
          ptrba += temp*4; // number of values in A
          ptrbb += temp*4; // number of values in B
#endif
#ifdef LEFT
          off += 4; // number of values in A
#endif
        }//i -> bm/4 loop tail
        if ( bm & 2 ){
#if BACKWARDS == 0
          ptrbb = bb;
#else
          ptrba += off*2;
          ptrbb = bb + off*4;
#endif
          INIT_m2n4
#if BACKWARDS == 1
          temp = bk-off;
#elif defined(LEFT)
          temp = off+2;	// number of values in A
#else
          temp = off+4;	// number of values in B
#endif
          for (k=0; k<temp; k++) KERNEL_k1m2n4
          SAVE_m2n4
#if BACKWARDS == 0
          temp = bk - off;
  #ifdef LEFT
          temp -= 2; // number of values in A
  #else
          temp -= 4; // number of values in B
  #endif
          ptrba += temp*2;
          ptrbb += temp*4;
#endif
#ifdef LEFT
          off += 2; // number of values in A
#endif
        }
        if ( bm & 1 ){
#if BACKWARDS == 0
          ptrbb = bb;
#else
          ptrba += off*1;
          ptrbb = bb + off*4;
#endif
          INIT_m1n4
#if BACKWARDS == 1
          temp = bk-off;
#elif defined(LEFT)
          temp = off+1;	// number of values in A
#else
          temp = off+4;	// number of values in B
#endif
          for (k=0; k<temp; k++) KERNEL_k1m1n4
          SAVE_m1n4
#if BACKWARDS == 0
          temp = bk - off;
  #ifdef LEFT
          temp -= 1; // number of values in A
  #else
          temp -= 4; // number of values in B
  #endif
          ptrba += temp*1;
          ptrbb += temp*4;
#endif
#ifdef LEFT
          off += 1; // number of values in A
#endif
        }
#if !defined(LEFT)
        off += 4;
#endif
        bb = bb + bk*4;
        c_ptr += ldc*4-bm;
      }// condition j -> bn&4 tail
      for (j=0; j<(bn&2); j+=2){
#if defined(LEFT)
        off = offset;
#endif
        ptrba = ba;
        for (i=0; i<bm/4; i+=1){
#if BACKWARDS == 0
          ptrbb = bb;
#else
          ptrba += off*4;
          ptrbb = bb + off*2;
#endif
          INIT_m2n4
#if BACKWARDS == 1
          temp = bk-off;
#elif defined(LEFT)
          temp = off+4;	// number of values in A
#else
          temp = off+2;	// number of values in B
#endif
          for (k=0; k<temp; k++) KERNEL_k1m2n4
          SAVE_m2n4
#if BACKWARDS == 0
          temp = bk - off;
#ifdef LEFT
          temp -= 4; // number of values in A
#else
          temp -= 2; // number of values in B
#endif
          ptrba += temp*4;
          ptrbb += temp*2;
#endif
#ifdef LEFT
          off += 4; // number of values in A
#endif
        }
        if ( bm & 2 ){
#if BACKWARDS == 0
          ptrbb = bb;
#else
          ptrba += off*2;
          ptrbb = bb + off*2;
#endif
          INIT_m2n2
#if BACKWARDS == 1
          temp = bk-off;
#elif defined(LEFT)
          temp = off+2;	// number of values in A
#else
          temp = off+2;	// number of values in B
#endif
          for (k=0; k<temp; k++) KERNEL_k1m2n2
          SAVE_m2n2
#if BACKWARDS == 0
          temp = bk - off;
#ifdef LEFT
          temp -= 2; // number of values in A
#else
          temp -= 2; // number of values in B
#endif
          ptrba += temp*2;
          ptrbb += temp*2;
#endif
#ifdef LEFT
          off += 2; // number of values in A
#endif
        }
        if ( bm & 1 ){
#if BACKWARDS == 0
          ptrbb = bb;
#else
          ptrba += off*1;
          ptrbb = bb + off*2;
#endif
          INIT_m1n2
#if BACKWARDS == 1
          temp = bk-off;
#elif defined(LEFT)
          temp = off+1;	// number of values in A
#else
          temp = off+2;	// number of values in B
#endif
          for (k=0; k<temp; k++) KERNEL_k1m1n2
          SAVE_m1n2
#if BACKWARDS == 0
          temp = bk - off;
#ifdef LEFT
          temp -= 1; // number of values in A
#else
          temp -= 2; // number of values in B
#endif
          ptrba += temp*1;
          ptrbb += temp*2;
#endif
#ifdef LEFT
          off += 1; // number of values in A
#endif
        }
#if !defined(LEFT)
        off += 2;
#endif
        bb = bb + bk*2;
        c_ptr += ldc*2-bm;
      }
      for (j=0; j<(bn&1); j+=1){
#if defined(LEFT)
        off = offset;
#endif
        ptrba = ba;
        for (i=0; i<bm/4; i+=1){
#if BACKWARDS == 0
          ptrbb = bb;
#else
          ptrba += off*4;
          ptrbb = bb + off*1;
#endif
          INIT_m4n1
#if BACKWARDS == 1
          temp = bk-off;
#elif defined(LEFT)
          temp = off+4;	// number of values in A
#else
          temp = off+1;	// number of values in B
#endif
          for (k=0; k<temp; k++) KERNEL_k1m4n1
          SAVE_m4n1
#if BACKWARDS == 0
          temp = bk - off;
#ifdef LEFT
          temp -= 4; // number of values in A
#else
          temp -= 1; // number of values in B
#endif
          ptrba += temp*4;
          ptrbb += temp*1;
#endif
#ifdef LEFT
          off += 4; // number of values in A
#endif
        }
        if ( bm & 2 ){
#if BACKWARDS == 0
          ptrbb = bb;
#else
          ptrba += off*2;
          ptrbb = bb + off*1;
#endif
          INIT_m2n1
#if BACKWARDS == 1
          temp = bk-off;
#elif defined(LEFT)
          temp = off+2;	// number of values in A
#else
          temp = off+1;	// number of values in B
#endif
          for (k=0; k<temp; k++) KERNEL_k1m2n1
          SAVE_m2n1
#if BACKWARDS == 0
          temp = bk - off;
#ifdef LEFT
          temp -= 2; // number of values in A
#else
          temp -= 1; // number of values in B
#endif
          ptrba += temp*2;
          ptrbb += temp*1;
#endif
#ifdef LEFT
          off += 2; // number of values in A
#endif
        }
        if ( bm & 1 ){
#if BACKWARDS == 0
          ptrbb = bb;
#else
          ptrba += off*1;
          ptrbb = bb + off*1;
#endif
          INIT_m1n1
#if BACKWARDS == 1
          temp = bk-off;
#elif defined(LEFT)
          temp = off+1;	// number of values in A
#else
          temp = off+1;	// number of values in B
#endif
          for (k=0; k<temp; k++) KERNEL_k1m1n1
          SAVE_m1n1
#if BACKWARDS == 0
          temp = bk - off;
#ifdef LEFT
          temp -= 1; // number of values in A
#else
          temp -= 1; // number of values in B
#endif
          ptrba += temp*1;
          ptrbb += temp*1;
#endif
#ifdef LEFT
          off += 1; // number of values in A
#endif
        }
#if !defined(LEFT)
        off += 1;
#endif
        bb = bb + bk;
        c_ptr += ldc-bm;
      }
      return 0;
}
