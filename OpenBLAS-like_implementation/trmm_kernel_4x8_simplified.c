#include "common.h"
#include <stdbool.h>
//this file is a modification of the file "trmm_kernel_4x8.c" from OpenBLAS, with definitions of some macros omitted.
int CNAME(BLASLONG bm,BLASLONG bn,BLASLONG bk,FLOAT alpha,FLOAT* ba,FLOAT* bb,FLOAT* C,BLASLONG ldc ,BLASLONG offset){
      BLASLONG i,j,k;
      FLOAT *c_ptr,*ptrba,*ptrbb;
      BLASLONG off, temp;
      bool left,transa,backwards;
#ifdef LEFT
      left = true;
#else
      left = false;
#endif
#ifdef TRANSA
      transa = true;
#else
      transa = false;
#endif
      backwards = (left != transa);
      if (!left) off = -offset;
      c_ptr = C;
      for (j=0; j<bn/8; j++){
        if (left) off = offset;
        ptrba = ba;
        for (i=0; i<bm/4; i+=1){
          ptrbb = bb;
          if (backwards){
            ptrba += off*4; // number of values in A
            ptrbb += off*8; // number of values in B
          }
          INIT_m4n8
          if(backwards) temp = bk - off;
          else{
             if(left) temp = off + 4;
             else temp = off + 8;
          }
          for (k=0; k<temp; k++)  KERNEL_k1m4n8
          SAVE_m4n8 //don't forget to update c_ptr
          if (!backwards) {
            temp = bk - off;
            if(left) temp -= 4;
            else temp -= 8;
            ptrba += temp*4; // number of values in A
            ptrbb += temp*8; // number of values in B
          }
#ifdef LEFT
          off += 4; // number of values in A
#endif
        }//i->bm/4 loop tail
        if ( bm & 2 ){
#if (defined(LEFT) &&  defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
          ptrbb = bb;
#else
          ptrba += off*2;
          ptrbb = bb + off*8;
#endif
          INIT_m2n8
#if (defined(LEFT) && !defined(TRANSA)) || (!defined(LEFT) && defined(TRANSA))
          temp = bk-off;
#elif defined(LEFT)
          temp = off+2;	// number of values in A
#else
          temp = off+8;	// number of values in B
#endif
          for (k=0; k<temp; k++) KERNEL _k1m2n8
          SAVE_m2n8
#if (defined(LEFT) && defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
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
#if (defined(LEFT) &&  defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
          ptrbb = bb;
#else
          ptrba += off*1;
          ptrbb = bb + off*8;
#endif
          INIT_m1n8
#if (defined(LEFT) && !defined(TRANSA)) || (!defined(LEFT) && defined(TRANSA))
          temp = bk-off;
#elif defined(LEFT)
          temp = off+1;	// number of values in A
#else
          temp = off+8;	// number of values in B
#endif
          for (k=0; k<temp; k++) KERNEL_k1m1n8
          SAVE_m1n8
#if (defined(LEFT) && defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
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
#if defined(TRMMKERNEL) && !defined(LEFT)
        off += 8;
#endif
        k = (bk<<3);
        bb = bb+k;
        c_ptr += ldc*8-bm;
      }//j -> bn/8 loop tail

      for (j=0; j<(bn&4); j+=4){
        if (left) off = offset;
        ptrba = ba;
        for (i=0; i<bm/4; i+=1){
          ptrbb = bb;
          if (backwards){
            ptrba += off*4; // number of values in A
            ptrbb += off*4; // number of values in B
          }
          INIT_m4n4
          if(backwards) temp = bk - off;
          else{
            if(left) temp = off + 4;
            else temp = off + 4;
          }
          for (k=0; k<temp; k++) KERNEL_k1m4n4
          SAVE_m4n4
          if (!backwards){
            temp = bk - off;
            if(left) temp -= 4;
            else temp -= 4;
            ptrba += temp*4; // number of values in A
            ptrbb += temp*4; // number of values in B
          }
#ifdef LEFT
          off += 4; // number of values in A
#endif
        }//i -> bm/4 loop tail
        if ( bm & 2 ){
#if (defined(LEFT) &&  defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
          ptrbb = bb;
#else
          ptrba += off*2;
          ptrbb = bb + off*4;
#endif
          INIT_m2n4
#if (defined(LEFT) && !defined(TRANSA)) || (!defined(LEFT) && defined(TRANSA))
          temp = bk-off;
#elif defined(LEFT)
          temp = off+2;	// number of values in A
#else
          temp = off+4;	// number of values in B
#endif
          for (k=0; k<temp; k++) KERNEL_k1m2n4
          SAVE_m2n4
#if (defined(LEFT) && defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
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
#if (defined(LEFT) &&  defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
          ptrbb = bb;
#else
          ptrba += off*1;
          ptrbb = bb + off*4;
#endif
          INIT_m1n4
#if (defined(LEFT) && !defined(TRANSA)) || (!defined(LEFT) && defined(TRANSA))
          temp = bk-off;
#elif defined(LEFT)
          temp = off+1;	// number of values in A
#else
          temp = off+4;	// number of values in B
#endif
          for (k=0; k<temp; k++) KERNEL_k1m1n4
          SAVE_m1n4
#if (defined(LEFT) && defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
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
#if defined(TRMMKERNEL) && !defined(LEFT)
        off += 4;
#endif
        k = (bk<<2);
        bb = bb+k;
        c_ptr += ldc*4-bm;
      }// condition j -> bn&4 tail
      for (j=0; j<(bn&2); j+=2){
#if defined(TRMMKERNEL) && defined(LEFT)
        off = offset;
#endif
        ptrba = ba;
        for (i=0; i<bm/4; i+=1){
#if (defined(LEFT) &&  defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
          ptrbb = bb;
#else
          ptrba += off*4;
          ptrbb = bb + off*2;
#endif
          INIT_m2n4
#if (defined(LEFT) && !defined(TRANSA)) || (!defined(LEFT) && defined(TRANSA))
          temp = bk-off;
#elif defined(LEFT)
          temp = off+4;	// number of values in A
#else
          temp = off+2;	// number of values in B
#endif
          for (k=0; k<temp; k++) KERNEL_k1m2n4
          SAVE_m2n4
#if (defined(LEFT) && defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
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
#if (defined(LEFT) &&  defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
          ptrbb = bb;
#else
          ptrba += off*2;
          ptrbb = bb + off*2;
#endif
          INIT_m2n2
#if (defined(LEFT) && !defined(TRANSA)) || (!defined(LEFT) && defined(TRANSA))
          temp = bk-off;
#elif defined(LEFT)
          temp = off+2;	// number of values in A
#else
          temp = off+2;	// number of values in B
#endif
          for (k=0; k<temp; k++) KERNEL_k1m2n2
          SAVE_m2n2
#if (defined(LEFT) && defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
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
#if (defined(LEFT) &&  defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
          ptrbb = bb;
#else
          ptrba += off*1;
          ptrbb = bb + off*2;
#endif
          INIT_m1n2
#if (defined(LEFT) && !defined(TRANSA)) || (!defined(LEFT) && defined(TRANSA))
          temp = bk-off;
#elif defined(LEFT)
          temp = off+1;	// number of values in A
#else
          temp = off+2;	// number of values in B
#endif
          for (k=0; k<temp; k++) KERNEL_k1m1n2
          SAVE_m1n2
#if (defined(LEFT) && defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
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
#if defined(TRMMKERNEL) && !defined(LEFT)
        off += 2;
#endif
        k = (bk<<1);
        bb = bb+k;
        i = (ldc<<1);
        c_ptr += ldc*2-bm;
      }
      for (j=0; j<(bn&1); j+=1){
#if defined(TRMMKERNEL) &&  defined(LEFT)
        off = offset;
#endif
        ptrba = ba;
        for (i=0; i<bm/4; i+=1){
#if (defined(LEFT) &&  defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
          ptrbb = bb;
#else
          ptrba += off*4;
          ptrbb = bb + off*1;
#endif
          INIT_m4n1
#if (defined(LEFT) && !defined(TRANSA)) || (!defined(LEFT) && defined(TRANSA))
          temp = bk-off;
#elif defined(LEFT)
          temp = off+4;	// number of values in A
#else
          temp = off+1;	// number of values in B
#endif
          for (k=0; k<temp; k++) KERNEL_k1m4n1
          SAVE_m4n1
#if (defined(LEFT) && defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
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
#if (defined(LEFT) &&  defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
          ptrbb = bb;
#else
          ptrba += off*2;
          ptrbb = bb + off*1;
#endif
          INIT_m2n1
#if (defined(LEFT) && !defined(TRANSA)) || (!defined(LEFT) && defined(TRANSA))
          temp = bk-off;
#elif defined(LEFT)
          temp = off+2;	// number of values in A
#else
          temp = off+1;	// number of values in B
#endif
          for (k=0; k<temp; k++) KERNEL_k1m2n1
          SAVE_m2n1
#if (defined(LEFT) && defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
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
#if (defined(LEFT) &&  defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
          ptrbb = bb;
#else
          ptrba += off*1;
          ptrbb = bb + off*1;
#endif
          INIT_m1n1
#if (defined(LEFT) && !defined(TRANSA)) || (!defined(LEFT) && defined(TRANSA))
          temp = bk-off;
#elif defined(LEFT)
          temp = off+1;	// number of values in A
#else
          temp = off+1;	// number of values in B
#endif
          for (k=0; k<temp; k++) KERNEL_k1m1n1
          SAVE_m1n1
#if (defined(LEFT) && defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
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
#if defined(TRMMKERNEL) && !defined(LEFT)
        off += 1;
#endif
        k = (bk<<0);
        bb = bb+k;
        c_ptr += ldc-bm;
      }
      return 0;
}
