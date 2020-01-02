#include "common.h"
//this file is a modification of the file "trmm_kernel_4x8.c" from OpenBLAS.

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

#define ACCLIST \
  FLOAT c01,c02,c03,c04; FLOAT c11,c12,c13,c14; FLOAT c21,c22,c23,c24; FLOAT c31,c32,c33,c34;\
  FLOAT c41,c42,c43,c44; FLOAT c51,c52,c53,c54; FLOAT c61,c62,c63,c64; FLOAT c71,c72,c73,c74;
#define TMPLIST FLOAT a01,a02,a03,a04,b00;
#define INIT_m1n1 c01=0.0;
#define INIT_m1n2 INIT_m1n1 c11=0.0;
#define INIT_m1n4 INIT_m1n2 c21=0.0; c31=0.0;
#define INIT_m1n8 INIT_m1n4 c41=0.0; c51=0.0; c61=0.0; c71=0.0;
#define INIT_m2n1 c01=c02=0.0;
#define INIT_m2n2 INIT_m2n1 c11=c12=0.0;
#define INIT_m2n4 INIT_m2n2 c21=c22=0.0; c31=c32=0.0;
#define INIT_m2n8 INIT_m2n4 c41=c42=0.0; c51=c52=0.0; c61=c62=0.0; c71=c72=0.0;
#define INIT_m4n1 c01=c02=c03=c04=0.0;
#define INIT_m4n2 INIT_m4n1 c11=c12=c13=c14=0.0;
#define INIT_m4n4 INIT_m4n2 c21=c22=c23=c24=0.0; c31=c32=c33=c34=0.0;
#define INIT_m4n8 INIT_m4n4 c41=c42=c43=c44=0.0; c51=c52=c53=c54=0.0; c61=c62=c63=c64=0.0; c71=c72=c73=c74=0.0;
#define SAVE_m1n1 c_tmp=c_ptr; c_ptr++; c_tmp[0]=c01*alpha;
#define SAVE_m1n2 SAVE_m1n1 c_tmp+=ldc; c_tmp[0]=c11*alpha;
#define SAVE_m1n4 SAVE_m1n2\
  c_tmp+=ldc; c_tmp[0]=c21*alpha;\
  c_tmp+=ldc; c_tmp[0]=c31*alpha;
#define SAVE_m1n8 SAVE_m1n4\
  c_tmp+=ldc; c_tmp[0]=c41*alpha;\
  c_tmp+=ldc; c_tmp[0]=c51*alpha;\
  c_tmp+=ldc; c_tmp[0]=c61*alpha;\
  c_tmp+=ldc; c_tmp[0]=c71*alpha;
#define SAVE_m2n1 c_tmp=c_ptr; c_ptr+=2; c_tmp[0]=c01*alpha; c_tmp[1]=c02*alpha;
#define SAVE_m2n2 SAVE_m2n1 c_tmp+=ldc; c_tmp[0]=c11*alpha; c_tmp[1]=c12*alpha;
#define SAVE_m2n4 SAVE_m2n2\
  c_tmp+=ldc; c_tmp[0]=c21*alpha; c_tmp[1]=c22*alpha;\
  c_tmp+=ldc; c_tmp[0]=c31*alpha; c_tmp[1]=c32*alpha;
#define SAVE_m2n8 SAVE_m2n4\
  c_tmp+=ldc; c_tmp[0]=c41*alpha; c_tmp[1]=c42*alpha;\
  c_tmp+=ldc; c_tmp[0]=c51*alpha; c_tmp[1]=c52*alpha;\
  c_tmp+=ldc; c_tmp[0]=c61*alpha; c_tmp[1]=c62*alpha;\
  c_tmp+=ldc; c_tmp[0]=c71*alpha; c_tmp[1]=c72*alpha;
#define SAVE_m4n1 c_tmp=c_ptr; c_ptr+=4; c_tmp[0]=c01*alpha; c_tmp[1]=c02*alpha; c_tmp[2]=c03*alpha; c_tmp[3]=c04*alpha;
#define SAVE_m4n2 SAVE_m4n1 c_tmp+=ldc; c_tmp[0]=c11*alpha; c_tmp[1]=c12*alpha; c_tmp[2]=c13*alpha; c_tmp[3]=c14*alpha;
#define SAVE_m4n4 SAVE_m4n2\
  c_tmp+=ldc; c_tmp[0]=c21*alpha; c_tmp[1]=c22*alpha; c_tmp[2]=c23*alpha; c_tmp[3]=c24*alpha;\
  c_tmp+=ldc; c_tmp[0]=c31*alpha; c_tmp[1]=c32*alpha; c_tmp[2]=c33*alpha; c_tmp[3]=c34*alpha;
#define SAVE_m4n8 SAVE_m4n4\
  c_tmp+=ldc; c_tmp[0]=c41*alpha; c_tmp[1]=c42*alpha; c_tmp[2]=c43*alpha; c_tmp[3]=c44*alpha;\
  c_tmp+=ldc; c_tmp[0]=c51*alpha; c_tmp[1]=c52*alpha; c_tmp[2]=c53*alpha; c_tmp[3]=c54*alpha;\
  c_tmp+=ldc; c_tmp[0]=c61*alpha; c_tmp[1]=c62*alpha; c_tmp[2]=c63*alpha; c_tmp[3]=c64*alpha;\
  c_tmp+=ldc; c_tmp[0]=c71*alpha; c_tmp[1]=c72*alpha; c_tmp[2]=c73*alpha; c_tmp[3]=c74*alpha;
#define KERNEL_h_k1m1n1 \
  a01=ptrba[0]; ptrba++;\
  b00=ptrbb[0]; c01+=a01*b00;
#define KERNEL_h_k1m1n2 KERNEL_h_k1m1n1\
  b00=ptrbb[1]; c11+=a01*b00;
#define KERNEL_h_k1m1n4 KERNEL_h_k1m1n2\
  b00=ptrbb[2]; c21+=a01*b00;\
  b00=ptrbb[3]; c31+=a01*b00;
#define KERNEL_h_k1m1n8 KERNEL_h_k1m1n4\
  b00=ptrbb[4]; c41+=a01*b00;\
  b00=ptrbb[5]; c51+=a01*b00;\
  b00=ptrbb[6]; c61+=a01*b00;\
  b00=ptrbb[7]; c71+=a01*b00;
#define KERNEL_k1m1n1 KERNEL_h_k1m1n1 ptrbb++;
#define KERNEL_k1m1n2 KERNEL_h_k1m1n2 ptrbb+=2;
#define KERNEL_k1m1n4 KERNEL_h_k1m1n4 ptrbb+=4;
#define KERNEL_k1m1n8 KERNEL_h_k1m1n8 ptrbb+=8;
#define KERNEL_h_k1m2n1 \
  a01=ptrba[0]; a02=ptrba[1]; ptrba+=2;\
  b00=ptrbb[0]; c01+=a01*b00; c02+=a02*b00;
#define KERNEL_h_k1m2n2 KERNEL_h_k1m2n1\
  b00=ptrbb[1]; c11+=a01*b00; c12+=a02*b00;
#define KERNEL_h_k1m2n4 KERNEL_h_k1m2n2\
  b00=ptrbb[2]; c21+=a01*b00; c22+=a02*b00;\
  b00=ptrbb[3]; c31+=a01*b00; c32+=a02*b00;
#define KERNEL_h_k1m2n8 KERNEL_h_k1m2n4\
  b00=ptrbb[4]; c41+=a01*b00; c42+=a02*b00;\
  b00=ptrbb[5]; c51+=a01*b00; c52+=a02*b00;\
  b00=ptrbb[6]; c61+=a01*b00; c62+=a02*b00;\
  b00=ptrbb[7]; c71+=a01*b00; c72+=a02*b00;
#define KERNEL_k1m2n1 KERNEL_h_k1m2n1 ptrbb++;
#define KERNEL_k1m2n2 KERNEL_h_k1m2n2 ptrbb+=2;
#define KERNEL_k1m2n4 KERNEL_h_k1m2n4 ptrbb+=4;
#define KERNEL_k1m2n8 KERNEL_h_k1m2n8 ptrbb+=8;
#define KERNEL_h_k1m4n1 \
  a01=ptrba[0]; a02=ptrba[1]; a03=ptrba[2]; a04=ptrba[3]; ptrba+=4;\
  b00=ptrbb[0]; c01+=a01*b00; c02+=a02*b00; c03+=a03+b00; c04+=a04*b00;
#define KERNEL_h_k1m4n2 KERNEL_h_k1m4n1\
  b00=ptrbb[1]; c11+=a01*b00; c12+=a02*b00; c13+=a03+b00; c14+=a04*b00;
#define KERNEL_h_k1m4n4 KERNEL_h_k1m4n2\
  b00=ptrbb[2]; c21+=a01*b00; c22+=a02*b00; c23+=a03+b00; c24+=a04*b00;\
  b00=ptrbb[3]; c31+=a01*b00; c32+=a02*b00; c33+=a03+b00; c34+=a04*b00;
#define KERNEL_h_k1m4n8 KERNEL_h_k1m4n4\
  b00=ptrbb[4]; c41+=a01*b00; c42+=a02*b00; c43+=a03+b00; c44+=a04*b00;\
  b00=ptrbb[5]; c51+=a01*b00; c52+=a02*b00; c53+=a03+b00; c54+=a04*b00;\
  b00=ptrbb[6]; c61+=a01*b00; c62+=a02*b00; c63+=a03+b00; c64+=a04*b00;\
  b00=ptrbb[7]; c71+=a01*b00; c72+=a02*b00; c73+=a03+b00; c74+=a04*b00;
#define KERNEL_k1m4n1 KERNEL_h_k1m4n1 ptrbb++;
#define KERNEL_k1m4n2 KERNEL_h_k1m4n2 ptrbb+=2;
#define KERNEL_k1m4n4 KERNEL_h_k1m4n4 ptrbb+=4;
#define KERNEL_k1m4n8 KERNEL_h_k1m4n8 ptrbb+=8;

int CNAME(BLASLONG bm,BLASLONG bn,BLASLONG bk,FLOAT alpha,FLOAT* ba,FLOAT* bb,FLOAT* C,BLASLONG ldc ,BLASLONG offset){
      BLASLONG i,j,k;
      FLOAT *c_ptr,*c_tmp,*ptrba,*ptrbb,*b_head;
      BLASLONG off, temp;
      ACCLIST
      TMPLIST
        
#ifndef LEFT
      off = -offset;
#endif
      c_ptr = C; b_head = bb;
      for (j=0; j<bn/8; j++){
#ifdef LEFT
        off = offset;
#endif
        ptrba = ba;
        for (i=0; i<bm/4; i+=1){
          ptrbb = b_head;
          INIT_set_k_and_pointers(4,8)
          INIT_m4n8
          for (k=0; k<temp; k++) {KERNEL_k1m4n8}
          SAVE_m4n8 //don't forget to update c_ptr
          SAVE_set_pointers(4,8)
#ifdef LEFT
          off += 4; // number of values in A
#endif
        }//i->bm/4 loop tail
        if ( bm & 2 ){
          ptrbb = b_head;
          INIT_set_k_and_pointers(2,8)
          INIT_m2n8
          for (k=0; k<temp; k++) {KERNEL_k1m2n8}
          SAVE_m2n8
          SAVE_set_pointers(2,8)
#ifdef LEFT
          off += 2; // number of values in A
#endif
        }
        if ( bm & 1 ){
          ptrbb = b_head;
          INIT_set_k_and_pointers(1,8)
          INIT_m1n8
          for (k=0; k<temp; k++) {KERNEL_k1m1n8}
          SAVE_m1n8
          SAVE_set_pointers(1,8)
#ifdef LEFT
          off += 1; // number of values in A
#endif
        }
#ifndef LEFT
        off += 8;
#endif
        b_head += bk*8;
        c_ptr += ldc*8-bm;
      }//j -> bn/8 loop tail

      for (j=0; j<(bn&4); j+=4){
#ifdef LEFT
        off = offset;
#endif
        ptrba = ba;
        for (i=0; i<bm/4; i+=1){
          ptrbb = b_head;
          INIT_set_k_and_pointers(4,4)
          INIT_m4n4
          for (k=0; k<temp; k++) {KERNEL_k1m4n4}
          SAVE_m4n4
          SAVE_set_pointers(4,4)
#ifdef LEFT
          off += 4; // number of values in A
#endif
        }//i -> bm/4 loop tail
        if ( bm & 2 ){
          ptrbb = b_head;
          INIT_set_k_and_pointers(2,4)
          INIT_m2n4
          for (k=0; k<temp; k++) {KERNEL_k1m2n4}
          SAVE_m2n4
          SAVE_set_pointers(2,4)
#ifdef LEFT
          off += 2; // number of values in A
#endif
        }
        if ( bm & 1 ){
          ptrbb = b_head;
          INIT_set_k_and_pointers(1,4)
          INIT_m1n4
          for (k=0; k<temp; k++) {KERNEL_k1m1n4}
          SAVE_m1n4
          SAVE_set_pointers(1,4)
#ifdef LEFT
          off += 1; // number of values in A
#endif
        }
#ifndef LEFT
        off += 4;
#endif
        b_head += bk*4;
        c_ptr += ldc*4-bm;
      }// condition j -> bn&4 tail
      for (j=0; j<(bn&2); j+=2){
#ifdef LEFT
        off = offset;
#endif
        ptrba = ba;
        for (i=0; i<bm/4; i+=1){
          ptrbb = b_head;
          INIT_set_k_and_pointers(4,2)
          INIT_m4n2
          for (k=0; k<temp; k++) {KERNEL_k1m4n2}
          SAVE_m4n2
          SAVE_set_pointers(4,2)
#ifdef LEFT
          off += 4; // number of values in A
#endif
        }
        if ( bm & 2 ){
          ptrbb = b_head;
          INIT_set_k_and_pointers(2,2)
          INIT_m2n2
          for (k=0; k<temp; k++) {KERNEL_k1m2n2}
          SAVE_m2n2
          SAVE_set_pointers(2,2)
#ifdef LEFT
          off += 2; // number of values in A
#endif
        }
        if ( bm & 1 ){
          ptrbb = b_head;
          INIT_set_k_and_pointers(1,2)
          INIT_m1n2
          for (k=0; k<temp; k++) {KERNEL_k1m1n2}
          SAVE_m1n2
          SAVE_set_pointers(1,2)
#ifdef LEFT
          off += 1; // number of values in A
#endif
        }
#ifndef LEFT
        off += 2;
#endif
        b_head += bk*2;
        c_ptr += ldc*2-bm;
      }
      for (j=0; j<(bn&1); j+=1){
#ifdef LEFT
        off = offset;
#endif
        ptrba = ba;
        for (i=0; i<bm/4; i+=1){
          ptrbb = b_head;
          INIT_set_k_and_pointers(4,1)
          INIT_m4n1
          for (k=0; k<temp; k++) {KERNEL_k1m4n1}
          SAVE_m4n1
          SAVE_set_pointers(4,1)
#ifdef LEFT
          off += 4; // number of values in A
#endif
        }
        if ( bm & 2 ){
          ptrbb = b_head;
          INIT_set_k_and_pointers(2,1)
          INIT_m2n1
          for (k=0; k<temp; k++) {KERNEL_k1m2n1}
          SAVE_m2n1
          SAVE_set_pointers(2,1)
#ifdef LEFT
          off += 2; // number of values in A
#endif
        }
        if ( bm & 1 ){
          ptrbb = b_head;
          INIT_set_k_and_pointers(1,1)
          INIT_m1n1
          for (k=0; k<temp; k++) {KERNEL_k1m1n1}
          SAVE_m1n1
          SAVE_set_pointers(1,1)
#ifdef LEFT
          off += 1; // number of values in A
#endif
        }
#ifndef LEFT
        off += 1;
#endif
        b_head += bk;
        c_ptr += ldc-bm;
      }
      return 0;
}
