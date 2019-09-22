# define INIT_1col {\
   c1=IRREG_VEC_ZERO();_mm_prefetch((char *)ctemp,_MM_HINT_T0);\
   c2=IRREG_VEC_ZERO();_mm_prefetch((char *)(ctemp+64/IRREG_SIZE),_MM_HINT_T0);\
   c3=IRREG_VEC_ZERO();_mm_prefetch((char *)(ctemp+128/IRREG_SIZE-1),_MM_HINT_T0);\
   c4=IRREG_VEC_ZERO();\
}
# define INIT_ncol {\
   c1=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c2=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE),_MM_HINT_T0);\
   c3=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+128/IRREG_SIZE-1),_MM_HINT_T0);\
   c4=IRREG_VEC_ZERO();cpref+=ldc;\
   c5=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c6=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE),_MM_HINT_T0);\
   c7=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+128/IRREG_SIZE-1),_MM_HINT_T0);\
   c8=IRREG_VEC_ZERO();cpref+=ldc;\
   c9=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c10=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE),_MM_HINT_T0);\
   c11=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+128/IRREG_SIZE-1),_MM_HINT_T0);\
   c12=IRREG_VEC_ZERO();\
}
# define KERNELkr {\
   t1=IRREG_VEC_BROAD(btemp);btemp++;\
   t4=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   c1=IRREG_VEC_FMADD(t4,t1,c1);\
   t4=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   c2=IRREG_VEC_FMADD(t4,t1,c2);\
   t4=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   c3=IRREG_VEC_FMADD(t4,t1,c3);\
   t4=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   c4=IRREG_VEC_FMADD(t4,t1,c4);\
}
# define KERNELk1 {\
   t1=IRREG_VEC_BROAD(btemp);btemp++;\
   t2=IRREG_VEC_BROAD(btemp);btemp++;\
   t3=IRREG_VEC_BROAD(btemp);btemp++;\
   t4=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   c1=IRREG_VEC_FMADD(t4,t1,c1);c5=IRREG_VEC_FMADD(t4,t2,c5);c9=IRREG_VEC_FMADD(t4,t3,c9);\
   t4=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   c2=IRREG_VEC_FMADD(t4,t1,c2);c6=IRREG_VEC_FMADD(t4,t2,c6);c10=IRREG_VEC_FMADD(t4,t3,c10);\
   t4=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   c3=IRREG_VEC_FMADD(t4,t1,c3);c7=IRREG_VEC_FMADD(t4,t2,c7);c11=IRREG_VEC_FMADD(t4,t3,c11);\
   t4=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   c4=IRREG_VEC_FMADD(t4,t1,c4);c8=IRREG_VEC_FMADD(t4,t2,c8);c12=IRREG_VEC_FMADD(t4,t3,c12);\
}
# define KERNELk2 {\
   _mm_prefetch((char *)(atemp+A_PR_BYTE/IRREG_SIZE),_MM_HINT_T0);\
   _mm_prefetch((char *)(atemp+(A_PR_BYTE+64)/IRREG_SIZE),_MM_HINT_T0);\
   _mm_prefetch((char *)(btemp+B_PR_ELEM),_MM_HINT_T0);\
   KERNELk1\
   _mm_prefetch((char *)(atemp+A_PR_BYTE/IRREG_SIZE),_MM_HINT_T0);\
   _mm_prefetch((char *)(atemp+(A_PR_BYTE+64)/IRREG_SIZE),_MM_HINT_T0);\
   KERNELk1\
}
# define sub_store_c_1col(c1,c2,c3,c4) {\
   c1=IRREG_VEC_ADD(IRREG_VEC_LOADU(ctemp),c1);\
   c2=IRREG_VEC_ADD(IRREG_VEC_LOADU(ctemp+32/IRREG_SIZE),c2);\
   c3=IRREG_VEC_ADD(IRREG_VEC_LOADU(ctemp+64/IRREG_SIZE),c3);\
   c4=IRREG_VEC_ADD(IRREG_VEC_LOADU(ctemp+96/IRREG_SIZE),c4);\
   IRREG_VEC_STOREU(ctemp,c1);\
   IRREG_VEC_STOREU(ctemp+32/IRREG_SIZE,c2);\
   IRREG_VEC_STOREU(ctemp+64/IRREG_SIZE,c3);\
   IRREG_VEC_STOREU(ctemp+96/IRREG_SIZE,c4);\
   ctemp+=ldc;\
}
# define sub_storeedgem_c_1col(c1,c2,c3,c4) {\
   c1=IRREG_VEC_ADD(IRREG_VEC_MASKLOAD(ctemp,ml1),c1);\
   c2=IRREG_VEC_ADD(IRREG_VEC_MASKLOAD(ctemp+32/IRREG_SIZE,ml2),c2);\
   c3=IRREG_VEC_ADD(IRREG_VEC_MASKLOAD(ctemp+64/IRREG_SIZE,ml3),c3);\
   c4=IRREG_VEC_ADD(IRREG_VEC_MASKLOAD(ctemp+96/IRREG_SIZE,ml4),c4);\
   IRREG_VEC_MASKSTORE(ctemp,ml1,c1);\
   IRREG_VEC_MASKSTORE(ctemp+32/IRREG_SIZE,ml2,c2);\
   IRREG_VEC_MASKSTORE(ctemp+64/IRREG_SIZE,ml3,c3);\
   IRREG_VEC_MASKSTORE(ctemp+96/IRREG_SIZE,ml4,c4);\
   ctemp+=ldc;\
}
# define STORE_C_1col sub_store_c_1col(c1,c2,c3,c4)
# define STOREEDGEM_C_1col sub_storeedgem_c_1col(c1,c2,c3,c4)
# define STORE_C_ncol {\
   sub_store_c_1col(c1,c2,c3,c4)\
   sub_store_c_1col(c5,c6,c7,c8)\
   sub_store_c_1col(c9,c10,c11,c12)\
}
# define STOREEDGEM_C_ncol {\
   sub_storeedgem_c_1col(c1,c2,c3,c4)\
   sub_storeedgem_c_1col(c5,c6,c7,c8)\
   sub_storeedgem_c_1col(c9,c10,c11,c12)\
}
