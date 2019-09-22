# define INIT_1col {\
   c1=IRREG_VEC_ZERO();_mm_prefetch((char *)ctemp,_MM_HINT_T0);\
   c2=IRREG_VEC_ZERO();_mm_prefetch((char *)(ctemp+64/IRREG_SIZE),_MM_HINT_T0);\
   c3=IRREG_VEC_ZERO();_mm_prefetch((char *)(ctemp+96/IRREG_SIZE-1),_MM_HINT_T0);\
}
# define INIT_ncol {\
   c1=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c2=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE),_MM_HINT_T0);\
   c3=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+96/IRREG_SIZE-1),_MM_HINT_T0);cpref+=ldc;\
   c4=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c5=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE),_MM_HINT_T0);\
   c6=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+96/IRREG_SIZE-1),_MM_HINT_T0);cpref+=ldc;\
   c7=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c8=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE),_MM_HINT_T0);\
   c9=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+96/IRREG_SIZE-1),_MM_HINT_T0);cpref+=ldc;\
   c10=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c11=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE),_MM_HINT_T0);\
   c12=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+96/IRREG_SIZE-1),_MM_HINT_T0);\
}
# define KERNELkr {\
   t1=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   t2=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   t3=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   t4=IRREG_VEC_BROAD(btemp);btemp++;\
   c1=IRREG_VEC_FMADD(t1,t4,c1);c2=IRREG_VEC_FMADD(t2,t4,c2);c3=IRREG_VEC_FMADD(t3,t4,c3);\
}
# define KERNELk1 {\
   KERNELkr\
   t4=IRREG_VEC_BROAD(btemp);btemp++;\
   c4=IRREG_VEC_FMADD(t1,t4,c4);c5=IRREG_VEC_FMADD(t2,t4,c5);c6=IRREG_VEC_FMADD(t3,t4,c6);\
   t4=IRREG_VEC_BROAD(btemp);btemp++;\
   c7=IRREG_VEC_FMADD(t1,t4,c7);c8=IRREG_VEC_FMADD(t2,t4,c8);c9=IRREG_VEC_FMADD(t3,t4,c9);\
   t4=IRREG_VEC_BROAD(btemp);btemp++;\
   c10=IRREG_VEC_FMADD(t1,t4,c10);c11=IRREG_VEC_FMADD(t2,t4,c11);c12=IRREG_VEC_FMADD(t3,t4,c12);\
}
# define KERNELk2 {\
   _mm_prefetch((char *)(atemp+A_PR_BYTE/IRREG_SIZE),_MM_HINT_T0);\
   _mm_prefetch((char *)(btemp+B_PR_ELEM),_MM_HINT_T0);\
   KERNELk1\
   _mm_prefetch((char *)(atemp+(A_PR_BYTE-32)/IRREG_SIZE),_MM_HINT_T0);\
   _mm_prefetch((char *)(atemp+(A_PR_BYTE+32)/IRREG_SIZE),_MM_HINT_T0);\
   KERNELk1\
}
# define sub_store_c_1col(c1,c2,c3) {\
   c1=IRREG_VEC_ADD(IRREG_VEC_LOADU(ctemp),c1);\
   c2=IRREG_VEC_ADD(IRREG_VEC_LOADU(ctemp+32/IRREG_SIZE),c2);\
   c3=IRREG_VEC_ADD(IRREG_VEC_LOADU(ctemp+64/IRREG_SIZE),c3);\
   IRREG_VEC_STOREU(ctemp,c1);\
   IRREG_VEC_STOREU(ctemp+32/IRREG_SIZE,c2);\
   IRREG_VEC_STOREU(ctemp+64/IRREG_SIZE,c3);\
   ctemp+=ldc;\
}
# define sub_storeedgem_c_1col(c1,c2,c3) {\
   c1=IRREG_VEC_ADD(IRREG_VEC_MASKLOAD(ctemp,ml1),c1);\
   c2=IRREG_VEC_ADD(IRREG_VEC_MASKLOAD(ctemp+32/IRREG_SIZE,ml2),c2);\
   c3=IRREG_VEC_ADD(IRREG_VEC_MASKLOAD(ctemp+64/IRREG_SIZE,ml3),c3);\
   IRREG_VEC_MASKSTORE(ctemp,ml1,c1);\
   IRREG_VEC_MASKSTORE(ctemp+32/IRREG_SIZE,ml2,c2);\
   IRREG_VEC_MASKSTORE(ctemp+64/IRREG_SIZE,ml3,c3);\
   ctemp+=ldc;\
}
# define STORE_C_1col sub_store_c_1col(c1,c2,c3)
# define STOREEDGEM_C_1col sub_storeedgem_c_1col(c1,c2,c3)
# define STORE_C_ncol {\
   sub_store_c_1col(c1,c2,c3)\
   sub_store_c_1col(c4,c5,c6)\
   sub_store_c_1col(c7,c8,c9)\
   sub_store_c_1col(c10,c11,c12)\
}
# define STOREEDGEM_C_ncol {\
   sub_storeedgem_c_1col(c1,c2,c3)\
   sub_storeedgem_c_1col(c4,c5,c6)\
   sub_storeedgem_c_1col(c7,c8,c9)\
   sub_storeedgem_c_1col(c10,c11,c12)\
}
