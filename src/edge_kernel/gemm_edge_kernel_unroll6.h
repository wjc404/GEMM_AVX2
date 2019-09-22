# define INIT_1col {\
   c1=IRREG_VEC_ZERO();_mm_prefetch((char *)ctemp,_MM_HINT_T0);\
   c2=IRREG_VEC_ZERO();_mm_prefetch((char *)(ctemp+64/IRREG_SIZE-1),_MM_HINT_T0);\
}
# define INIT_ncol {\
   c1=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c2=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE-1),_MM_HINT_T0);cpref+=ldc;\
   c3=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c4=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE-1),_MM_HINT_T0);cpref+=ldc;\
   c5=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c6=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE-1),_MM_HINT_T0);cpref+=ldc;\
   c7=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c8=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE-1),_MM_HINT_T0);cpref+=ldc;\
   c9=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c10=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE-1),_MM_HINT_T0);cpref+=ldc;\
   c11=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c12=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE-1),_MM_HINT_T0);\
}
# define KERNELkr {\
   t1=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   t2=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   t3=IRREG_VEC_BROAD(btemp);btemp++;\
   c1=IRREG_VEC_FMADD(t1,t3,c1);c2=IRREG_VEC_FMADD(t2,t3,c2);\
}
# define KERNELk1 {\
   KERNELkr\
   t3=IRREG_VEC_BROAD(btemp);btemp++;\
   c3=IRREG_VEC_FMADD(t1,t3,c3);c4=IRREG_VEC_FMADD(t2,t3,c4);\
   t3=IRREG_VEC_BROAD(btemp);btemp++;\
   c5=IRREG_VEC_FMADD(t1,t3,c5);c6=IRREG_VEC_FMADD(t2,t3,c6);\
   t3=IRREG_VEC_BROAD(btemp);btemp++;\
   c7=IRREG_VEC_FMADD(t1,t3,c7);c8=IRREG_VEC_FMADD(t2,t3,c8);\
   t3=IRREG_VEC_BROAD(btemp);btemp++;\
   c9=IRREG_VEC_FMADD(t1,t3,c9);c10=IRREG_VEC_FMADD(t2,t3,c10);\
   t3=IRREG_VEC_BROAD(btemp);btemp++;\
   c11=IRREG_VEC_FMADD(t1,t3,c11);c12=IRREG_VEC_FMADD(t2,t3,c12);\
}
# define KERNELk2 {\
    _mm_prefetch((char *)(atemp+A_PR_BYTE/IRREG_SIZE),_MM_HINT_T0);\
    _mm_prefetch((char *)(btemp+B_PR_ELEM),_MM_HINT_T0);\
    KERNELk1\
    _mm_prefetch((char *)(atemp+A_PR_BYTE/IRREG_SIZE),_MM_HINT_T0);\
    _mm_prefetch((char *)(btemp+B_PR_ELEM),_MM_HINT_T0);\
    KERNELk1\
}
# define sub_store_c_1col(c1,c2) {\
   c1=IRREG_VEC_ADD(IRREG_VEC_LOADU(ctemp),c1);\
   c2=IRREG_VEC_ADD(IRREG_VEC_LOADU(ctemp+32/IRREG_SIZE),c2);\
   IRREG_VEC_STOREU(ctemp,c1);\
   IRREG_VEC_STOREU(ctemp+32/IRREG_SIZE,c2);\
   ctemp+=ldc;\
}
# define sub_storeedgem_c_1col(c1,c2) {\
   c1=IRREG_VEC_ADD(IRREG_VEC_MASKLOAD(ctemp,ml1),c1);\
   c2=IRREG_VEC_ADD(IRREG_VEC_MASKLOAD(ctemp+32/IRREG_SIZE,ml2),c2);\
   IRREG_VEC_MASKSTORE(ctemp,ml1,c1);\
   IRREG_VEC_MASKSTORE(ctemp+32/IRREG_SIZE,ml2,c2);\
   ctemp+=ldc;\
}
# define STORE_C_1col sub_store_c_1col(c1,c2)
# define STOREEDGEM_C_1col sub_storeedgem_c_1col(c1,c2)
# define STORE_C_ncol {\
   sub_store_c_1col(c1,c2)\
   sub_store_c_1col(c3,c4)\
   sub_store_c_1col(c5,c6)\
   sub_store_c_1col(c7,c8)\
   sub_store_c_1col(c9,c10)\
   sub_store_c_1col(c11,c12)\
}
# define STOREEDGEM_C_ncol {\
   sub_storeedgem_c_1col(c1,c2)\
   sub_storeedgem_c_1col(c3,c4)\
   sub_storeedgem_c_1col(c5,c6)\
   sub_storeedgem_c_1col(c7,c8)\
   sub_storeedgem_c_1col(c9,c10)\
   sub_storeedgem_c_1col(c11,c12)\
}
