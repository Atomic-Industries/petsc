/* $Id: baij.h,v 1.14 1998/12/17 22:10:39 bsmith Exp bsmith $ */

#include "src/mat/matimpl.h"

#if !defined(__BAIJ_H)
#define __BAIJ_H

/*  
  MATSEQBAIJ format - Block compressed row storage. The i[] and j[] 
  arrays start at 0.
*/

typedef struct {
  int              sorted;       /* if true, rows are sorted by increasing columns */
  int              roworiented;  /* if true, row-oriented input, default */
  int              nonew;        /* 1 don't add new nonzeros, -1 generate error on new */
  int              singlemalloc; /* if true a, i, and j have been obtained with
                                        one big malloc */
  int              m,n;          /* rows, columns */
  int              bs,bs2;       /* block size, square of block size */
  int              mbs,nbs;      /* rows/bs, columns/bs */
  int              nz,maxnz;     /* nonzeros, allocated nonzeros */
  int              *diag;        /* pointers to diagonal elements */
  int              *i;           /* pointer to beginning of each row */
  int              *imax;        /* maximum space allocated for each row */
  int              *ilen;        /* actual length of each row */
  int              *j;           /* column values: j + i[k] - 1 is start of row k */
  MatScalar        *a;           /* nonzero elements */
  IS               row,col,icol; /* index sets, used for reorderings */
  Scalar           *solve_work;  /* work space used in MatSolve */
  void             *spptr;       /* pointer for special library like SuperLU */
  int              reallocs;     /* number of mallocs done during MatSetValues() 
                                    as more values are set then were preallocated */
  Scalar           *mult_work;   /* work array for matrix vector product*/
} Mat_SeqBAIJ;

extern int MatILUFactorSymbolic_SeqBAIJ(Mat,IS,IS,MatILUInfo*,Mat *);
extern int MatConvert_SeqBAIJ(Mat,MatType,Mat *);
extern int MatDuplicate_SeqBAIJ(Mat,MatDuplicateOption, Mat*);
extern int MatMarkDiag_SeqBAIJ(Mat);
extern int MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering(Mat,Mat*);
extern int MatSolve_SeqBAIJ_4_NaturalOrdering(Mat,Vec,Vec);
extern int MatLUFactorNumeric_SeqBAIJ_5_NaturalOrdering(Mat,Mat*);
extern int MatSolve_SeqBAIJ_5_NaturalOrdering(Mat,Vec,Vec);


#endif
