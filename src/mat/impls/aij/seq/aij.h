/* $Id: aij.h,v 1.31 1998/03/06 00:14:28 bsmith Exp bsmith $ */

#include "src/mat/matimpl.h"
#include <math.h>

#if !defined(__AIJ_H)
#define __AIJ_H

/* Info about i-nodes (identical nodes) */
typedef struct {
  int node_count;                    /* number of inodes */
  int *size;                         /* size of each inode */
  int limit;                         /* inode limit */
  int max_limit;                     /* maximum supported inode limit */
} Mat_SeqAIJ_Inode;

/*  
  MATSEQAIJ format - Compressed row storage (also called Yale sparse matrix
  format), compatible with Fortran.  The i[] and j[] arrays start at 1,
  or 0, depending on the value of shift.  For example, in Fortran 
  j[i[k]+p+shift] is the pth column in row k.  Note that the diagonal
  matrix elements are stored with the rest of the nonzeros (not separately).
*/

typedef struct {
  int              sorted;           /* if true, rows are sorted by increasing columns */
  int              roworiented;      /* if true, row-oriented input, default */
  int              nonew;            /* 1 don't add new nonzeros, -1 generate error on new */
  int              singlemalloc;     /* if true a, i, and j have been obtained with
                                        one big malloc */
  int              m, n;             /* rows, columns */
  int              nz, maxnz;        /* nonzeros, allocated nonzeros */
  int              *diag;            /* pointers to diagonal elements */
  int              *i;               /* pointer to beginning of each row */
  int              *imax;            /* maximum space allocated for each row */
  int              *ilen;            /* actual length of each row */
  int              *j;               /* column values: j + i[k] - 1 is start of row k */
  Scalar           *a;               /* nonzero elements */
  IS               row, col, icol;   /* index sets, used for reorderings */
  Scalar           *solve_work;      /* work space used in MatSolve */
  void             *spptr;           /* pointer for special library like SuperLU */
  int              indexshift;       /* zero or -one for C or Fortran indexing */
  Mat_SeqAIJ_Inode inode;            /* identical node informaton */
  int              reallocs;         /* number of mallocs done during MatSetValues() 
                                        as more values are set than were prealloced */
  PetscTruth       ilu_preserve_row_sums;
  Scalar           *saved_values;     /* location for stashing nonzero values of matrix */
} Mat_SeqAIJ;

extern int MatILUFactorSymbolic_SeqAIJ(Mat,IS,IS,double,int,Mat *);
extern int MatConvert_SeqAIJ(Mat,MatType,Mat *);
extern int MatDuplicate_SeqAIJ(Mat,MatDuplicateOption, Mat*);
extern int MatMarkDiag_SeqAIJ(Mat);

extern int MatMult_SeqAIJ(Mat A,Vec,Vec);
extern int MatMultAdd_SeqAIJ(Mat A,Vec,Vec,Vec);
extern int MatMultTrans_SeqAIJ(Mat A,Vec,Vec);
extern int MatMultTransAdd_SeqAIJ(Mat A,Vec,Vec,Vec);
extern int MatRelax_SeqAIJ(Mat,Vec,double,MatSORType,double,int,Vec);

#endif
