#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: bdiag.c,v 1.167 1998/12/03 04:00:32 bsmith Exp bsmith $";
#endif

/* Block diagonal matrix format */

#include "sys.h"
#include "src/mat/impls/bdiag/seq/bdiag.h"
#include "src/vec/vecimpl.h"
#include "src/inline/ilu.h"

extern int MatSetValues_SeqBDiag_1(Mat,int,int *,int,int *,Scalar *,InsertMode);
extern int MatSetValues_SeqBDiag_N(Mat,int,int *,int,int *,Scalar *,InsertMode);
extern int MatGetValues_SeqBDiag_1(Mat,int,int *,int,int *,Scalar *);
extern int MatGetValues_SeqBDiag_N(Mat,int,int *,int,int *,Scalar *);
extern int MatMult_SeqBDiag_1(Mat,Vec,Vec);
extern int MatMult_SeqBDiag_2(Mat,Vec,Vec);
extern int MatMult_SeqBDiag_3(Mat,Vec,Vec);
extern int MatMult_SeqBDiag_4(Mat,Vec,Vec);
extern int MatMult_SeqBDiag_5(Mat,Vec,Vec);
extern int MatMult_SeqBDiag_N(Mat,Vec,Vec);
extern int MatMultAdd_SeqBDiag_1(Mat,Vec,Vec,Vec);
extern int MatMultAdd_SeqBDiag_2(Mat,Vec,Vec,Vec);
extern int MatMultAdd_SeqBDiag_3(Mat,Vec,Vec,Vec);
extern int MatMultAdd_SeqBDiag_4(Mat,Vec,Vec,Vec);
extern int MatMultAdd_SeqBDiag_5(Mat,Vec,Vec,Vec);
extern int MatMultAdd_SeqBDiag_N(Mat,Vec,Vec,Vec);
extern int MatMultTrans_SeqBDiag_1(Mat,Vec,Vec);
extern int MatMultTrans_SeqBDiag_N(Mat,Vec,Vec);
extern int MatMultTransAdd_SeqBDiag_1(Mat,Vec,Vec,Vec);
extern int MatMultTransAdd_SeqBDiag_N(Mat,Vec,Vec,Vec);
extern int MatRelax_SeqBDiag_N(Mat,Vec,double,MatSORType,double,int,Vec);
extern int MatRelax_SeqBDiag_1(Mat,Vec,double,MatSORType,double,int,Vec);
extern int MatView_SeqBDiag(Mat,Viewer);
extern int MatGetInfo_SeqBDiag(Mat,MatInfoType,MatInfo*);
extern int MatGetRow_SeqBDiag(Mat,int,int *,int **,Scalar **);
extern int MatRestoreRow_SeqBDiag(Mat,int,int *,int **,Scalar **);
extern int MatTranspose_SeqBDiag(Mat,Mat *);
extern int MatNorm_SeqBDiag(Mat,NormType,double *);
extern int MatGetOwnershipRange_SeqBDiag(Mat,int*,int *);

#undef __FUNC__  
#define __FUNC__ "MatDestroy_SeqBDiag"
int MatDestroy_SeqBDiag(Mat A)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          i, bs = a->bs,ierr;

  PetscFunctionBegin;
  if (--A->refct > 0) PetscFunctionReturn(0);

  if (A->mapping) {
    ierr = ISLocalToGlobalMappingDestroy(A->mapping); CHKERRQ(ierr);
  }
  if (A->bmapping) {
    ierr = ISLocalToGlobalMappingDestroy(A->bmapping); CHKERRQ(ierr);
  }
  if (A->rmap) {
    ierr = MapDestroy(A->rmap);CHKERRQ(ierr);
  }
  if (A->cmap) {
    ierr = MapDestroy(A->cmap);CHKERRQ(ierr);
  }
#if defined(USE_PETSC_LOG)
  PLogObjectState((PetscObject)A,"Rows=%d, Cols=%d, NZ=%d, BSize=%d, NDiag=%d",
                       a->m,a->n,a->nz,a->bs,a->nd);
#endif
  if (!a->user_alloc) { /* Free the actual diagonals */
    for (i=0; i<a->nd; i++) {
      if (a->diag[i] > 0) {
        PetscFree( a->diagv[i] + bs*bs*a->diag[i]  );
      } else {
        PetscFree( a->diagv[i] );
      }
    }
  }
  if (a->pivot) PetscFree(a->pivot);
  PetscFree(a->diagv); PetscFree(a->diag);
  PetscFree(a->colloc);
  PetscFree(a->dvalue);
  PetscFree(a);
  PLogObjectDestroy(A);
  PetscHeaderDestroy(A);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatAssemblyEnd_SeqBDiag"
int MatAssemblyEnd_SeqBDiag(Mat A,MatAssemblyType mode)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          i, k, temp, *diag = a->diag, *bdlen = a->bdlen;
  Scalar       *dtemp, **dv = a->diagv;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  /* Sort diagonals */
  for (i=0; i<a->nd; i++) {
    for (k=i+1; k<a->nd; k++) {
      if (diag[i] < diag[k]) {
        temp     = diag[i];   
        diag[i]  = diag[k];
        diag[k]  = temp;
        temp     = bdlen[i];   
        bdlen[i] = bdlen[k];
        bdlen[k] = temp;
        dtemp    = dv[i];
        dv[i]    = dv[k];
        dv[k]    = dtemp;
      }
    }
  }

  /* Set location of main diagonal */
  for (i=0; i<a->nd; i++) {
    if (a->diag[i] == 0) {a->mainbd = i; break;}
  }
  PLogInfo(A,"MatAssemblyEnd_SeqBDiag:Number diagonals %d, memory used %d, block size %d\n", 
           a->nd,a->maxnz,a->bs);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSetOption_SeqBDiag"
int MatSetOption_SeqBDiag(Mat A,MatOption op)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;

  PetscFunctionBegin;
  if (op == MAT_NO_NEW_NONZERO_LOCATIONS)       a->nonew       = 1;
  else if (op == MAT_YES_NEW_NONZERO_LOCATIONS) a->nonew       = 0;
  else if (op == MAT_NO_NEW_DIAGONALS)          a->nonew_diag  = 1;
  else if (op == MAT_YES_NEW_DIAGONALS)         a->nonew_diag  = 0;
  else if (op == MAT_COLUMN_ORIENTED)           a->roworiented = 0;
  else if (op == MAT_ROW_ORIENTED)              a->roworiented = 1;
  else if (op == MAT_ROWS_SORTED || 
           op == MAT_ROWS_UNSORTED || 
           op == MAT_COLUMNS_SORTED || 
           op == MAT_COLUMNS_UNSORTED || 
           op == MAT_SYMMETRIC ||
           op == MAT_IGNORE_OFF_PROC_ENTRIES ||
           op == MAT_STRUCTURALLY_SYMMETRIC ||
           op == MAT_NEW_NONZERO_LOCATION_ERROR ||
           op == MAT_NEW_NONZERO_ALLOCATION_ERROR ||
           op == MAT_USE_HASH_TABLE)
    PLogInfo(A,"MatSetOption_SeqBDiag:Option ignored\n");
  else { 
    SETERRQ(PETSC_ERR_SUP,0,"unknown option");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPrintHelp_SeqBDiag"
int MatPrintHelp_SeqBDiag(Mat A)
{
  static int called = 0; 
  MPI_Comm   comm = A->comm;

  PetscFunctionBegin;
  if (called) {PetscFunctionReturn(0);} else called = 1;
  (*PetscHelpPrintf)(comm," Options for MATSEQBDIAG and MATMPIBDIAG matrix formats:\n");
  (*PetscHelpPrintf)(comm,"  -mat_block_size <block_size>\n");
  (*PetscHelpPrintf)(comm,"  -mat_bdiag_diags <d1,d2,d3,...> (diagonal numbers)\n"); 
  (*PetscHelpPrintf)(comm,"   (for example) -mat_bdiag_diags -5,-1,0,1,5\n"); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetDiagonal_SeqBDiag_N"
static int MatGetDiagonal_SeqBDiag_N(Mat A,Vec v)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          ierr,i, j, n, len, ibase, bs = a->bs, iloc;
  Scalar       *x, *dd, zero = 0.0;

  PetscFunctionBegin;
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix");  
  ierr = VecSet(&zero,v);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  if (n != a->m) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Nonconforming mat and vec");
  if (a->mainbd == -1) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Main diagonal not set");
  len = PetscMin(a->mblock,a->nblock);
  dd = a->diagv[a->mainbd];
  ierr = VecGetArray(v,&x); CHKERRQ(ierr);
  for (i=0; i<len; i++) {
    ibase = i*bs*bs;  iloc = i*bs;
    for (j=0; j<bs; j++) x[j + iloc] = dd[ibase + j*(bs+1)];
  }
  ierr = VecRestoreArray(v,&x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetDiagonal_SeqBDiag_1"
static int MatGetDiagonal_SeqBDiag_1(Mat A,Vec v)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          ierr,i, n, len;
  Scalar       *x, *dd, zero = 0.0;

  PetscFunctionBegin;
  ierr = VecSet(&zero,v); CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  if (n != a->m) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Nonconforming mat and vec");
  if (a->mainbd == -1) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Main diagonal not set");
  dd = a->diagv[a->mainbd];
  len = PetscMin(a->m,a->n);
  ierr = VecGetArray(v,&x); CHKERRQ(ierr);
  for (i=0; i<len; i++) x[i] = dd[i];
  ierr = VecRestoreArray(v,&x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatZeroEntries_SeqBDiag"
int MatZeroEntries_SeqBDiag(Mat A)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          d, i, len, bs = a->bs;
  Scalar       *dv;

  PetscFunctionBegin;
  for (d=0; d<a->nd; d++) {
    dv  = a->diagv[d];
    if (a->diag[d] > 0) {
      dv += bs*bs*a->diag[d];
    }
    len = a->bdlen[d]*bs*bs;
    for (i=0; i<len; i++) dv[i] = 0.0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetBlockSize_SeqBDiag"
int MatGetBlockSize_SeqBDiag(Mat A,int *bs)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;

  PetscFunctionBegin;
  *bs = a->bs;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatZeroRows_SeqBDiag"
int MatZeroRows_SeqBDiag(Mat A,IS is,Scalar *diag)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          i, ierr, N, *rows, m = a->m - 1, nz, *col;
  Scalar       *dd, *val;

  PetscFunctionBegin;
  ierr = ISGetSize(is,&N); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows); CHKERRQ(ierr);
  for ( i=0; i<N; i++ ) {
    if (rows[i]<0 || rows[i]>m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"row out of range");
    ierr = MatGetRow(A,rows[i],&nz,&col,&val); CHKERRQ(ierr);
    PetscMemzero(val,nz*sizeof(Scalar));
    ierr = MatSetValues(A,1,&rows[i],nz,col,val,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(A,rows[i],&nz,&col,&val); CHKERRQ(ierr);
  }
  if (diag) {
    if (a->mainbd == -1) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Main diagonal does not exist");
    dd = a->diagv[a->mainbd];
    for ( i=0; i<N; i++ ) dd[rows[i]] = *diag;
  }
  ISRestoreIndices(is,&rows);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSize_SeqBDiag"
int MatGetSize_SeqBDiag(Mat A,int *m,int *n)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;

  PetscFunctionBegin;
  *m = a->m; *n = a->n;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSubMatrix_SeqBDiag"
int MatGetSubMatrix_SeqBDiag(Mat A,IS isrow,IS iscol,MatGetSubMatrixCall scall,
                                    Mat *submat)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          nznew, *smap, i, j, ierr, oldcols = a->n;
  int          *irow, *icol, newr, newc, *cwork, *col,nz, bs;
  Scalar       *vwork, *val;
  Mat          newmat;

  PetscFunctionBegin;
  ierr = ISGetIndices(isrow,&irow); CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&icol); CHKERRQ(ierr);
  ierr = ISGetSize(isrow,&newr); CHKERRQ(ierr);
  ierr = ISGetSize(iscol,&newc); CHKERRQ(ierr);

  smap  = (int *) PetscMalloc((oldcols+1)*sizeof(int)); CHKPTRQ(smap);
  cwork = (int *) PetscMalloc((newc+1)*sizeof(int)); CHKPTRQ(cwork);
  vwork = (Scalar *) PetscMalloc((newc+1)*sizeof(Scalar)); CHKPTRQ(vwork);
  PetscMemzero((char*)smap,oldcols*sizeof(int));
  for ( i=0; i<newc; i++ ) smap[icol[i]] = i+1;

  /* Determine diagonals; then create submatrix */
  bs = a->bs; /* Default block size remains the same */
  ierr = MatCreateSeqBDiag(A->comm,newr,newc,0,bs,0,0,&newmat); CHKERRQ(ierr); 

  /* Fill new matrix */
  for (i=0; i<newr; i++) {
    ierr = MatGetRow(A,irow[i],&nz,&col,&val); CHKERRQ(ierr);
    nznew = 0;
    for (j=0; j<nz; j++) {
      if (smap[col[j]]) {
        cwork[nznew]   = smap[col[j]] - 1;
        vwork[nznew++] = val[j];
      }
    }
    ierr = MatSetValues(newmat,1,&i,nznew,cwork,vwork,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i,&nz,&col,&val); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(newmat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(newmat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  /* Free work space */
  PetscFree(smap); PetscFree(cwork); PetscFree(vwork);
  ierr = ISRestoreIndices(isrow,&irow); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&icol); CHKERRQ(ierr);
  *submat = newmat;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSubMatrices_SeqBDiag"
int MatGetSubMatrices_SeqBDiag(Mat A,int n, IS *irow,IS *icol,MatGetSubMatrixCall scall,
                                    Mat **B)
{
  int ierr,i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    *B = (Mat *) PetscMalloc( (n+1)*sizeof(Mat) ); CHKPTRQ(*B);
  }

  for ( i=0; i<n; i++ ) {
    ierr = MatGetSubMatrix_SeqBDiag(A,irow[i],icol[i],scall,&(*B)[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatScale_SeqBDiag"
int MatScale_SeqBDiag(Scalar *alpha,Mat inA)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) inA->data;
  int          one = 1, i, len, bs = a->bs;

  PetscFunctionBegin;
  for (i=0; i<a->nd; i++) {
    len = bs*bs*a->bdlen[i];
    if (a->diag[i] > 0) {
      BLscal_( &len, alpha, a->diagv[i] + bs*bs*a->diag[i], &one );
    } else {
      BLscal_( &len, alpha, a->diagv[i], &one );
    }
  }
  PLogFlops(a->nz);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDiagonalScale_SeqBDiag"
int MatDiagonalScale_SeqBDiag(Mat A,Vec ll,Vec rr)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  Scalar       *l,*r, *dv;
  int          d, j, len,ierr;
  int          nd = a->nd, bs = a->bs, diag, m, n;

  PetscFunctionBegin;
  if (ll) {
    ierr = VecGetSize(ll,&m);CHKERRQ(ierr);
    if (m != a->m) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Left scaling vector wrong length");
    if (bs == 1) {
      ierr = VecGetArray(ll,&l);CHKERRQ(ierr); 
      for (d=0; d<nd; d++) {
        dv   = a->diagv[d];
        diag = a->diag[d];
        len  = a->bdlen[d];
        if (diag > 0) for (j=0; j<len; j++) dv[j+diag] *= l[j+diag];
        else          for (j=0; j<len; j++) dv[j]      *= l[j];
      }
      ierr = VecRestoreArray(ll,&l);CHKERRQ(ierr); 
      PLogFlops(a->nz);
    } else SETERRQ(PETSC_ERR_SUP,0,"Not yet done for bs>1");
  }
  if (rr) {
    ierr = VecGetSize(rr,&n);CHKERRQ(ierr);
    if (n != a->n) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Right scaling vector wrong length");
    if (bs == 1) {
      ierr = VecGetArray(rr,&r);CHKERRQ(ierr);  
      for (d=0; d<nd; d++) {
        dv   = a->diagv[d];
        diag = a->diag[d];
        len  = a->bdlen[d];
        if (diag > 0) for (j=0; j<len; j++) dv[j+diag] *= r[j];
        else          for (j=0; j<len; j++) dv[j]      *= r[j-diag];
      }
      ierr = VecRestoreArray(rr,&r);CHKERRQ(ierr);  
      PLogFlops(a->nz);
    } else SETERRQ(PETSC_ERR_SUP,0,"Not yet done for bs>1");
  }
  PetscFunctionReturn(0);
}

static int MatDuplicate_SeqBDiag(Mat,MatDuplicateOption,Mat *);
extern int MatLUFactorSymbolic_SeqBDiag(Mat,IS,IS,double,Mat*);
extern int MatILUFactorSymbolic_SeqBDiag(Mat,IS,IS,double,int,Mat*);
extern int MatILUFactor_SeqBDiag(Mat,IS,IS,double,int);
extern int MatLUFactorNumeric_SeqBDiag_N(Mat,Mat*);
extern int MatLUFactorNumeric_SeqBDiag_1(Mat,Mat*);
extern int MatSolve_SeqBDiag_1(Mat,Vec,Vec);
extern int MatSolve_SeqBDiag_2(Mat,Vec,Vec);
extern int MatSolve_SeqBDiag_3(Mat,Vec,Vec);
extern int MatSolve_SeqBDiag_4(Mat,Vec,Vec);
extern int MatSolve_SeqBDiag_5(Mat,Vec,Vec);
extern int MatSolve_SeqBDiag_N(Mat,Vec,Vec);

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_SeqBDiag_N,
       MatGetRow_SeqBDiag,
       MatRestoreRow_SeqBDiag,
       MatMult_SeqBDiag_N,
       MatMultAdd_SeqBDiag_N, 
       MatMultTrans_SeqBDiag_N,
       MatMultTransAdd_SeqBDiag_N, 
       MatSolve_SeqBDiag_N,
       0,
       0,
       0,
       0,
       0,
       MatRelax_SeqBDiag_N,
       MatTranspose_SeqBDiag,
       MatGetInfo_SeqBDiag,
       0,
       MatGetDiagonal_SeqBDiag_N,
       MatDiagonalScale_SeqBDiag,
       MatNorm_SeqBDiag,
       0,
       MatAssemblyEnd_SeqBDiag,
       0,
       MatSetOption_SeqBDiag,
       MatZeroEntries_SeqBDiag,
       MatZeroRows_SeqBDiag,
       0,
       MatLUFactorNumeric_SeqBDiag_N,
       0,
       0,
       MatGetSize_SeqBDiag,
       MatGetSize_SeqBDiag,
       MatGetOwnershipRange_SeqBDiag,
       MatILUFactorSymbolic_SeqBDiag,
       0,
       0,
       0,
       MatDuplicate_SeqBDiag,
       0,
       0,
       MatILUFactor_SeqBDiag,
       0,
       0,
       MatGetSubMatrices_SeqBDiag,
       0,
       MatGetValues_SeqBDiag_N,
       0,
       MatPrintHelp_SeqBDiag,
       MatScale_SeqBDiag,
       0,
       0,
       0,
       MatGetBlockSize_SeqBDiag,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       MatGetMaps_Petsc};

#undef __FUNC__  
#define __FUNC__ "MatCreateSeqBDiag"
/*@C
   MatCreateSeqBDiag - Creates a sequential block diagonal matrix.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nd - number of block diagonals (optional)
.  bs - each element of a diagonal is an bs x bs dense matrix
.  diag - optional array of block diagonal numbers (length nd).
   For a matrix element A[i,j], where i=row and j=column, the
   diagonal number is
$     diag = i/bs - j/bs  (integer division)
   Set diag=PETSC_NULL on input for PETSc to dynamically allocate memory as 
   needed (expensive).
-  diagv - pointer to actual diagonals (in same order as diag array), 
   if allocated by user.  Otherwise, set diagv=PETSC_NULL on input for PETSc
   to control memory allocation.

   Output Parameters:
.  A - the matrix

   Options Database Keys:
.  -mat_block_size <bs> - Sets blocksize
.  -mat_bdiag_diags <s1,s2,s3,...> - Sets diagonal numbers

   Notes:
   See the users manual for further details regarding this storage format.

   Fortran Note:
   Fortran programmers cannot set diagv; this value is ignored.

.keywords: matrix, block, diagonal, sparse

.seealso: MatCreate(), MatCreateMPIBDiag(), MatSetValues()
@*/
int MatCreateSeqBDiag(MPI_Comm comm,int m,int n,int nd,int bs,int *diag,Scalar **diagv,Mat *A)
{
  Mat          B;
  Mat_SeqBDiag *b;
  int          i, nda, sizetot, ierr,  nd2 = 128,flg1,idiag[128],size;

  PetscFunctionBegin;
  MPI_Comm_size(comm,&size);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_WRONG,0,"Comm must be of size 1");

  *A = 0;
  if (bs == PETSC_DEFAULT) bs = 1;
  if (nd == PETSC_DEFAULT) nd = 0;
  ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,&flg1);CHKERRQ(ierr);
  ierr = OptionsGetIntArray(PETSC_NULL,"-mat_bdiag_diags",idiag,&nd2,&flg1);CHKERRQ(ierr);
  if (flg1) {
    diag = idiag;
    nd   = nd2;
  }

  if ((n%bs) || (m%bs)) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Invalid block size");
  if (!nd) nda = nd + 1;
  else     nda = nd;
  PetscHeaderCreate(B,_p_Mat,struct _MatOps,MAT_COOKIE,MATSEQBDIAG,"Mat",comm,MatDestroy,MatView);
  PLogObjectCreate(B);
  B->data    = (void *) (b = PetscNew(Mat_SeqBDiag)); CHKPTRQ(b);
  PetscMemzero(b,sizeof(Mat_SeqBDiag));
  PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));
  B->ops->destroy = MatDestroy_SeqBDiag;
  B->ops->view    = MatView_SeqBDiag;
  B->factor  = 0;
  B->mapping = 0;

  ierr = OptionsHasName(PETSC_NULL,"-mat_no_unroll",&flg1); CHKERRQ(ierr);
  if (!flg1) {
    switch (bs) {
      case 1:
        B->ops->setvalues       = MatSetValues_SeqBDiag_1;
        B->ops->getvalues       = MatGetValues_SeqBDiag_1;
        B->ops->getdiagonal     = MatGetDiagonal_SeqBDiag_1;
        B->ops->mult            = MatMult_SeqBDiag_1;
        B->ops->multadd         = MatMultAdd_SeqBDiag_1;
        B->ops->multtrans       = MatMultTrans_SeqBDiag_1;
        B->ops->multtransadd    = MatMultTransAdd_SeqBDiag_1;
        B->ops->relax           = MatRelax_SeqBDiag_1;
        B->ops->solve           = MatSolve_SeqBDiag_1;
        B->ops->lufactornumeric = MatLUFactorNumeric_SeqBDiag_1;
        break;
      case 2:
	B->ops->mult            = MatMult_SeqBDiag_2; 
        B->ops->multadd         = MatMultAdd_SeqBDiag_2;
        B->ops->solve           = MatSolve_SeqBDiag_2;
        break;
      case 3:
	B->ops->mult            = MatMult_SeqBDiag_3; 
        B->ops->multadd         = MatMultAdd_SeqBDiag_3;
	B->ops->solve           = MatSolve_SeqBDiag_3; 
        break;
      case 4:
	B->ops->mult            = MatMult_SeqBDiag_4; 
        B->ops->multadd         = MatMultAdd_SeqBDiag_4;
	B->ops->solve           = MatSolve_SeqBDiag_4; 
        break;
      case 5:
	B->ops->mult            = MatMult_SeqBDiag_5; 
        B->ops->multadd         = MatMultAdd_SeqBDiag_5;
	B->ops->solve           = MatSolve_SeqBDiag_5; 
        break;
   }
  }

  b->m      = m; B->m = m; B->M = m;
  b->n      = n; B->n = n; B->N = n;

  ierr = MapCreateMPI(comm,m,m,&B->rmap);CHKERRQ(ierr);
  ierr = MapCreateMPI(comm,n,n,&B->cmap);CHKERRQ(ierr);

  b->mblock = m/bs;
  b->nblock = n/bs;
  b->nd     = nd;
  b->bs     = bs;
  b->ndim   = 0;
  b->mainbd = -1;
  b->pivot  = 0;

  b->diag   = (int *)PetscMalloc(2*nda*sizeof(int)); CHKPTRQ(b->diag);
  b->bdlen  = b->diag + nda;
  b->colloc = (int *)PetscMalloc((n+1)*sizeof(int)); CHKPTRQ(b->colloc);
  b->diagv  = (Scalar**)PetscMalloc(nda*sizeof(Scalar*)); CHKPTRQ(b->diagv);
  sizetot   = 0;

  if (diagv != PETSC_NULL) { /* user allocated space */
    b->user_alloc = 1;
    for (i=0; i<nd; i++) b->diagv[i] = diagv[i];
  } else b->user_alloc = 0;

  for (i=0; i<nd; i++) {
    b->diag[i] = diag[i];
    if (diag[i] > 0) { /* lower triangular */
      b->bdlen[i] = PetscMin(b->nblock,b->mblock - diag[i]);
    } else {           /* upper triangular */
      b->bdlen[i] = PetscMin(b->mblock,b->nblock + diag[i]);
    }
    sizetot += b->bdlen[i];
  }
  sizetot   *= bs*bs;
  b->maxnz  =  sizetot;
  b->dvalue = (Scalar *) PetscMalloc((n+1)*sizeof(Scalar)); CHKPTRQ(b->dvalue);
  PLogObjectMemory(B,(nda*(bs+2))*sizeof(int) + bs*nda*sizeof(Scalar)
                    + nda*sizeof(Scalar*) + sizeof(Mat_SeqBDiag)
                    + sizeof(struct _p_Mat) + sizetot*sizeof(Scalar));

  if (!b->user_alloc) {
    for (i=0; i<nd; i++) {
      b->diagv[i] = (Scalar*)PetscMalloc(bs*bs*b->bdlen[i]*sizeof(Scalar));CHKPTRQ(b->diagv[i]);
      PetscMemzero(b->diagv[i],bs*bs*b->bdlen[i]*sizeof(Scalar));
    }
    b->nonew = 0; b->nonew_diag = 0;
  } else { /* diagonals are set on input; don't allow dynamic allocation */
    b->nonew = 1; b->nonew_diag = 1;
  }

  /* adjust diagv so one may access rows with diagv[diag][row] for all rows */
  for (i=0; i<nd; i++) {
    if (diag[i] > 0) {
      b->diagv[i] -= bs*bs*diag[i];
    }
  }

  b->nz          = b->maxnz; /* Currently not keeping track of exact count */
  b->roworiented = 1;
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg1); CHKERRQ(ierr);
  if (flg1) {ierr = MatPrintHelp(B); CHKERRQ(ierr);}
  B->info.nz_unneeded = (double)b->maxnz;

  *A = B;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDuplicate_SeqBDiag"
static int MatDuplicate_SeqBDiag(Mat A,MatDuplicateOption cpvalues,Mat *matout)
{ 
  Mat_SeqBDiag *newmat, *a = (Mat_SeqBDiag *) A->data;
  int          i, ierr, len,diag,bs = a->bs;
  Mat          mat;

  PetscFunctionBegin;
  ierr = MatCreateSeqBDiag(A->comm,a->m,a->n,a->nd,bs,a->diag,PETSC_NULL,matout);CHKERRQ(ierr);

  /* Copy contents of diagonals */
  mat = *matout;
  newmat = (Mat_SeqBDiag *) mat->data;
  if (cpvalues == MAT_COPY_VALUES) {
    for (i=0; i<a->nd; i++) {
      len = a->bdlen[i] * bs * bs * sizeof(Scalar);
      diag = a->diag[i];
      if (diag > 0) {
        PetscMemcpy(newmat->diagv[i]+bs*bs*diag,a->diagv[i]+bs*bs*diag,len);
      } else {
        PetscMemcpy(newmat->diagv[i],a->diagv[i],len);
      }
    }
  } 
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatLoad_SeqBDiag"
int MatLoad_SeqBDiag(Viewer viewer,MatType type,Mat *A)
{
  Mat          B;
  int          *scols, i, nz, ierr, fd, header[4], size,nd = 128;
  int          bs, *rowlengths = 0,M,N,*cols,flg,extra_rows,*diag = 0;
  int          idiag[128];
  Scalar       *vals, *svals;
  MPI_Comm     comm;
  
  PetscFunctionBegin;
  PetscObjectGetComm((PetscObject)viewer,&comm);
  MPI_Comm_size(comm,&size);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_SIZ,0,"view must have one processor");
  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,header,4,PETSC_INT); CHKERRQ(ierr);
  if (header[0] != MAT_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,0,"Not matrix object");
  M = header[1]; N = header[2]; nz = header[3];
  if (M != N) SETERRQ(PETSC_ERR_SUP,0,"Can only load square matrices");
  if (header[3] < 0) {
    SETERRQ(PETSC_ERR_FILE_UNEXPECTED,1,"Matrix stored in special format, cannot load as SeqBDiag");
  }

  /* 
     This code adds extra rows to make sure the number of rows is 
    divisible by the blocksize
  */
  bs = 1;
  ierr = OptionsGetInt(PETSC_NULL,"-matload_block_size",&bs,&flg);CHKERRQ(ierr);
  extra_rows = bs - M + bs*(M/bs);
  if (extra_rows == bs) extra_rows = 0;
  if (extra_rows) {
    PLogInfo(0,"MatLoad_SeqBDiag:Padding loaded matrix to match blocksize\n");
  }

  /* read row lengths */
  rowlengths = (int*) PetscMalloc((M+extra_rows)*sizeof(int));CHKPTRQ(rowlengths);
  ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT); CHKERRQ(ierr);
  for ( i=0; i<extra_rows; i++ ) rowlengths[M+i] = 1;

  /* load information about diagonals */
  ierr = OptionsGetIntArray(PETSC_NULL,"-matload_bdiag_diags",idiag,&nd,&flg);
         CHKERRQ(ierr);
  if (flg) {
    diag = idiag;
  }

  /* create our matrix */
  ierr = MatCreateSeqBDiag(comm,M+extra_rows,M+extra_rows,nd,bs,diag,
                           PETSC_NULL,A); CHKERRQ(ierr);
  B = *A;

  /* read column indices and nonzeros */
  cols = scols = (int *) PetscMalloc( nz*sizeof(int) ); CHKPTRQ(cols);
  ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT); CHKERRQ(ierr);
  vals = svals = (Scalar *) PetscMalloc( nz*sizeof(Scalar) ); CHKPTRQ(vals);
  ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR); CHKERRQ(ierr);
  /* insert into matrix */

  for ( i=0; i<M; i++ ) {
    ierr = MatSetValues(B,1,&i,rowlengths[i],scols,svals,INSERT_VALUES);CHKERRQ(ierr);
    scols += rowlengths[i]; svals += rowlengths[i];
  }
  vals[0] = 1.0;
  for ( i=M; i<M+extra_rows; i++ ) {
    ierr = MatSetValues(B,1,&i,1,&i,vals,INSERT_VALUES);CHKERRQ(ierr);
  }

  PetscFree(cols);
  PetscFree(vals);
  PetscFree(rowlengths);   

  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
