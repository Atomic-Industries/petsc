#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: bdiag3.c,v 1.3 1998/12/03 04:00:32 bsmith Exp bsmith $";
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


#undef __FUNC__  
#define __FUNC__ "MatGetInfo_SeqBDiag"
int MatGetInfo_SeqBDiag(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;

  PetscFunctionBegin;
  info->rows_global       = (double)a->m;
  info->columns_global    = (double)a->n;
  info->rows_local        = (double)a->m;
  info->columns_local     = (double)a->n;
  info->block_size        = a->bs;
  info->nz_allocated      = (double)a->maxnz;
  info->nz_used           = (double)a->nz;
  info->nz_unneeded       = (double)(a->maxnz - a->nz);
  info->assemblies        = (double)A->num_ass;
  info->mallocs           = (double)a->reallocs;
  info->memory            = A->mem;
  info->fill_ratio_given  = 0; /* supports ILU(0) only */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetOwnershipRange_SeqBDiag" 
int MatGetOwnershipRange_SeqBDiag(Mat A,int *m,int *n)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;

  PetscFunctionBegin;
  *m = 0; *n = a->m;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetRow_SeqBDiag"
int MatGetRow_SeqBDiag(Mat A,int row,int *nz,int **col,Scalar **v)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          nd = a->nd, bs = a->bs;
  int          nc = a->n, *diag = a->diag, pcol, shift, i, j, k;

  PetscFunctionBegin;
  /* For efficiency, if ((nz) && (col) && (v)) then do all at once */
  if ((nz) && (col) && (v)) {
    *col = a->colloc;
    *v   = a->dvalue;
    k    = 0;
    if (bs == 1) { 
      for (j=0; j<nd; j++) {
        pcol = row - diag[j];
        if (pcol > -1 && pcol < nc) {
	  (*v)[k]   = (a->diagv[j])[row];
          (*col)[k] = pcol;  k++;
	}
      }
      *nz = k;
    } else {
      shift = (row/bs)*bs*bs + row%bs;
      for (j=0; j<nd; j++) {
        pcol = bs * (row/bs - diag[j]);
        if (pcol > -1 && pcol < nc) {
          for (i=0; i<bs; i++) {
	    (*v)[k+i]   = (a->diagv[j])[shift + i*bs];
	    (*col)[k+i] = pcol + i;
	  }
          k += bs;
        } 
      }
      *nz = k;
    }
  } else {
    if (bs == 1) { 
      if (nz) {
        k = 0;
        for (j=0; j<nd; j++) {
          pcol = row - diag[j];
          if (pcol > -1 && pcol < nc) k++; 
        }
        *nz = k;
      }
      if (col) {
        *col = a->colloc;
        k = 0;
        for (j=0; j<nd; j++) {
          pcol = row - diag[j];
          if (pcol > -1 && pcol < nc) {
            (*col)[k] = pcol;  k++;
          }
        }
      }
      if (v) {
        *v = a->dvalue;
        k = 0;
        for (j=0; j<nd; j++) {
          pcol = row - diag[j];
          if (pcol > -1 && pcol < nc) {
	    (*v)[k] = (a->diagv[j])[row]; k++;
          }
        }
      }
    } else {
      if (nz) {
        k = 0;
        for (j=0; j<nd; j++) {
          pcol = bs * (row/bs- diag[j]);
          if (pcol > -1 && pcol < nc) k += bs; 
        }
        *nz = k;
      }
      if (col) {
        *col = a->colloc;
        k = 0;
        for (j=0; j<nd; j++) {
          pcol = bs * (row/bs - diag[j]);
          if (pcol > -1 && pcol < nc) {
            for (i=0; i<bs; i++) {
	      (*col)[k+i] = pcol + i;
            }
	    k += bs;
          }
        }
      }
      if (v) {
        shift = (row/bs)*bs*bs + row%bs;
        *v = a->dvalue;
        k = 0;
        for (j=0; j<nd; j++) {
	  pcol = bs * (row/bs - diag[j]);
	  if (pcol > -1 && pcol < nc) {
	    for (i=0; i<bs; i++) {
	     (*v)[k+i] = (a->diagv[j])[shift + i*bs];
            }
	    k += bs;
	  }
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreRow_SeqBDiag"
int MatRestoreRow_SeqBDiag(Mat A,int row,int *ncols,int **cols,Scalar **vals)
{
  PetscFunctionBegin;
  /* Work space is allocated during matrix creation and freed
     when matrix is destroyed */
  PetscFunctionReturn(0);
}

/* 
   MatNorm_SeqBDiag_Columns - Computes the column norms of a block diagonal
   matrix.  We code this separately from MatNorm_SeqBDiag() so that the
   routine can be used for the parallel version as well.
 */
#undef __FUNC__  
#define __FUNC__ "MatNorm_SeqBDiag_Columns"
int MatNorm_SeqBDiag_Columns(Mat A,double *tmp,int n)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          d, i, j, k, nd = a->nd, bs = a->bs, diag, kshift, kloc, len;
  Scalar       *dv;

  PetscFunctionBegin;
  PetscMemzero(tmp,a->n*sizeof(double));
  if (bs == 1) {
    for (d=0; d<nd; d++) {
      dv   = a->diagv[d];
      diag = a->diag[d];
      len  = a->bdlen[d];
      if (diag > 0) {	/* lower triangle */
        for (i=0; i<len; i++) {
          tmp[i] += PetscAbsScalar(dv[i+diag]); 
        }
      } else {	/* upper triangle */
        for (i=0; i<len; i++) {
          tmp[i-diag] += PetscAbsScalar(dv[i]); 
        }
      }
    }
  } else { 
    for (d=0; d<nd; d++) {
      dv   = a->diagv[d];
      diag = a->diag[d];
      len  = a->bdlen[d];

      if (diag > 0) {	/* lower triangle */
        for (k=0; k<len; k++) {
          kloc = k*bs; kshift = kloc*bs + diag*bs; 
          for (i=0; i<bs; i++) {	/* i = local row */
            for (j=0; j<bs; j++) {	/* j = local column */
              tmp[kloc + j] += PetscAbsScalar(dv[kshift + j*bs + i]);
            }
          }
        }
      } else {	/* upper triangle */
        for (k=0; k<len; k++) {
          kloc = k*bs; kshift = kloc*bs; 
          for (i=0; i<bs; i++) {	/* i = local row */
            for (j=0; j<bs; j++) {	/* j = local column */
              tmp[kloc + j - bs*diag] += PetscAbsScalar(dv[kshift + j*bs + i]);
            }
          }
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatNorm_SeqBDiag"
int MatNorm_SeqBDiag(Mat A,NormType type,double *norm)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  double       sum = 0.0, *tmp;
  int          ierr,d,i,j,k,nd = a->nd,bs = a->bs,diag, kshift, kloc, len;
  Scalar       *dv;

  PetscFunctionBegin;
  if (type == NORM_FROBENIUS) {
    for (d=0; d<nd; d++) {
      dv   = a->diagv[d];
      len  = a->bdlen[d]*bs*bs;
      diag = a->diag[d];
      if (diag > 0) {
        for (i=0; i<len; i++) {
#if defined(USE_PETSC_COMPLEX)
          sum += PetscReal(PetscConj(dv[i+diag])*dv[i+diag]);
#else
          sum += dv[i+diag]*dv[i+diag];
#endif
        }
      } else {
        for (i=0; i<len; i++) {
#if defined(USE_PETSC_COMPLEX)
          sum += PetscReal(PetscConj(dv[i])*dv[i]);
#else
          sum += dv[i]*dv[i];
#endif
        }
      }
    }
    *norm = sqrt(sum);
  } else if (type == NORM_1) { /* max column norm */
    tmp = (double *) PetscMalloc( (a->n+1)*sizeof(double) ); CHKPTRQ(tmp);
    ierr = MatNorm_SeqBDiag_Columns(A,tmp,a->n); CHKERRQ(ierr);
    *norm = 0.0;
    for ( j=0; j<a->n; j++ ) {
      if (tmp[j] > *norm) *norm = tmp[j];
    }
    PetscFree(tmp);
  } else if (type == NORM_INFINITY) { /* max row norm */
    tmp = (double *) PetscMalloc( (a->m+1)*sizeof(double) ); CHKPTRQ(tmp);
    PetscMemzero(tmp,a->m*sizeof(double));
    *norm = 0.0;
    if (bs == 1) {
      for (d=0; d<nd; d++) {
        dv   = a->diagv[d];
        diag = a->diag[d];
        len  = a->bdlen[d];
        if (diag > 0) {	/* lower triangle */
          for (i=0; i<len; i++) {
            tmp[i+diag] += PetscAbsScalar(dv[i+diag]); 
          }
        } else {	/* upper triangle */
          for (i=0; i<len; i++) {
            tmp[i] += PetscAbsScalar(dv[i]); 
          }
        }
      }
    } else { 
      for (d=0; d<nd; d++) {
        dv   = a->diagv[d];
        diag = a->diag[d];
        len  = a->bdlen[d];
        if (diag > 0) {
          for (k=0; k<len; k++) {
            kloc = k*bs; kshift = kloc*bs + bs*diag; 
            for (i=0; i<bs; i++) {	/* i = local row */
              for (j=0; j<bs; j++) {	/* j = local column */
                tmp[kloc + i + bs*diag] += PetscAbsScalar(dv[kshift+j*bs+i]);
              }
            }
          }
        } else {
          for (k=0; k<len; k++) {
            kloc = k*bs; kshift = kloc*bs; 
            for (i=0; i<bs; i++) {	/* i = local row */
              for (j=0; j<bs; j++) {	/* j = local column */
                tmp[kloc + i] += PetscAbsScalar(dv[kshift + j*bs + i]);
              }
            }
          }
        }
      }
    }
    for ( j=0; j<a->m; j++ ) {
      if (tmp[j] > *norm) *norm = tmp[j];
    }
    PetscFree(tmp);
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"No support for two norm");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatTranspose_SeqBDiag"
int MatTranspose_SeqBDiag(Mat A,Mat *matout)
{ 
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data, *anew;
  Mat          tmat;
  int          i, j, k, d, ierr, nd = a->nd, *diag = a->diag, *diagnew;
  int          bs = a->bs, kshift,shifto,shiftn;
  Scalar       *dwork, *dvnew;

  PetscFunctionBegin;
  diagnew = (int *) PetscMalloc((nd+1)*sizeof(int)); CHKPTRQ(diagnew);
  for (i=0; i<nd; i++) {
    diagnew[i] = -diag[nd-i-1]; /* assume sorted in descending order */
  }
  ierr = MatCreateSeqBDiag(A->comm,a->n,a->m,nd,bs,diagnew,0,&tmat);CHKERRQ(ierr);
  PetscFree(diagnew);
  anew = (Mat_SeqBDiag *) tmat->data;
  for (d=0; d<nd; d++) {
    dvnew = anew->diagv[d];
    dwork = a->diagv[nd-d-1];
    if (anew->bdlen[d] != a->bdlen[nd-d-1]) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Incompatible diagonal lengths");
    shifto = a->diag[nd-d-1];
    shiftn = anew->diag[d];
    if (shifto > 0)  shifto = bs*bs*shifto; else shifto = 0;
    if (shiftn > 0)  shiftn = bs*bs*shiftn; else shiftn = 0;
    if (bs == 1) {
      for (k=0; k<anew->bdlen[d]; k++) dvnew[shiftn+k] = dwork[shifto+k];
    } else {
      for (k=0; k<anew->bdlen[d]; k++) {
        kshift = k*bs*bs;
        for (i=0; i<bs; i++) {	/* i = local row */
          for (j=0; j<bs; j++) {	/* j = local column */
            dvnew[shiftn + kshift + j + i*bs] = dwork[shifto + kshift + j*bs + i];
          }
        }
      }
    }
  }
  ierr = MatAssemblyBegin(tmat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(tmat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  if (matout != PETSC_NULL) {
    *matout = tmat;
  } else {
    /* This isn't really an in-place transpose ... but free data 
       structures from a.  We should fix this. */
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
    PetscFree(a->colloc); PetscFree(a->dvalue);
    PetscFree(a);
    PetscMemcpy(A,tmat,sizeof(struct _p_Mat)); 
    PetscHeaderDestroy(tmat);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------*/


#undef __FUNC__  
#define __FUNC__ "MatView_SeqBDiag_Binary"
int MatView_SeqBDiag_Binary(Mat A,Viewer viewer)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          i, ict, fd, *col_lens, *cval, *col, ierr, nz;
  Scalar       *anonz, *val;

  PetscFunctionBegin;
  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);

  /* For MATSEQBDIAG format, maxnz = nz */
  col_lens    = (int *) PetscMalloc( (4+a->m)*sizeof(int) ); CHKPTRQ(col_lens);
  col_lens[0] = MAT_COOKIE;
  col_lens[1] = a->m;
  col_lens[2] = a->n;
  col_lens[3] = a->maxnz;

  /* Should do translation using less memory; this is just a quick initial version */
  cval  = (int *) PetscMalloc( (a->maxnz)*sizeof(int) ); CHKPTRQ(cval);
  anonz = (Scalar *) PetscMalloc( (a->maxnz)*sizeof(Scalar) ); CHKPTRQ(anonz);

  ict = 0;
  for (i=0; i<a->m; i++) {
    ierr = MatGetRow(A,i,&nz,&col,&val); CHKERRQ(ierr);
    col_lens[4+i] = nz;
    PetscMemcpy(&cval[ict],col,nz*sizeof(int)); CHKERRQ(ierr);
    PetscMemcpy(&anonz[ict],anonz,nz*sizeof(Scalar)); CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i,&nz,&col,&val); CHKERRQ(ierr);
    ict += nz;
  }
  if (ict != a->maxnz) SETERRQ(PETSC_ERR_PLIB,0,"Error in nonzero count");

  /* Store lengths of each row and write (including header) to file */
  ierr = PetscBinaryWrite(fd,col_lens,4+a->m,PETSC_INT,1); CHKERRQ(ierr);
  PetscFree(col_lens);

  /* Store column indices (zero start index) */
  ierr = PetscBinaryWrite(fd,cval,a->maxnz,PETSC_INT,0); CHKERRQ(ierr);

  /* Store nonzero values */
  ierr = PetscBinaryWrite(fd,anonz,a->maxnz,PETSC_SCALAR,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqBDiag_ASCII"
int MatView_SeqBDiag_ASCII(Mat A,Viewer viewer)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  FILE         *fd;
  char         *outputname;
  int          ierr, *col, i, j, len, diag, nr = a->m, bs = a->bs, format, iprint, nz;
  Scalar       *val, *dv, zero = 0.0;

  PetscFunctionBegin;
  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  ierr = ViewerGetOutputname(viewer,&outputname); CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format); CHKERRQ(ierr);
  if (format == VIEWER_FORMAT_ASCII_INFO || format == VIEWER_FORMAT_ASCII_INFO_LONG) {
    int nline = PetscMin(10,a->nd), k, nk, np;
    if (a->user_alloc)
      fprintf(fd,"  block size=%d, number of diagonals=%d, user-allocated storage\n",bs,a->nd);
    else
      fprintf(fd,"  block size=%d, number of diagonals=%d, PETSc-allocated storage\n",bs,a->nd);
    nk = (a->nd-1)/nline + 1;
    for (k=0; k<nk; k++) {
      fprintf(fd,"  diag numbers:");
      np = PetscMin(nline,a->nd - nline*k);
      for (i=0; i<np; i++) 
        fprintf(fd,"  %d",a->diag[i+nline*k]);
      fprintf(fd,"\n");        
    }
  } else if (format == VIEWER_FORMAT_ASCII_MATLAB) {
    fprintf(fd,"%% Size = %d %d \n",nr, a->n);
    fprintf(fd,"%% Nonzeros = %d \n",a->nz);
    fprintf(fd,"zzz = zeros(%d,3);\n",a->nz);
    fprintf(fd,"zzz = [\n");
    for ( i=0; i<a->m; i++ ) {
      ierr = MatGetRow( A, i, &nz, &col, &val ); CHKERRQ(ierr);
      for (j=0; j<nz; j++) {
        if (val[j] != zero)
#if defined(USE_PETSC_COMPLEX)
          fprintf(fd,"%d %d  %18.16e  %18.16e \n",
             i+1, col[j]+1, PetscReal(val[j]), PetscImaginary(val[j]) );
#else
          fprintf(fd,"%d %d  %18.16e\n", i+1, col[j]+1, val[j]);
#endif
      }
      ierr = MatRestoreRow(A,i,&nz,&col,&val); CHKERRQ(ierr);
    }
    fprintf(fd,"];\n %s = spconvert(zzz);\n",outputname);
  } else if (format == VIEWER_FORMAT_ASCII_IMPL) {
    if (bs == 1) { /* diagonal format */
      for (i=0; i<a->nd; i++) {
        dv   = a->diagv[i];
        diag = a->diag[i];
        fprintf(fd,"\n<diagonal %d>\n",diag);
        /* diag[i] is (row-col)/bs */
        if (diag > 0) {  /* lower triangle */
          len  = a->bdlen[i];
          for (j=0; j<len; j++) {
            if (dv[diag+j] != zero) {
#if defined(USE_PETSC_COMPLEX)
              if (PetscImaginary(dv[diag+j]) != 0.0) fprintf(fd,"A[ %d , %d ] = %e + %e i\n",
                                     j+diag,j,PetscReal(dv[diag+j]),PetscImaginary(dv[diag+j]));
              else fprintf(fd,"A[ %d , %d ] = %e\n",j+diag,j,PetscReal(dv[diag+j]));
#else
              fprintf(fd,"A[ %d , %d ] = %e\n",j+diag,j,dv[diag+j]);

#endif
            }
          }
        } else {         /* upper triangle, including main diagonal */
          len  = a->bdlen[i];
          for (j=0; j<len; j++) {
            if (dv[j] != zero) {
#if defined(USE_PETSC_COMPLEX)
              if (PetscImaginary(dv[j]) != 0.0) fprintf(fd,"A[ %d , %d ] = %e + %e i\n",
                                         j,j-diag,PetscReal(dv[j]),PetscImaginary(dv[j]));
              else fprintf(fd,"A[ %d , %d ] = %e\n",j,j-diag,PetscReal(dv[j]));
#else
              fprintf(fd,"A[ %d , %d ] = %e\n",j,j-diag,dv[j]);
#endif
            }
          }
        }
      }
    } else {  /* Block diagonals */
      int d, k, kshift;
      for (d=0; d< a->nd; d++) {
        dv   = a->diagv[d];
        diag = a->diag[d];
        len  = a->bdlen[d];
	fprintf(fd,"\n<diagonal %d>\n", diag);
	if (diag > 0) {		/* lower triangle */
	  for (k=0; k<len; k++) {
	    kshift = (diag+k)*bs*bs;
	    for (i=0; i<bs; i++) {
              iprint = 0;
	      for (j=0; j<bs; j++) {
		if (dv[kshift + j*bs + i] != zero) {
                  iprint = 1;
#if defined(USE_PETSC_COMPLEX)
                  if (PetscImaginary(dv[kshift + j*bs + i]))
                    fprintf(fd,"A[%d,%d]=%5.2e + %5.2e i  ",(k+diag)*bs+i,k*bs+j,
                      PetscReal(dv[kshift + j*bs + i]),PetscImaginary(dv[kshift + j*bs + i]));
                  else
                    fprintf(fd,"A[%d,%d]=%5.2e   ",(k+diag)*bs+i,k*bs+j,
                      PetscReal(dv[kshift + j*bs + i]));
#else
		  fprintf(fd,"A[%d,%d]=%5.2e   ", (k+diag)*bs+i,k*bs+j,
                      dv[kshift + j*bs + i]);
#endif
                }
              }
              if (iprint) fprintf(fd,"\n");
            }
          }
        } else {		/* upper triangle, including main diagonal */
	  for (k=0; k<len; k++) {
	    kshift = k*bs*bs;
            for (i=0; i<bs; i++) {
              iprint = 0;
              for (j=0; j<bs; j++) {
                if (dv[kshift + j*bs + i] != zero) {
                  iprint = 1;
#if defined(USE_PETSC_COMPLEX)
                  if (PetscImaginary(dv[kshift + j*bs + i]))
                    fprintf(fd,"A[%d,%d]=%5.2e + %5.2e i  ", k*bs+i,(k-diag)*bs+j,
                       PetscReal(dv[kshift + j*bs + i]),PetscImaginary(dv[kshift + j*bs + i]));
                  else
                    fprintf(fd,"A[%d,%d]=%5.2e   ", k*bs+i,(k-diag)*bs+j,
                       PetscReal(dv[kshift + j*bs + i]));
#else
                  fprintf(fd,"A[%d,%d]=%5.2e   ", k*bs+i,(k-diag)*bs+j,
                     dv[kshift + j*bs + i]);
#endif
                }
              }
              if (iprint) fprintf(fd,"\n");
            }
          }
        }
      }
    }
  } else {
    /* the usual row format (VIEWER_FORMAT_ASCII_NONZERO_ONLY) */
    for (i=0; i<a->m; i++) {
      fprintf(fd,"row %d:",i);
      ierr = MatGetRow(A,i,&nz,&col,&val); CHKERRQ(ierr);
      for (j=0; j<nz; j++) {
#if defined(USE_PETSC_COMPLEX)
        if (PetscImaginary(val[j]) != 0.0 && PetscReal(val[j]) != 0.0)
          fprintf(fd," %d %g + %g i ",col[j],PetscReal(val[j]),PetscImaginary(val[j]));
        else if (PetscReal(val[j]) != 0.0)
	  fprintf(fd," %d %g ",col[j],PetscReal(val[j]));
#else
        if (val[j] != 0.0) fprintf(fd," %d %g ",col[j],val[j]);
#endif
      }
      fprintf(fd,"\n");
      ierr = MatRestoreRow(A,i,&nz,&col,&val); CHKERRQ(ierr);
    }
  }
  fflush(fd);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqBDiag_Draw"
static int MatView_SeqBDiag_Draw(Mat A,Viewer viewer)
{
  Mat_SeqBDiag  *a = (Mat_SeqBDiag *) A->data;
  Draw          draw;
  double        xl, yl, xr, yr, w, h;
  int           ierr, nz, *col, i, j, nr = a->m;
  PetscTruth    isnull;

  PetscFunctionBegin;
  ierr = ViewerDrawGetDraw(viewer,0,&draw); CHKERRQ(ierr);
  ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);

  xr = a->n; yr = a->m; h = yr/10.0; w = xr/10.0;
  xr += w; yr += h; xl = -w; yl = -h;
  ierr = DrawSetCoordinates(draw,xl,yl,xr,yr); CHKERRQ(ierr);

  /* loop over matrix elements drawing boxes; we really should do this
     by diagonals.  What do we really want to draw here: nonzeros, 
     allocated space? */
  for ( i=0; i<nr; i++ ) {
    yl = nr - i - 1.0; yr = yl + 1.0;
    ierr = MatGetRow(A,i,&nz,&col,0); CHKERRQ(ierr);
    for ( j=0; j<nz; j++ ) {
      xl = col[j]; xr = xl + 1.0;
      ierr = DrawRectangle(draw,xl,yl,xr,yr,DRAW_BLACK,DRAW_BLACK,
			   DRAW_BLACK,DRAW_BLACK); CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(A,i,&nz,&col,0); CHKERRQ(ierr);
  }
  ierr = DrawFlush(draw); CHKERRQ(ierr);
  DrawPause(draw); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqBDiag"
int MatView_SeqBDiag(Mat A,Viewer viewer)
{
  ViewerType  vtype;
  int         ierr;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (PetscTypeCompare(vtype,MATLAB_VIEWER)) {
    SETERRQ(PETSC_ERR_SUP,0,"Matlab viewer");
  } else if (PetscTypeCompare(vtype,ASCII_VIEWER)) {
    ierr = MatView_SeqBDiag_ASCII(A,viewer);CHKERRQ(ierr);
  } else if (PetscTypeCompare(vtype,BINARY_VIEWER)) {
    ierr = MatView_SeqBDiag_Binary(A,viewer);CHKERRQ(ierr);
  } else if (PetscTypeCompare(vtype,DRAW_VIEWER)) {
    ierr = MatView_SeqBDiag_Draw(A,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Viewer type not supported by PETSc object");
  }
  PetscFunctionReturn(0);
}

