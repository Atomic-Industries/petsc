#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: baij.c,v 1.145 1998/10/09 19:22:47 bsmith Exp bsmith $";
#endif

/*
    Defines the basic matrix operations for the BAIJ (compressed row)
  matrix storage format.
*/

#include "pinclude/pviewer.h"
#include "sys.h"
#include "src/mat/impls/baij/seq/baij.h"
#include "src/vec/vecimpl.h"
#include "src/inline/spops.h"
#include "petsc.h"

#define CHUNKSIZE  10

#undef __FUNC__  
#define __FUNC__ "MatMarkDiag_SeqBAIJ"
int MatMarkDiag_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data; 
  int         i,j, *diag, m = a->mbs;

  PetscFunctionBegin;
  diag = (int *) PetscMalloc( (m+1)*sizeof(int)); CHKPTRQ(diag);
  PLogObjectMemory(A,(m+1)*sizeof(int));
  for ( i=0; i<m; i++ ) {
    diag[i] = a->i[i+1];
    for ( j=a->i[i]; j<a->i[i+1]; j++ ) {
      if (a->j[j] == i) {
        diag[i] = j;
        break;
      }
    }
  }
  a->diag = diag;
  PetscFunctionReturn(0);
}


extern int MatToSymmetricIJ_SeqAIJ(int,int*,int*,int,int,int**,int**);

#undef __FUNC__  
#define __FUNC__ "MatGetRowIJ_SeqBAIJ"
static int MatGetRowIJ_SeqBAIJ(Mat A,int oshift,PetscTruth symmetric,int *nn,int **ia,int **ja,
                            PetscTruth *done)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int         ierr, n = a->mbs,i;

  PetscFunctionBegin;
  *nn = n;
  if (!ia) PetscFunctionReturn(0);
  if (symmetric) {
    ierr = MatToSymmetricIJ_SeqAIJ(n,a->i,a->j,0,oshift,ia,ja);CHKERRQ(ierr);
  } else if (oshift == 1) {
    /* temporarily add 1 to i and j indices */
    int nz = a->i[n] + 1; 
    for ( i=0; i<nz; i++ ) a->j[i]++;
    for ( i=0; i<n+1; i++ ) a->i[i]++;
    *ia = a->i; *ja = a->j;
  } else {
    *ia = a->i; *ja = a->j;
  }

  PetscFunctionReturn(0); 
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreRowIJ_SeqBAIJ" 
static int MatRestoreRowIJ_SeqBAIJ(Mat A,int oshift,PetscTruth symmetric,int *nn,int **ia,int **ja,
                                PetscTruth *done)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int         i,n = a->mbs;

  PetscFunctionBegin;
  if (!ia) PetscFunctionReturn(0);
  if (symmetric) {
    PetscFree(*ia);
    PetscFree(*ja);
  } else if (oshift == 1) {
    int nz = a->i[n]; 
    for ( i=0; i<nz; i++ ) a->j[i]--;
    for ( i=0; i<n+1; i++ ) a->i[i]--;
  }
  PetscFunctionReturn(0); 
}

#undef __FUNC__  
#define __FUNC__ "MatGetBlockSize_SeqBAIJ"
int MatGetBlockSize_SeqBAIJ(Mat mat, int *bs)
{
  Mat_SeqBAIJ *baij = (Mat_SeqBAIJ *) mat->data;

  PetscFunctionBegin;
  *bs = baij->bs;
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "MatDestroy_SeqBAIJ"
int MatDestroy_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int         ierr;

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
  PLogObjectState((PetscObject) A,"Rows=%d, Cols=%d, NZ=%d",a->m,a->n,a->nz);
#endif
  PetscFree(a->a); 
  if (!a->singlemalloc) { PetscFree(a->i); PetscFree(a->j);}
  if (a->diag) PetscFree(a->diag);
  if (a->ilen) PetscFree(a->ilen);
  if (a->imax) PetscFree(a->imax);
  if (a->solve_work) PetscFree(a->solve_work);
  if (a->mult_work) PetscFree(a->mult_work);
  if (a->icol) {ierr = ISDestroy(a->icol);CHKERRQ(ierr);}
  PetscFree(a); 
  PLogObjectDestroy(A);
  PetscHeaderDestroy(A);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSetOption_SeqBAIJ"
int MatSetOption_SeqBAIJ(Mat A,MatOption op)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;

  PetscFunctionBegin;
  if      (op == MAT_ROW_ORIENTED)                 a->roworiented = 1;
  else if (op == MAT_COLUMN_ORIENTED)              a->roworiented = 0;
  else if (op == MAT_COLUMNS_SORTED)               a->sorted      = 1;
  else if (op == MAT_COLUMNS_UNSORTED)             a->sorted      = 0;
  else if (op == MAT_NO_NEW_NONZERO_LOCATIONS)     a->nonew       = 1;
  else if (op == MAT_NEW_NONZERO_LOCATION_ERROR)   a->nonew       = -1;
  else if (op == MAT_NEW_NONZERO_ALLOCATION_ERROR) a->nonew       = -2;
  else if (op == MAT_YES_NEW_NONZERO_LOCATIONS)    a->nonew       = 0;
  else if (op == MAT_ROWS_SORTED || 
           op == MAT_ROWS_UNSORTED ||
           op == MAT_SYMMETRIC ||
           op == MAT_STRUCTURALLY_SYMMETRIC ||
           op == MAT_YES_NEW_DIAGONALS ||
           op == MAT_IGNORE_OFF_PROC_ENTRIES ||
           op == MAT_USE_HASH_TABLE) {
    PLogInfo(A,"MatSetOption_SeqBAIJ:Option ignored\n");
  } else if (op == MAT_NO_NEW_DIAGONALS) {
    SETERRQ(PETSC_ERR_SUP,0,"MAT_NO_NEW_DIAGONALS");
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"unknown option");
  }
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "MatGetSize_SeqBAIJ"
int MatGetSize_SeqBAIJ(Mat A,int *m,int *n)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;

  PetscFunctionBegin;
  if (m) *m = a->m; 
  if (n) *n = a->n;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetOwnershipRange_SeqBAIJ"
int MatGetOwnershipRange_SeqBAIJ(Mat A,int *m,int *n)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;

  PetscFunctionBegin;
  *m = 0; *n = a->m;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetRow_SeqBAIJ"
int MatGetRow_SeqBAIJ(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int          itmp,i,j,k,M,*ai,*aj,bs,bn,bp,*idx_i,bs2;
  Scalar      *aa,*v_i,*aa_i;

  PetscFunctionBegin;
  bs  = a->bs;
  ai  = a->i;
  aj  = a->j;
  aa  = a->a;
  bs2 = a->bs2;
  
  if (row < 0 || row >= a->m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row out of range");
  
  bn  = row/bs;   /* Block number */
  bp  = row % bs; /* Block Position */
  M   = ai[bn+1] - ai[bn];
  *nz = bs*M;
  
  if (v) {
    *v = 0;
    if (*nz) {
      *v = (Scalar *) PetscMalloc( (*nz)*sizeof(Scalar) ); CHKPTRQ(*v);
      for ( i=0; i<M; i++ ) { /* for each block in the block row */
        v_i  = *v + i*bs;
        aa_i = aa + bs2*(ai[bn] + i);
        for ( j=bp,k=0; j<bs2; j+=bs,k++ ) {v_i[k] = aa_i[j];}
      }
    }
  }

  if (idx) {
    *idx = 0;
    if (*nz) {
      *idx = (int *) PetscMalloc( (*nz)*sizeof(int) ); CHKPTRQ(*idx);
      for ( i=0; i<M; i++ ) { /* for each block in the block row */
        idx_i = *idx + i*bs;
        itmp  = bs*aj[ai[bn] + i];
        for ( j=0; j<bs; j++ ) {idx_i[j] = itmp++;}
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreRow_SeqBAIJ"
int MatRestoreRow_SeqBAIJ(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  PetscFunctionBegin;
  if (idx) {if (*idx) PetscFree(*idx);}
  if (v)   {if (*v)   PetscFree(*v);}
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatTranspose_SeqBAIJ"
int MatTranspose_SeqBAIJ(Mat A,Mat *B)
{ 
  Mat_SeqBAIJ *a=(Mat_SeqBAIJ *)A->data;
  Mat         C;
  int         i,j,k,ierr,*aj=a->j,*ai=a->i,bs=a->bs,mbs=a->mbs,nbs=a->nbs,len,*col;
  int         *rows,*cols,bs2=a->bs2;
  Scalar      *array=a->a;

  PetscFunctionBegin;
  if (B==PETSC_NULL && mbs!=nbs) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Square matrix only for in-place");
  col = (int *) PetscMalloc((1+nbs)*sizeof(int)); CHKPTRQ(col);
  PetscMemzero(col,(1+nbs)*sizeof(int));

  for ( i=0; i<ai[mbs]; i++ ) col[aj[i]] += 1;
  ierr = MatCreateSeqBAIJ(A->comm,bs,a->n,a->m,PETSC_NULL,col,&C); CHKERRQ(ierr);
  PetscFree(col);
  rows = (int *) PetscMalloc(2*bs*sizeof(int)); CHKPTRQ(rows);
  cols = rows + bs;
  for ( i=0; i<mbs; i++ ) {
    cols[0] = i*bs;
    for (k=1; k<bs; k++ ) cols[k] = cols[k-1] + 1;
    len = ai[i+1] - ai[i];
    for ( j=0; j<len; j++ ) {
      rows[0] = (*aj++)*bs;
      for (k=1; k<bs; k++ ) rows[k] = rows[k-1] + 1;
      ierr = MatSetValues(C,bs,rows,bs,cols,array,INSERT_VALUES); CHKERRQ(ierr);
      array += bs2;
    }
  }
  PetscFree(rows);
  
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  
  if (B != PETSC_NULL) {
    *B = C;
  } else {
    PetscOps *Abops;
    MatOps   Aops;

    /* This isn't really an in-place transpose */
    PetscFree(a->a); 
    if (!a->singlemalloc) {PetscFree(a->i); PetscFree(a->j);}
    if (a->diag) PetscFree(a->diag);
    if (a->ilen) PetscFree(a->ilen);
    if (a->imax) PetscFree(a->imax);
    if (a->solve_work) PetscFree(a->solve_work);
    PetscFree(a); 


    ierr = MapDestroy(A->rmap);CHKERRQ(ierr);
    ierr = MapDestroy(A->cmap);CHKERRQ(ierr);

    /*
       This is horrible, horrible code. We need to keep the 
      A pointers for the bops and ops but copy everything 
      else from C.
    */
    Abops    = A->bops;
    Aops     = A->ops;
    PetscMemcpy(A,C,sizeof(struct _p_Mat));
    A->bops  = Abops;
    A->ops   = Aops;
    A->qlist = 0;

    PetscHeaderDestroy(C);
  }
  PetscFunctionReturn(0);
}




#undef __FUNC__  
#define __FUNC__ "MatView_SeqBAIJ_Binary"
static int MatView_SeqBAIJ_Binary(Mat A,Viewer viewer)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int         i, fd, *col_lens, ierr, bs = a->bs,count,*jj,j,k,l,bs2=a->bs2;
  Scalar      *aa;

  PetscFunctionBegin;
  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
  col_lens = (int *) PetscMalloc((4+a->m)*sizeof(int));CHKPTRQ(col_lens);
  col_lens[0] = MAT_COOKIE;

  col_lens[1] = a->m;
  col_lens[2] = a->n;
  col_lens[3] = a->nz*bs2;

  /* store lengths of each row and write (including header) to file */
  count = 0;
  for ( i=0; i<a->mbs; i++ ) {
    for ( j=0; j<bs; j++ ) {
      col_lens[4+count++] = bs*(a->i[i+1] - a->i[i]);
    }
  }
  ierr = PetscBinaryWrite(fd,col_lens,4+a->m,PETSC_INT,1); CHKERRQ(ierr);
  PetscFree(col_lens);

  /* store column indices (zero start index) */
  jj = (int *) PetscMalloc( (a->nz+1)*bs2*sizeof(int) ); CHKPTRQ(jj);
  count = 0;
  for ( i=0; i<a->mbs; i++ ) {
    for ( j=0; j<bs; j++ ) {
      for ( k=a->i[i]; k<a->i[i+1]; k++ ) {
        for ( l=0; l<bs; l++ ) {
          jj[count++] = bs*a->j[k] + l;
        }
      }
    }
  }
  ierr = PetscBinaryWrite(fd,jj,bs2*a->nz,PETSC_INT,0); CHKERRQ(ierr);
  PetscFree(jj);

  /* store nonzero values */
  aa = (Scalar *) PetscMalloc((a->nz+1)*bs2*sizeof(Scalar)); CHKPTRQ(aa);
  count = 0;
  for ( i=0; i<a->mbs; i++ ) {
    for ( j=0; j<bs; j++ ) {
      for ( k=a->i[i]; k<a->i[i+1]; k++ ) {
        for ( l=0; l<bs; l++ ) {
          aa[count++] = a->a[bs2*k + l*bs + j];
        }
      }
    }
  }
  ierr = PetscBinaryWrite(fd,aa,bs2*a->nz,PETSC_SCALAR,0); CHKERRQ(ierr);
  PetscFree(aa);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqBAIJ_ASCII"
static int MatView_SeqBAIJ_ASCII(Mat A,Viewer viewer)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int         ierr, i,j,format,bs = a->bs,k,l,bs2=a->bs2;
  FILE        *fd;
  char        *outputname;

  PetscFunctionBegin;
  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  ierr = ViewerFileGetOutputname_Private(viewer,&outputname);CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format);
  if (format == VIEWER_FORMAT_ASCII_INFO || format == VIEWER_FORMAT_ASCII_INFO_LONG) {
    fprintf(fd,"  block size is %d\n",bs);
  } 
  else if (format == VIEWER_FORMAT_ASCII_MATLAB) {
    SETERRQ(PETSC_ERR_SUP,0,"Matlab format not supported");
  } 
  else if (format == VIEWER_FORMAT_ASCII_COMMON) {
    for ( i=0; i<a->mbs; i++ ) {
      for ( j=0; j<bs; j++ ) {
        fprintf(fd,"row %d:",i*bs+j);
        for ( k=a->i[i]; k<a->i[i+1]; k++ ) {
          for ( l=0; l<bs; l++ ) {
#if defined(USE_PETSC_COMPLEX)
          if (PetscImaginary(a->a[bs2*k + l*bs + j]) > 0.0 && PetscReal(a->a[bs2*k + l*bs + j]) != 0.0)
            fprintf(fd," %d %g + %g i",bs*a->j[k]+l,
              PetscReal(a->a[bs2*k + l*bs + j]),PetscImaginary(a->a[bs2*k + l*bs + j]));
          else if (PetscImaginary(a->a[bs2*k + l*bs + j]) < 0.0 && PetscReal(a->a[bs2*k + l*bs + j]) != 0.0)
            fprintf(fd," %d %g - %g i",bs*a->j[k]+l,
              PetscReal(a->a[bs2*k + l*bs + j]),-PetscImaginary(a->a[bs2*k + l*bs + j]));
          else if (PetscReal(a->a[bs2*k + l*bs + j]) != 0.0)
            fprintf(fd," %d %g ",bs*a->j[k]+l,PetscReal(a->a[bs2*k + l*bs + j]));
#else
          if (a->a[bs2*k + l*bs + j] != 0.0)
            fprintf(fd," %d %g ",bs*a->j[k]+l,a->a[bs2*k + l*bs + j]);
#endif
          }
        }
        fprintf(fd,"\n");
      }
    } 
  }
  else {
    for ( i=0; i<a->mbs; i++ ) {
      for ( j=0; j<bs; j++ ) {
        fprintf(fd,"row %d:",i*bs+j);
        for ( k=a->i[i]; k<a->i[i+1]; k++ ) {
          for ( l=0; l<bs; l++ ) {
#if defined(USE_PETSC_COMPLEX)
          if (PetscImaginary(a->a[bs2*k + l*bs + j]) > 0.0) {
            fprintf(fd," %d %g + %g i",bs*a->j[k]+l,
              PetscReal(a->a[bs2*k + l*bs + j]),PetscImaginary(a->a[bs2*k + l*bs + j]));
          }
          else if (PetscImaginary(a->a[bs2*k + l*bs + j]) < 0.0) {
            fprintf(fd," %d %g - %g i",bs*a->j[k]+l,
              PetscReal(a->a[bs2*k + l*bs + j]),-PetscImaginary(a->a[bs2*k + l*bs + j]));
          }
          else {
            fprintf(fd," %d %g ",bs*a->j[k]+l,PetscReal(a->a[bs2*k + l*bs + j]));
          }
#else
            fprintf(fd," %d %g ",bs*a->j[k]+l,a->a[bs2*k + l*bs + j]);
#endif
          }
        }
        fprintf(fd,"\n");
      }
    } 
  }
  fflush(fd);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqBAIJ_Draw"
static int MatView_SeqBAIJ_Draw(Mat A,Viewer viewer)
{
  Mat_SeqBAIJ  *a=(Mat_SeqBAIJ *) A->data;
  int          row,ierr,i,j,k,l,mbs=a->mbs,pause,color,bs=a->bs,bs2=a->bs2;
  double       xl,yl,xr,yr,w,h,xc,yc,scale = 1.0,x_l,x_r,y_l,y_r;
  Scalar       *aa;
  Draw         draw;
  DrawButton   button;
  PetscTruth   isnull;

  PetscFunctionBegin;
  ierr = ViewerDrawGetDraw(viewer,&draw);CHKERRQ(ierr);
  ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);

  xr  = a->n; yr = a->m; h = yr/10.0; w = xr/10.0; 
  xr += w;    yr += h;  xl = -w;     yl = -h;
  ierr = DrawSetCoordinates(draw,xl,yl,xr,yr); CHKERRQ(ierr);
  /* loop over matrix elements drawing boxes */
  color = DRAW_BLUE;
  for ( i=0,row=0; i<mbs; i++,row+=bs ) {
    for ( j=a->i[i]; j<a->i[i+1]; j++ ) {
      y_l = a->m - row - 1.0; y_r = y_l + 1.0;
      x_l = a->j[j]*bs; x_r = x_l + 1.0;
      aa = a->a + j*bs2;
      for ( k=0; k<bs; k++ ) {
        for ( l=0; l<bs; l++ ) {
          if (PetscReal(*aa++) >=  0.) continue;
          DrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color);
        }
      }
    } 
  }
  color = DRAW_CYAN;
  for ( i=0,row=0; i<mbs; i++,row+=bs ) {
    for ( j=a->i[i]; j<a->i[i+1]; j++ ) {
      y_l = a->m - row - 1.0; y_r = y_l + 1.0;
      x_l = a->j[j]*bs; x_r = x_l + 1.0;
      aa = a->a + j*bs2;
      for ( k=0; k<bs; k++ ) {
        for ( l=0; l<bs; l++ ) {
          if (PetscReal(*aa++) != 0.) continue;
          DrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color);
        }
      }
    } 
  }

  color = DRAW_RED;
  for ( i=0,row=0; i<mbs; i++,row+=bs ) {
    for ( j=a->i[i]; j<a->i[i+1]; j++ ) {
      y_l = a->m - row - 1.0; y_r = y_l + 1.0;
      x_l = a->j[j]*bs; x_r = x_l + 1.0;
      aa = a->a + j*bs2;
      for ( k=0; k<bs; k++ ) {
        for ( l=0; l<bs; l++ ) {
          if (PetscReal(*aa++) <= 0.) continue;
          DrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color);
        }
      }
    } 
  }

  DrawSynchronizedFlush(draw); 
  DrawGetPause(draw,&pause);
  if (pause >= 0) { PetscSleep(pause); PetscFunctionReturn(0);}

  /* allow the matrix to zoom or shrink */
  ierr = DrawSynchronizedGetMouseButton(draw,&button,&xc,&yc,0,0); 
  while (button != BUTTON_RIGHT) {
    DrawSynchronizedClear(draw);
    if (button == BUTTON_LEFT) scale = .5;
    else if (button == BUTTON_CENTER) scale = 2.;
    xl = scale*(xl + w - xc) + xc - w*scale;
    xr = scale*(xr - w - xc) + xc + w*scale;
    yl = scale*(yl + h - yc) + yc - h*scale;
    yr = scale*(yr - h - yc) + yc + h*scale;
    w *= scale; h *= scale;
    ierr = DrawSetCoordinates(draw,xl,yl,xr,yr); CHKERRQ(ierr);
    color = DRAW_BLUE;
    for ( i=0,row=0; i<mbs; i++,row+=bs ) {
      for ( j=a->i[i]; j<a->i[i+1]; j++ ) {
        y_l = a->m - row - 1.0; y_r = y_l + 1.0;
        x_l = a->j[j]*bs; x_r = x_l + 1.0;
        aa = a->a + j*bs2;
        for ( k=0; k<bs; k++ ) {
          for ( l=0; l<bs; l++ ) {
            if (PetscReal(*aa++) >=  0.) continue;
            DrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color);
          }
        }
      } 
    }
    color = DRAW_CYAN;
    for ( i=0,row=0; i<mbs; i++,row+=bs ) {
      for ( j=a->i[i]; j<a->i[i+1]; j++ ) {
        y_l = a->m - row - 1.0; y_r = y_l + 1.0;
        x_l = a->j[j]*bs; x_r = x_l + 1.0;
        aa = a->a + j*bs2;
        for ( k=0; k<bs; k++ ) {
          for ( l=0; l<bs; l++ ) {
          if (PetscReal(*aa++) != 0.) continue;
          DrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color);
          }
        }
      } 
    }
    
    color = DRAW_RED;
    for ( i=0,row=0; i<mbs; i++,row+=bs ) {
      for ( j=a->i[i]; j<a->i[i+1]; j++ ) {
        y_l = a->m - row - 1.0; y_r = y_l + 1.0;
        x_l = a->j[j]*bs; x_r = x_l + 1.0;
        aa = a->a + j*bs2;
        for ( k=0; k<bs; k++ ) {
          for ( l=0; l<bs; l++ ) {
            if (PetscReal(*aa++) <= 0.) continue;
            DrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color);
          }
        }
      } 
    }
    ierr = DrawSynchronizedGetMouseButton(draw,&button,&xc,&yc,0,0); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqBAIJ"
int MatView_SeqBAIJ(Mat A,Viewer viewer)
{
  ViewerType  vtype;
  int         ierr;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == MATLAB_VIEWER) {
    SETERRQ(PETSC_ERR_SUP,0,"Matlab viewer not supported");
  } else if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER){
    ierr = MatView_SeqBAIJ_ASCII(A,viewer);CHKERRQ(ierr);
  } else if (vtype == BINARY_FILE_VIEWER) {
    ierr = MatView_SeqBAIJ_Binary(A,viewer);CHKERRQ(ierr);
  } else if (vtype == DRAW_VIEWER) {
    ierr = MatView_SeqBAIJ_Draw(A,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Viewer type not supported by PETSc object");
  }
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "MatGetValues_SeqBAIJ"
int MatGetValues_SeqBAIJ(Mat A,int m,int *im,int n,int *in,Scalar *v)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int        *rp, k, low, high, t, row, nrow, i, col, l, *aj = a->j;
  int        *ai = a->i, *ailen = a->ilen;
  int        brow,bcol,ridx,cidx,bs=a->bs,bs2=a->bs2;
  Scalar     *ap, *aa = a->a, zero = 0.0;

  PetscFunctionBegin;
  for ( k=0; k<m; k++ ) { /* loop over rows */
    row  = im[k]; brow = row/bs;  
    if (row < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative row");
    if (row >= a->m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row too large");
    rp   = aj + ai[brow] ; ap = aa + bs2*ai[brow] ;
    nrow = ailen[brow]; 
    for ( l=0; l<n; l++ ) { /* loop over columns */
      if (in[l] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative column");
      if (in[l] >= a->n) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Column too large");
      col  = in[l] ; 
      bcol = col/bs;
      cidx = col%bs; 
      ridx = row%bs;
      high = nrow; 
      low  = 0; /* assume unsorted */
      while (high-low > 5) {
        t = (low+high)/2;
        if (rp[t] > bcol) high = t;
        else             low  = t;
      }
      for ( i=low; i<high; i++ ) {
        if (rp[i] > bcol) break;
        if (rp[i] == bcol) {
          *v++ = ap[bs2*i+bs*cidx+ridx];
          goto finished;
        }
      } 
      *v++ = zero;
      finished:;
    }
  }
  PetscFunctionReturn(0);
} 


#undef __FUNC__  
#define __FUNC__ "MatSetValuesBlocked_SeqBAIJ"
int MatSetValuesBlocked_SeqBAIJ(Mat A,int m,int *im,int n,int *in,Scalar *v,InsertMode is)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int         *rp,k,low,high,t,ii,jj,row,nrow,i,col,l,rmax,N,sorted=a->sorted;
  int         *imax=a->imax,*ai=a->i,*ailen=a->ilen, roworiented=a->roworiented; 
  int         *aj=a->j,nonew=a->nonew,bs2=a->bs2,bs=a->bs,stepval;
  Scalar      *ap,*value=v,*aa=a->a,*bap;

  PetscFunctionBegin;
  if (roworiented) { 
    stepval = (n-1)*bs;
  } else {
    stepval = (m-1)*bs;
  }
  for ( k=0; k<m; k++ ) { /* loop over added rows */
    row  = im[k]; 
#if defined(USE_PETSC_BOPT_g)  
    if (row < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative row");
    if (row >= a->mbs) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row too large");
#endif
    rp   = aj + ai[row]; 
    ap   = aa + bs2*ai[row];
    rmax = imax[row]; 
    nrow = ailen[row]; 
    low  = 0;
    for ( l=0; l<n; l++ ) { /* loop over added columns */
#if defined(USE_PETSC_BOPT_g)  
      if (in[l] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative column");
      if (in[l] >= a->nbs) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Column too large");
#endif
      col = in[l]; 
      if (roworiented) { 
        value = v + k*(stepval+bs)*bs + l*bs;
      } else {
        value = v + l*(stepval+bs)*bs + k*bs;
      }
      if (!sorted) low = 0; high = nrow;
      while (high-low > 7) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else             low  = t;
      }
      for ( i=low; i<high; i++ ) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          bap  = ap +  bs2*i;
          if (roworiented) { 
            if (is == ADD_VALUES) {
              for ( ii=0; ii<bs; ii++,value+=stepval ) {
                for (jj=ii; jj<bs2; jj+=bs ) {
                  bap[jj] += *value++; 
                }
              }
            } else {
              for ( ii=0; ii<bs; ii++,value+=stepval ) {
                for (jj=ii; jj<bs2; jj+=bs ) {
                  bap[jj] = *value++; 
                }
              }
            }
          } else {
            if (is == ADD_VALUES) {
              for ( ii=0; ii<bs; ii++,value+=stepval ) {
                for (jj=0; jj<bs; jj++ ) {
                  *bap++ += *value++; 
                }
              }
            } else {
              for ( ii=0; ii<bs; ii++,value+=stepval ) {
                for (jj=0; jj<bs; jj++ ) {
                  *bap++  = *value++; 
                }
              }
            }
          }
          goto noinsert2;
        }
      } 
      if (nonew == 1) goto noinsert2;
      else if (nonew == -1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Inserting a new nonzero in the matrix");
      if (nrow >= rmax) {
        /* there is no extra room in row, therefore enlarge */
        int    new_nz = ai[a->mbs] + CHUNKSIZE,len,*new_i,*new_j;
        Scalar *new_a;

        if (nonew == -2) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Inserting a new nonzero in the matrix");

        /* malloc new storage space */
        len     = new_nz*(sizeof(int)+bs2*sizeof(Scalar))+(a->mbs+1)*sizeof(int);
        new_a   = (Scalar *) PetscMalloc( len ); CHKPTRQ(new_a);
        new_j   = (int *) (new_a + bs2*new_nz);
        new_i   = new_j + new_nz;

        /* copy over old data into new slots */
        for ( ii=0; ii<row+1; ii++ ) {new_i[ii] = ai[ii];}
        for ( ii=row+1; ii<a->mbs+1; ii++ ) {new_i[ii] = ai[ii]+CHUNKSIZE;}
        PetscMemcpy(new_j,aj,(ai[row]+nrow)*sizeof(int));
        len = (new_nz - CHUNKSIZE - ai[row] - nrow);
        PetscMemcpy(new_j+ai[row]+nrow+CHUNKSIZE,aj+ai[row]+nrow,len*sizeof(int));
        PetscMemcpy(new_a,aa,(ai[row]+nrow)*bs2*sizeof(Scalar));
        PetscMemzero(new_a+bs2*(ai[row]+nrow),bs2*CHUNKSIZE*sizeof(Scalar));
        PetscMemcpy(new_a+bs2*(ai[row]+nrow+CHUNKSIZE),aa+bs2*(ai[row]+nrow),bs2*len*sizeof(Scalar)); 
        /* free up old matrix storage */
        PetscFree(a->a); 
        if (!a->singlemalloc) {PetscFree(a->i);PetscFree(a->j);}
        aa = a->a = new_a; ai = a->i = new_i; aj = a->j = new_j; 
        a->singlemalloc = 1;

        rp   = aj + ai[row]; ap = aa + bs2*ai[row];
        rmax = imax[row] = imax[row] + CHUNKSIZE;
        PLogObjectMemory(A,CHUNKSIZE*(sizeof(int) + bs2*sizeof(Scalar)));
        a->maxnz += bs2*CHUNKSIZE;
        a->reallocs++;
        a->nz++;
      }
      N = nrow++ - 1; 
      /* shift up all the later entries in this row */
      for ( ii=N; ii>=i; ii-- ) {
        rp[ii+1] = rp[ii];
        PetscMemcpy(ap+bs2*(ii+1),ap+bs2*(ii),bs2*sizeof(Scalar));
      }
      if (N >= i) PetscMemzero(ap+bs2*i,bs2*sizeof(Scalar)); 
      rp[i] = col; 
      bap   = ap +  bs2*i;
      if (roworiented) { 
        for ( ii=0; ii<bs; ii++,value+=stepval) {
          for (jj=ii; jj<bs2; jj+=bs ) {
            bap[jj] = *value++; 
          }
        }
      } else {
        for ( ii=0; ii<bs; ii++,value+=stepval ) {
          for (jj=0; jj<bs; jj++ ) {
            *bap++  = *value++; 
          }
        }
      }
      noinsert2:;
      low = i;
    }
    ailen[row] = nrow;
  }
  PetscFunctionReturn(0);
} 


#undef __FUNC__  
#define __FUNC__ "MatAssemblyEnd_SeqBAIJ"
int MatAssemblyEnd_SeqBAIJ(Mat A,MatAssemblyType mode)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int        fshift = 0,i,j,*ai = a->i, *aj = a->j, *imax = a->imax;
  int        m = a->m,*ip, N, *ailen = a->ilen;
  int        mbs = a->mbs, bs2 = a->bs2,rmax = 0;
  Scalar     *aa = a->a, *ap;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  if (m) rmax = ailen[0];
  for ( i=1; i<mbs; i++ ) {
    /* move each row back by the amount of empty slots (fshift) before it*/
    fshift += imax[i-1] - ailen[i-1];
    rmax   = PetscMax(rmax,ailen[i]);
    if (fshift) {
      ip = aj + ai[i]; ap = aa + bs2*ai[i];
      N = ailen[i];
      for ( j=0; j<N; j++ ) {
        ip[j-fshift] = ip[j];
        PetscMemcpy(ap+(j-fshift)*bs2,ap+j*bs2,bs2*sizeof(Scalar));
      }
    } 
    ai[i] = ai[i-1] + ailen[i-1];
  }
  if (mbs) {
    fshift += imax[mbs-1] - ailen[mbs-1];
    ai[mbs] = ai[mbs-1] + ailen[mbs-1];
  }
  /* reset ilen and imax for each row */
  for ( i=0; i<mbs; i++ ) {
    ailen[i] = imax[i] = ai[i+1] - ai[i];
  }
  a->nz = ai[mbs]; 

  /* diagonals may have moved, so kill the diagonal pointers */
  if (fshift && a->diag) {
    PetscFree(a->diag);
    PLogObjectMemory(A,-(m+1)*sizeof(int));
    a->diag = 0;
  } 
  PLogInfo(A,"MatAssemblyEnd_SeqBAIJ:Matrix size: %d X %d, block size %d; storage space: %d unneeded, %d used\n",
           m,a->n,a->bs,fshift*bs2,a->nz*bs2);
  PLogInfo(A,"MatAssemblyEnd_SeqBAIJ:Number of mallocs during MatSetValues is %d\n",
           a->reallocs);
  PLogInfo(A,"MatAssemblyEnd_SeqBAIJ:Most nonzeros blocks in any row is %d\n",rmax);
  a->reallocs          = 0;
  A->info.nz_unneeded  = (double)fshift*bs2;

  PetscFunctionReturn(0);
}


/* idx should be of length atlease bs */
#undef __FUNC__  
#define __FUNC__ "MatZeroRows_SeqBAIJ_Check_Block"
static int MatZeroRows_SeqBAIJ_Check_Block(int *idx, int bs, PetscTruth *flg)
{
  int i,row;

  PetscFunctionBegin;
  row = idx[0];
  if (row%bs!=0) { *flg = PETSC_FALSE; PetscFunctionReturn(0); }
  
  for ( i=1; i<bs; i++ ) {
    if (row+i != idx[i]) { *flg = PETSC_FALSE; PetscFunctionReturn(0); }
  }
  *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}
  
#undef __FUNC__  
#define __FUNC__ "MatZeroRows_SeqBAIJ"
int MatZeroRows_SeqBAIJ(Mat A,IS is, Scalar *diag)
{
  Mat_SeqBAIJ *baij=(Mat_SeqBAIJ*)A->data;
  IS          is_local;
  int         ierr,i,j,count,m=baij->m,is_n,*is_idx,*rows,bs=baij->bs,bs2=baij->bs2;
  PetscTruth  flg;
  Scalar      zero = 0.0,*aa;

  PetscFunctionBegin;
  /* Make a copy of the IS and  sort it */
  ierr = ISGetSize(is,&is_n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&is_idx);CHKERRQ(ierr);
  ierr = ISCreateGeneral(A->comm,is_n,is_idx,&is_local); CHKERRQ(ierr);
  ierr = ISSort(is_local); CHKERRQ(ierr);
  ierr = ISGetIndices(is_local,&rows); CHKERRQ(ierr);

  i = 0;
  while (i < is_n) {
    if (rows[i]<0 || rows[i]>m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"row out of range");
    flg = PETSC_FALSE;
    if (i+bs <= is_n) {ierr = MatZeroRows_SeqBAIJ_Check_Block(rows+i,bs,&flg); CHKERRQ(ierr); }
    count = (baij->i[rows[i]/bs +1] - baij->i[rows[i]/bs])*bs;
    aa    = baij->a + baij->i[rows[i]/bs]*bs2 + (rows[i]%bs);
    if (flg) { /* There exists a block of rows to be Zerowed */
      if (baij->ilen[rows[i]/bs] > 0) {
        PetscMemzero(aa,count*bs*sizeof(Scalar));
        baij->ilen[rows[i]/bs] = 1;
        baij->j[baij->i[rows[i]/bs]] = rows[i]/bs;
      }
      i += bs;
    } else { /* Zero out only the requested row */
      for ( j=0; j<count; j++ ) { 
        aa[0] = zero; 
        aa+=bs;
      }
      i++;
    }
  } 
  if (diag) {
    for ( j=0; j<is_n; j++ ) {
      ierr = (*A->ops->setvalues)(A,1,rows+j,1,rows+j,diag,INSERT_VALUES);CHKERRQ(ierr);
      /* ierr = MatSetValues(A,1,rows+j,1,rows+j,diag,INSERT_VALUES);CHKERRQ(ierr); */
    }
  }
  ierr = ISRestoreIndices(is,&is_idx); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is_local,&rows); CHKERRQ(ierr);
  ierr = ISDestroy(is_local); CHKERRQ(ierr);
  ierr = MatAssemblyEnd_SeqBAIJ(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSetValues_SeqBAIJ"
int MatSetValues_SeqBAIJ(Mat A,int m,int *im,int n,int *in,Scalar *v,InsertMode is)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int         *rp,k,low,high,t,ii,row,nrow,i,col,l,rmax,N,sorted=a->sorted;
  int         *imax=a->imax,*ai=a->i,*ailen=a->ilen,roworiented=a->roworiented;
  int         *aj=a->j,nonew=a->nonew,bs=a->bs,brow,bcol;
  int          ridx,cidx,bs2=a->bs2;
  Scalar      *ap,value,*aa=a->a,*bap;

  PetscFunctionBegin;
  for ( k=0; k<m; k++ ) { /* loop over added rows */
    row  = im[k]; brow = row/bs;  
#if defined(USE_PETSC_BOPT_g)  
    if (row < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative row");
    if (row >= a->m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row too large");
#endif
    rp   = aj + ai[brow]; 
    ap   = aa + bs2*ai[brow];
    rmax = imax[brow]; 
    nrow = ailen[brow]; 
    low  = 0;
    for ( l=0; l<n; l++ ) { /* loop over added columns */
#if defined(USE_PETSC_BOPT_g)  
      if (in[l] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative column");
      if (in[l] >= a->n) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Column too large");
#endif
      col = in[l]; bcol = col/bs;
      ridx = row % bs; cidx = col % bs;
      if (roworiented) {
        value = *v++; 
      } else {
        value = v[k + l*m];
      }
      if (!sorted) low = 0; high = nrow;
      while (high-low > 7) {
        t = (low+high)/2;
        if (rp[t] > bcol) high = t;
        else              low  = t;
      }
      for ( i=low; i<high; i++ ) {
        if (rp[i] > bcol) break;
        if (rp[i] == bcol) {
          bap  = ap +  bs2*i + bs*cidx + ridx;
          if (is == ADD_VALUES) *bap += value;  
          else                  *bap  = value; 
          goto noinsert1;
        }
      } 
      if (nonew == 1) goto noinsert1;
      else if (nonew == -1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Inserting a new nonzero in the matrix");
      if (nrow >= rmax) {
        /* there is no extra room in row, therefore enlarge */
        int    new_nz = ai[a->mbs] + CHUNKSIZE,len,*new_i,*new_j;
        Scalar *new_a;

        if (nonew == -2) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Inserting a new nonzero in the matrix");

        /* Malloc new storage space */
        len     = new_nz*(sizeof(int)+bs2*sizeof(Scalar))+(a->mbs+1)*sizeof(int);
        new_a   = (Scalar *) PetscMalloc( len ); CHKPTRQ(new_a);
        new_j   = (int *) (new_a + bs2*new_nz);
        new_i   = new_j + new_nz;

        /* copy over old data into new slots */
        for ( ii=0; ii<brow+1; ii++ ) {new_i[ii] = ai[ii];}
        for ( ii=brow+1; ii<a->mbs+1; ii++ ) {new_i[ii] = ai[ii]+CHUNKSIZE;}
        PetscMemcpy(new_j,aj,(ai[brow]+nrow)*sizeof(int));
        len = (new_nz - CHUNKSIZE - ai[brow] - nrow);
        PetscMemcpy(new_j+ai[brow]+nrow+CHUNKSIZE,aj+ai[brow]+nrow,
                                                           len*sizeof(int));
        PetscMemcpy(new_a,aa,(ai[brow]+nrow)*bs2*sizeof(Scalar));
        PetscMemzero(new_a+bs2*(ai[brow]+nrow),bs2*CHUNKSIZE*sizeof(Scalar));
        PetscMemcpy(new_a+bs2*(ai[brow]+nrow+CHUNKSIZE),
                    aa+bs2*(ai[brow]+nrow),bs2*len*sizeof(Scalar)); 
        /* free up old matrix storage */
        PetscFree(a->a); 
        if (!a->singlemalloc) {PetscFree(a->i);PetscFree(a->j);}
        aa = a->a = new_a; ai = a->i = new_i; aj = a->j = new_j; 
        a->singlemalloc = 1;

        rp   = aj + ai[brow]; ap = aa + bs2*ai[brow];
        rmax = imax[brow] = imax[brow] + CHUNKSIZE;
        PLogObjectMemory(A,CHUNKSIZE*(sizeof(int) + bs2*sizeof(Scalar)));
        a->maxnz += bs2*CHUNKSIZE;
        a->reallocs++;
        a->nz++;
      }
      N = nrow++ - 1; 
      /* shift up all the later entries in this row */
      for ( ii=N; ii>=i; ii-- ) {
        rp[ii+1] = rp[ii];
        PetscMemcpy(ap+bs2*(ii+1),ap+bs2*(ii),bs2*sizeof(Scalar));
      }
      if (N>=i) PetscMemzero(ap+bs2*i,bs2*sizeof(Scalar)); 
      rp[i]                      = bcol; 
      ap[bs2*i + bs*cidx + ridx] = value; 
      noinsert1:;
      low = i;
    }
    ailen[brow] = nrow;
  }
  PetscFunctionReturn(0);
} 

extern int MatLUFactorSymbolic_SeqBAIJ(Mat,IS,IS,double,Mat*);
extern int MatLUFactor_SeqBAIJ(Mat,IS,IS,double);
extern int MatIncreaseOverlap_SeqBAIJ(Mat,int,IS*,int);
extern int MatGetSubMatrix_SeqBAIJ(Mat,IS,IS,int,MatGetSubMatrixCall,Mat*);
extern int MatGetSubMatrices_SeqBAIJ(Mat,int,IS*,IS*,MatGetSubMatrixCall,Mat**);
extern int MatMultTrans_SeqBAIJ(Mat,Vec,Vec);
extern int MatMultTransAdd_SeqBAIJ(Mat,Vec,Vec,Vec);
extern int MatScale_SeqBAIJ(Scalar*,Mat);
extern int MatNorm_SeqBAIJ(Mat,NormType,double *);
extern int MatEqual_SeqBAIJ(Mat,Mat, PetscTruth*);
extern int MatGetDiagonal_SeqBAIJ(Mat,Vec);
extern int MatDiagonalScale_SeqBAIJ(Mat,Vec,Vec);
extern int MatGetInfo_SeqBAIJ(Mat,MatInfoType,MatInfo *);
extern int MatZeroEntries_SeqBAIJ(Mat);

extern int MatSolve_SeqBAIJ_N(Mat,Vec,Vec);
extern int MatSolve_SeqBAIJ_1(Mat,Vec,Vec);
extern int MatSolve_SeqBAIJ_2(Mat,Vec,Vec);
extern int MatSolve_SeqBAIJ_3(Mat,Vec,Vec);
extern int MatSolve_SeqBAIJ_4(Mat,Vec,Vec);
extern int MatSolve_SeqBAIJ_5(Mat,Vec,Vec);
extern int MatSolve_SeqBAIJ_7(Mat,Vec,Vec);

extern int MatLUFactorNumeric_SeqBAIJ_N(Mat,Mat*);
extern int MatLUFactorNumeric_SeqBAIJ_1(Mat,Mat*);
extern int MatLUFactorNumeric_SeqBAIJ_2(Mat,Mat*);
extern int MatLUFactorNumeric_SeqBAIJ_3(Mat,Mat*);
extern int MatLUFactorNumeric_SeqBAIJ_4(Mat,Mat*);
extern int MatLUFactorNumeric_SeqBAIJ_5(Mat,Mat*);
extern int MatSolve_SeqBAIJ_4_NaturalOrdering(Mat,Vec,Vec);

extern int MatMult_SeqBAIJ_1(Mat,Vec,Vec);
extern int MatMult_SeqBAIJ_2(Mat,Vec,Vec);
extern int MatMult_SeqBAIJ_3(Mat,Vec,Vec);
extern int MatMult_SeqBAIJ_4(Mat,Vec,Vec);
extern int MatMult_SeqBAIJ_5(Mat,Vec,Vec);
extern int MatMult_SeqBAIJ_7(Mat,Vec,Vec);
extern int MatMult_SeqBAIJ_N(Mat,Vec,Vec);

extern int MatMultAdd_SeqBAIJ_1(Mat,Vec,Vec,Vec);
extern int MatMultAdd_SeqBAIJ_2(Mat,Vec,Vec,Vec);
extern int MatMultAdd_SeqBAIJ_3(Mat,Vec,Vec,Vec);
extern int MatMultAdd_SeqBAIJ_4(Mat,Vec,Vec,Vec);
extern int MatMultAdd_SeqBAIJ_5(Mat,Vec,Vec,Vec);
extern int MatMultAdd_SeqBAIJ_7(Mat,Vec,Vec,Vec);
extern int MatMultAdd_SeqBAIJ_N(Mat,Vec,Vec,Vec);

/*
     note: This can only work for identity for row and col. It would 
   be good to check this and otherwise generate an error.
*/
#undef __FUNC__  
#define __FUNC__ "MatILUFactor_SeqBAIJ"
int MatILUFactor_SeqBAIJ(Mat inA,IS row,IS col,double efill,int fill)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) inA->data;
  Mat         outA;
  int         ierr;

  PetscFunctionBegin;
  if (fill != 0) SETERRQ(PETSC_ERR_SUP,0,"Only fill=0 supported");

  outA          = inA; 
  inA->factor   = FACTOR_LU;
  a->row        = row;
  a->col        = col;

  /* Create the invert permutation so that it can be used in MatLUFactorNumeric() */
  ierr = ISInvertPermutation(col,&(a->icol)); CHKERRQ(ierr);
  PLogObjectParent(inA,a->icol);

  if (!a->solve_work) {
    a->solve_work = (Scalar *) PetscMalloc((a->m+a->bs)*sizeof(Scalar));CHKPTRQ(a->solve_work);
    PLogObjectMemory(inA,(a->m+a->bs)*sizeof(Scalar));
  }

  if (!a->diag) {
    ierr = MatMarkDiag_SeqBAIJ(inA); CHKERRQ(ierr);
  }
  ierr = MatLUFactorNumeric(inA,&outA); CHKERRQ(ierr);
  
  /*
      Blocksize 4 has a special faster solver for ILU(0) factorization 
    with natural ordering 
  */
  if (a->bs == 4) {
    inA->ops->solve = MatSolve_SeqBAIJ_4_NaturalOrdering;
  }

  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ "MatPrintHelp_SeqBAIJ"
int MatPrintHelp_SeqBAIJ(Mat A)
{
  static int called = 0; 
  MPI_Comm   comm = A->comm;

  PetscFunctionBegin;
  if (called) {PetscFunctionReturn(0);} else called = 1;
  (*PetscHelpPrintf)(comm," Options for MATSEQBAIJ and MATMPIBAIJ matrix formats (the defaults):\n");
  (*PetscHelpPrintf)(comm,"  -mat_block_size <block_size>\n");
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "MatSeqBAIJSetColumnIndices_SeqBAIJ"
int MatSeqBAIJSetColumnIndices_SeqBAIJ(Mat mat,int *indices)
{
  Mat_SeqBAIJ *baij = (Mat_SeqBAIJ *)mat->data;
  int         i,nz,n;

  PetscFunctionBegin;
  nz = baij->maxnz;
  n  = baij->n;
  for (i=0; i<nz; i++) {
    baij->j[i] = indices[i];
  }
  baij->nz = nz;
  for ( i=0; i<n; i++ ) {
    baij->ilen[i] = baij->imax[i];
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "MatSeqBAIJSetColumnIndices"
/*@
    MatSeqBAIJSetColumnIndices - Set the column indices for all the rows
       in the matrix.

  Input Parameters:
+  mat - the SeqBAIJ matrix
-  indices - the column indices

  Notes:
    This can be called if you have precomputed the nonzero structure of the 
  matrix and want to provide it to the matrix object to improve the performance
  of the MatSetValues() operation.

    You MUST have set the correct numbers of nonzeros per row in the call to 
  MatCreateSeqBAIJ().

    MUST be called before any calls to MatSetValues();

@*/ 
int MatSeqBAIJSetColumnIndices(Mat mat,int *indices)
{
  int ierr,(*f)(Mat,int *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatSeqBAIJSetColumnIndices_C",(void **)&f); 
         CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat,indices);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Wrong type of matrix to set column indices");
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_SeqBAIJ,
       MatGetRow_SeqBAIJ,
       MatRestoreRow_SeqBAIJ,
       MatMult_SeqBAIJ_N,
       MatMultAdd_SeqBAIJ_N,
       MatMultTrans_SeqBAIJ,
       MatMultTransAdd_SeqBAIJ,
       MatSolve_SeqBAIJ_N,
       0,
       0,
       0,
       MatLUFactor_SeqBAIJ,
       0,
       0,
       MatTranspose_SeqBAIJ,
       MatGetInfo_SeqBAIJ,
       MatEqual_SeqBAIJ,
       MatGetDiagonal_SeqBAIJ,
       MatDiagonalScale_SeqBAIJ,
       MatNorm_SeqBAIJ,
       0,
       MatAssemblyEnd_SeqBAIJ,
       0,
       MatSetOption_SeqBAIJ,
       MatZeroEntries_SeqBAIJ,
       MatZeroRows_SeqBAIJ,
       MatLUFactorSymbolic_SeqBAIJ,
       MatLUFactorNumeric_SeqBAIJ_N,
       0,
       0,
       MatGetSize_SeqBAIJ,
       MatGetSize_SeqBAIJ,
       MatGetOwnershipRange_SeqBAIJ,
       MatILUFactorSymbolic_SeqBAIJ,
       0,
       0,
       0,
       MatDuplicate_SeqBAIJ,
       0,
       0,
       MatILUFactor_SeqBAIJ,
       0,
       0,
       MatGetSubMatrices_SeqBAIJ,
       MatIncreaseOverlap_SeqBAIJ,
       MatGetValues_SeqBAIJ,
       0,
       MatPrintHelp_SeqBAIJ,
       MatScale_SeqBAIJ,
       0,
       0,
       0,
       MatGetBlockSize_SeqBAIJ,
       MatGetRowIJ_SeqBAIJ,
       MatRestoreRowIJ_SeqBAIJ,
       0,
       0,
       0,
       0,
       0,
       0,
       MatSetValuesBlocked_SeqBAIJ,
       MatGetSubMatrix_SeqBAIJ,
       0,
       0,
       MatGetMaps_Petsc};

#undef __FUNC__  
#define __FUNC__ "MatCreateSeqBAIJ"
/*@C
   MatCreateSeqBAIJ - Creates a sparse matrix in block AIJ (block
   compressed row) format.  For good matrix assembly performance the
   user should preallocate the matrix storage by setting the parameter nz
   (or the array nzz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  bs - size of block
.  m - number of rows
.  n - number of columns
.  nz - number of block nonzeros per block row (same for all rows)
-  nzz - array containing the number of block nonzeros in the various block rows 
         (possibly different for each block row) or PETSC_NULL

   Output Parameter:
.  A - the matrix 

   Options Database Keys:
.   -mat_no_unroll - uses code that does not unroll the loops in the 
                     block calculations (much slower)
.    -mat_block_size - size of the blocks to use

   Notes:
   The block AIJ format is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=PETSC_NULL for PETSc to control dynamic memory 
   allocation.  For additional details, see the users manual chapter on
   matrices.

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatCreateMPIBAIJ()
@*/
int MatCreateSeqBAIJ(MPI_Comm comm,int bs,int m,int n,int nz,int *nnz, Mat *A)
{
  Mat         B;
  Mat_SeqBAIJ *b;
  int         i,len,ierr,flg,mbs=m/bs,nbs=n/bs,bs2=bs*bs,size;

  PetscFunctionBegin;
  MPI_Comm_size(comm,&size);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_WRONG,0,"Comm must be of size 1");

  ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,&flg);CHKERRQ(ierr);

  if (mbs*bs!=m || nbs*bs!=n) {
    SETERRQ(PETSC_ERR_ARG_SIZ,0,"Number rows, cols must be divisible by blocksize");
  }

  *A = 0;
  PetscHeaderCreate(B,_p_Mat,struct _MatOps,MAT_COOKIE,MATSEQBAIJ,comm,MatDestroy,MatView);
  PLogObjectCreate(B);
  B->data = (void *) (b = PetscNew(Mat_SeqBAIJ)); CHKPTRQ(b);
  PetscMemzero(b,sizeof(Mat_SeqBAIJ));
  PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));
  ierr = OptionsHasName(PETSC_NULL,"-mat_no_unroll",&flg); CHKERRQ(ierr);
  if (!flg) {
    switch (bs) {
    case 1:
      B->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_1;  
      B->ops->solve           = MatSolve_SeqBAIJ_1;
      B->ops->mult            = MatMult_SeqBAIJ_1;
      B->ops->multadd         = MatMultAdd_SeqBAIJ_1;
      break;
    case 2:
      B->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_2;  
      B->ops->solve           = MatSolve_SeqBAIJ_2;
      B->ops->mult            = MatMult_SeqBAIJ_2;
      B->ops->multadd         = MatMultAdd_SeqBAIJ_2;
      break;
    case 3:
      B->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_3;  
      B->ops->solve           = MatSolve_SeqBAIJ_3;
      B->ops->mult            = MatMult_SeqBAIJ_3;
      B->ops->multadd         = MatMultAdd_SeqBAIJ_3;
      break;
    case 4:
      B->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4;  
      B->ops->solve           = MatSolve_SeqBAIJ_4;
      B->ops->mult            = MatMult_SeqBAIJ_4;
      B->ops->multadd         = MatMultAdd_SeqBAIJ_4;
      break;
    case 5:
      B->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_5;  
      B->ops->solve           = MatSolve_SeqBAIJ_5; 
      B->ops->mult            = MatMult_SeqBAIJ_5;
      B->ops->multadd         = MatMultAdd_SeqBAIJ_5;
      break;
    case 7:
      B->ops->mult            = MatMult_SeqBAIJ_7; 
      B->ops->solve           = MatSolve_SeqBAIJ_7;
      B->ops->multadd         = MatMultAdd_SeqBAIJ_7;
      break;
    }
  }
  B->ops->destroy     = MatDestroy_SeqBAIJ;
  B->ops->view        = MatView_SeqBAIJ;
  B->factor           = 0;
  B->lupivotthreshold = 1.0;
  B->mapping          = 0;
  b->row              = 0;
  b->col              = 0;
  b->icol             = 0;
  b->reallocs         = 0;
  
  b->m       = m; B->m = m; B->M = m;
  b->n       = n; B->n = n; B->N = n;

  ierr = MapCreateMPI(comm,m,m,&B->rmap);CHKERRQ(ierr);
  ierr = MapCreateMPI(comm,n,n,&B->cmap);CHKERRQ(ierr);

  b->mbs     = mbs;
  b->nbs     = nbs;
  b->imax    = (int *) PetscMalloc( (mbs+1)*sizeof(int) ); CHKPTRQ(b->imax);
  if (nnz == PETSC_NULL) {
    if (nz == PETSC_DEFAULT) nz = 5;
    else if (nz <= 0)        nz = 1;
    for ( i=0; i<mbs; i++ ) b->imax[i] = nz;
    nz = nz*mbs;
  } else {
    nz = 0;
    for ( i=0; i<mbs; i++ ) {b->imax[i] = nnz[i]; nz += nnz[i];}
  }

  /* allocate the matrix space */
  len   = nz*sizeof(int) + nz*bs2*sizeof(Scalar) + (b->m+1)*sizeof(int);
  b->a  = (Scalar *) PetscMalloc( len ); CHKPTRQ(b->a);
  PetscMemzero(b->a,nz*bs2*sizeof(Scalar));
  b->j  = (int *) (b->a + nz*bs2);
  PetscMemzero(b->j,nz*sizeof(int));
  b->i  = b->j + nz;
  b->singlemalloc = 1;

  b->i[0] = 0;
  for (i=1; i<mbs+1; i++) {
    b->i[i] = b->i[i-1] + b->imax[i-1];
  }

  /* b->ilen will count nonzeros in each block row so far. */
  b->ilen = (int *) PetscMalloc((mbs+1)*sizeof(int)); 
  PLogObjectMemory(B,len+2*(mbs+1)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_SeqBAIJ));
  for ( i=0; i<mbs; i++ ) { b->ilen[i] = 0;}

  b->bs               = bs;
  b->bs2              = bs2;
  b->mbs              = mbs;
  b->nz               = 0;
  b->maxnz            = nz*bs2;
  b->sorted           = 0;
  b->roworiented      = 1;
  b->nonew            = 0;
  b->diag             = 0;
  b->solve_work       = 0;
  b->mult_work        = 0;
  b->spptr            = 0;
  B->info.nz_unneeded = (double)b->maxnz;

  *A = B;
  ierr = OptionsHasName(PETSC_NULL,"-help", &flg); CHKERRQ(ierr);
  if (flg) {ierr = MatPrintHelp(B); CHKERRQ(ierr); }

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqBAIJSetColumnIndices_C",
                                     "MatSeqBAIJSetColumnIndices_SeqBAIJ",
                                     (void*)MatSeqBAIJSetColumnIndices_SeqBAIJ);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDuplicate_SeqBAIJ"
int MatDuplicate_SeqBAIJ(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  Mat         C;
  Mat_SeqBAIJ *c,*a = (Mat_SeqBAIJ *) A->data;
  int         i,len, mbs = a->mbs,nz = a->nz,bs2 =a->bs2,ierr;

  PetscFunctionBegin;
  if (a->i[mbs] != nz) SETERRQ(PETSC_ERR_PLIB,0,"Corrupt matrix");

  *B = 0;
  PetscHeaderCreate(C,_p_Mat,struct _MatOps,MAT_COOKIE,MATSEQBAIJ,A->comm,MatDestroy,MatView);
  PLogObjectCreate(C);
  C->data       = (void *) (c = PetscNew(Mat_SeqBAIJ)); CHKPTRQ(c);
  PetscMemcpy(C->ops,A->ops,sizeof(struct _MatOps));
  C->ops->destroy    = MatDestroy_SeqBAIJ;
  C->ops->view       = MatView_SeqBAIJ;
  C->factor          = A->factor;
  c->row             = 0;
  c->col             = 0;
  C->assembled       = PETSC_TRUE;

  c->m = C->m   = a->m;
  c->n = C->n   = a->n;
  C->M          = a->m;
  C->N          = a->n;

  c->bs         = a->bs;
  c->bs2        = a->bs2;
  c->mbs        = a->mbs;
  c->nbs        = a->nbs;

  c->imax       = (int *) PetscMalloc((mbs+1)*sizeof(int)); CHKPTRQ(c->imax);
  c->ilen       = (int *) PetscMalloc((mbs+1)*sizeof(int)); CHKPTRQ(c->ilen);
  for ( i=0; i<mbs; i++ ) {
    c->imax[i] = a->imax[i];
    c->ilen[i] = a->ilen[i]; 
  }

  /* allocate the matrix space */
  c->singlemalloc = 1;
  len   = (mbs+1)*sizeof(int) + nz*(bs2*sizeof(Scalar) + sizeof(int));
  c->a  = (Scalar *) PetscMalloc( len ); CHKPTRQ(c->a);
  c->j  = (int *) (c->a + nz*bs2);
  c->i  = c->j + nz;
  PetscMemcpy(c->i,a->i,(mbs+1)*sizeof(int));
  if (mbs > 0) {
    PetscMemcpy(c->j,a->j,nz*sizeof(int));
    if (cpvalues == MAT_COPY_VALUES) {
      PetscMemcpy(c->a,a->a,bs2*nz*sizeof(Scalar));
    } else {
      PetscMemzero(c->a,bs2*nz*sizeof(Scalar));
    }
  }

  PLogObjectMemory(C,len+2*(mbs+1)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_SeqBAIJ));  
  c->sorted      = a->sorted;
  c->roworiented = a->roworiented;
  c->nonew       = a->nonew;

  if (a->diag) {
    c->diag = (int *) PetscMalloc( (mbs+1)*sizeof(int) ); CHKPTRQ(c->diag);
    PLogObjectMemory(C,(mbs+1)*sizeof(int));
    for ( i=0; i<mbs; i++ ) {
      c->diag[i] = a->diag[i];
    }
  }
  else c->diag          = 0;
  c->nz                 = a->nz;
  c->maxnz              = a->maxnz;
  c->solve_work         = 0;
  c->spptr              = 0;      /* Dangerous -I'm throwing away a->spptr */
  c->mult_work          = 0;
  *B = C;
  ierr = PetscObjectComposeFunction((PetscObject)C,"MatSeqBAIJSetColumnIndices_C",
                                     "MatSeqBAIJSetColumnIndices_SeqBAIJ",
                                     (void*)MatSeqBAIJSetColumnIndices_SeqBAIJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatLoad_SeqBAIJ"
int MatLoad_SeqBAIJ(Viewer viewer,MatType type,Mat *A)
{
  Mat_SeqBAIJ  *a;
  Mat          B;
  int          i,nz,ierr,fd,header[4],size,*rowlengths=0,M,N,bs=1,flg;
  int          *mask,mbs,*jj,j,rowcount,nzcount,k,*browlengths,maskcount;
  int          kmax,jcount,block,idx,point,nzcountb,extra_rows;
  int          *masked, nmask,tmp,bs2,ishift;
  Scalar       *aa;
  MPI_Comm     comm = ((PetscObject) viewer)->comm;

  PetscFunctionBegin;
  ierr = OptionsGetInt(PETSC_NULL,"-matload_block_size",&bs,&flg);CHKERRQ(ierr);
  bs2  = bs*bs;

  MPI_Comm_size(comm,&size);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_WRONG,0,"view must have one processor");
  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,header,4,PETSC_INT); CHKERRQ(ierr);
  if (header[0] != MAT_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,0,"not Mat object");
  M = header[1]; N = header[2]; nz = header[3];

  if (header[3] < 0) {
    SETERRQ(PETSC_ERR_FILE_UNEXPECTED,1,"Matrix stored in special format, cannot load as SeqBAIJ");
  }

  if (M != N) SETERRQ(PETSC_ERR_SUP,0,"Can only do square matrices");

  /* 
     This code adds extra rows to make sure the number of rows is 
    divisible by the blocksize
  */
  mbs        = M/bs;
  extra_rows = bs - M + bs*(mbs);
  if (extra_rows == bs) extra_rows = 0;
  else                  mbs++;
  if (extra_rows) {
    PLogInfo(0,"MatLoad_SeqBAIJ:Padding loaded matrix to match blocksize\n");
  }

  /* read in row lengths */
  rowlengths = (int*) PetscMalloc((M+extra_rows)*sizeof(int));CHKPTRQ(rowlengths);
  ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT); CHKERRQ(ierr);
  for ( i=0; i<extra_rows; i++ ) rowlengths[M+i] = 1;

  /* read in column indices */
  jj = (int*) PetscMalloc( (nz+extra_rows)*sizeof(int) ); CHKPTRQ(jj);
  ierr = PetscBinaryRead(fd,jj,nz,PETSC_INT); CHKERRQ(ierr);
  for ( i=0; i<extra_rows; i++ ) jj[nz+i] = M+i;

  /* loop over row lengths determining block row lengths */
  browlengths = (int *) PetscMalloc(mbs*sizeof(int));CHKPTRQ(browlengths);
  PetscMemzero(browlengths,mbs*sizeof(int));
  mask   = (int *) PetscMalloc( 2*mbs*sizeof(int) ); CHKPTRQ(mask);
  PetscMemzero(mask,mbs*sizeof(int));
  masked = mask + mbs;
  rowcount = 0; nzcount = 0;
  for ( i=0; i<mbs; i++ ) {
    nmask = 0;
    for ( j=0; j<bs; j++ ) {
      kmax = rowlengths[rowcount];
      for ( k=0; k<kmax; k++ ) {
        tmp = jj[nzcount++]/bs;
        if (!mask[tmp]) {masked[nmask++] = tmp; mask[tmp] = 1;}
      }
      rowcount++;
    }
    browlengths[i] += nmask;
    /* zero out the mask elements we set */
    for ( j=0; j<nmask; j++ ) mask[masked[j]] = 0;
  }

  /* create our matrix */
  ierr = MatCreateSeqBAIJ(comm,bs,M+extra_rows,N+extra_rows,0,browlengths,A);CHKERRQ(ierr);
  B = *A;
  a = (Mat_SeqBAIJ *) B->data;

  /* set matrix "i" values */
  a->i[0] = 0;
  for ( i=1; i<= mbs; i++ ) {
    a->i[i]      = a->i[i-1] + browlengths[i-1];
    a->ilen[i-1] = browlengths[i-1];
  }
  a->nz         = 0;
  for ( i=0; i<mbs; i++ ) a->nz += browlengths[i];

  /* read in nonzero values */
  aa = (Scalar *) PetscMalloc((nz+extra_rows)*sizeof(Scalar));CHKPTRQ(aa);
  ierr = PetscBinaryRead(fd,aa,nz,PETSC_SCALAR); CHKERRQ(ierr);
  for ( i=0; i<extra_rows; i++ ) aa[nz+i] = 1.0;

  /* set "a" and "j" values into matrix */
  nzcount = 0; jcount = 0;
  for ( i=0; i<mbs; i++ ) {
    nzcountb = nzcount;
    nmask    = 0;
    for ( j=0; j<bs; j++ ) {
      kmax = rowlengths[i*bs+j];
      for ( k=0; k<kmax; k++ ) {
        tmp = jj[nzcount++]/bs;
	if (!mask[tmp]) { masked[nmask++] = tmp; mask[tmp] = 1;}
      }
      rowcount++;
    }
    /* sort the masked values */
    PetscSortInt(nmask,masked);

    /* set "j" values into matrix */
    maskcount = 1;
    for ( j=0; j<nmask; j++ ) {
      a->j[jcount++]  = masked[j];
      mask[masked[j]] = maskcount++; 
    }
    /* set "a" values into matrix */
    ishift = bs2*a->i[i];
    for ( j=0; j<bs; j++ ) {
      kmax = rowlengths[i*bs+j];
      for ( k=0; k<kmax; k++ ) {
        tmp    = jj[nzcountb]/bs ;
        block  = mask[tmp] - 1;
        point  = jj[nzcountb] - bs*tmp;
        idx    = ishift + bs2*block + j + bs*point;
        a->a[idx] = aa[nzcountb++];
      }
    }
    /* zero out the mask elements we set */
    for ( j=0; j<nmask; j++ ) mask[masked[j]] = 0;
  }
  if (jcount != a->nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,0,"Bad binary matrix");

  PetscFree(rowlengths);   
  PetscFree(browlengths);
  PetscFree(aa);
  PetscFree(jj);
  PetscFree(mask);

  B->assembled = PETSC_TRUE;

  ierr = MatView_Private(B); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



