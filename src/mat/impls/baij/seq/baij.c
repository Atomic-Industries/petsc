#ifndef lint
static char vcid[] = "$Id: baij.c,v 1.79 1996/12/17 23:57:30 balay Exp balay $";
#endif

/*
    Defines the basic matrix operations for the BAIJ (compressed row)
  matrix storage format.
*/
#include "src/mat/impls/baij/seq/baij.h"
#include "src/vec/vecimpl.h"
#include "src/inline/spops.h"
#include "petsc.h"


/*
     Adds diagonal pointers to sparse matrix structure.
*/

#undef __FUNCTION__  
#define __FUNCTION__ "MatMarkDiag_SeqBAIJ"
int MatMarkDiag_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data; 
  int         i,j, *diag, m = a->mbs;

  diag = (int *) PetscMalloc( (m+1)*sizeof(int)); CHKPTRQ(diag);
  PLogObjectMemory(A,(m+1)*sizeof(int));
  for ( i=0; i<m; i++ ) {
    for ( j=a->i[i]; j<a->i[i+1]; j++ ) {
      if (a->j[j] == i) {
        diag[i] = j;
        break;
      }
    }
  }
  a->diag = diag;
  return 0;
}

#include "draw.h"
#include "pinclude/pviewer.h"
#include "sys.h"

extern int MatToSymmetricIJ_SeqAIJ(int,int*,int*,int,int,int**,int**);

#undef __FUNCTION__  
#define __FUNCTION__ "MatGetRowIJ_SeqBAIJ"
static int MatGetRowIJ_SeqBAIJ(Mat A,int oshift,PetscTruth symmetric,int *nn,int **ia,int **ja,
                            PetscTruth *done)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int         ierr, n = a->mbs,i;

  *nn = n;
  if (!ia) return 0;
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

  return 0; 
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatRestoreRowIJ_SeqBAIJ"
static int MatRestoreRowIJ_SeqBAIJ(Mat A,int oshift,PetscTruth symmetric,int *nn,int **ia,int **ja,
                                PetscTruth *done)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int         i,n = a->mbs;

  if (!ia) return 0;
  if (symmetric) {
    PetscFree(*ia);
    PetscFree(*ja);
  } else if (oshift == 1) {
    int nz = a->i[n]; 
    for ( i=0; i<nz; i++ ) a->j[i]--;
    for ( i=0; i<n+1; i++ ) a->i[i]--;
  }
  return 0; 
}


#undef __FUNCTION__  
#define __FUNCTION__ "MatView_SeqBAIJ_Binary"
static int MatView_SeqBAIJ_Binary(Mat A,Viewer viewer)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int         i, fd, *col_lens, ierr, bs = a->bs,count,*jj,j,k,l,bs2=a->bs2;
  Scalar      *aa;

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
  ierr = PetscBinaryWrite(fd,col_lens,4+a->m,BINARY_INT,1); CHKERRQ(ierr);
  PetscFree(col_lens);

  /* store column indices (zero start index) */
  jj = (int *) PetscMalloc( a->nz*bs2*sizeof(int) ); CHKPTRQ(jj);
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
  ierr = PetscBinaryWrite(fd,jj,bs2*a->nz,BINARY_INT,0); CHKERRQ(ierr);
  PetscFree(jj);

  /* store nonzero values */
  aa = (Scalar *) PetscMalloc(a->nz*bs2*sizeof(Scalar)); CHKPTRQ(aa);
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
  ierr = PetscBinaryWrite(fd,aa,bs2*a->nz,BINARY_SCALAR,0); CHKERRQ(ierr);
  PetscFree(aa);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatView_SeqBAIJ_ASCII"
static int MatView_SeqBAIJ_ASCII(Mat A,Viewer viewer)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int         ierr, i,j,format,bs = a->bs,k,l,bs2=a->bs2;
  FILE        *fd;
  char        *outputname;

  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  ierr = ViewerFileGetOutputname_Private(viewer,&outputname);CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format);
  if (format == VIEWER_FORMAT_ASCII_INFO || format == VIEWER_FORMAT_ASCII_INFO_LONG) {
    fprintf(fd,"  block size is %d\n",bs);
  } 
  else if (format == VIEWER_FORMAT_ASCII_MATLAB) {
    SETERRQ(1,"Matlab format not supported");
  } 
  else if (format == VIEWER_FORMAT_ASCII_COMMON) {
    for ( i=0; i<a->mbs; i++ ) {
      for ( j=0; j<bs; j++ ) {
        fprintf(fd,"row %d:",i*bs+j);
        for ( k=a->i[i]; k<a->i[i+1]; k++ ) {
          for ( l=0; l<bs; l++ ) {
#if defined(PETSC_COMPLEX)
          if (imag(a->a[bs2*k + l*bs + j]) != 0.0 && real(a->a[bs2*k + l*bs + j]) != 0.0)
            fprintf(fd," %d %g + %g i",bs*a->j[k]+l,
              real(a->a[bs2*k + l*bs + j]),imag(a->a[bs2*k + l*bs + j]));
          else if (real(a->a[bs2*k + l*bs + j]) != 0.0)
            fprintf(fd," %d %g ",bs*a->j[k]+l,real(a->a[bs2*k + l*bs + j]));
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
#if defined(PETSC_COMPLEX)
          if (imag(a->a[bs2*k + l*bs + j]) != 0.0) {
            fprintf(fd," %d %g + %g i",bs*a->j[k]+l,
              real(a->a[bs2*k + l*bs + j]),imag(a->a[bs2*k + l*bs + j]));
          }
          else {
            fprintf(fd," %d %g ",bs*a->j[k]+l,real(a->a[bs2*k + l*bs + j]));
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
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatView_SeqBAIJ_Draw"
static int MatView_SeqBAIJ_Draw(Mat A,Viewer viewer)
{
  Mat_SeqBAIJ  *a=(Mat_SeqBAIJ *) A->data;
  int          row,ierr,i,j,k,l,mbs=a->mbs,pause,color,bs=a->bs,bs2=a->bs2;
  double       xl,yl,xr,yr,w,h,xc,yc,scale = 1.0,x_l,x_r,y_l,y_r;
  Scalar       *aa;
  Draw         draw;
  DrawButton   button;
  PetscTruth   isnull;

  ViewerDrawGetDraw(viewer,&draw);
  ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) return 0;

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
          DrawRectangle(draw,x_l+k,y_l+l,x_r+k,y_r+l,color,color,color,color);
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
          DrawRectangle(draw,x_l+k,y_l+l,x_r+k,y_r+l,color,color,color,color);
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
          DrawRectangle(draw,x_l+k,y_l+l,x_r+k,y_r+l,color,color,color,color);
        }
      }
    } 
  }

  DrawFlush(draw); 
  DrawGetPause(draw,&pause);
  if (pause >= 0) { PetscSleep(pause); return 0;}

  /* allow the matrix to zoom or shrink */
  ierr = DrawGetMouseButton(draw,&button,&xc,&yc,0,0); 
  while (button != BUTTON_RIGHT) {
    DrawClear(draw);
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
            DrawRectangle(draw,x_l+k,y_l+l,x_r+k,y_r+l,color,color,color,color);
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
          DrawRectangle(draw,x_l+k,y_l+l,x_r+k,y_r+l,color,color,color,color);
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
            DrawRectangle(draw,x_l+k,y_l+l,x_r+k,y_r+l,color,color,color,color);
          }
        }
      } 
    }
    ierr = DrawGetMouseButton(draw,&button,&xc,&yc,0,0); 
  }
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatView_SeqBAIJ"
static int MatView_SeqBAIJ(PetscObject obj,Viewer viewer)
{
  Mat         A = (Mat) obj;
  ViewerType  vtype;
  int         ierr;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == MATLAB_VIEWER) {
    SETERRQ(1,"Matlab viewer not supported");
  }
  else if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER){
    return MatView_SeqBAIJ_ASCII(A,viewer);
  }
  else if (vtype == BINARY_FILE_VIEWER) {
    return MatView_SeqBAIJ_Binary(A,viewer);
  }
  else if (vtype == DRAW_VIEWER) {
    return MatView_SeqBAIJ_Draw(A,viewer);
  }
  return 0;
}

#define CHUNKSIZE  10

#undef __FUNCTION__  
#define __FUNCTION__ "MatSetValues_SeqBAIJ"
int MatSetValues_SeqBAIJ(Mat A,int m,int *im,int n,int *in,Scalar *v,InsertMode is)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int         *rp,k,low,high,t,ii,row,nrow,i,col,l,rmax,N,sorted=a->sorted;
  int         *imax=a->imax,*ai=a->i,*ailen=a->ilen,roworiented=a->roworiented;
  int         *aj=a->j,nonew=a->nonew,bs=a->bs,brow,bcol;
  int          ridx,cidx,bs2=a->bs2;
  Scalar      *ap,value,*aa=a->a,*bap;

  for ( k=0; k<m; k++ ) { /* loop over added rows */
    row  = im[k]; brow = row/bs;  
#if defined(PETSC_BOPT_g)  
    if (row < 0) SETERRQ(1,"Negative row");
    if (row >= a->m) SETERRQ(1,"Row too large");
#endif
    rp   = aj + ai[brow]; 
    ap   = aa + bs2*ai[brow];
    rmax = imax[brow]; 
    nrow = ailen[brow]; 
    low  = 0;
    for ( l=0; l<n; l++ ) { /* loop over added columns */
#if defined(PETSC_BOPT_g)  
      if (in[l] < 0) SETERRQ(1,"Negative column");
      if (in[l] >= a->n) SETERRQ(1,"Column too large");
#endif
      col = in[l]; bcol = col/bs;
      ridx = row % bs; cidx = col % bs;
      if (roworiented) {
        value = *v++; 
      }
      else {
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
          goto noinsert;
        }
      } 
      if (nonew) goto noinsert;
      if (nrow >= rmax) {
        /* there is no extra room in row, therefore enlarge */
        int    new_nz = ai[a->mbs] + CHUNKSIZE,len,*new_i,*new_j;
        Scalar *new_a;

        /* malloc new storage space */
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
      noinsert:;
      low = i;
    }
    ailen[brow] = nrow;
  }
  return 0;
} 

#undef __FUNCTION__  
#define __FUNCTION__ "MatGetSize_SeqBAIJ"
static int MatGetSize_SeqBAIJ(Mat A,int *m,int *n)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  *m = a->m; *n = a->n;
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatGetOwnershipRange_SeqBAIJ"
static int MatGetOwnershipRange_SeqBAIJ(Mat A,int *m,int *n)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  *m = 0; *n = a->m;
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatGetRow_SeqBAIJ"
int MatGetRow_SeqBAIJ(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int          itmp,i,j,k,M,*ai,*aj,bs,bn,bp,*idx_i,bs2;
  Scalar      *aa,*v_i,*aa_i;

  bs  = a->bs;
  ai  = a->i;
  aj  = a->j;
  aa  = a->a;
  bs2 = a->bs2;
  
  if (row < 0 || row >= a->m) SETERRQ(1,"Row out of range");
  
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
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatRestoreRow_SeqBAIJ"
int MatRestoreRow_SeqBAIJ(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  if (idx) {if (*idx) PetscFree(*idx);}
  if (v)   {if (*v)   PetscFree(*v);}
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatTranspose_SeqBAIJ"
static int MatTranspose_SeqBAIJ(Mat A,Mat *B)
{ 
  Mat_SeqBAIJ *a=(Mat_SeqBAIJ *)A->data;
  Mat         C;
  int         i,j,k,ierr,*aj=a->j,*ai=a->i,bs=a->bs,mbs=a->mbs,nbs=a->nbs,len,*col;
  int         *rows,*cols,bs2=a->bs2;
  Scalar      *array=a->a;

  if (B==PETSC_NULL && mbs!=nbs)
    SETERRQ(1,"Square matrix only for in-place");
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
    /* This isn't really an in-place transpose */
    PetscFree(a->a); 
    if (!a->singlemalloc) {PetscFree(a->i); PetscFree(a->j);}
    if (a->diag) PetscFree(a->diag);
    if (a->ilen) PetscFree(a->ilen);
    if (a->imax) PetscFree(a->imax);
    if (a->solve_work) PetscFree(a->solve_work);
    PetscFree(a); 
    PetscMemcpy(A,C,sizeof(struct _Mat)); 
    PetscHeaderDestroy(C);
  }
  return 0;
}


#undef __FUNCTION__  
#define __FUNCTION__ "MatAssemblyEnd_SeqBAIJ"
static int MatAssemblyEnd_SeqBAIJ(Mat A,MatAssemblyType mode)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int        fshift = 0,i,j,*ai = a->i, *aj = a->j, *imax = a->imax;
  int        m = a->m,*ip, N, *ailen = a->ilen;
  int        mbs = a->mbs, bs2 = a->bs2;
  Scalar     *aa = a->a, *ap;

  if (mode == MAT_FLUSH_ASSEMBLY) return 0;

  for ( i=1; i<mbs; i++ ) {
    /* move each row back by the amount of empty slots (fshift) before it*/
    fshift += imax[i-1] - ailen[i-1];
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
  a->reallocs          = 0;
  A->info.nz_unneeded  = (double)fshift*bs2;

  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatZeroEntries_SeqBAIJ"
static int MatZeroEntries_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data; 
  PetscMemzero(a->a,a->bs2*a->i[a->mbs]*sizeof(Scalar));
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatDestroy_SeqBAIJ"
int MatDestroy_SeqBAIJ(PetscObject obj)
{
  Mat         A  = (Mat) obj;
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int         ierr;

#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows=%d, Cols=%d, NZ=%d",a->m,a->n,a->nz);
#endif
  PetscFree(a->a); 
  if (!a->singlemalloc) { PetscFree(a->i); PetscFree(a->j);}
  if (a->diag) PetscFree(a->diag);
  if (a->ilen) PetscFree(a->ilen);
  if (a->imax) PetscFree(a->imax);
  if (a->solve_work) PetscFree(a->solve_work);
  if (a->mult_work) PetscFree(a->mult_work);
  PetscFree(a); 
  if (A->mapping) {
    ierr = ISLocalToGlobalMappingDestroy(A->mapping); CHKERRQ(ierr);
  }
  PLogObjectDestroy(A);
  PetscHeaderDestroy(A);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatSetOption_SeqBAIJ"
static int MatSetOption_SeqBAIJ(Mat A,MatOption op)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  if      (op == MAT_ROW_ORIENTED)              a->roworiented = 1;
  else if (op == MAT_COLUMN_ORIENTED)           a->roworiented = 0;
  else if (op == MAT_COLUMNS_SORTED)            a->sorted      = 1;
  else if (op == MAT_COLUMNS_UNSORTED)          a->sorted      = 0;
  else if (op == MAT_NO_NEW_NONZERO_LOCATIONS)  a->nonew       = 1;
  else if (op == MAT_YES_NEW_NONZERO_LOCATIONS) a->nonew       = 0;
  else if (op == MAT_ROWS_SORTED || 
           op == MAT_ROWS_UNSORTED ||
           op == MAT_SYMMETRIC ||
           op == MAT_STRUCTURALLY_SYMMETRIC ||
           op == MAT_YES_NEW_DIAGONALS ||
           op == MAT_IGNORE_OFF_PROCESSOR_ENTRIES)
    PLogInfo(A,"Info:MatSetOption_SeqBAIJ:Option ignored\n");
  else if (op == MAT_NO_NEW_DIAGONALS)
    {SETERRQ(PETSC_ERR_SUP,"MAT_NO_NEW_DIAGONALS");}
  else 
    {SETERRQ(PETSC_ERR_SUP,"unknown option");}
  return 0;
}


/* -------------------------------------------------------*/
/* Should check that shapes of vectors and matrices match */
/* -------------------------------------------------------*/
#include "pinclude/plapack.h"

#undef __FUNCTION__  
#define __FUNCTION__ "MatMult_SeqBAIJ_1"
static int MatMult_SeqBAIJ_1(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data;
  register Scalar *x,*z,*v,sum;
  int             mbs=a->mbs,i,*idx,*ii,n;

  VecGetArray_Fast(xx,x);
  VecGetArray_Fast(zz,z);

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

  for ( i=0; i<mbs; i++ ) {
    n    = ii[1] - ii[0]; ii++;
    sum  = 0.0;
    while (n--) sum += *v++ * x[*idx++];
    z[i] = sum;
  }
  VecRestoreArray_Fast(xx,x);
  VecRestoreArray_Fast(zz,z);
  PLogFlops(2*a->nz - a->m);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatMult_SeqBAIJ_2"
static int MatMult_SeqBAIJ_2(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data;
  register Scalar *x,*z,*v,*xb,sum1,sum2;
  register Scalar x1,x2;
  int             mbs=a->mbs,i,*idx,*ii;
  int             j,n;

  VecGetArray_Fast(xx,x);
  VecGetArray_Fast(zz,z);

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

  for ( i=0; i<mbs; i++ ) {
    n  = ii[1] - ii[0]; ii++; 
    sum1 = 0.0; sum2 = 0.0;
    for ( j=0; j<n; j++ ) {
      xb = x + 2*(*idx++); x1 = xb[0]; x2 = xb[1];
      sum1 += v[0]*x1 + v[2]*x2;
      sum2 += v[1]*x1 + v[3]*x2;
      v += 4;
    }
    z[0] = sum1; z[1] = sum2;
    z += 2;
  }
  VecRestoreArray_Fast(xx,x);
  VecRestoreArray_Fast(zz,z);
  PLogFlops(4*a->nz - a->m);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatMult_SeqBAIJ_3"
static int MatMult_SeqBAIJ_3(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data;
  register Scalar *x,*z,*v,*xb,sum1,sum2,sum3,x1,x2,x3;
  int             mbs=a->mbs,i,*idx,*ii,j,n;

  VecGetArray_Fast(xx,x);
  VecGetArray_Fast(zz,z);

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

  for ( i=0; i<mbs; i++ ) {
    n  = ii[1] - ii[0]; ii++; 
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0;
    for ( j=0; j<n; j++ ) {
      xb = x + 3*(*idx++); x1 = xb[0]; x2 = xb[1]; x3 = xb[2];
      sum1 += v[0]*x1 + v[3]*x2 + v[6]*x3;
      sum2 += v[1]*x1 + v[4]*x2 + v[7]*x3;
      sum3 += v[2]*x1 + v[5]*x2 + v[8]*x3;
      v += 9;
    }
    z[0] = sum1; z[1] = sum2; z[2] = sum3;
    z += 3;
  }
  VecRestoreArray_Fast(xx,x);
  VecRestoreArray_Fast(zz,z);
  PLogFlops(18*a->nz - a->m);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatMult_SeqBAIJ_4"
static int MatMult_SeqBAIJ_4(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data;
  register Scalar *x,*z,*v,*xb,sum1,sum2,sum3,sum4;
  register Scalar x1,x2,x3,x4;
  int             mbs=a->mbs,i,*idx,*ii;
  int             j,n;

  VecGetArray_Fast(xx,x);
  VecGetArray_Fast(zz,z);

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

  for ( i=0; i<mbs; i++ ) {
    n  = ii[1] - ii[0]; ii++; 
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; sum4 = 0.0;
    for ( j=0; j<n; j++ ) {
      xb = x + 4*(*idx++);
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3];
      sum1 += v[0]*x1 + v[4]*x2 + v[8]*x3   + v[12]*x4;
      sum2 += v[1]*x1 + v[5]*x2 + v[9]*x3   + v[13]*x4;
      sum3 += v[2]*x1 + v[6]*x2 + v[10]*x3  + v[14]*x4;
      sum4 += v[3]*x1 + v[7]*x2 + v[11]*x3  + v[15]*x4;
      v += 16;
    }
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4;
    z += 4;
  }
  VecRestoreArray_Fast(xx,x);
  VecRestoreArray_Fast(zz,z);
  PLogFlops(32*a->nz - a->m);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatMult_SeqBAIJ_5"
static int MatMult_SeqBAIJ_5(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data;
  register Scalar *x,*z,*v,*xb,sum1,sum2,sum3,sum4,sum5;
  register Scalar x1,x2,x3,x4,x5;
  int             mbs=a->mbs,i,*idx,*ii,j,n;

  VecGetArray_Fast(xx,x);
  VecGetArray_Fast(zz,z);

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

  for ( i=0; i<mbs; i++ ) {
    n  = ii[1] - ii[0]; ii++; 
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; sum4 = 0.0; sum5 = 0.0;
    for ( j=0; j<n; j++ ) {
      xb = x + 5*(*idx++);
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5 = xb[4];
      sum1 += v[0]*x1 + v[5]*x2 + v[10]*x3  + v[15]*x4 + v[20]*x5;
      sum2 += v[1]*x1 + v[6]*x2 + v[11]*x3  + v[16]*x4 + v[21]*x5;
      sum3 += v[2]*x1 + v[7]*x2 + v[12]*x3  + v[17]*x4 + v[22]*x5;
      sum4 += v[3]*x1 + v[8]*x2 + v[13]*x3  + v[18]*x4 + v[23]*x5;
      sum5 += v[4]*x1 + v[9]*x2 + v[14]*x3  + v[19]*x4 + v[24]*x5;
      v += 25;
    }
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4; z[4] = sum5;
    z += 5;
  }
  VecRestoreArray_Fast(xx,x);
  VecRestoreArray_Fast(zz,z);
  PLogFlops(50*a->nz - a->m);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatMult_SeqBAIJ_N"
static int MatMult_SeqBAIJ_N(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data;
  register Scalar *x,*z,*v,*xb;
  int             mbs=a->mbs,i,*idx,*ii,bs=a->bs,j,n,bs2=a->bs2;
  int             _One = 1,ncols,k;
  Scalar          _DOne = 1.0,*work,*workt,_DZero = 0.0;

  VecGetArray_Fast(xx,x);
  VecGetArray_Fast(zz,z);

  idx   = a->j;
  v     = a->a;
  ii    = a->i;


  if (!a->mult_work) {
    k = PetscMax(a->m,a->n);
    a->mult_work = (Scalar *) PetscMalloc((k+1)*sizeof(Scalar));CHKPTRQ(a->mult_work);
  }
  work = a->mult_work;
  for ( i=0; i<mbs; i++ ) {
    n     = ii[1] - ii[0]; ii++;
    ncols = n*bs;
    workt = work;
    for ( j=0; j<n; j++ ) {
      xb = x + bs*(*idx++);
      for ( k=0; k<bs; k++ ) workt[k] = xb[k];
      workt += bs;
    }
    LAgemv_("N",&bs,&ncols,&_DOne,v,&bs,work,&_One,&_DZero,z,&_One);
    v += n*bs2;
    z += bs;
  }
  VecRestoreArray_Fast(xx,x);
  VecRestoreArray_Fast(zz,z);
  PLogFlops(2*a->nz*bs2 - a->m);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatMultAdd_SeqBAIJ_1"
static int MatMultAdd_SeqBAIJ_1(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data;
  register Scalar *x,*y,*z,*v,sum;
  int             mbs=a->mbs,i,*idx,*ii,n;

  VecGetArray_Fast(xx,x);
  VecGetArray_Fast(yy,y);
  VecGetArray_Fast(zz,z);

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

  for ( i=0; i<mbs; i++ ) {
    n    = ii[1] - ii[0]; ii++;
    sum  = y[i];
    while (n--) sum += *v++ * x[*idx++];
    z[i] = sum;
  }
  VecRestoreArray_Fast(xx,x);
  VecRestoreArray_Fast(yy,y);
  VecRestoreArray_Fast(zz,z);
  PLogFlops(2*a->nz);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatMultAdd_SeqBAIJ_2"
static int MatMultAdd_SeqBAIJ_2(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data;
  register Scalar *x,*y,*z,*v,*xb,sum1,sum2;
  register Scalar x1,x2;
  int             mbs=a->mbs,i,*idx,*ii;
  int             j,n;

  VecGetArray_Fast(xx,x);
  VecGetArray_Fast(yy,y);
  VecGetArray_Fast(zz,z);

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

  for ( i=0; i<mbs; i++ ) {
    n  = ii[1] - ii[0]; ii++; 
    sum1 = y[0]; sum2 = y[1];
    for ( j=0; j<n; j++ ) {
      xb = x + 2*(*idx++); x1 = xb[0]; x2 = xb[1];
      sum1 += v[0]*x1 + v[2]*x2;
      sum2 += v[1]*x1 + v[3]*x2;
      v += 4;
    }
    z[0] = sum1; z[1] = sum2;
    z += 2; y += 2;
  }
  VecRestoreArray_Fast(xx,x);
  VecRestoreArray_Fast(yy,y);
  VecRestoreArray_Fast(zz,z);
  PLogFlops(4*a->nz);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatMultAdd_SeqBAIJ_3"
static int MatMultAdd_SeqBAIJ_3(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data;
  register Scalar *x,*y,*z,*v,*xb,sum1,sum2,sum3,x1,x2,x3;
  int             mbs=a->mbs,i,*idx,*ii,j,n;

  VecGetArray_Fast(xx,x);
  VecGetArray_Fast(yy,y);
  VecGetArray_Fast(zz,z);

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

  for ( i=0; i<mbs; i++ ) {
    n  = ii[1] - ii[0]; ii++; 
    sum1 = y[0]; sum2 = y[1]; sum3 = y[2];
    for ( j=0; j<n; j++ ) {
      xb = x + 3*(*idx++); x1 = xb[0]; x2 = xb[1]; x3 = xb[2];
      sum1 += v[0]*x1 + v[3]*x2 + v[6]*x3;
      sum2 += v[1]*x1 + v[4]*x2 + v[7]*x3;
      sum3 += v[2]*x1 + v[5]*x2 + v[8]*x3;
      v += 9;
    }
    z[0] = sum1; z[1] = sum2; z[2] = sum3;
    z += 3; y += 3;
  }
  VecRestoreArray_Fast(xx,x);
  VecRestoreArray_Fast(yy,y);
  VecRestoreArray_Fast(zz,z);
  PLogFlops(18*a->nz);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatMultAdd_SeqBAIJ_4"
static int MatMultAdd_SeqBAIJ_4(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data;
  register Scalar *x,*y,*z,*v,*xb,sum1,sum2,sum3,sum4;
  register Scalar x1,x2,x3,x4;
  int             mbs=a->mbs,i,*idx,*ii;
  int             j,n;

  VecGetArray_Fast(xx,x);
  VecGetArray_Fast(yy,y);
  VecGetArray_Fast(zz,z);

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

  for ( i=0; i<mbs; i++ ) {
    n  = ii[1] - ii[0]; ii++; 
    sum1 = y[0]; sum2 = y[1]; sum3 = y[2]; sum4 = y[3];
    for ( j=0; j<n; j++ ) {
      xb = x + 4*(*idx++);
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3];
      sum1 += v[0]*x1 + v[4]*x2 + v[8]*x3   + v[12]*x4;
      sum2 += v[1]*x1 + v[5]*x2 + v[9]*x3   + v[13]*x4;
      sum3 += v[2]*x1 + v[6]*x2 + v[10]*x3  + v[14]*x4;
      sum4 += v[3]*x1 + v[7]*x2 + v[11]*x3  + v[15]*x4;
      v += 16;
    }
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4;
    z += 4; y += 4;
  }
  VecRestoreArray_Fast(xx,x);
  VecRestoreArray_Fast(yy,y);
  VecRestoreArray_Fast(zz,z);
  PLogFlops(32*a->nz);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatMultAdd_SeqBAIJ_5"
static int MatMultAdd_SeqBAIJ_5(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data;
  register Scalar *x,*y,*z,*v,*xb,sum1,sum2,sum3,sum4,sum5;
  register Scalar x1,x2,x3,x4,x5;
  int             mbs=a->mbs,i,*idx,*ii,j,n;

  VecGetArray_Fast(xx,x);
  VecGetArray_Fast(yy,y);
  VecGetArray_Fast(zz,z);

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

  for ( i=0; i<mbs; i++ ) {
    n  = ii[1] - ii[0]; ii++; 
    sum1 = y[0]; sum2 = y[1]; sum3 = y[2]; sum4 = y[3]; sum5 = y[4];
    for ( j=0; j<n; j++ ) {
      xb = x + 5*(*idx++);
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5 = xb[4];
      sum1 += v[0]*x1 + v[5]*x2 + v[10]*x3  + v[15]*x4 + v[20]*x5;
      sum2 += v[1]*x1 + v[6]*x2 + v[11]*x3  + v[16]*x4 + v[21]*x5;
      sum3 += v[2]*x1 + v[7]*x2 + v[12]*x3  + v[17]*x4 + v[22]*x5;
      sum4 += v[3]*x1 + v[8]*x2 + v[13]*x3  + v[18]*x4 + v[23]*x5;
      sum5 += v[4]*x1 + v[9]*x2 + v[14]*x3  + v[19]*x4 + v[24]*x5;
      v += 25;
    }
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4; z[4] = sum5;
    z += 5; y += 5;
  }
  VecRestoreArray_Fast(xx,x);
  VecRestoreArray_Fast(yy,y);
  VecRestoreArray_Fast(zz,z);
  PLogFlops(50*a->nz);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatMultAdd_SeqBAIJ_N"
static int MatMultAdd_SeqBAIJ_N(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data;
  register Scalar *x,*z,*v,*xb;
  int             mbs=a->mbs,i,*idx,*ii,bs=a->bs,j,n,bs2=a->bs2,ierr;
  int             _One = 1,ncols,k; Scalar _DOne = 1.0, *work,*workt;

  if ( xx != yy) { ierr = VecCopy(yy,zz); CHKERRQ(ierr); }

  VecGetArray_Fast(xx,x);
  VecGetArray_Fast(zz,z);
 
  idx   = a->j;
  v     = a->a;
  ii    = a->i;


  if (!a->mult_work) {
    k = PetscMax(a->m,a->n);
    a->mult_work = (Scalar *) PetscMalloc(k*sizeof(Scalar));CHKPTRQ(a->mult_work);
  }
  work = a->mult_work;
  for ( i=0; i<mbs; i++ ) {
    n     = ii[1] - ii[0]; ii++;
    ncols = n*bs;
    workt = work;
    for ( j=0; j<n; j++ ) {
      xb = x + bs*(*idx++);
      for ( k=0; k<bs; k++ ) workt[k] = xb[k];
      workt += bs;
    }
    LAgemv_("N",&bs,&ncols,&_DOne,v,&bs,work,&_One,&_DOne,z,&_One);
    v += n*bs2;
    z += bs;
  }
  VecRestoreArray_Fast(xx,x);
  VecRestoreArray_Fast(zz,z);
  PLogFlops(2*a->nz*bs2 );
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatMultTrans_SeqBAIJ"
static int MatMultTrans_SeqBAIJ(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data;
  Scalar          *xg,*zg,*zb;
  register Scalar *x,*z,*v,*xb,x1,x2,x3,x4,x5;
  int             mbs=a->mbs,i,*idx,*ii,*ai=a->i,rval,N=a->n;
  int             bs=a->bs,j,n,bs2=a->bs2,*ib,ierr;


  VecGetArray_Fast(xx,xg); x = xg;
  VecGetArray_Fast(zz,zg); z = zg;
  PetscMemzero(z,N*sizeof(Scalar));

  idx   = a->j;
  v     = a->a;
  ii    = a->i;
  
  switch (bs) {
  case 1:
    for ( i=0; i<mbs; i++ ) {
      n  = ii[1] - ii[0]; ii++;
      xb = x + i; x1 = xb[0];
      ib = idx + ai[i];
      for ( j=0; j<n; j++ ) {
        rval    = ib[j];
        z[rval] += *v++ * x1;
      }
    }
    break;
  case 2:
    for ( i=0; i<mbs; i++ ) {
      n  = ii[1] - ii[0]; ii++; 
      xb = x + 2*i; x1 = xb[0]; x2 = xb[1];
      ib = idx + ai[i];
      for ( j=0; j<n; j++ ) {
        rval      = ib[j]*2;
        z[rval++] += v[0]*x1 + v[1]*x2;
        z[rval++] += v[2]*x1 + v[3]*x2;
        v += 4;
      }
    }
    break;
  case 3:
    for ( i=0; i<mbs; i++ ) {
      n  = ii[1] - ii[0]; ii++; 
      xb = x + 3*i; x1 = xb[0]; x2 = xb[1]; x3 = xb[2];
      ib = idx + ai[i];
      for ( j=0; j<n; j++ ) {
        rval      = ib[j]*3;
        z[rval++] += v[0]*x1 + v[1]*x2 + v[2]*x3;
        z[rval++] += v[3]*x1 + v[4]*x2 + v[5]*x3;
        z[rval++] += v[6]*x1 + v[7]*x2 + v[8]*x3;
        v += 9;
      }
    }
    break;
  case 4:
    for ( i=0; i<mbs; i++ ) {
      n  = ii[1] - ii[0]; ii++; 
      xb = x + 4*i; x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3];
      ib = idx + ai[i];
      for ( j=0; j<n; j++ ) {
        rval      = ib[j]*4;
        z[rval++] +=  v[0]*x1 +  v[1]*x2 +  v[2]*x3 +  v[3]*x4;
        z[rval++] +=  v[4]*x1 +  v[5]*x2 +  v[6]*x3 +  v[7]*x4;
        z[rval++] +=  v[8]*x1 +  v[9]*x2 + v[10]*x3 + v[11]*x4;
        z[rval++] += v[12]*x1 + v[13]*x2 + v[14]*x3 + v[15]*x4;
        v += 16;
      }
    }
    break;
  case 5:
    for ( i=0; i<mbs; i++ ) {
      n  = ii[1] - ii[0]; ii++; 
      xb = x + 5*i; x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; 
      x4 = xb[3];   x5 = xb[4];
      ib = idx + ai[i];
      for ( j=0; j<n; j++ ) {
        rval      = ib[j]*5;
        z[rval++] +=  v[0]*x1 +  v[1]*x2 +  v[2]*x3 +  v[3]*x4 +  v[4]*x5;
        z[rval++] +=  v[5]*x1 +  v[6]*x2 +  v[7]*x3 +  v[8]*x4 +  v[9]*x5;
        z[rval++] += v[10]*x1 + v[11]*x2 + v[12]*x3 + v[13]*x4 + v[14]*x5;
        z[rval++] += v[15]*x1 + v[16]*x2 + v[17]*x3 + v[18]*x4 + v[19]*x5;
        z[rval++] += v[20]*x1 + v[21]*x2 + v[22]*x3 + v[23]*x4 + v[24]*x5;
        v += 25;
      }
    }
    break;
      /* block sizes larger then 5 by 5 are handled by BLAS */
    default: {
      int  _One = 1,ncols,k; Scalar _DOne = 1.0, *work,*workt;
      if (!a->mult_work) {
        k = PetscMax(a->m,a->n);
        a->mult_work = (Scalar *) PetscMalloc(k*sizeof(Scalar));
        CHKPTRQ(a->mult_work);
      }
      work = a->mult_work;
      for ( i=0; i<mbs; i++ ) {
        n     = ii[1] - ii[0]; ii++;
        ncols = n*bs;
        PetscMemzero(work,ncols*sizeof(Scalar));
        LAgemv_("T",&bs,&ncols,&_DOne,v,&bs,x,&_One,&_DOne,work,&_One);
        v += n*bs2;
        x += bs;
        workt = work;
        for ( j=0; j<n; j++ ) {
          zb = z + bs*(*idx++);
          for ( k=0; k<bs; k++ ) zb[k] += workt[k] ;
          workt += bs;
        }
      }
    }
  }
  ierr = VecRestoreArray(xx,&xg); CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zg); CHKERRQ(ierr);
  PLogFlops(2*a->nz*a->bs2 - a->n);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatMultTransAdd_SeqBAIJ"
static int MatMultTransAdd_SeqBAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data;
  Scalar          *xg,*zg,*zb;
  register Scalar *x,*z,*v,*xb,x1,x2,x3,x4,x5;
  int             mbs=a->mbs,i,*idx,*ii,*ai=a->i,rval,N=a->n;
  int             bs=a->bs,j,n,bs2=a->bs2,*ib,ierr;



  VecGetArray_Fast(xx,xg); x = xg;
  VecGetArray_Fast(zz,zg); z = zg;

  if ( yy != zz ) { ierr = VecCopy(yy,zz); CHKERRQ(ierr); }
  else PetscMemzero(z,N*sizeof(Scalar));


  idx   = a->j;
  v     = a->a;
  ii    = a->i;
  
  switch (bs) {
  case 1:
    for ( i=0; i<mbs; i++ ) {
      n  = ii[1] - ii[0]; ii++;
      xb = x + i; x1 = xb[0];
      ib = idx + ai[i];
      for ( j=0; j<n; j++ ) {
        rval    = ib[j];
        z[rval] += *v++ * x1;
      }
    }
    break;
  case 2:
    for ( i=0; i<mbs; i++ ) {
      n  = ii[1] - ii[0]; ii++; 
      xb = x + 2*i; x1 = xb[0]; x2 = xb[1];
      ib = idx + ai[i];
      for ( j=0; j<n; j++ ) {
        rval      = ib[j]*2;
        z[rval++] += v[0]*x1 + v[1]*x2;
        z[rval++] += v[2]*x1 + v[3]*x2;
        v += 4;
      }
    }
    break;
  case 3:
    for ( i=0; i<mbs; i++ ) {
      n  = ii[1] - ii[0]; ii++; 
      xb = x + 3*i; x1 = xb[0]; x2 = xb[1]; x3 = xb[2];
      ib = idx + ai[i];
      for ( j=0; j<n; j++ ) {
        rval      = ib[j]*3;
        z[rval++] += v[0]*x1 + v[1]*x2 + v[2]*x3;
        z[rval++] += v[3]*x1 + v[4]*x2 + v[5]*x3;
        z[rval++] += v[6]*x1 + v[7]*x2 + v[8]*x3;
        v += 9;
      }
    }
    break;
  case 4:
    for ( i=0; i<mbs; i++ ) {
      n  = ii[1] - ii[0]; ii++; 
      xb = x + 4*i; x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3];
      ib = idx + ai[i];
      for ( j=0; j<n; j++ ) {
        rval      = ib[j]*4;
        z[rval++] +=  v[0]*x1 +  v[1]*x2 +  v[2]*x3 +  v[3]*x4;
        z[rval++] +=  v[4]*x1 +  v[5]*x2 +  v[6]*x3 +  v[7]*x4;
        z[rval++] +=  v[8]*x1 +  v[9]*x2 + v[10]*x3 + v[11]*x4;
        z[rval++] += v[12]*x1 + v[13]*x2 + v[14]*x3 + v[15]*x4;
        v += 16;
      }
    }
    break;
  case 5:
    for ( i=0; i<mbs; i++ ) {
      n  = ii[1] - ii[0]; ii++; 
      xb = x + 5*i; x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; 
      x4 = xb[3];   x5 = xb[4];
      ib = idx + ai[i];
      for ( j=0; j<n; j++ ) {
        rval      = ib[j]*5;
        z[rval++] +=  v[0]*x1 +  v[1]*x2 +  v[2]*x3 +  v[3]*x4 +  v[4]*x5;
        z[rval++] +=  v[5]*x1 +  v[6]*x2 +  v[7]*x3 +  v[8]*x4 +  v[9]*x5;
        z[rval++] += v[10]*x1 + v[11]*x2 + v[12]*x3 + v[13]*x4 + v[14]*x5;
        z[rval++] += v[15]*x1 + v[16]*x2 + v[17]*x3 + v[18]*x4 + v[19]*x5;
        z[rval++] += v[20]*x1 + v[21]*x2 + v[22]*x3 + v[23]*x4 + v[24]*x5;
        v += 25;
      }
    }
    break;
      /* block sizes larger then 5 by 5 are handled by BLAS */
    default: {
      int  _One = 1,ncols,k; Scalar _DOne = 1.0, *work,*workt;
      if (!a->mult_work) {
        k = PetscMax(a->m,a->n);
        a->mult_work = (Scalar *) PetscMalloc(k*sizeof(Scalar));
        CHKPTRQ(a->mult_work);
      }
      work = a->mult_work;
      for ( i=0; i<mbs; i++ ) {
        n     = ii[1] - ii[0]; ii++;
        ncols = n*bs;
        PetscMemzero(work,ncols*sizeof(Scalar));
        LAgemv_("T",&bs,&ncols,&_DOne,v,&bs,x,&_One,&_DOne,work,&_One);
        v += n*bs2;
        x += bs;
        workt = work;
        for ( j=0; j<n; j++ ) {
          zb = z + bs*(*idx++);
          for ( k=0; k<bs; k++ ) zb[k] += workt[k] ;
          workt += bs;
        }
      }
    }
  }
  ierr = VecRestoreArray(xx,&xg); CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zg); CHKERRQ(ierr);
  PLogFlops(2*a->nz*a->bs2);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatGetInfo_SeqBAIJ"
static int MatGetInfo_SeqBAIJ(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;

  info->rows_global    = (double)a->m;
  info->columns_global = (double)a->n;
  info->rows_local     = (double)a->m;
  info->columns_local  = (double)a->n;
  info->block_size     = a->bs2;
  info->nz_allocated   = a->maxnz;
  info->nz_used        = a->bs2*a->nz;
  info->nz_unneeded    = (double)(info->nz_allocated - info->nz_used);
  /*  if (info->nz_unneeded != A->info.nz_unneeded) 
    printf("space descrepancy: maxnz-nz = %d, nz_unneeded = %d\n",(int)info->nz_unneeded,(int)A->info.nz_unneeded); */
  info->assemblies   = A->num_ass;
  info->mallocs      = a->reallocs;
  info->memory       = A->mem;
  if (A->factor) {
    info->fill_ratio_given  = A->info.fill_ratio_given;
    info->fill_ratio_needed = A->info.fill_ratio_needed;
    info->factor_mallocs    = A->info.factor_mallocs;
  } else {
    info->fill_ratio_given  = 0;
    info->fill_ratio_needed = 0;
    info->factor_mallocs    = 0;
  }
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatEqual_SeqBAIJ"
static int MatEqual_SeqBAIJ(Mat A,Mat B, PetscTruth* flg)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *)A->data, *b = (Mat_SeqBAIJ *)B->data;

  if (B->type !=MATSEQBAIJ)SETERRQ(1,"Matrices must be same type");

  /* If the  matrix/block dimensions are not equal, or no of nonzeros or shift */
  if ((a->m != b->m) || (a->n !=b->n) || (a->bs != b->bs)|| 
      (a->nz != b->nz)) {
    *flg = PETSC_FALSE; return 0; 
  }
  
  /* if the a->i are the same */
  if (PetscMemcmp(a->i,b->i, (a->mbs+1)*sizeof(int))) { 
    *flg = PETSC_FALSE; return 0;
  }
  
  /* if a->j are the same */
  if (PetscMemcmp(a->j,b->j,(a->nz)*sizeof(int))) { 
    *flg = PETSC_FALSE; return 0;
  }
  
  /* if a->a are the same */
  if (PetscMemcmp(a->a, b->a,(a->nz)*(a->bs)*(a->bs)*sizeof(Scalar))) {
    *flg = PETSC_FALSE; return 0;
  }
  *flg = PETSC_TRUE; 
  return 0;
  
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatGetDiagonal_SeqBAIJ"
static int MatGetDiagonal_SeqBAIJ(Mat A,Vec v)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int         i,j,k,n,row,bs,*ai,*aj,ambs,bs2;
  Scalar      *x, zero = 0.0,*aa,*aa_j;

  bs   = a->bs;
  aa   = a->a;
  ai   = a->i;
  aj   = a->j;
  ambs = a->mbs;
  bs2  = a->bs2;

  VecSet(&zero,v);
  VecGetArray_Fast(v,x); VecGetLocalSize_Fast(v,n);
  if (n != a->m) SETERRQ(1,"Nonconforming matrix and vector");
  for ( i=0; i<ambs; i++ ) {
    for ( j=ai[i]; j<ai[i+1]; j++ ) {
      if (aj[j] == i) {
        row  = i*bs;
        aa_j = aa+j*bs2;
        for (k=0; k<bs2; k+=(bs+1),row++) x[row] = aa_j[k];
        break;
      }
    }
  }
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatDiagonalScale_SeqBAIJ"
static int MatDiagonalScale_SeqBAIJ(Mat A,Vec ll,Vec rr)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  Scalar      *l,*r,x,*v,*aa,*li,*ri;
  int         i,j,k,lm,rn,M,m,n,*ai,*aj,mbs,tmp,bs,bs2;

  ai  = a->i;
  aj  = a->j;
  aa  = a->a;
  m   = a->m;
  n   = a->n;
  bs  = a->bs;
  mbs = a->mbs;
  bs2 = a->bs2;
  if (ll) {
    VecGetArray_Fast(ll,l); VecGetLocalSize_Fast(ll,lm);
    if (lm != m) SETERRQ(1,"Left scaling vector wrong length");
    for ( i=0; i<mbs; i++ ) { /* for each block row */
      M  = ai[i+1] - ai[i];
      li = l + i*bs;
      v  = aa + bs2*ai[i];
      for ( j=0; j<M; j++ ) { /* for each block */
        for ( k=0; k<bs2; k++ ) {
          (*v++) *= li[k%bs];
        } 
      }  
    }
  }
  
  if (rr) {
    VecGetArray_Fast(rr,r); VecGetLocalSize_Fast(rr,rn);
    if (rn != n) SETERRQ(1,"Right scaling vector wrong length");
    for ( i=0; i<mbs; i++ ) { /* for each block row */
      M  = ai[i+1] - ai[i];
      v  = aa + bs2*ai[i];
      for ( j=0; j<M; j++ ) { /* for each block */
        ri = r + bs*aj[ai[i]+j];
        for ( k=0; k<bs; k++ ) {
          x = ri[k];
          for ( tmp=0; tmp<bs; tmp++ ) (*v++) *= x;
        } 
      }  
    }
  }
  return 0;
}


extern int MatLUFactorSymbolic_SeqBAIJ(Mat,IS,IS,double,Mat*);
extern int MatLUFactor_SeqBAIJ(Mat,IS,IS,double);
extern int MatIncreaseOverlap_SeqBAIJ(Mat,int,IS*,int);
extern int MatGetSubMatrix_SeqBAIJ(Mat,IS,IS,MatGetSubMatrixCall,Mat*);
extern int MatGetSubMatrices_SeqBAIJ(Mat,int,IS*,IS*,MatGetSubMatrixCall,Mat**);

extern int MatSolve_SeqBAIJ_N(Mat,Vec,Vec);
extern int MatSolve_SeqBAIJ_1(Mat,Vec,Vec);
extern int MatSolve_SeqBAIJ_2(Mat,Vec,Vec);
extern int MatSolve_SeqBAIJ_3(Mat,Vec,Vec);
extern int MatSolve_SeqBAIJ_4(Mat,Vec,Vec);
extern int MatSolve_SeqBAIJ_5(Mat,Vec,Vec);

extern int MatLUFactorNumeric_SeqBAIJ_N(Mat,Mat*);
extern int MatLUFactorNumeric_SeqBAIJ_1(Mat,Mat*);
extern int MatLUFactorNumeric_SeqBAIJ_2(Mat,Mat*);
extern int MatLUFactorNumeric_SeqBAIJ_3(Mat,Mat*);
extern int MatLUFactorNumeric_SeqBAIJ_4(Mat,Mat*);
extern int MatLUFactorNumeric_SeqBAIJ_5(Mat,Mat*);

#undef __FUNCTION__  
#define __FUNCTION__ "MatNorm_SeqBAIJ"
static int MatNorm_SeqBAIJ(Mat A,NormType type,double *norm)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  Scalar      *v = a->a;
  double      sum = 0.0;
  int         i,nz=a->nz,bs2=a->bs2;

  if (type == NORM_FROBENIUS) {
    for (i=0; i< bs2*nz; i++ ) {
#if defined(PETSC_COMPLEX)
      sum += real(conj(*v)*(*v)); v++;
#else
      sum += (*v)*(*v); v++;
#endif
    }
    *norm = sqrt(sum);
  }
  else {
    SETERRQ(PETSC_ERR_SUP,"No support for this norm yet");
  }
  return 0;
}

/*
     note: This can only work for identity for row and col. It would 
   be good to check this and otherwise generate an error.
*/
#undef __FUNCTION__  
#define __FUNCTION__ "MatILUFactor_SeqBAIJ"
static int MatILUFactor_SeqBAIJ(Mat inA,IS row,IS col,double efill,int fill)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) inA->data;
  Mat         outA;
  int         ierr;

  if (fill != 0) SETERRQ(1,"Only fill=0 supported");

  outA          = inA; 
  inA->factor   = FACTOR_LU;
  a->row        = row;
  a->col        = col;

  a->solve_work = (Scalar *) PetscMalloc((a->m+a->bs)*sizeof(Scalar));CHKPTRQ(a->solve_work);

  if (!a->diag) {
    ierr = MatMarkDiag_SeqBAIJ(inA); CHKERRQ(ierr);
  }
  ierr = MatLUFactorNumeric(inA,&outA); CHKERRQ(ierr);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatScale_SeqBAIJ"
static int MatScale_SeqBAIJ(Scalar *alpha,Mat inA)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) inA->data;
  int         one = 1, totalnz = a->bs2*a->nz;
  BLscal_( &totalnz, alpha, a->a, &one );
  PLogFlops(totalnz);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatGetValues_SeqBAIJ"
static int MatGetValues_SeqBAIJ(Mat A,int m,int *im,int n,int *in,Scalar *v)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int        *rp, k, low, high, t, row, nrow, i, col, l, *aj = a->j;
  int        *ai = a->i, *ailen = a->ilen;
  int        brow,bcol,ridx,cidx,bs=a->bs,bs2=a->bs2;
  Scalar     *ap, *aa = a->a, zero = 0.0;

  for ( k=0; k<m; k++ ) { /* loop over rows */
    row  = im[k]; brow = row/bs;  
    if (row < 0) SETERRQ(1,"Negative row");
    if (row >= a->m) SETERRQ(1,"Row too large");
    rp   = aj + ai[brow] ; ap = aa + bs2*ai[brow] ;
    nrow = ailen[brow]; 
    for ( l=0; l<n; l++ ) { /* loop over columns */
      if (in[l] < 0) SETERRQ(1,"Negative column");
      if (in[l] >= a->n) SETERRQ(1,"Column too large");
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
  return 0;
} 

#undef __FUNCTION__  
#define __FUNCTION__ "MatGetBlockSize_SeqBAIJ"
static int MatGetBlockSize_SeqBAIJ(Mat mat, int *bs)
{
  Mat_SeqBAIJ *baij = (Mat_SeqBAIJ *) mat->data;
  *bs = baij->bs;
  return 0;
}

/* idx should be of length atlease bs */
#undef __FUNCTION__  
#define __FUNCTION__ "MatZeroRows_SeqBAIJ_Check_Block"
static int MatZeroRows_SeqBAIJ_Check_Block(int *idx, int bs, PetscTruth *flg)
{
  int i,row;
  row = idx[0];
  if (row%bs!=0) { *flg = PETSC_FALSE; return 0; }
  
  for ( i=1; i<bs; i++ ) {
    if (row+i != idx[i]) { *flg = PETSC_FALSE; return 0; }
  }
  *flg = PETSC_TRUE;
  return 0;
}
  
#undef __FUNCTION__  
#define __FUNCTION__ "MatZeroRows_SeqBAIJ"
static int MatZeroRows_SeqBAIJ(Mat A,IS is, Scalar *diag)
{
  Mat_SeqBAIJ *baij=(Mat_SeqBAIJ*)A->data;
  IS          is_local;
  int         ierr,i,j,count,m=baij->m,is_n,*is_idx,*rows,bs=baij->bs,bs2=baij->bs2;
  PetscTruth  flg;
  Scalar      zero = 0.0,*aa;

  /* Make a copy of the IS and  sort it */
  ierr = ISGetSize(is,&is_n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&is_idx);CHKERRQ(ierr);
  ierr = ISCreateGeneral(A->comm,is_n,is_idx,&is_local); CHKERRQ(ierr);
  ierr = ISSort(is_local); CHKERRQ(ierr);
  ierr = ISGetIndices(is_local,&rows); CHKERRQ(ierr);

  i = 0;
  while (i < is_n) {
    if (rows[i]<0 || rows[i]>m) SETERRQ(1,"row out of range");
    flg = PETSC_FALSE;
    if (i+bs <= is_n) {ierr = MatZeroRows_SeqBAIJ_Check_Block(rows+i,bs,&flg); CHKERRQ(ierr); }
    if (flg) { /* There exists a block of rows to be Zerowed */
      baij->ilen[rows[i]/bs] = 0;
      i += bs;
    } else { /* Zero out only the requested row */
      count = (baij->i[rows[i]/bs +1] - baij->i[rows[i]/bs])*bs;
      aa    = baij->a + baij->i[rows[i]/bs]*bs2 + (rows[i]%bs);
      for ( j=0; j<count; j++ ) { 
        aa[0] = zero; 
        aa+=bs;
      }
      i++;
    }
  } 
  if (diag) {
    for ( j=0; j<is_n; j++ ) {
      ierr = (*A->ops.setvalues)(A,1,rows+j,1,rows+j,diag,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = ISRestoreIndices(is,&is_idx); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is_local,&rows); CHKERRQ(ierr);
  ierr = ISDestroy(is_local); CHKERRQ(ierr);
  ierr = MatAssemblyEnd_SeqBAIJ(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatPrintHelp_SeqBAIJ"
int MatPrintHelp_SeqBAIJ(Mat A)
{
  static int called = 0; 
  MPI_Comm   comm = A->comm;

  if (called) return 0; else called = 1;
  PetscPrintf(comm," Options for MATSEQBAIJ and MATMPIBAIJ matrix formats (the defaults):\n");
  PetscPrintf(comm,"  -mat_block_size <block_size>\n");
  return 0;
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps = {MatSetValues_SeqBAIJ,
       MatGetRow_SeqBAIJ,MatRestoreRow_SeqBAIJ,
       MatMult_SeqBAIJ_N,MatMultAdd_SeqBAIJ_N,
       MatMultTrans_SeqBAIJ,MatMultTransAdd_SeqBAIJ,
       MatSolve_SeqBAIJ_N,0,
       0,0,
       MatLUFactor_SeqBAIJ,0,
       0,
       MatTranspose_SeqBAIJ,
       MatGetInfo_SeqBAIJ,MatEqual_SeqBAIJ,
       MatGetDiagonal_SeqBAIJ,MatDiagonalScale_SeqBAIJ,MatNorm_SeqBAIJ,
       0,MatAssemblyEnd_SeqBAIJ,
       0,
       MatSetOption_SeqBAIJ,MatZeroEntries_SeqBAIJ,MatZeroRows_SeqBAIJ,
       MatLUFactorSymbolic_SeqBAIJ,MatLUFactorNumeric_SeqBAIJ_N,0,0,
       MatGetSize_SeqBAIJ,MatGetSize_SeqBAIJ,MatGetOwnershipRange_SeqBAIJ,
       MatILUFactorSymbolic_SeqBAIJ,0,
       0,0,/*  MatConvert_SeqBAIJ  */ 0,
       MatConvertSameType_SeqBAIJ,0,0,
       MatILUFactor_SeqBAIJ,0,0,
       MatGetSubMatrices_SeqBAIJ,MatIncreaseOverlap_SeqBAIJ,
       MatGetValues_SeqBAIJ,0,
       MatPrintHelp_SeqBAIJ,MatScale_SeqBAIJ,
       0,0,0,MatGetBlockSize_SeqBAIJ,
       MatGetRowIJ_SeqBAIJ,
       MatRestoreRowIJ_SeqBAIJ};

#undef __FUNCTION__  
#define __FUNCTION__ "MatCreateSeqBAIJ"
/*@C
   MatCreateSeqBAIJ - Creates a sparse matrix in block AIJ (block
   compressed row) format.  For good matrix assembly performance the
   user should preallocate the matrix storage by setting the parameter nz
   (or the array nzz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Input Parameters:
.  comm - MPI communicator, set to MPI_COMM_SELF
.  bs - size of block
.  m - number of rows
.  n - number of columns
.  nz - number of block nonzeros per block row (same for all rows)
.  nzz - array containing the number of block nonzeros in the various block rows 
         (possibly different for each block row) or PETSC_NULL

   Output Parameter:
.  A - the matrix 

   Notes:
   The block AIJ format is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=PETSC_NULL for PETSc to control dynamic memory 
   allocation.  For additional details, see the users manual chapter on
   matrices.

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues()
@*/
int MatCreateSeqBAIJ(MPI_Comm comm,int bs,int m,int n,int nz,int *nnz, Mat *A)
{
  Mat         B;
  Mat_SeqBAIJ *b;
  int         i,len,ierr,flg,mbs=m/bs,nbs=n/bs,bs2=bs*bs,size;

  MPI_Comm_size(comm,&size);
  if (size > 1) SETERRQ(1,"Comm must be of size 1");

  if (mbs*bs!=m || nbs*bs!=n) 
    SETERRQ(1,"Number rows, cols must be divisible by blocksize");

  *A = 0;
  PetscHeaderCreate(B,_Mat,MAT_COOKIE,MATSEQBAIJ,comm);
  PLogObjectCreate(B);
  B->data = (void *) (b = PetscNew(Mat_SeqBAIJ)); CHKPTRQ(b);
  PetscMemzero(b,sizeof(Mat_SeqBAIJ));
  PetscMemcpy(&B->ops,&MatOps,sizeof(struct _MatOps));
  ierr = OptionsHasName(PETSC_NULL,"-mat_no_unroll",&flg); CHKERRQ(ierr);
  if (!flg) {
    switch (bs) {
      case 1:
        B->ops.lufactornumeric = MatLUFactorNumeric_SeqBAIJ_1;  
        B->ops.solve           = MatSolve_SeqBAIJ_1;
        B->ops.mult            = MatMult_SeqBAIJ_1;
        B->ops.multadd         = MatMultAdd_SeqBAIJ_1;
       break;
      case 2:
        B->ops.lufactornumeric = MatLUFactorNumeric_SeqBAIJ_2;  
        B->ops.solve           = MatSolve_SeqBAIJ_2;
        B->ops.mult            = MatMult_SeqBAIJ_2;
        B->ops.multadd         = MatMultAdd_SeqBAIJ_2;
        break;
      case 3:
        B->ops.lufactornumeric = MatLUFactorNumeric_SeqBAIJ_3;  
        B->ops.solve           = MatSolve_SeqBAIJ_3;
        B->ops.mult            = MatMult_SeqBAIJ_3;
        B->ops.multadd         = MatMultAdd_SeqBAIJ_3;
        break;
      case 4:
        B->ops.lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4;  
        B->ops.solve           = MatSolve_SeqBAIJ_4;
        B->ops.mult            = MatMult_SeqBAIJ_4;
        B->ops.multadd         = MatMultAdd_SeqBAIJ_4;
        break;
      case 5:
        B->ops.lufactornumeric = MatLUFactorNumeric_SeqBAIJ_5;  
        B->ops.solve           = MatSolve_SeqBAIJ_5; 
        B->ops.mult            = MatMult_SeqBAIJ_5;
        B->ops.multadd         = MatMultAdd_SeqBAIJ_5;
        break;
    }
  }
  B->destroy          = MatDestroy_SeqBAIJ;
  B->view             = MatView_SeqBAIJ;
  B->factor           = 0;
  B->lupivotthreshold = 1.0;
  B->mapping          = 0;
  b->row              = 0;
  b->col              = 0;
  b->reallocs         = 0;
  
  b->m       = m; B->m = m; B->M = m;
  b->n       = n; B->n = n; B->N = n;
  b->mbs     = mbs;
  b->nbs     = nbs;
  b->imax    = (int *) PetscMalloc( (mbs+1)*sizeof(int) ); CHKPTRQ(b->imax);
  if (nnz == PETSC_NULL) {
    if (nz == PETSC_DEFAULT) nz = 5;
    else if (nz <= 0)        nz = 1;
    for ( i=0; i<mbs; i++ ) b->imax[i] = nz;
    nz = nz*mbs;
  }
  else {
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
  PLogObjectMemory(B,len+2*(mbs+1)*sizeof(int)+sizeof(struct _Mat)+sizeof(Mat_SeqBAIJ));
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
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatConvertSameType_SeqBAIJ"
int MatConvertSameType_SeqBAIJ(Mat A,Mat *B,int cpvalues)
{
  Mat         C;
  Mat_SeqBAIJ *c,*a = (Mat_SeqBAIJ *) A->data;
  int         i,len, mbs = a->mbs,nz = a->nz,bs2 =a->bs2;

  if (a->i[mbs] != nz) SETERRQ(1,"Corrupt matrix");

  *B = 0;
  PetscHeaderCreate(C,_Mat,MAT_COOKIE,MATSEQBAIJ,A->comm);
  PLogObjectCreate(C);
  C->data       = (void *) (c = PetscNew(Mat_SeqBAIJ)); CHKPTRQ(c);
  PetscMemcpy(&C->ops,&A->ops,sizeof(struct _MatOps));
  C->destroy    = MatDestroy_SeqBAIJ;
  C->view       = MatView_SeqBAIJ;
  C->factor     = A->factor;
  c->row        = 0;
  c->col        = 0;
  C->assembled  = PETSC_TRUE;

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
    if (cpvalues == COPY_VALUES) {
      PetscMemcpy(c->a,a->a,bs2*nz*sizeof(Scalar));
    }
  }

  PLogObjectMemory(C,len+2*(mbs+1)*sizeof(int)+sizeof(struct _Mat)+sizeof(Mat_SeqBAIJ));  
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
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatLoad_SeqBAIJ"
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

  ierr = OptionsGetInt(PETSC_NULL,"-matload_block_size",&bs,&flg);CHKERRQ(ierr);
  bs2  = bs*bs;

  MPI_Comm_size(comm,&size);
  if (size > 1) SETERRQ(1,"view must have one processor");
  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,header,4,BINARY_INT); CHKERRQ(ierr);
  if (header[0] != MAT_COOKIE) SETERRQ(1,"not Mat object");
  M = header[1]; N = header[2]; nz = header[3];

  if (M != N) SETERRQ(1,"Can only do square matrices");

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
  ierr = PetscBinaryRead(fd,rowlengths,M,BINARY_INT); CHKERRQ(ierr);
  for ( i=0; i<extra_rows; i++ ) rowlengths[M+i] = 1;

  /* read in column indices */
  jj = (int*) PetscMalloc( (nz+extra_rows)*sizeof(int) ); CHKPTRQ(jj);
  ierr = PetscBinaryRead(fd,jj,nz,BINARY_INT); CHKERRQ(ierr);
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
  ierr = MatCreateSeqBAIJ(comm,bs,M+extra_rows,N+extra_rows,0,browlengths,A);
         CHKERRQ(ierr);
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
  ierr = PetscBinaryRead(fd,aa,nz,BINARY_SCALAR); CHKERRQ(ierr);
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
  if (jcount != a->nz) SETERRQ(1,"Error bad binary matrix");

  PetscFree(rowlengths);   
  PetscFree(browlengths);
  PetscFree(aa);
  PetscFree(jj);
  PetscFree(mask);

  B->assembled = PETSC_TRUE;

  ierr = OptionsHasName(PETSC_NULL,"-mat_view_info",&flg); CHKERRQ(ierr);
  if (flg) {
    Viewer tviewer;
    ierr = ViewerFileOpenASCII(B->comm,"stdout",&tviewer);CHKERRQ(ierr);
    ierr = ViewerSetFormat(tviewer,VIEWER_FORMAT_ASCII_INFO,0);CHKERRQ(ierr);
    ierr = MatView(B,tviewer); CHKERRQ(ierr);
    ierr = ViewerDestroy(tviewer); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-mat_view_info_detailed",&flg);CHKERRQ(ierr);
  if (flg) {
    Viewer tviewer;
    ierr = ViewerFileOpenASCII(B->comm,"stdout",&tviewer);CHKERRQ(ierr);
    ierr = ViewerSetFormat(tviewer,VIEWER_FORMAT_ASCII_INFO_LONG,0);CHKERRQ(ierr);
    ierr = MatView(B,tviewer); CHKERRQ(ierr);
    ierr = ViewerDestroy(tviewer); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-mat_view",&flg); CHKERRQ(ierr);
  if (flg) {
    Viewer tviewer;
    ierr = ViewerFileOpenASCII(B->comm,"stdout",&tviewer);CHKERRQ(ierr);
    ierr = MatView(B,tviewer); CHKERRQ(ierr);
    ierr = ViewerDestroy(tviewer); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-mat_view_matlab",&flg); CHKERRQ(ierr);
  if (flg) {
    Viewer tviewer;
    ierr = ViewerFileOpenASCII(B->comm,"stdout",&tviewer);CHKERRQ(ierr);
    ierr = ViewerSetFormat(tviewer,VIEWER_FORMAT_ASCII_MATLAB,"M");CHKERRQ(ierr);
    ierr = MatView(B,tviewer); CHKERRQ(ierr);
    ierr = ViewerDestroy(tviewer); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-mat_view_draw",&flg); CHKERRQ(ierr);
  if (flg) {
    Viewer tviewer;
    ierr = ViewerDrawOpenX(B->comm,0,0,0,0,300,300,&tviewer); CHKERRQ(ierr);
    ierr = MatView(B,(Viewer)tviewer); CHKERRQ(ierr);
    ierr = ViewerFlush(tviewer); CHKERRQ(ierr);
    ierr = ViewerDestroy(tviewer); CHKERRQ(ierr);
  }
  return 0;
}



