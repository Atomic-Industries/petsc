#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: aij.c,v 1.252 1998/03/13 21:43:00 balay Exp bsmith $";
#endif

/*
    Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format.
*/

#include "pinclude/pviewer.h"
#include "sys.h"
#include "src/mat/impls/aij/seq/aij.h"
#include "src/vec/vecimpl.h"
#include "src/inline/spops.h"
#include "src/inline/dot.h"
#include "src/inline/bitarray.h"

/*
    Basic AIJ format ILU based on drop tolerance 
*/
#undef __FUNC__  
#define __FUNC__ "MatILUDTFactor_SeqAIJ"
int MatILUDTFactor_SeqAIJ(Mat A,double dt,int maxnz,IS row,IS col,Mat *fact)
{
  /* Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data; */
  int        ierr = 1;

  PetscFunctionBegin;  
  SETERRQ(ierr,0,"Not implemented");
#if !defined(USE_PETSC_DEBUG)
  PetscFunctionReturn(0);
#endif
}

extern int MatToSymmetricIJ_SeqAIJ(int,int*,int*,int,int,int**,int**);

#undef __FUNC__  
#define __FUNC__ "MatGetRowIJ_SeqAIJ"
int MatGetRowIJ_SeqAIJ(Mat A,int oshift,PetscTruth symmetric,int *m,int **ia,int **ja,
                           PetscTruth *done)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        ierr,i,ishift;
 
  PetscFunctionBegin;  
  *m     = A->m;
  if (!ia) PetscFunctionReturn(0);
  ishift = a->indexshift;
  if (symmetric) {
    ierr = MatToSymmetricIJ_SeqAIJ(a->m,a->i,a->j,ishift,oshift,ia,ja); CHKERRQ(ierr);
  } else if (oshift == 0 && ishift == -1) {
    int nz = a->i[a->m]; 
    /* malloc space and  subtract 1 from i and j indices */
    *ia = (int *) PetscMalloc( (a->m+1)*sizeof(int) ); CHKPTRQ(*ia);
    *ja = (int *) PetscMalloc( (nz+1)*sizeof(int) ); CHKPTRQ(*ja);
    for ( i=0; i<nz; i++ ) (*ja)[i] = a->j[i] - 1;
    for ( i=0; i<a->m+1; i++ ) (*ia)[i] = a->i[i] - 1;
  } else if (oshift == 1 && ishift == 0) {
    int nz = a->i[a->m] + 1; 
    /* malloc space and  add 1 to i and j indices */
    *ia = (int *) PetscMalloc( (a->m+1)*sizeof(int) ); CHKPTRQ(*ia);
    *ja = (int *) PetscMalloc( (nz+1)*sizeof(int) ); CHKPTRQ(*ja);
    for ( i=0; i<nz; i++ ) (*ja)[i] = a->j[i] + 1;
    for ( i=0; i<a->m+1; i++ ) (*ia)[i] = a->i[i] + 1;
  } else {
    *ia = a->i; *ja = a->j;
  }
  
  PetscFunctionReturn(0); 
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreRowIJ_SeqAIJ"
int MatRestoreRowIJ_SeqAIJ(Mat A,int oshift,PetscTruth symmetric,int *n,int **ia,int **ja,
                               PetscTruth *done)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        ishift = a->indexshift;
 
  PetscFunctionBegin;  
  if (!ia) PetscFunctionReturn(0);
  if (symmetric || (oshift == 0 && ishift == -1) || (oshift == 1 && ishift == 0)) {
    PetscFree(*ia);
    PetscFree(*ja);
  }
  PetscFunctionReturn(0); 
}

#undef __FUNC__  
#define __FUNC__ "MatGetColumnIJ_SeqAIJ"
int MatGetColumnIJ_SeqAIJ(Mat A,int oshift,PetscTruth symmetric,int *nn,int **ia,int **ja,
                           PetscTruth *done)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        ierr,i,ishift = a->indexshift,*collengths,*cia,*cja,n = A->n,m = A->m;
  int        nz = a->i[m]+ishift,row,*jj,mr,col;
 
  PetscFunctionBegin;  
  *nn     = A->n;
  if (!ia) PetscFunctionReturn(0);
  if (symmetric) {
    ierr = MatToSymmetricIJ_SeqAIJ(a->m,a->i,a->j,ishift,oshift,ia,ja); CHKERRQ(ierr);
  } else {
    collengths = (int *) PetscMalloc( (n+1)*sizeof(int) ); CHKPTRQ(collengths);
    PetscMemzero(collengths,n*sizeof(int));
    cia        = (int *) PetscMalloc( (n+1)*sizeof(int) ); CHKPTRQ(cia);
    cja        = (int *) PetscMalloc( (nz+1)*sizeof(int) ); CHKPTRQ(cja);
    jj = a->j;
    for ( i=0; i<nz; i++ ) {
      collengths[jj[i] + ishift]++;
    }
    cia[0] = oshift;
    for ( i=0; i<n; i++) {
      cia[i+1] = cia[i] + collengths[i];
    }
    PetscMemzero(collengths,n*sizeof(int));
    jj = a->j;
    for ( row=0; row<m; row++ ) {
      mr = a->i[row+1] - a->i[row];
      for ( i=0; i<mr; i++ ) {
        col = *jj++ + ishift;
        cja[cia[col] + collengths[col]++ - oshift] = row + oshift;  
      }
    }
    PetscFree(collengths);
    *ia = cia; *ja = cja;
  }
  
  PetscFunctionReturn(0); 
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreColumnIJ_SeqAIJ"
int MatRestoreColumnIJ_SeqAIJ(Mat A,int oshift,PetscTruth symmetric,int *n,int **ia,
                                     int **ja,PetscTruth *done)
{
  PetscFunctionBegin;  
  if (!ia) PetscFunctionReturn(0);

  PetscFree(*ia);
  PetscFree(*ja);
  
  PetscFunctionReturn(0); 
}

#define CHUNKSIZE   15

#undef __FUNC__  
#define __FUNC__ "MatSetValues_SeqAIJ"
int MatSetValues_SeqAIJ(Mat A,int m,int *im,int n,int *in,Scalar *v,InsertMode is)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        *rp,k,low,high,t,ii,row,nrow,i,col,l,rmax, N, sorted = a->sorted;
  int        *imax = a->imax, *ai = a->i, *ailen = a->ilen,roworiented = a->roworiented;
  int        *aj = a->j, nonew = a->nonew,shift = a->indexshift;
  Scalar     *ap,value, *aa = a->a;

  PetscFunctionBegin;  
  for ( k=0; k<m; k++ ) { /* loop over added rows */
    row  = im[k]; 
#if defined(USE_PETSC_BOPT_g)  
    if (row < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative row");
    if (row >= a->m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row too large");
#endif
    rp   = aj + ai[row] + shift; ap = aa + ai[row] + shift;
    rmax = imax[row]; nrow = ailen[row]; 
    low = 0;
    for ( l=0; l<n; l++ ) { /* loop over added columns */
#if defined(USE_PETSC_BOPT_g)  
      if (in[l] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative column");
      if (in[l] >= a->n) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Column too large");
#endif
      col = in[l] - shift;
      if (roworiented) {
        value = *v++; 
      }
      else {
        value = v[k + l*m];
      }
      if (!sorted) low = 0; high = nrow;
      while (high-low > 5) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else             low  = t;
      }
      for ( i=low; i<high; i++ ) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          if (is == ADD_VALUES) ap[i] += value;  
          else                  ap[i] = value;
          goto noinsert;
        }
      } 
      if (nonew == 1) goto noinsert;
      else if (nonew == -1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Inserting a new nonzero in the matrix");
      if (nrow >= rmax) {
        /* there is no extra room in row, therefore enlarge */
        int    new_nz = ai[a->m] + CHUNKSIZE,len,*new_i,*new_j;
        Scalar *new_a;

        if (nonew == -2) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Inserting a new nonzero in the matrix");

        /* malloc new storage space */
        len     = new_nz*(sizeof(int)+sizeof(Scalar))+(a->m+1)*sizeof(int);
        new_a   = (Scalar *) PetscMalloc( len ); CHKPTRQ(new_a);
        new_j   = (int *) (new_a + new_nz);
        new_i   = new_j + new_nz;

        /* copy over old data into new slots */
        for ( ii=0; ii<row+1; ii++ ) {new_i[ii] = ai[ii];}
        for ( ii=row+1; ii<a->m+1; ii++ ) {new_i[ii] = ai[ii]+CHUNKSIZE;}
        PetscMemcpy(new_j,aj,(ai[row]+nrow+shift)*sizeof(int));
        len = (new_nz - CHUNKSIZE - ai[row] - nrow - shift);
        PetscMemcpy(new_j+ai[row]+shift+nrow+CHUNKSIZE,aj+ai[row]+shift+nrow,
                                                           len*sizeof(int));
        PetscMemcpy(new_a,aa,(ai[row]+nrow+shift)*sizeof(Scalar));
        PetscMemcpy(new_a+ai[row]+shift+nrow+CHUNKSIZE,aa+ai[row]+shift+nrow,
                                                           len*sizeof(Scalar)); 
        /* free up old matrix storage */
        PetscFree(a->a); 
        if (!a->singlemalloc) {PetscFree(a->i);PetscFree(a->j);}
        aa = a->a = new_a; ai = a->i = new_i; aj = a->j = new_j; 
        a->singlemalloc = 1;

        rp   = aj + ai[row] + shift; ap = aa + ai[row] + shift;
        rmax = imax[row] = imax[row] + CHUNKSIZE;
        PLogObjectMemory(A,CHUNKSIZE*(sizeof(int) + sizeof(Scalar)));
        a->maxnz += CHUNKSIZE;
        a->reallocs++;
      }
      N = nrow++ - 1; a->nz++;
      /* shift up all the later entries in this row */
      for ( ii=N; ii>=i; ii-- ) {
        rp[ii+1] = rp[ii];
        ap[ii+1] = ap[ii];
      }
      rp[i] = col; 
      ap[i] = value; 
      noinsert:;
      low = i + 1;
    }
    ailen[row] = nrow;
  }
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ "MatGetValues_SeqAIJ"
int MatGetValues_SeqAIJ(Mat A,int m,int *im,int n,int *in,Scalar *v)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        *rp, k, low, high, t, row, nrow, i, col, l, *aj = a->j;
  int        *ai = a->i, *ailen = a->ilen, shift = a->indexshift;
  Scalar     *ap, *aa = a->a, zero = 0.0;

  PetscFunctionBegin;  
  for ( k=0; k<m; k++ ) { /* loop over rows */
    row  = im[k];   
    if (row < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative row");
    if (row >= a->m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row too large");
    rp   = aj + ai[row] + shift; ap = aa + ai[row] + shift;
    nrow = ailen[row]; 
    for ( l=0; l<n; l++ ) { /* loop over columns */
      if (in[l] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative column");
      if (in[l] >= a->n) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Column too large");
      col = in[l] - shift;
      high = nrow; low = 0; /* assume unsorted */
      while (high-low > 5) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else             low  = t;
      }
      for ( i=low; i<high; i++ ) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          *v++ = ap[i];
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
#define __FUNC__ "MatView_SeqAIJ_Binary"
extern int MatView_SeqAIJ_Binary(Mat A,Viewer viewer)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        i, fd, *col_lens, ierr;

  PetscFunctionBegin;  
  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
  col_lens = (int *) PetscMalloc( (4+a->m)*sizeof(int) ); CHKPTRQ(col_lens);
  col_lens[0] = MAT_COOKIE;
  col_lens[1] = a->m;
  col_lens[2] = a->n;
  col_lens[3] = a->nz;

  /* store lengths of each row and write (including header) to file */
  for ( i=0; i<a->m; i++ ) {
    col_lens[4+i] = a->i[i+1] - a->i[i];
  }
  ierr = PetscBinaryWrite(fd,col_lens,4+a->m,PETSC_INT,1); CHKERRQ(ierr);
  PetscFree(col_lens);

  /* store column indices (zero start index) */
  if (a->indexshift) {
    for ( i=0; i<a->nz; i++ ) a->j[i]--;
  }
  ierr = PetscBinaryWrite(fd,a->j,a->nz,PETSC_INT,0); CHKERRQ(ierr);
  if (a->indexshift) {
    for ( i=0; i<a->nz; i++ ) a->j[i]++;
  }

  /* store nonzero values */
  ierr = PetscBinaryWrite(fd,a->a,a->nz,PETSC_SCALAR,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqAIJ_ASCII"
extern int MatView_SeqAIJ_ASCII(Mat A,Viewer viewer)
{
  Mat_SeqAIJ  *a = (Mat_SeqAIJ *) A->data;
  int         ierr, i,j, m = a->m, shift = a->indexshift, format, flg1,flg2;
  FILE        *fd;
  char        *outputname;

  PetscFunctionBegin;  
  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  ierr = ViewerFileGetOutputname_Private(viewer,&outputname); CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format);
  if (format == VIEWER_FORMAT_ASCII_INFO) {
    PetscFunctionReturn(0);
  } else if (format == VIEWER_FORMAT_ASCII_INFO_LONG) {
    ierr = OptionsHasName(PETSC_NULL,"-mat_aij_no_inode",&flg1); CHKERRQ(ierr);
    ierr = OptionsHasName(PETSC_NULL,"-mat_no_unroll",&flg2); CHKERRQ(ierr);
    if (flg1 || flg2) fprintf(fd,"  not using I-node routines\n");
    else     fprintf(fd,"  using I-node routines: found %d nodes, limit used is %d\n",
        a->inode.node_count,a->inode.limit);
  } else if (format == VIEWER_FORMAT_ASCII_MATLAB) {
    int nofinalvalue = 0;
    if ((a->i[m] == a->i[m-1]) || (a->j[a->nz-1] != a->n-!shift)) {
      nofinalvalue = 1;
    }
    fprintf(fd,"%% Size = %d %d \n",m,a->n);
    fprintf(fd,"%% Nonzeros = %d \n",a->nz);
    fprintf(fd,"zzz = zeros(%d,3);\n",a->nz+nofinalvalue);
    fprintf(fd,"zzz = [\n");

    for (i=0; i<m; i++) {
      for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
#if defined(USE_PETSC_COMPLEX)
        fprintf(fd,"%d %d  %18.16e + %18.16e i \n",i+1,a->j[j]+!shift,real(a->a[j]),imag(a->a[j]));
#else
        fprintf(fd,"%d %d  %18.16e\n", i+1, a->j[j]+!shift, a->a[j]);
#endif
      }
    }
    if (nofinalvalue) {
      fprintf(fd,"%d %d  %18.16e\n", m, a->n, 0.0);
    } 
    fprintf(fd,"];\n %s = spconvert(zzz);\n",outputname);
  } else if (format == VIEWER_FORMAT_ASCII_COMMON) {
    for ( i=0; i<m; i++ ) {
      fprintf(fd,"row %d:",i);
      for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
#if defined(USE_PETSC_COMPLEX)
        if (imag(a->a[j]) > 0.0 && real(a->a[j]) != 0.0)
          fprintf(fd," %d %g + %g i",a->j[j]+shift,real(a->a[j]),imag(a->a[j]));
        else if (imag(a->a[j]) < 0.0 && real(a->a[j]) != 0.0)
          fprintf(fd," %d %g - %g i",a->j[j]+shift,real(a->a[j]),-imag(a->a[j]));
        else if (real(a->a[j]) != 0.0)
          fprintf(fd," %d %g ",a->j[j]+shift,real(a->a[j]));
#else
        if (a->a[j] != 0.0) fprintf(fd," %d %g ",a->j[j]+shift,a->a[j]);
#endif
      }
      fprintf(fd,"\n");
    }
  } 
  else if (format == VIEWER_FORMAT_ASCII_SYMMODU) {
    int nzd=0, fshift=1, *sptr;
    sptr = (int *) PetscMalloc( (m+1)*sizeof(int) ); CHKPTRQ(sptr);
    for ( i=0; i<m; i++ ) {
      sptr[i] = nzd+1;
      for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
        if (a->j[j] >= i) {
#if defined(USE_PETSC_COMPLEX)
          if (imag(a->a[j]) != 0.0 || real(a->a[j]) != 0.0) nzd++;
#else
          if (a->a[j] != 0.0) nzd++;
#endif
        }
      }
    }
    sptr[m] = nzd+1;
    fprintf(fd," %d %d\n\n",m,nzd);
    for ( i=0; i<m+1; i+=6 ) {
      if (i+4<m) fprintf(fd," %d %d %d %d %d %d\n",sptr[i],sptr[i+1],sptr[i+2],sptr[i+3],sptr[i+4],sptr[i+5]);
      else if (i+3<m) fprintf(fd," %d %d %d %d %d\n",sptr[i],sptr[i+1],sptr[i+2],sptr[i+3],sptr[i+4]);
      else if (i+2<m) fprintf(fd," %d %d %d %d\n",sptr[i],sptr[i+1],sptr[i+2],sptr[i+3]);
      else if (i+1<m) fprintf(fd," %d %d %d\n",sptr[i],sptr[i+1],sptr[i+2]);
      else if (i<m)   fprintf(fd," %d %d\n",sptr[i],sptr[i+1]);
      else            fprintf(fd," %d\n",sptr[i]);
    }
    fprintf(fd,"\n");
    PetscFree(sptr);
    for ( i=0; i<m; i++ ) {
      for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
        if (a->j[j] >= i) fprintf(fd," %d ",a->j[j]+fshift);
      }
      fprintf(fd,"\n");
    }
    fprintf(fd,"\n");
    for ( i=0; i<m; i++ ) {
      for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
        if (a->j[j] >= i) {
#if defined(USE_PETSC_COMPLEX)
          if (imag(a->a[j]) != 0.0 || real(a->a[j]) != 0.0)
            fprintf(fd," %18.16e %18.16e ",real(a->a[j]),imag(a->a[j]));
#else
          if (a->a[j] != 0.0) fprintf(fd," %18.16e ",a->a[j]);
#endif
        }
      }
      fprintf(fd,"\n");
    }
  } else {
    for ( i=0; i<m; i++ ) {
      fprintf(fd,"row %d:",i);
      for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
#if defined(USE_PETSC_COMPLEX)
        if (imag(a->a[j]) > 0.0) {
          fprintf(fd," %d %g + %g i",a->j[j]+shift,real(a->a[j]),imag(a->a[j]));
        } else if (imag(a->a[j]) < 0.0) {
          fprintf(fd," %d %g - %g i",a->j[j]+shift,real(a->a[j]),-imag(a->a[j]));
        } else {
          fprintf(fd," %d %g ",a->j[j]+shift,real(a->a[j]));
        }
#else
        fprintf(fd," %d %g ",a->j[j]+shift,a->a[j]);
#endif
      }
      fprintf(fd,"\n");
    }
  }
  fflush(fd);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqAIJ_Draw"
extern int MatView_SeqAIJ_Draw(Mat A,Viewer viewer)
{
  Mat_SeqAIJ  *a = (Mat_SeqAIJ *) A->data;
  int         ierr, i,j, m = a->m, shift = a->indexshift,pause,color;
  int         format;
  double      xl,yl,xr,yr,w,h,xc,yc,scale = 1.0,x_l,x_r,y_l,y_r,maxv = 0.0;
  Draw        draw;
  DrawButton  button;
  PetscTruth  isnull;

  PetscFunctionBegin;  
  ierr = ViewerDrawGetDraw(viewer,&draw); CHKERRQ(ierr);
  ierr = DrawSynchronizedClear(draw); CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format); CHKERRQ(ierr);
  ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);

  xr  = a->n; yr = a->m; h = yr/10.0; w = xr/10.0; 
  xr += w;    yr += h;  xl = -w;     yl = -h;
  ierr = DrawSetCoordinates(draw,xl,yl,xr,yr); CHKERRQ(ierr);
  /* loop over matrix elements drawing boxes */

  if (format != VIEWER_FORMAT_DRAW_CONTOUR) {
    /* Blue for negative, Cyan for zero and  Red for positive */
    color = DRAW_BLUE;
    for ( i=0; i<m; i++ ) {
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
        x_l = a->j[j] + shift; x_r = x_l + 1.0;
#if defined(USE_PETSC_COMPLEX)
        if (real(a->a[j]) >=  0.) continue;
#else
        if (a->a[j] >=  0.) continue;
#endif
        DrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);
      } 
    }
    color = DRAW_CYAN;
    for ( i=0; i<m; i++ ) {
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
        x_l = a->j[j] + shift; x_r = x_l + 1.0;
        if (a->a[j] !=  0.) continue;
        DrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);
      } 
    }
    color = DRAW_RED;
    for ( i=0; i<m; i++ ) {
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
        x_l = a->j[j] + shift; x_r = x_l + 1.0;
#if defined(USE_PETSC_COMPLEX)
        if (real(a->a[j]) <=  0.) continue;
#else
        if (a->a[j] <=  0.) continue;
#endif
        DrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);
      } 
    }
  } else {
    /* use contour shading to indicate magnitude of values */
    /* first determine max of all nonzero values */
    int    nz = a->nz,count;
    Draw   popup;

    for ( i=0; i<nz; i++ ) {
      if (PetscAbsScalar(a->a[i]) > maxv) maxv = PetscAbsScalar(a->a[i]);
    }
    ierr = DrawGetPopup(draw,&popup); CHKERRQ(ierr);
    ierr = DrawScalePopup(popup,0.0,maxv); CHKERRQ(ierr);
    count = 0;
    for ( i=0; i<m; i++ ) {
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
        x_l = a->j[j] + shift; x_r = x_l + 1.0;
        color = 32 + (int) ((200.0 - 32.0)*PetscAbsScalar(a->a[count])/maxv);
        DrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);
        count++;
      } 
    }
  }
  DrawSynchronizedFlush(draw); 
  DrawGetPause(draw,&pause);
  if (pause >= 0) { PetscSleep(pause); PetscFunctionReturn(0);}

  /* allow the matrix to zoom or shrink */
  ierr = DrawCheckResizedWindow(draw);
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
    if (format != VIEWER_FORMAT_DRAW_CONTOUR) {
      /* Blue for negative, Cyan for zero and  Red for positive */
      color = DRAW_BLUE;
      for ( i=0; i<m; i++ ) {
        y_l = m - i - 1.0; y_r = y_l + 1.0;
        for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
          x_l = a->j[j] + shift; x_r = x_l + 1.0;
#if defined(USE_PETSC_COMPLEX)
          if (real(a->a[j]) >=  0.) continue;
#else
          if (a->a[j] >=  0.) continue;
#endif
          DrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);
        } 
      }
      color = DRAW_CYAN;
      for ( i=0; i<m; i++ ) {
        y_l = m - i - 1.0; y_r = y_l + 1.0;
        for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
          x_l = a->j[j] + shift; x_r = x_l + 1.0;
          if (a->a[j] !=  0.) continue;
          DrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);
        } 
      }
      color = DRAW_RED;
      for ( i=0; i<m; i++ ) {
        y_l = m - i - 1.0; y_r = y_l + 1.0;
        for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
          x_l = a->j[j] + shift; x_r = x_l + 1.0;
#if defined(USE_PETSC_COMPLEX)
          if (real(a->a[j]) <=  0.) continue;
#else
          if (a->a[j] <=  0.) continue;
#endif
          DrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);
        } 
      }
    } else {
      /* use contour shading to indicate magnitude of values */
      int count = 0;
      for ( i=0; i<m; i++ ) {
        y_l = m - i - 1.0; y_r = y_l + 1.0;
        for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
          x_l = a->j[j] + shift; x_r = x_l + 1.0;
          color = 32 + (int) ((200.0 - 32.0)*PetscAbsScalar(a->a[count])/maxv);
          DrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color); CHKERRQ(ierr);
          count++;
        } 
      }
    }

    ierr = DrawCheckResizedWindow(draw); CHKERRQ(ierr);
    ierr = DrawSynchronizedGetMouseButton(draw,&button,&xc,&yc,0,0);  CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqAIJ"
int MatView_SeqAIJ(PetscObject obj,Viewer viewer)
{
  Mat         A = (Mat) obj;
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*) A->data;
  ViewerType  vtype;
  int         ierr;

  PetscFunctionBegin;  
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == MATLAB_VIEWER) {
    ierr = ViewerMatlabPutSparse_Private(viewer,a->m,a->n,a->nz,a->a,a->i,a->j);  CHKERRQ(ierr);
  }
  else if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER){
    ierr = MatView_SeqAIJ_ASCII(A,viewer); CHKERRQ(ierr);
  }
  else if (vtype == BINARY_FILE_VIEWER) {
    ierr = MatView_SeqAIJ_Binary(A,viewer); CHKERRQ(ierr);
  }
  else if (vtype == DRAW_VIEWER) {
    ierr = MatView_SeqAIJ_Draw(A,viewer); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

extern int Mat_AIJ_CheckInode(Mat);
#undef __FUNC__  
#define __FUNC__ "MatAssemblyEnd_SeqAIJ"
int MatAssemblyEnd_SeqAIJ(Mat A,MatAssemblyType mode)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        fshift = 0,i,j,*ai = a->i, *aj = a->j, *imax = a->imax,ierr;
  int        m = a->m, *ip, N, *ailen = a->ilen,shift = a->indexshift,rmax = 0;
  Scalar     *aa = a->a, *ap;

  PetscFunctionBegin;  
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  if (m) rmax = ailen[0]; /* determine row with most nonzeros */
  for ( i=1; i<m; i++ ) {
    /* move each row back by the amount of empty slots (fshift) before it*/
    fshift += imax[i-1] - ailen[i-1];
    rmax   = PetscMax(rmax,ailen[i]);
    if (fshift) {
      ip = aj + ai[i] + shift; ap = aa + ai[i] + shift;
      N = ailen[i];
      for ( j=0; j<N; j++ ) {
        ip[j-fshift] = ip[j];
        ap[j-fshift] = ap[j]; 
      }
    } 
    ai[i] = ai[i-1] + ailen[i-1];
  }
  if (m) {
    fshift += imax[m-1] - ailen[m-1];
    ai[m] = ai[m-1] + ailen[m-1];
  }
  /* reset ilen and imax for each row */
  for ( i=0; i<m; i++ ) {
    ailen[i] = imax[i] = ai[i+1] - ai[i];
  }
  a->nz = ai[m] + shift; 

  /* diagonals may have moved, so kill the diagonal pointers */
  if (fshift && a->diag) {
    PetscFree(a->diag);
    PLogObjectMemory(A,-(m+1)*sizeof(int));
    a->diag = 0;
  } 
  PLogInfo(A,"MatAssemblyEnd_SeqAIJ:Matrix size: %d X %d; storage space: %d unneeded, %d used\n",
           m,a->n,fshift,a->nz);
  PLogInfo(A,"MatAssemblyEnd_SeqAIJ:Number of mallocs during MatSetValues is %d\n",
           a->reallocs);
  PLogInfo(A,"MatAssemblyEnd_SeqAIJ:Most nonzeros in any row is %d\n",rmax);
  a->reallocs          = 0;
  A->info.nz_unneeded  = (double)fshift;

  /* check out for identical nodes. If found, use inode functions */
  ierr = Mat_AIJ_CheckInode(A); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatZeroEntries_SeqAIJ"
int MatZeroEntries_SeqAIJ(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data; 

  PetscFunctionBegin;  
  PetscMemzero(a->a,(a->i[a->m]+a->indexshift)*sizeof(Scalar));
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDestroy_SeqAIJ"
int MatDestroy_SeqAIJ(PetscObject obj)
{
  Mat        A  = (Mat) obj;
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        ierr;

  PetscFunctionBegin;  
#if defined(USE_PETSC_LOG)
  PLogObjectState(obj,"Rows=%d, Cols=%d, NZ=%d",a->m,a->n,a->nz);
#endif
  PetscFree(a->a); 
  if (!a->singlemalloc) { PetscFree(a->i); PetscFree(a->j);}
  if (a->diag) PetscFree(a->diag);
  if (a->ilen) PetscFree(a->ilen);
  if (a->imax) PetscFree(a->imax);
  if (a->solve_work) PetscFree(a->solve_work);
  if (a->inode.size) PetscFree(a->inode.size);
  if (a->icol) {ierr = ISDestroy(a->icol);CHKERRQ(ierr);}
  PetscFree(a); 

  PLogObjectDestroy(A);
  PetscHeaderDestroy(A);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatCompress_SeqAIJ"
int MatCompress_SeqAIJ(Mat A)
{
  PetscFunctionBegin;  
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSetOption_SeqAIJ"
int MatSetOption_SeqAIJ(Mat A,MatOption op)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;

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
           op == MAT_IGNORE_OFF_PROC_ENTRIES||
           op == MAT_USE_HASH_TABLE)
    PLogInfo(A,"MatSetOption_SeqAIJ:Option ignored\n");
  else if (op == MAT_NO_NEW_DIAGONALS) {
    SETERRQ(PETSC_ERR_SUP,0,"MAT_NO_NEW_DIAGONALS");
  } else if (op == MAT_INODE_LIMIT_1)            a->inode.limit  = 1;
  else if (op == MAT_INODE_LIMIT_2)            a->inode.limit  = 2;
  else if (op == MAT_INODE_LIMIT_3)            a->inode.limit  = 3;
  else if (op == MAT_INODE_LIMIT_4)            a->inode.limit  = 4;
  else if (op == MAT_INODE_LIMIT_5)            a->inode.limit  = 5;
  else SETERRQ(PETSC_ERR_SUP,0,"unknown option");
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetDiagonal_SeqAIJ"
int MatGetDiagonal_SeqAIJ(Mat A,Vec v)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        i,j, n,shift = a->indexshift,ierr;
  Scalar     *x, zero = 0.0;

  PetscFunctionBegin;
  ierr = VecSet(&zero,v);CHKERRQ(ierr);
  VecGetArray_Fast(v,x); VecGetLocalSize(v,&n);
  if (n != a->m) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Nonconforming matrix and vector");
  for ( i=0; i<a->m; i++ ) {
    for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
      if (a->j[j]+shift == i) {
        x[i] = a->a[j];
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------*/
/* Should check that shapes of vectors and matrices match */
/* -------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "MatMultTrans_SeqAIJ"
int MatMultTrans_SeqAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *x, *y, *v, alpha;
  int        m = a->m, n, i, *idx, shift = a->indexshift;

  PetscFunctionBegin; 
  VecGetArray_Fast(xx,x); VecGetArray_Fast(yy,y);
  PetscMemzero(y,a->n*sizeof(Scalar));
  y = y + shift; /* shift for Fortran start by 1 indexing */
  for ( i=0; i<m; i++ ) {
    idx   = a->j + a->i[i] + shift;
    v     = a->a + a->i[i] + shift;
    n     = a->i[i+1] - a->i[i];
    alpha = x[i];
    while (n-->0) {y[*idx++] += alpha * *v++;}
  }
  PLogFlops(2*a->nz - a->n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMultTransAdd_SeqAIJ"
int MatMultTransAdd_SeqAIJ(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *x, *y, *v, alpha;
  int        m = a->m, n, i, *idx,shift = a->indexshift;

  PetscFunctionBegin;
  VecGetArray_Fast(xx,x); VecGetArray_Fast(yy,y);
  if (zz != yy) VecCopy(zz,yy);
  y = y + shift; /* shift for Fortran start by 1 indexing */
  for ( i=0; i<m; i++ ) {
    idx   = a->j + a->i[i] + shift;
    v     = a->a + a->i[i] + shift;
    n     = a->i[i+1] - a->i[i];
    alpha = x[i];
    while (n-->0) {y[*idx++] += alpha * *v++;}
  }
  PLogFlops(2*a->nz);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMult_SeqAIJ"
int MatMult_SeqAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *x, *y, *v, sum;
  int        m = a->m, n, i, *idx, shift = a->indexshift,*ii,jrow,j;

  PetscFunctionBegin;
  VecGetArray_Fast(xx,x); VecGetArray_Fast(yy,y);
  x    = x + shift;    /* shift for Fortran start by 1 indexing */
  idx  = a->j;
  v    = a->a;
  ii   = a->i;
#if defined(USE_FORTRAN_KERNELS)
  fortranmultaij_(&m,x,ii,idx+shift,v+shift,y);
#else
  v    += shift; /* shift for Fortran start by 1 indexing */
  idx  += shift;
  for ( i=0; i<m; i++ ) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sum  = 0.0;
    for ( j=0; j<n; j++) {
      sum += v[jrow]*x[idx[jrow]]; jrow++;
     }
    y[i] = sum;
  }
#endif
  PLogFlops(2*a->nz - m);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMultAdd_SeqAIJ"
int MatMultAdd_SeqAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *x, *y, *z, *v, sum;
  int        m = a->m, n, i, *idx, shift = a->indexshift,*ii,jrow,j;

  PetscFunctionBegin;
  VecGetArray_Fast(xx,x); VecGetArray_Fast(yy,y); VecGetArray_Fast(zz,z); 
  x    = x + shift; /* shift for Fortran start by 1 indexing */
  idx  = a->j;
  v    = a->a;
  ii   = a->i;
#if defined(USE_FORTRAN_KERNELS)
  fortranmultaddaij_(&m,x,ii,idx+shift,v+shift,y,z);
#else
  v   += shift; /* shift for Fortran start by 1 indexing */
  idx += shift;
  for ( i=0; i<m; i++ ) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sum  = y[i];
    for ( j=0; j<n; j++) {
      sum += v[jrow]*x[idx[jrow]]; jrow++;
     }
    z[i] = sum;
  }
#endif
  PLogFlops(2*a->nz);
  PetscFunctionReturn(0);
}

/*
     Adds diagonal pointers to sparse matrix structure.
*/

#undef __FUNC__  
#define __FUNC__ "MatMarkDiag_SeqAIJ"
int MatMarkDiag_SeqAIJ(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data; 
  int        i,j, *diag, m = a->m,shift = a->indexshift;

  PetscFunctionBegin;
  diag = (int *) PetscMalloc( (m+1)*sizeof(int)); CHKPTRQ(diag);
  PLogObjectMemory(A,(m+1)*sizeof(int));
  for ( i=0; i<a->m; i++ ) {
    for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
      if (a->j[j]+shift == i) {
        diag[i] = j - shift;
        break;
      }
    }
  }
  a->diag = diag;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatRelax_SeqAIJ"
int MatRelax_SeqAIJ(Mat A,Vec bb,double omega,MatSORType flag,
                           double fshift,int its,Vec xx)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *x, *b, *bs,  d, *xs, sum, *v = a->a,*t,scale,*ts, *xb;
  int        ierr, *idx, *diag,n = a->n, m = a->m, i, shift = a->indexshift;

  PetscFunctionBegin;
  VecGetArray_Fast(xx,x); VecGetArray_Fast(bb,b);
  if (!a->diag) {ierr = MatMarkDiag_SeqAIJ(A);CHKERRQ(ierr);}
  diag = a->diag;
  xs   = x + shift; /* shifted by one for index start of a or a->j*/
  if (flag == SOR_APPLY_UPPER) {
   /* apply ( U + D/omega) to the vector */
    bs = b + shift;
    for ( i=0; i<m; i++ ) {
        d    = fshift + a->a[diag[i] + shift];
        n    = a->i[i+1] - diag[i] - 1;
        idx  = a->j + diag[i] + (!shift);
        v    = a->a + diag[i] + (!shift);
        sum  = b[i]*d/omega;
        SPARSEDENSEDOT(sum,bs,v,idx,n); 
        x[i] = sum;
    }
    PetscFunctionReturn(0);
  }
  if (flag == SOR_APPLY_LOWER) {
    SETERRQ(PETSC_ERR_SUP,0,"SOR_APPLY_LOWER is not done");
  } else if (flag & SOR_EISENSTAT) {
    /* Let  A = L + U + D; where L is lower trianglar,
    U is upper triangular, E is diagonal; This routine applies

            (L + E)^{-1} A (U + E)^{-1}

    to a vector efficiently using Eisenstat's trick. This is for
    the case of SSOR preconditioner, so E is D/omega where omega
    is the relaxation factor.
    */
    t = (Scalar *) PetscMalloc( m*sizeof(Scalar) ); CHKPTRQ(t);
    scale = (2.0/omega) - 1.0;

    /*  x = (E + U)^{-1} b */
    for ( i=m-1; i>=0; i-- ) {
      d    = fshift + a->a[diag[i] + shift];
      n    = a->i[i+1] - diag[i] - 1;
      idx  = a->j + diag[i] + (!shift);
      v    = a->a + diag[i] + (!shift);
      sum  = b[i];
      SPARSEDENSEMDOT(sum,xs,v,idx,n); 
      x[i] = omega*(sum/d);
    }

    /*  t = b - (2*E - D)x */
    v = a->a;
    for ( i=0; i<m; i++ ) { t[i] = b[i] - scale*(v[*diag++ + shift])*x[i]; }

    /*  t = (E + L)^{-1}t */
    ts = t + shift; /* shifted by one for index start of a or a->j*/
    diag = a->diag;
    for ( i=0; i<m; i++ ) {
      d    = fshift + a->a[diag[i]+shift];
      n    = diag[i] - a->i[i];
      idx  = a->j + a->i[i] + shift;
      v    = a->a + a->i[i] + shift;
      sum  = t[i];
      SPARSEDENSEMDOT(sum,ts,v,idx,n); 
      t[i] = omega*(sum/d);
    }

    /*  x = x + t */
    for ( i=0; i<m; i++ ) { x[i] += t[i]; }
    PetscFree(t);
    PetscFunctionReturn(0);
  }
  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      for ( i=0; i<m; i++ ) {
        d    = fshift + a->a[diag[i]+shift];
        n    = diag[i] - a->i[i];
        idx  = a->j + a->i[i] + shift;
        v    = a->a + a->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = omega*(sum/d);
      }
      xb = x;
    } else xb = b;
    if ((flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) && 
        (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP)) {
      for ( i=0; i<m; i++ ) {
        x[i] *= a->a[diag[i]+shift];
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      for ( i=m-1; i>=0; i-- ) {
        d    = fshift + a->a[diag[i] + shift];
        n    = a->i[i+1] - diag[i] - 1;
        idx  = a->j + diag[i] + (!shift);
        v    = a->a + diag[i] + (!shift);
        sum  = xb[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = omega*(sum/d);
      }
    }
    its--;
  }
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      for ( i=0; i<m; i++ ) {
        d    = fshift + a->a[diag[i]+shift];
        n    = a->i[i+1] - a->i[i]; 
        idx  = a->j + a->i[i] + shift;
        v    = a->a + a->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum + a->a[diag[i]+shift]*x[i])/d;
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      for ( i=m-1; i>=0; i-- ) {
        d    = fshift + a->a[diag[i] + shift];
        n    = a->i[i+1] - a->i[i]; 
        idx  = a->j + a->i[i] + shift;
        v    = a->a + a->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum + a->a[diag[i]+shift]*x[i])/d;
      }
    }
  }
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ "MatGetInfo_SeqAIJ"
int MatGetInfo_SeqAIJ(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;

  PetscFunctionBegin;
  info->rows_global    = (double)a->m;
  info->columns_global = (double)a->n;
  info->rows_local     = (double)a->m;
  info->columns_local  = (double)a->n;
  info->block_size     = 1.0;
  info->nz_allocated   = (double)a->maxnz;
  info->nz_used        = (double)a->nz;
  info->nz_unneeded    = (double)(a->maxnz - a->nz);
  /*  if (info->nz_unneeded != A->info.nz_unneeded) 
    printf("space descrepancy: maxnz-nz = %d, nz_unneeded = %d\n",(int)info->nz_unneeded,(int)A->info.nz_unneeded); */
  info->assemblies     = (double)A->num_ass;
  info->mallocs        = (double)a->reallocs;
  info->memory         = A->mem;
  if (A->factor) {
    info->fill_ratio_given  = A->info.fill_ratio_given;
    info->fill_ratio_needed = A->info.fill_ratio_needed;
    info->factor_mallocs    = A->info.factor_mallocs;
  } else {
    info->fill_ratio_given  = 0;
    info->fill_ratio_needed = 0;
    info->factor_mallocs    = 0;
  }
  PetscFunctionReturn(0);
}

extern int MatLUFactorSymbolic_SeqAIJ(Mat,IS,IS,double,Mat*);
extern int MatLUFactorNumeric_SeqAIJ(Mat,Mat*);
extern int MatLUFactor_SeqAIJ(Mat,IS,IS,double);
extern int MatSolve_SeqAIJ(Mat,Vec,Vec);
extern int MatSolveAdd_SeqAIJ(Mat,Vec,Vec,Vec);
extern int MatSolveTrans_SeqAIJ(Mat,Vec,Vec);
extern int MatSolveTransAdd_SeqAIJ(Mat,Vec,Vec,Vec);

#undef __FUNC__  
#define __FUNC__ "MatZeroRows_SeqAIJ"
int MatZeroRows_SeqAIJ(Mat A,IS is,Scalar *diag)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int         i,ierr,N, *rows,m = a->m - 1,shift = a->indexshift;

  PetscFunctionBegin;
  ierr = ISGetSize(is,&N); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows); CHKERRQ(ierr);
  if (diag) {
    for ( i=0; i<N; i++ ) {
      if (rows[i] < 0 || rows[i] > m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"row out of range");
      if (a->ilen[rows[i]] > 0) { /* in case row was completely empty */
        a->ilen[rows[i]] = 1; 
        a->a[a->i[rows[i]]+shift] = *diag;
        a->j[a->i[rows[i]]+shift] = rows[i]+shift;
      } else {
        ierr = MatSetValues_SeqAIJ(A,1,&rows[i],1,&rows[i],diag,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  } else {
    for ( i=0; i<N; i++ ) {
      if (rows[i] < 0 || rows[i] > m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"row out of range");
      a->ilen[rows[i]] = 0; 
    }
  }
  ISRestoreIndices(is,&rows);
  ierr = MatAssemblyEnd_SeqAIJ(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSize_SeqAIJ"
int MatGetSize_SeqAIJ(Mat A,int *m,int *n)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;

  PetscFunctionBegin;
  if (m) *m = a->m; 
  if (n) *n = a->n;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetOwnershipRange_SeqAIJ"
int MatGetOwnershipRange_SeqAIJ(Mat A,int *m,int *n)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;

  PetscFunctionBegin;
  *m = 0; *n = a->m;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetRow_SeqAIJ"
int MatGetRow_SeqAIJ(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        *itmp,i,shift = a->indexshift;

  PetscFunctionBegin;
  if (row < 0 || row >= a->m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row out of range");

  *nz = a->i[row+1] - a->i[row];
  if (v) *v = a->a + a->i[row] + shift;
  if (idx) {
    itmp = a->j + a->i[row] + shift;
    if (*nz && shift) {
      *idx = (int *) PetscMalloc( (*nz)*sizeof(int) ); CHKPTRQ(*idx);
      for ( i=0; i<(*nz); i++ ) {(*idx)[i] = itmp[i] + shift;}
    } else if (*nz) {
      *idx = itmp;
    }
    else *idx = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreRow_SeqAIJ"
int MatRestoreRow_SeqAIJ(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;

  PetscFunctionBegin;
  if (idx) {if (*idx && a->indexshift) PetscFree(*idx);}
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatNorm_SeqAIJ"
int MatNorm_SeqAIJ(Mat A,NormType type,double *norm)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *v = a->a;
  double     sum = 0.0;
  int        i, j,shift = a->indexshift;

  PetscFunctionBegin;
  if (type == NORM_FROBENIUS) {
    for (i=0; i<a->nz; i++ ) {
#if defined(USE_PETSC_COMPLEX)
      sum += real(conj(*v)*(*v)); v++;
#else
      sum += (*v)*(*v); v++;
#endif
    }
    *norm = sqrt(sum);
  } else if (type == NORM_1) {
    double *tmp;
    int    *jj = a->j;
    tmp = (double *) PetscMalloc( (a->n+1)*sizeof(double) ); CHKPTRQ(tmp);
    PetscMemzero(tmp,a->n*sizeof(double));
    *norm = 0.0;
    for ( j=0; j<a->nz; j++ ) {
        tmp[*jj++ + shift] += PetscAbsScalar(*v);  v++;
    }
    for ( j=0; j<a->n; j++ ) {
      if (tmp[j] > *norm) *norm = tmp[j];
    }
    PetscFree(tmp);
  } else if (type == NORM_INFINITY) {
    *norm = 0.0;
    for ( j=0; j<a->m; j++ ) {
      v = a->a + a->i[j] + shift;
      sum = 0.0;
      for ( i=0; i<a->i[j+1]-a->i[j]; i++ ) {
        sum += PetscAbsScalar(*v); v++;
      }
      if (sum > *norm) *norm = sum;
    }
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"No support for two norm");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatTranspose_SeqAIJ"
int MatTranspose_SeqAIJ(Mat A,Mat *B)
{ 
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Mat        C;
  int        i, ierr, *aj = a->j, *ai = a->i, m = a->m, len, *col;
  int        shift = a->indexshift;
  Scalar     *array = a->a;

  PetscFunctionBegin;
  if (B == PETSC_NULL && m != a->n) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Square matrix only for in-place");
  col = (int *) PetscMalloc((1+a->n)*sizeof(int)); CHKPTRQ(col);
  PetscMemzero(col,(1+a->n)*sizeof(int));
  if (shift) {
    for ( i=0; i<ai[m]-1; i++ ) aj[i] -= 1;
  }
  for ( i=0; i<ai[m]+shift; i++ ) col[aj[i]] += 1;
  ierr = MatCreateSeqAIJ(A->comm,a->n,m,0,col,&C); CHKERRQ(ierr);
  PetscFree(col);
  for ( i=0; i<m; i++ ) {
    len = ai[i+1]-ai[i];
    ierr = MatSetValues(C,len,aj,1,&i,array,INSERT_VALUES); CHKERRQ(ierr);
    array += len; aj += len;
  }
  if (shift) { 
    for ( i=0; i<ai[m]-1; i++ ) aj[i] += 1;
  }

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  if (B != PETSC_NULL) {
    *B = C;
  } else {
    PetscOps       *Abops;
    struct _MatOps *Aops;

    /* This isn't really an in-place transpose */
    PetscFree(a->a); 
    if (!a->singlemalloc) {PetscFree(a->i); PetscFree(a->j);}
    if (a->diag) PetscFree(a->diag);
    if (a->ilen) PetscFree(a->ilen);
    if (a->imax) PetscFree(a->imax);
    if (a->solve_work) PetscFree(a->solve_work);
    if (a->inode.size) PetscFree(a->inode.size);
    PetscFree(a);
 
    /*
        This is horrible, horrible code. We need to keep the 
      A pointers for the bops and ops but copy everything 
      else from C.
    */
    Abops = A->bops;
    Aops  = A->ops;
    PetscMemcpy(A,C,sizeof(struct _p_Mat));
    A->bops = Abops;
    A->ops  = Aops;

    PetscHeaderDestroy(C);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDiagonalScale_SeqAIJ"
int MatDiagonalScale_SeqAIJ(Mat A,Vec ll,Vec rr)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *l,*r,x,*v;
  int        i,j,m = a->m, n = a->n, M, nz = a->nz, *jj,shift = a->indexshift;

  PetscFunctionBegin;
  if (ll) {
    /* The local size is used so that VecMPI can be passed to this routine
       by MatDiagonalScale_MPIAIJ */
    VecGetLocalSize_Fast(ll,m);
    if (m != a->m) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Left scaling vector wrong length");
    VecGetArray_Fast(ll,l); 
    v = a->a;
    for ( i=0; i<m; i++ ) {
      x = l[i];
      M = a->i[i+1] - a->i[i];
      for ( j=0; j<M; j++ ) { (*v++) *= x;} 
    }
    PLogFlops(nz);
  }
  if (rr) {
    VecGetLocalSize_Fast(rr,n);
    if (n != a->n) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Right scaling vector wrong length");
    VecGetArray_Fast(rr,r); 
    v = a->a; jj = a->j;
    for ( i=0; i<nz; i++ ) {
      (*v++) *= r[*jj++ + shift]; 
    }
    PLogFlops(nz);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSubMatrix_SeqAIJ"
int MatGetSubMatrix_SeqAIJ(Mat A,IS isrow,IS iscol,int csize,MatGetSubMatrixCall scall,Mat *B)
{
  Mat_SeqAIJ   *a = (Mat_SeqAIJ *) A->data,*c;
  int          *smap, i, k, kstart, kend, ierr, oldcols = a->n,*lens;
  int          row,mat_i,*mat_j,tcol,first,step,*mat_ilen;
  register int sum,lensi;
  int          *irow, *icol, nrows, ncols, shift = a->indexshift,*ssmap;
  int          *starts,*j_new,*i_new,*aj = a->j, *ai = a->i,ii,*ailen = a->ilen;
  Scalar       *a_new,*mat_a;
  Mat          C;

  PetscFunctionBegin;
  ierr = ISSorted(isrow,(PetscTruth*)&i);CHKERRQ(ierr);
  if (!i) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"ISrow is not sorted");
  ierr = ISSorted(iscol,(PetscTruth*)&i);CHKERRQ(ierr);
  if (!i) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"IScol is not sorted");

  ierr = ISGetIndices(isrow,&irow); CHKERRQ(ierr);
  ierr = ISGetSize(isrow,&nrows); CHKERRQ(ierr);
  ierr = ISGetSize(iscol,&ncols); CHKERRQ(ierr);

  if (ISStrideGetInfo(iscol,&first,&step) && step == 1) { /* no need to sort */
    /* special case of contiguous rows */
    lens   = (int *) PetscMalloc((ncols+nrows+1)*sizeof(int)); CHKPTRQ(lens);
    starts = lens + ncols;
    /* loop over new rows determining lens and starting points */
    for (i=0; i<nrows; i++) {
      kstart  = ai[irow[i]]+shift; 
      kend    = kstart + ailen[irow[i]];
      for ( k=kstart; k<kend; k++ ) {
        if (aj[k]+shift >= first) {
          starts[i] = k;
          break;
	}
      }
      sum = 0;
      while (k < kend) {
        if (aj[k++]+shift >= first+ncols) break;
        sum++;
      }
      lens[i] = sum;
    }
    /* create submatrix */
    if (scall == MAT_REUSE_MATRIX) {
      int n_cols,n_rows;
      ierr = MatGetSize(*B,&n_rows,&n_cols); CHKERRQ(ierr);
      if (n_rows != nrows || n_cols != ncols) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Reused submatrix wrong size");
      ierr = MatZeroEntries(*B); CHKERRQ(ierr);
      C = *B;
    } else {  
      ierr = MatCreateSeqAIJ(A->comm,nrows,ncols,0,lens,&C);CHKERRQ(ierr);
    }
    c = (Mat_SeqAIJ*) C->data;

    /* loop over rows inserting into submatrix */
    a_new    = c->a;
    j_new    = c->j;
    i_new    = c->i;
    i_new[0] = -shift;
    for (i=0; i<nrows; i++) {
      ii    = starts[i];
      lensi = lens[i];
      for ( k=0; k<lensi; k++ ) {
        *j_new++ = aj[ii+k] - first;
      }
      PetscMemcpy(a_new,a->a + starts[i],lensi*sizeof(Scalar));
      a_new      += lensi;
      i_new[i+1]  = i_new[i] + lensi;
      c->ilen[i]  = lensi;
    }
    PetscFree(lens);
  } else {
    ierr = ISGetIndices(iscol,&icol); CHKERRQ(ierr);
    smap  = (int *) PetscMalloc((1+oldcols)*sizeof(int)); CHKPTRQ(smap);
    ssmap = smap + shift;
    lens  = (int *) PetscMalloc((1+nrows)*sizeof(int)); CHKPTRQ(lens);
    PetscMemzero(smap,oldcols*sizeof(int));
    for ( i=0; i<ncols; i++ ) smap[icol[i]] = i+1;
    /* determine lens of each row */
    for (i=0; i<nrows; i++) {
      kstart  = ai[irow[i]]+shift; 
      kend    = kstart + a->ilen[irow[i]];
      lens[i] = 0;
      for ( k=kstart; k<kend; k++ ) {
        if (ssmap[aj[k]]) {
          lens[i]++;
        }
      }
    }
    /* Create and fill new matrix */
    if (scall == MAT_REUSE_MATRIX) {
      c = (Mat_SeqAIJ *)((*B)->data);

      if (c->m  != nrows || c->n != ncols) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Cannot reuse matrix. wrong size");
      if (PetscMemcmp(c->ilen,lens, c->m *sizeof(int))) {
        SETERRQ(PETSC_ERR_ARG_SIZ,0,"Cannot reuse matrix. wrong no of nonzeros");
      }
      PetscMemzero(c->ilen,c->m*sizeof(int));
      C = *B;
    } else {  
      ierr = MatCreateSeqAIJ(A->comm,nrows,ncols,0,lens,&C);CHKERRQ(ierr);
    }
    c = (Mat_SeqAIJ *)(C->data);
    for (i=0; i<nrows; i++) {
      row    = irow[i];
      kstart = ai[row]+shift; 
      kend   = kstart + a->ilen[row];
      mat_i  = c->i[i]+shift;
      mat_j  = c->j + mat_i; 
      mat_a  = c->a + mat_i;
      mat_ilen = c->ilen + i;
      for ( k=kstart; k<kend; k++ ) {
        if ((tcol=ssmap[a->j[k]])) {
          *mat_j++ = tcol - (!shift);
          *mat_a++ = a->a[k];
          (*mat_ilen)++;

        }
      }
    }
    /* Free work space */
    ierr = ISRestoreIndices(iscol,&icol); CHKERRQ(ierr);
    PetscFree(smap); PetscFree(lens);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = ISRestoreIndices(isrow,&irow); CHKERRQ(ierr);
  *B = C;
  PetscFunctionReturn(0);
}

/*
     note: This can only work for identity for row and col. It would 
   be good to check this and otherwise generate an error.
*/
#undef __FUNC__  
#define __FUNC__ "MatILUFactor_SeqAIJ"
int MatILUFactor_SeqAIJ(Mat inA,IS row,IS col,double efill,int fill)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) inA->data;
  int        ierr;
  Mat        outA;

  PetscFunctionBegin;
  if (fill != 0) SETERRQ(PETSC_ERR_SUP,0,"Only fill=0 supported");

  outA          = inA; 
  inA->factor   = FACTOR_LU;
  a->row        = row;
  a->col        = col;

  /* Create the invert permutation so that it can be used in MatLUFactorNumeric() */
  ierr = ISInvertPermutation(col,&(a->icol)); CHKERRQ(ierr);

  if (!a->solve_work) { /* this matrix may have been factored before */
    a->solve_work = (Scalar *) PetscMalloc( (a->m+1)*sizeof(Scalar)); CHKPTRQ(a->solve_work);
  }

  if (!a->diag) {
    ierr = MatMarkDiag_SeqAIJ(inA); CHKERRQ(ierr);
  }
  ierr = MatLUFactorNumeric_SeqAIJ(inA,&outA); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include "pinclude/plapack.h"
#undef __FUNC__  
#define __FUNC__ "MatScale_SeqAIJ"
int MatScale_SeqAIJ(Scalar *alpha,Mat inA)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) inA->data;
  int        one = 1;

  PetscFunctionBegin;
  BLscal_( &a->nz, alpha, a->a, &one );
  PLogFlops(a->nz);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSubMatrices_SeqAIJ"
int MatGetSubMatrices_SeqAIJ(Mat A,int n, IS *irow,IS *icol,MatGetSubMatrixCall scall,Mat **B)
{
  int ierr,i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    *B = (Mat *) PetscMalloc( (n+1)*sizeof(Mat) ); CHKPTRQ(*B);
  }

  for ( i=0; i<n; i++ ) {
    ierr = MatGetSubMatrix_SeqAIJ(A,irow[i],icol[i],PETSC_DECIDE,scall,&(*B)[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetBlockSize_SeqAIJ"
int MatGetBlockSize_SeqAIJ(Mat A, int *bs)
{
  PetscFunctionBegin;
  *bs = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatIncreaseOverlap_SeqAIJ"
int MatIncreaseOverlap_SeqAIJ(Mat A, int is_max, IS *is, int ov)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        shift, row, i,j,k,l,m,n, *idx,ierr, *nidx, isz, val;
  int        start, end, *ai, *aj;
  BT         table;

  PetscFunctionBegin;
  shift = a->indexshift;
  m     = a->m;
  ai    = a->i;
  aj    = a->j+shift;

  if (ov < 0)  SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"illegal overlap value used");

  nidx  = (int *) PetscMalloc((m+1)*sizeof(int)); CHKPTRQ(nidx); 
  ierr  = BTCreate(m,table); CHKERRQ(ierr);

  for ( i=0; i<is_max; i++ ) {
    /* Initialize the two local arrays */
    isz  = 0;
    BTMemzero(m,table);
                 
    /* Extract the indices, assume there can be duplicate entries */
    ierr = ISGetIndices(is[i],&idx);  CHKERRQ(ierr);
    ierr = ISGetSize(is[i],&n);  CHKERRQ(ierr);
    
    /* Enter these into the temp arrays. I.e., mark table[row], enter row into new index */
    for ( j=0; j<n ; ++j){
      if(!BTLookupSet(table, idx[j])) { nidx[isz++] = idx[j];}
    }
    ierr = ISRestoreIndices(is[i],&idx);  CHKERRQ(ierr);
    ierr = ISDestroy(is[i]); CHKERRQ(ierr);
    
    k = 0;
    for ( j=0; j<ov; j++){ /* for each overlap */
      n = isz;
      for ( ; k<n ; k++){ /* do only those rows in nidx[k], which are not done yet */
        row   = nidx[k];
        start = ai[row];
        end   = ai[row+1];
        for ( l = start; l<end ; l++){
          val = aj[l] + shift;
          if (!BTLookupSet(table,val)) {nidx[isz++] = val;}
        }
      }
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF, isz, nidx, (is+i)); CHKERRQ(ierr);
  }
  BTDestroy(table);
  PetscFree(nidx);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "MatPermute_SeqAIJ"
int MatPermute_SeqAIJ(Mat A, IS rowp, IS colp, Mat *B)
{ 
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *vwork;
  int        i, ierr, nz, m = a->m, n = a->n, *cwork;
  int        *row,*col,*cnew,j,*lens;
  IS         icolp,irowp;

  PetscFunctionBegin;
  ierr = ISInvertPermutation(rowp,&irowp); CHKERRQ(ierr);
  ierr = ISGetIndices(irowp,&row); CHKERRQ(ierr);
  ierr = ISInvertPermutation(colp,&icolp); CHKERRQ(ierr);
  ierr = ISGetIndices(icolp,&col); CHKERRQ(ierr);
  
  /* determine lengths of permuted rows */
  lens = (int *) PetscMalloc( (m+1)*sizeof(int) ); CHKPTRQ(lens);
  for (i=0; i<m; i++ ) {
    lens[row[i]] = a->i[i+1] - a->i[i];
  }
  ierr = MatCreateSeqAIJ(A->comm,m,n,0,lens,B);CHKERRQ(ierr);
  PetscFree(lens);

  cnew = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(cnew);
  for (i=0; i<m; i++) {
    ierr = MatGetRow(A,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
    for (j=0; j<nz; j++ ) { cnew[j] = col[cwork[j]];}
    ierr = MatSetValues(*B,1,&row[i],nz,cnew,vwork,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
  }
  PetscFree(cnew);
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = ISRestoreIndices(irowp,&row); CHKERRQ(ierr);
  ierr = ISRestoreIndices(icolp,&col); CHKERRQ(ierr);
  ierr = ISDestroy(irowp); CHKERRQ(ierr);
  ierr = ISDestroy(icolp); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPrintHelp_SeqAIJ"
int MatPrintHelp_SeqAIJ(Mat A)
{
  static int called = 0; 
  MPI_Comm   comm = A->comm;

  PetscFunctionBegin;
  if (called) {PetscFunctionReturn(0);} else called = 1;
  (*PetscHelpPrintf)(comm," Options for MATSEQAIJ and MATMPIAIJ matrix formats (the defaults):\n");
  (*PetscHelpPrintf)(comm,"  -mat_lu_pivotthreshold <threshold>: Set pivoting threshold\n");
  (*PetscHelpPrintf)(comm,"  -mat_aij_oneindex: internal indices begin at 1 instead of the default 0.\n");
  (*PetscHelpPrintf)(comm,"  -mat_aij_no_inode: Do not use inodes\n");
  (*PetscHelpPrintf)(comm,"  -mat_aij_inode_limit <limit>: Set inode limit (max limit=5)\n");
#if defined(HAVE_ESSL)
  (*PetscHelpPrintf)(comm,"  -mat_aij_essl: Use IBM sparse LU factorization and solve.\n");
#endif
  PetscFunctionReturn(0);
}
extern int MatEqual_SeqAIJ(Mat A,Mat B, PetscTruth* flg);
extern int MatFDColoringCreate_SeqAIJ(Mat,ISColoring,MatFDColoring);
extern int MatColoringPatch_SeqAIJ(Mat,int,int *,ISColoring *);

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps = {MatSetValues_SeqAIJ,
       MatGetRow_SeqAIJ,MatRestoreRow_SeqAIJ,
       MatMult_SeqAIJ,MatMultAdd_SeqAIJ,
       MatMultTrans_SeqAIJ,MatMultTransAdd_SeqAIJ,
       MatSolve_SeqAIJ,MatSolveAdd_SeqAIJ,
       MatSolveTrans_SeqAIJ,MatSolveTransAdd_SeqAIJ,
       MatLUFactor_SeqAIJ,0,
       MatRelax_SeqAIJ,
       MatTranspose_SeqAIJ,
       MatGetInfo_SeqAIJ,MatEqual_SeqAIJ,
       MatGetDiagonal_SeqAIJ,MatDiagonalScale_SeqAIJ,MatNorm_SeqAIJ,
       0,MatAssemblyEnd_SeqAIJ,
       MatCompress_SeqAIJ,
       MatSetOption_SeqAIJ,MatZeroEntries_SeqAIJ,MatZeroRows_SeqAIJ,
       MatLUFactorSymbolic_SeqAIJ,MatLUFactorNumeric_SeqAIJ,0,0,
       MatGetSize_SeqAIJ,MatGetSize_SeqAIJ,MatGetOwnershipRange_SeqAIJ,
       MatILUFactorSymbolic_SeqAIJ,0,
       0,0,
       MatConvertSameType_SeqAIJ,0,0,
       MatILUFactor_SeqAIJ,0,0,
       MatGetSubMatrices_SeqAIJ,MatIncreaseOverlap_SeqAIJ,
       MatGetValues_SeqAIJ,0,
       MatPrintHelp_SeqAIJ,
       MatScale_SeqAIJ,0,0,
       MatILUDTFactor_SeqAIJ,
       MatGetBlockSize_SeqAIJ,
       MatGetRowIJ_SeqAIJ,
       MatRestoreRowIJ_SeqAIJ,
       MatGetColumnIJ_SeqAIJ,
       MatRestoreColumnIJ_SeqAIJ,
       MatFDColoringCreate_SeqAIJ,
       MatColoringPatch_SeqAIJ,
       0,
       MatPermute_SeqAIJ};

extern int MatUseSuperLU_SeqAIJ(Mat);
extern int MatUseEssl_SeqAIJ(Mat);
extern int MatUseDXML_SeqAIJ(Mat);

#undef __FUNC__  
#define __FUNC__ "MatCreateSeqAIJ"
/*@C
   MatCreateSeqAIJ - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format).  For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameter nz
   (or the array nzz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Input Parameters:
.  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
.  nzz - array containing the number of nonzeros in the various rows
         (possibly different for each row) or PETSC_NULL

   Output Parameter:
.  A - the matrix 

   Notes:
   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=PETSC_NULL for PETSc to control dynamic memory 
   allocation.  For large problems you MUST preallocate memory or you 
   will get TERRIBLE performance, see the users' manual chapter on matrices.

   By default, this format uses inodes (identical nodes) when possible, to 
   improve numerical efficiency of matrix-vector products and solves. We 
   search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Options Database Keys:
$    -mat_aij_no_inode  - Do not use inodes
$    -mat_aij_inode_limit <limit> - Set inode limit.
$        (max limit=5)
$    -mat_aij_oneindex - Internally use indexing starting at 1
$        rather than 0.  Note: When calling MatSetValues(),
$        the user still MUST index entries starting at 0!

.seealso: MatCreate(), MatCreateMPIAIJ(), MatSetValues()
@*/
int MatCreateSeqAIJ(MPI_Comm comm,int m,int n,int nz,int *nnz, Mat *A)
{
  Mat        B;
  Mat_SeqAIJ *b;
  int        i, len, ierr, flg,size;

  PetscFunctionBegin;
  MPI_Comm_size(comm,&size);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Comm must be of size 1");

  *A                  = 0;
  PetscHeaderCreate(B,_p_Mat,struct _MatOps,MAT_COOKIE,MATSEQAIJ,comm,MatDestroy,MatView);
  PLogObjectCreate(B);
  B->data             = (void *) (b = PetscNew(Mat_SeqAIJ)); CHKPTRQ(b);
  PetscMemzero(b,sizeof(Mat_SeqAIJ));
  PetscMemcpy(B->ops,&MatOps,sizeof(struct _MatOps));
  B->destroy          = MatDestroy_SeqAIJ;
  B->view             = MatView_SeqAIJ;
  B->factor           = 0;
  B->lupivotthreshold = 1.0;
  B->mapping          = 0;
  ierr = OptionsGetDouble(PETSC_NULL,"-mat_lu_pivotthreshold",&B->lupivotthreshold,
                          &flg); CHKERRQ(ierr);
  b->ilu_preserve_row_sums = PETSC_FALSE;
  ierr = OptionsHasName(PETSC_NULL,"-pc_ilu_preserve_row_sums",
                        (int*) &b->ilu_preserve_row_sums); CHKERRQ(ierr);
  b->row              = 0;
  b->col              = 0;
  b->icol             = 0;
  b->indexshift       = 0;
  b->reallocs         = 0;
  ierr = OptionsHasName(PETSC_NULL,"-mat_aij_oneindex", &flg); CHKERRQ(ierr);
  if (flg) b->indexshift = -1;
  
  b->m = m; B->m = m; B->M = m;
  b->n = n; B->n = n; B->N = n;
  b->imax = (int *) PetscMalloc( (m+1)*sizeof(int) ); CHKPTRQ(b->imax);
  if (nnz == PETSC_NULL) {
    if (nz == PETSC_DEFAULT) nz = 10;
    else if (nz <= 0)        nz = 1;
    for ( i=0; i<m; i++ ) b->imax[i] = nz;
    nz = nz*m;
  } else {
    nz = 0;
    for ( i=0; i<m; i++ ) {b->imax[i] = nnz[i]; nz += nnz[i];}
  }

  /* allocate the matrix space */
  len     = nz*(sizeof(int) + sizeof(Scalar)) + (b->m+1)*sizeof(int);
  b->a  = (Scalar *) PetscMalloc( len ); CHKPTRQ(b->a);
  b->j  = (int *) (b->a + nz);
  PetscMemzero(b->j,nz*sizeof(int));
  b->i  = b->j + nz;
  b->singlemalloc = 1;

  b->i[0] = -b->indexshift;
  for (i=1; i<m+1; i++) {
    b->i[i] = b->i[i-1] + b->imax[i-1];
  }

  /* b->ilen will count nonzeros in each row so far. */
  b->ilen = (int *) PetscMalloc((m+1)*sizeof(int)); 
  PLogObjectMemory(B,len+2*(m+1)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_SeqAIJ));
  for ( i=0; i<b->m; i++ ) { b->ilen[i] = 0;}

  b->nz               = 0;
  b->maxnz            = nz;
  b->sorted           = 0;
  b->roworiented      = 1;
  b->nonew            = 0;
  b->diag             = 0;
  b->solve_work       = 0;
  b->spptr            = 0;
  b->inode.node_count = 0;
  b->inode.size       = 0;
  b->inode.limit      = 5;
  b->inode.max_limit  = 5;
  B->info.nz_unneeded = (double)b->maxnz;

  *A = B;

  /*  SuperLU is not currently supported through PETSc */
#if defined(HAVE_SUPERLU)
  ierr = OptionsHasName(PETSC_NULL,"-mat_aij_superlu", &flg); CHKERRQ(ierr);
  if (flg) { ierr = MatUseSuperLU_SeqAIJ(B); CHKERRQ(ierr); }
#endif
  ierr = OptionsHasName(PETSC_NULL,"-mat_aij_essl", &flg); CHKERRQ(ierr);
  if (flg) { ierr = MatUseEssl_SeqAIJ(B); CHKERRQ(ierr); }
  ierr = OptionsHasName(PETSC_NULL,"-mat_aij_dxml", &flg); CHKERRQ(ierr);
  if (flg) {
    if (!b->indexshift) SETERRQ( PETSC_ERR_LIB,0,"need -mat_aij_oneindex with -mat_aij_dxml");
    ierr = MatUseDXML_SeqAIJ(B); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-help", &flg); CHKERRQ(ierr);
  if (flg) {ierr = MatPrintHelp(B); CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatConvertSameType_SeqAIJ"
int MatConvertSameType_SeqAIJ(Mat A,Mat *B,int cpvalues)
{
  Mat        C;
  Mat_SeqAIJ *c,*a = (Mat_SeqAIJ *) A->data;
  int        i,len, m = a->m,shift = a->indexshift;

  PetscFunctionBegin;
  *B = 0;
  PetscHeaderCreate(C,_p_Mat,struct _MatOps,MAT_COOKIE,MATSEQAIJ,A->comm,MatDestroy,MatView);
  PLogObjectCreate(C);
  C->data       = (void *) (c = PetscNew(Mat_SeqAIJ)); CHKPTRQ(c);
  PetscMemcpy(C->ops,A->ops,sizeof(struct _MatOps));
  C->destroy    = MatDestroy_SeqAIJ;
  C->view       = MatView_SeqAIJ;
  C->factor     = A->factor;
  c->row        = 0;
  c->col        = 0;
  c->icol       = 0;
  c->indexshift = shift;
  C->assembled  = PETSC_TRUE;

  c->m = C->m   = a->m;
  c->n = C->n   = a->n;
  C->M          = a->m;
  C->N          = a->n;

  c->imax       = (int *) PetscMalloc((m+1)*sizeof(int)); CHKPTRQ(c->imax);
  c->ilen       = (int *) PetscMalloc((m+1)*sizeof(int)); CHKPTRQ(c->ilen);
  for ( i=0; i<m; i++ ) {
    c->imax[i] = a->imax[i];
    c->ilen[i] = a->ilen[i]; 
  }

  /* allocate the matrix space */
  c->singlemalloc = 1;
  len     = (m+1)*sizeof(int)+(a->i[m])*(sizeof(Scalar)+sizeof(int));
  c->a  = (Scalar *) PetscMalloc( len ); CHKPTRQ(c->a);
  c->j  = (int *) (c->a + a->i[m] + shift);
  c->i  = c->j + a->i[m] + shift;
  PetscMemcpy(c->i,a->i,(m+1)*sizeof(int));
  if (m > 0) {
    PetscMemcpy(c->j,a->j,(a->i[m]+shift)*sizeof(int));
    if (cpvalues == COPY_VALUES) {
      PetscMemcpy(c->a,a->a,(a->i[m]+shift)*sizeof(Scalar));
    }
  }

  PLogObjectMemory(C,len+2*(m+1)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_SeqAIJ));  
  c->sorted      = a->sorted;
  c->roworiented = a->roworiented;
  c->nonew       = a->nonew;
  c->ilu_preserve_row_sums = a->ilu_preserve_row_sums;

  if (a->diag) {
    c->diag = (int *) PetscMalloc( (m+1)*sizeof(int) ); CHKPTRQ(c->diag);
    PLogObjectMemory(C,(m+1)*sizeof(int));
    for ( i=0; i<m; i++ ) {
      c->diag[i] = a->diag[i];
    }
  } else c->diag          = 0;
  c->inode.limit        = a->inode.limit;
  c->inode.max_limit    = a->inode.max_limit;
  if (a->inode.size){
    c->inode.size       = (int *) PetscMalloc( (m+1)*sizeof(int) ); CHKPTRQ(c->inode.size);
    c->inode.node_count = a->inode.node_count;
    PetscMemcpy( c->inode.size, a->inode.size, (m+1)*sizeof(int));
  } else {
    c->inode.size       = 0;
    c->inode.node_count = 0;
  }
  c->nz                 = a->nz;
  c->maxnz              = a->maxnz;
  c->solve_work         = 0;
  c->spptr              = 0;      /* Dangerous -I'm throwing away a->spptr */

  *B = C;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatLoad_SeqAIJ"
int MatLoad_SeqAIJ(Viewer viewer,MatType type,Mat *A)
{
  Mat_SeqAIJ   *a;
  Mat          B;
  int          i, nz, ierr, fd, header[4],size,*rowlengths = 0,M,N,shift;
  MPI_Comm     comm;
  
  PetscFunctionBegin;
  PetscObjectGetComm((PetscObject) viewer,&comm);
  MPI_Comm_size(comm,&size);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_SIZ,0,"view must have one processor");
  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,header,4,PETSC_INT); CHKERRQ(ierr);
  if (header[0] != MAT_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,0,"not matrix object in file");
  M = header[1]; N = header[2]; nz = header[3];

  if (nz < 0) {
    SETERRQ(PETSC_ERR_FILE_UNEXPECTED,1,"Matrix stored in special format on disk, cannot load as SeqAIJ");
  }

  /* read in row lengths */
  rowlengths = (int*) PetscMalloc( M*sizeof(int) ); CHKPTRQ(rowlengths);
  ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT); CHKERRQ(ierr);

  /* create our matrix */
  ierr = MatCreateSeqAIJ(comm,M,N,0,rowlengths,A); CHKERRQ(ierr);
  B = *A;
  a = (Mat_SeqAIJ *) B->data;
  shift = a->indexshift;

  /* read in column indices and adjust for Fortran indexing*/
  ierr = PetscBinaryRead(fd,a->j,nz,PETSC_INT); CHKERRQ(ierr);
  if (shift) {
    for ( i=0; i<nz; i++ ) {
      a->j[i] += 1;
    }
  }

  /* read in nonzero values */
  ierr = PetscBinaryRead(fd,a->a,nz,PETSC_SCALAR); CHKERRQ(ierr);

  /* set matrix "i" values */
  a->i[0] = -shift;
  for ( i=1; i<= M; i++ ) {
    a->i[i]      = a->i[i-1] + rowlengths[i-1];
    a->ilen[i-1] = rowlengths[i-1];
  }
  PetscFree(rowlengths);   

  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatEqual_SeqAIJ"
int MatEqual_SeqAIJ(Mat A,Mat B, PetscTruth* flg)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data, *b = (Mat_SeqAIJ *)B->data;

  PetscFunctionBegin;
  if (B->type !=MATSEQAIJ)SETERRQ(PETSC_ERR_ARG_INCOMP,0,"Matrices must be same type");

  /* If the  matrix dimensions are not equal, or no of nonzeros or shift */
  if ((a->m != b->m ) || (a->n !=b->n) ||( a->nz != b->nz)|| 
      (a->indexshift != b->indexshift)) {
    *flg = PETSC_FALSE; PetscFunctionReturn(0); 
  }
  
  /* if the a->i are the same */
  if (PetscMemcmp(a->i,b->i,(a->m+1)*sizeof(int))) { 
    *flg = PETSC_FALSE; PetscFunctionReturn(0);
  }
  
  /* if a->j are the same */
  if (PetscMemcmp(a->j, b->j, (a->nz)*sizeof(int))) { 
    *flg = PETSC_FALSE; PetscFunctionReturn(0);
  }
  
  /* if a->a are the same */
  if (PetscMemcmp(a->a, b->a, (a->nz)*sizeof(Scalar))) {
    *flg = PETSC_FALSE; PetscFunctionReturn(0);
  }
  *flg = PETSC_TRUE; 
  PetscFunctionReturn(0);
  
}
