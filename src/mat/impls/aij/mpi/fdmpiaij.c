

#ifndef lint
static char vcid[] = "$Id: fdmpiaij.c,v 1.8 1997/02/22 02:25:15 bsmith Exp bsmith $";
#endif

#include "src/mat/impls/aij/mpi/mpiaij.h"
#include "src/vec/vecimpl.h"
#include "petsc.h"

extern int CreateColmap_MPIAIJ_Private(Mat);
extern int MatGetColumnIJ_SeqAIJ(Mat,int,PetscTruth,int*,int**,int**,PetscTruth*);
extern int MatRestoreColumnIJ_SeqAIJ(Mat,int,PetscTruth,int*,int**,int**,PetscTruth*);

#undef __FUNC__  
#define __FUNC__ "MatFDColoringCreate_MPIAIJ" /* ADIC Ignore */
int MatFDColoringCreate_MPIAIJ(Mat mat,ISColoring iscoloring,MatFDColoring c)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  int        i,*is,n,nrows,j,k,m,*rows = 0,ierr,*A_ci,*A_cj,ncols,col,flg;
  int        nis = iscoloring->n,*ncolsonproc,size,nctot,*cols,*disp,*B_ci,*B_cj;
  int        *rowhit, M = mat->m,cstart = aij->cstart, cend = aij->cend,colb;
  int        *columnsforrow;
  IS         *isa = iscoloring->is;
  PetscTruth done;

  c->ncolors       = nis;
  c->ncolumns      = (int *) PetscMalloc( nis*sizeof(int) );   CHKPTRQ(c->ncolumns);
  c->columns       = (int **) PetscMalloc( nis*sizeof(int *)); CHKPTRQ(c->columns); 
  c->nrows         = (int *) PetscMalloc( nis*sizeof(int) );   CHKPTRQ(c->nrows);
  c->rows          = (int **) PetscMalloc( nis*sizeof(int *)); CHKPTRQ(c->rows);
  c->columnsforrow = (int **) PetscMalloc( nis*sizeof(int *)); CHKPTRQ(c->columnsforrow);
  PLogObjectMemory(c,5*nis*sizeof(int));

  /* Allow access to data structures of local part of matrix */
  if (!aij->colmap) {
    ierr = CreateColmap_MPIAIJ_Private(mat);CHKERRQ(ierr);
  }
  /*
      Calls the _SeqAIJ() version of these routines to make sure it does not 
     get the reduced (by inodes) version of I and J
  */
  ierr = MatGetColumnIJ_SeqAIJ(aij->A,0,PETSC_FALSE,&ncols,&A_ci,&A_cj,&done); CHKERRQ(ierr); 
  ierr = MatGetColumnIJ_SeqAIJ(aij->B,0,PETSC_FALSE,&ncols,&B_ci,&B_cj,&done); CHKERRQ(ierr); 

  MPI_Comm_size(mat->comm,&size);
  ncolsonproc = (int *) PetscMalloc( 2*size*sizeof(int *) ); CHKPTRQ(ncolsonproc);
  disp        = ncolsonproc + size;

  rowhit        = (int *) PetscMalloc( (M+1)*sizeof(int) ); CHKPTRQ(rowhit);
  columnsforrow = (int *) PetscMalloc( (M+1)*sizeof(int) );CHKPTRQ(columnsforrow);

  /*
     Temporary option to allow for debugging/testing
  */
  ierr = OptionsHasName(0,"-matfdcoloring_slow",&flg);

  for ( i=0; i<nis; i++ ) {
    ierr = ISGetSize(isa[i],&n); CHKERRQ(ierr);
    ierr = ISGetIndices(isa[i],&is); CHKERRQ(ierr);
    c->ncolumns[i] = n;
    c->ncolumns[i] = n;
    if (n) {
      c->columns[i]  = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(c->columns[i]);
      PLogObjectMemory(c,n*sizeof(int));
      PetscMemcpy(c->columns[i],is,n*sizeof(int)); 
    } else {
      c->columns[i]  = 0;
    }

    /* Determine the total (parallel) number of columns of this color */
    MPI_Allgather(&n,1,MPI_INT,ncolsonproc,1,MPI_INT,mat->comm);
    nctot = 0; for ( j=0; j<size; j++ ) {nctot += ncolsonproc[j];}
    if (!nctot) SETERRQ(1,0,"Invalid coloring");

    disp[0] = 0;
    for ( j=1; j<size; j++ ) {
      disp[j] = disp[j-1] + ncolsonproc[j-1];
    }
    
    /* Get complete list of columns for color on each processor */
    cols = (int *) PetscMalloc( nctot*sizeof(int) ); CHKPTRQ(cols);
    MPI_Allgatherv(is,n,MPI_INT,cols,ncolsonproc,disp,MPI_INT,mat->comm);

/*
for ( j=0; j<nctot; j++ ) {
  printf("color %d %d col %d\n",i,j,cols[j]);
}
*/

    /*
       Mark all rows affect by these columns
    */
    if (flg) {/*-----------------------------------------------------------------------------*/
      /* crude, slow version */
      PetscMemzero(rowhit,M*sizeof(int));
      /* loop over columns*/
      for ( j=0; j<nctot; j++ ) {
        col  = cols[j];
        if (col >= cstart && col < cend) {
          /* column is in diagonal block of matrix */
          rows = A_cj + A_ci[col-cstart]; 
          m    = A_ci[col-cstart+1] - A_ci[col-cstart];
        } else {
          colb = aij->colmap[col] - 1;
          if (colb == -1) {
            m = 0; 
          } else {
            rows = B_cj + B_ci[colb]; 
            m    = B_ci[colb+1] - B_ci[colb];
          }
        }
        /* loop over columns marking them in rowhit */
        for ( k=0; k<m; k++ ) {
          rowhit[*rows++] = col + 1;
        }
      }

/*
printf("for col %d found rows \n",i);
for ( j=0; j<M; j++ ) printf("rhow hit %d %d\n",j,rowhit[j]);
*/

      /* count the number of hits */
      nrows = 0;
      for ( j=0; j<M; j++ ) {
        if (rowhit[j]) nrows++;
      }
      c->nrows[i]         = nrows;
      c->rows[i]          = (int *) PetscMalloc((nrows+1)*sizeof(int)); CHKPTRQ(c->rows[i]);
      c->columnsforrow[i] = (int *) PetscMalloc((nrows+1)*sizeof(int)); CHKPTRQ(c->columnsforrow[i]);
      PLogObjectMemory(c,2*(nrows+1)*sizeof(int));
      nrows = 0;
      for ( j=0; j<M; j++ ) {
        if (rowhit[j]) {
          c->rows[i][nrows]           = j;
          c->columnsforrow[i][nrows] = rowhit[j] - 1;
          nrows++;
        }
      }
    } else {/*-------------------------------------------------------------------------------*/
      /* efficient version, using rowhit as a linked list */
      int currentcol,fm,mfm;
      rowhit[M] = M;
      nrows     = 0;
      /* loop over columns*/
      for ( j=0; j<nctot; j++ ) {
        col  = cols[j];
        if (col >= cstart && col < cend) {
          /* column is in diagonal block of matrix */
          rows = A_cj + A_ci[col-cstart]; 
          m    = A_ci[col-cstart+1] - A_ci[col-cstart];
        } else {
          colb = aij->colmap[col] - 1;
          if (colb == -1) {
            m = 0; 
          } else {
            rows = B_cj + B_ci[colb]; 
            m    = B_ci[colb+1] - B_ci[colb];
          }
        }
        /* loop over columns marking them in rowhit */
        fm    = M; /* fm points to first entry in linked list */
        for ( k=0; k<m; k++ ) {
          currentcol = *rows++;
	  /* is it already in the list? */
          do {
            mfm  = fm;
            fm   = rowhit[fm];
          } while (fm < currentcol);
          /* not in list so add it */
          if (fm != currentcol) {
            nrows++;
            columnsforrow[currentcol] = col;
            /* next three lines insert new entry into linked list */
            rowhit[mfm]               = currentcol;
            rowhit[currentcol]        = fm;
            fm                        = currentcol; 
            /* fm points to present position in list since we know the columns are sorted */
          } else {
            SETERRQ(1,0,"Invalid coloring");
          }
        }
      }
      c->nrows[i]         = nrows;
      c->rows[i]          = (int *)PetscMalloc((nrows+1)*sizeof(int));CHKPTRQ(c->rows[i]);
      c->columnsforrow[i] = (int *)PetscMalloc((nrows+1)*sizeof(int));CHKPTRQ(c->columnsforrow[i]);
      PLogObjectMemory(c,(nrows+1)*sizeof(int));
      /* now store the linked list of rows into c->rows[i] */
      nrows = 0;
      fm    = rowhit[M];
      do {
        c->rows[i][nrows]            = fm;
        c->columnsforrow[i][nrows++] = columnsforrow[fm];
        fm                           = rowhit[fm];
      } while (fm < M);
    } /* ---------------------------------------------------------------------------------------*/
    PetscFree(cols);
  }
  PetscFree(rowhit);
  PetscFree(columnsforrow);
  PetscFree(ncolsonproc);
  ierr = MatRestoreColumnIJ_SeqAIJ(aij->A,0,PETSC_FALSE,&ncols,&A_ci,&A_cj,&done); CHKERRQ(ierr); 
  ierr = MatRestoreColumnIJ_SeqAIJ(aij->B,0,PETSC_FALSE,&ncols,&B_ci,&B_cj,&done); CHKERRQ(ierr); 

  c->scale  = (Scalar *) PetscMalloc( 2*mat->N*sizeof(Scalar) ); CHKPTRQ(c->scale);
  PLogObjectMemory(c,2*mat->N*sizeof(Scalar));
  c->wscale = c->scale + mat->N;
  return 0;
}

