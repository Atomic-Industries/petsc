
#ifndef lint
static char vcid[] = "$Id: mmbdiag.c,v 1.22 1996/08/13 02:41:06 bsmith Exp bsmith $";
#endif

/*
   Support for the MPIBDIAG matrix-vector multiply
*/
#include "src/mat/impls/bdiag/mpi/mpibdiag.h"
#include "src/vec/vecimpl.h"

int MatSetUpMultiply_MPIBDiag(Mat mat)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  Mat_SeqBDiag *lmbd = (Mat_SeqBDiag *) mbd->A->data;
  int          ierr, N = mbd->N, *indices, *garray, ec=0;
  int          bs = lmbd->bs, d, i, j, diag;
  IS           tofrom;
  Vec          gvec;

  /* We make an array as long as the number of columns */
  /* mark those columns that are in mbd->A */
  indices = (int *) PetscMalloc( N*sizeof(int) ); CHKPTRQ(indices);
  PetscMemzero(indices,N*sizeof(int));

  if (bs == 1) {
    for (d=0; d<lmbd->nd; d++) {
      diag = lmbd->diag[d];
      if (diag > 0) { /* col = loc */
        for (j=0; j<lmbd->bdlen[d]; j++) {
          if (!indices[j]) ec++; 
          indices[j] = 1;
        }
      } else { /* col = loc-diag */
        for (j=0; j<lmbd->bdlen[d]; j++) {
          if (!indices[j-diag]) ec++; 
          indices[j-diag] = 1;
        }
      }
    }
  } else {
    for (d=0; d<lmbd->nd; d++) {
      diag = lmbd->diag[d];
      if (diag > 0) { /* col = loc */
        for (j=0; j<lmbd->bdlen[d]; j++) {
          if (!indices[bs*j]) ec += bs; 
          for (i=0; i<bs; i++) indices[bs*j+i] = 1;
        }
      } else { /* col = loc-diag */
        for (j=0; j<lmbd->bdlen[d]; j++) {
          if (!indices[bs*(j-diag)]) ec += bs; 
          for (i=0; i<bs; i++) indices[bs*(j-diag)+i] = 1;
        }
      }
    }
  }

  /* form array of columns we need */
  garray = (int *) PetscMalloc( (ec+1)*sizeof(int) ); CHKPTRQ(garray);
  ec = 0;
  for ( i=0; i<N; i++ ) {
    if (indices[i]) garray[ec++] = i;
  }
  PetscFree(indices);

  /* create local vector that is used to scatter into */
  ierr = VecCreateSeq(MPI_COMM_SELF,N,&mbd->lvec); CHKERRQ(ierr);

  /* create temporary index set for building scatter-gather */
  ierr = ISCreateGeneral(MPI_COMM_SELF,ec,garray,&tofrom); CHKERRQ(ierr);
  PetscFree(garray);

  /* create temporary global vector to generate scatter context */
  /* this is inefficient, but otherwise we must do either 
     1) save garray until the first actual scatter when the vector is known or
     2) have another way of generating a scatter context without a vector.*/
  /*
     This is not correct for a rectangular matrix mbd->m? 
  */
  ierr = VecCreateMPI(mat->comm,mbd->m,mbd->N,&gvec); CHKERRQ(ierr);

  /* generate the scatter context */
  ierr = VecScatterCreate(gvec,tofrom,mbd->lvec,tofrom,&mbd->Mvctx);CHKERRQ(ierr);
  PLogObjectParent(mat,mbd->Mvctx);
  PLogObjectParent(mat,mbd->lvec);
  PLogObjectParent(mat,tofrom);
  PLogObjectParent(mat,gvec);

  ierr = ISDestroy(tofrom); CHKERRQ(ierr);
  ierr = VecDestroy(gvec); CHKERRQ(ierr);
  return 0;
}
