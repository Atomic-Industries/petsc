#ifndef lint
static char vcid[] = "$Id: mmaij.c,v 1.15 1995/07/20 04:26:23 bsmith Exp bsmith $";
#endif


/*
   Support for the parallel AIJ matrix vector multiply
*/
#include "mpiaij.h"
#include "vec/vecimpl.h"
#include "../seq/aij.h"

int MatSetUpMultiply_MPIAIJ(Mat mat)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  Mat_AIJ    *B = (Mat_AIJ *) (aij->B->data);  
  int        N = aij->N,i,j,*indices,*aj = B->j;
  int        ierr,ec = 0,*garray;
  IS         from,to;
  Vec        gvec;

  /* For the first stab we make an array as long as the number of columns */
  /* mark those columns that are in aij->B */
  indices = (int *) PETSCMALLOC( N*sizeof(int) ); CHKPTRQ(indices);
  PETSCMEMSET(indices,0,N*sizeof(int));
  for ( i=0; i<B->m; i++ ) {
    for ( j=0; j<B->ilen[i]; j++ ) {
     if (!indices[aj[B->i[i] - 1 + j]-1]) ec++; 
     indices[aj[B->i[i] - 1 + j]-1] = 1;}
  }

  /* form array of columns we need */
  garray = (int *) PETSCMALLOC( (ec+1)*sizeof(int) ); CHKPTRQ(garray);
  ec = 0;
  for ( i=0; i<N; i++ ) {
    if (indices[i]) garray[ec++] = i;
  }

  /* make indices now point into garray */
  for ( i=0; i<ec; i++ ) {
    indices[garray[i]] = i+1;
  }

  /* compact out the extra columns in B */
  for ( i=0; i<B->m; i++ ) {
    for ( j=0; j<B->ilen[i]; j++ ) {
      aj[B->i[i] - 1 + j] = indices[aj[B->i[i] - 1 + j]-1];
    }
  }
  B->n = ec;
  PETSCFREE(indices);
  
  /* create local vector that is used to scatter into */
  ierr = VecCreateSequential(MPI_COMM_SELF,ec,&aij->lvec); CHKERRQ(ierr);

  /* create two temporary Index sets for build scatter gather */
  ierr = ISCreateSequential(MPI_COMM_SELF,ec,garray,&from); CHKERRQ(ierr);
  ierr = ISCreateStrideSequential(MPI_COMM_SELF,ec,0,1,&to); CHKERRQ(ierr);

  /* create temporary global vector to generate scatter context */
  /* this is inefficient, but otherwise we must do either 
     1) save garray until the first actual scatter when the vector is known or
     2) have another way of generating a scatter context without a vector.*/
  ierr = VecCreateMPI(mat->comm,aij->n,aij->N,&gvec); CHKERRQ(ierr);

  /* gnerate the scatter context */
  ierr = VecScatterCtxCreate(gvec,from,aij->lvec,to,&aij->Mvctx); CHKERRQ(ierr);
  PLogObjectParent(mat,aij->Mvctx);
  PLogObjectParent(mat,aij->lvec);
  PLogObjectParent(mat,from);
  PLogObjectParent(mat,to);
  aij->garray = garray;
  PLogObjectMemory(mat,(ec+1)*sizeof(int));
  ierr = ISDestroy(from); CHKERRQ(ierr);
  ierr = ISDestroy(to); CHKERRQ(ierr);
  ierr = VecDestroy(gvec);
  return 0;
}


/*
     Takes the local part of an already assembled MPIAIJ matrix
   and disassembles it. This is to allow new nonzeros into the matrix
   that require more communication in the matrix vector multiply. 
   Thus certain data-structures must be rebuilt.

   Kind of slow! But that's what application programmers get when 
   they are sloppy.
*/
int DisAssemble_MPIAIJ(Mat A)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) A->data;
  Mat        B = aij->B,Bnew;
  Mat_AIJ    *Baij = (Mat_AIJ*)B->data;
  int        ierr,i,j,m=Baij->m,n = aij->N,col,ct = 0,*garray = aij->garray;
  int        *nz,ec;
  Scalar     v;

  /* free stuff related to matrix-vec multiply */
  ierr = VecGetSize(aij->lvec,&ec); /* needed for PLogObjectMemory below */
  ierr = VecDestroy(aij->lvec); CHKERRQ(ierr); aij->lvec = 0;
  ierr = VecScatterCtxDestroy(aij->Mvctx); CHKERRQ(ierr); aij->Mvctx = 0;
  if (aij->colmap) {
    PETSCFREE(aij->colmap); aij->colmap = 0;
    PLogObjectMemory(A,-Baij->n*sizeof(int));
  }

  /* make sure that B is assembled so we can access its values */
  ierr = MatAssemblyBegin(B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  MatAssemblyEnd(B,FINAL_ASSEMBLY); CHKERRQ(ierr);

  /* invent new B and copy stuff over */
  nz = (int *) PETSCMALLOC( m*sizeof(int) ); CHKPTRQ(nz);
  for ( i=0; i<m; i++ ) {
    nz[i] = Baij->i[i+1]-Baij->i[i];
  }
  ierr = MatCreateSequentialAIJ(MPI_COMM_SELF,m,n,0,nz,&Bnew); CHKERRQ(ierr);
  PETSCFREE(nz);
  for ( i=0; i<m; i++ ) {
    for ( j=Baij->i[i]-1; j<Baij->i[i+1]-1; j++ ) {
      col = garray[Baij->j[ct]-1];
      v = Baij->a[ct++];
      ierr = MatSetValues(Bnew,1,&i,1,&col,&v,INSERTVALUES); CHKERRQ(ierr);
    }
  }
  PETSCFREE(aij->garray); aij->garray = 0;
  PLogObjectMemory(A,-ec*sizeof(int));
  ierr = MatDestroy(B); CHKERRQ(ierr);
  PLogObjectParent(A,Bnew);
  aij->B = Bnew;
  aij->assembled = 0;
  return 0;
}


