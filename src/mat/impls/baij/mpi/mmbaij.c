#ifndef lint
static char vcid[] = "$Id: mmbaij.c,v 1.12 1997/04/05 00:18:13 balay Exp bsmith $";
#endif


/*
   Support for the parallel BAIJ matrix vector multiply
*/
#include "src/mat/impls/baij/mpi/mpibaij.h"
#include "src/vec/vecimpl.h"

#undef __FUNC__  
#define __FUNC__ "MatSetUpMultiply_MPIBAIJ"
int MatSetUpMultiply_MPIBAIJ(Mat mat)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  Mat_SeqBAIJ *B = (Mat_SeqBAIJ *) (baij->B->data);  
  int        Nbs = baij->Nbs,i,j,*indices,*aj = B->j,ierr,ec = 0,*garray;
  int        col,bs = baij->bs,*tmp,*stmp;
  IS         from,to;
  Vec        gvec;

  /* For the first stab we make an array as long as the number of columns */
  /* mark those columns that are in baij->B */
  indices = (int *) PetscMalloc( (Nbs+1)*sizeof(int) ); CHKPTRQ(indices);
  PetscMemzero(indices,Nbs*sizeof(int));
  for ( i=0; i<B->mbs; i++ ) {
    for ( j=0; j<B->ilen[i]; j++ ) {
      if (!indices[aj[B->i[i] + j]]) ec++; 
      indices[aj[B->i[i] + j] ] = 1;
    }
  }

  /* form array of columns we need */
  garray = (int *) PetscMalloc( (ec+1)*sizeof(int) ); CHKPTRQ(garray);
  tmp    = (int *) PetscMalloc( (ec*bs+1)*sizeof(int) ); CHKPTRQ(tmp)
  ec = 0;
  for ( i=0; i<Nbs; i++ ) {
    if (indices[i]) garray[ec++] = i;
  }

  /* make indices now point into garray */
  for ( i=0; i<ec; i++ ) {
    indices[garray[i]] = i;
  }

  /* compact out the extra columns in B */
  for ( i=0; i<B->mbs; i++ ) {
    for ( j=0; j<B->ilen[i]; j++ ) {
      aj[B->i[i] + j] = indices[aj[B->i[i] + j]];
    }
  }
  B->nbs = ec;
  B->n   = ec*B->bs;
  PetscFree(indices);
  
  for ( i=0,col=0; i<ec; i++ ) {
    for ( j=0; j<bs; j++,col++) tmp[col] = garray[i]*bs+j;
  }
  /* create local vector that is used to scatter into */
  ierr = VecCreateSeq(PETSC_COMM_SELF,ec*bs,&baij->lvec); CHKERRQ(ierr);

  /* create two temporary index sets for building scatter-gather */

  /* ierr = ISCreateGeneral(PETSC_COMM_SELF,ec*bs,tmp,&from); CHKERRQ(ierr); */
  for ( i=0,col=0; i<ec; i++ ) {
    garray[i] = bs*garray[i];
  }
  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,ec,garray,&from);CHKERRQ(ierr);   
  for ( i=0,col=0; i<ec; i++ ) {
    garray[i] = garray[i]/bs;
  }

  stmp = (int *) PetscMalloc( (ec+1)*sizeof(int) ); CHKPTRQ(stmp);
  for ( i=0; i<ec; i++ ) { stmp[i] = bs*i; } 
  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,ec,stmp,&to);CHKERRQ(ierr);
  PetscFree(stmp);

  /* create temporary global vector to generate scatter context */
  /* this is inefficient, but otherwise we must do either 
     1) save garray until the first actual scatter when the vector is known or
     2) have another way of generating a scatter context without a vector.*/
  ierr = VecCreateMPI(mat->comm,baij->n,baij->N,&gvec); CHKERRQ(ierr);

  /* gnerate the scatter context */
  ierr = VecScatterCreate(gvec,from,baij->lvec,to,&baij->Mvctx);CHKERRQ(ierr);

  /*
      Post the receives for the first matrix vector product. We sync-chronize after
    this on the chance that the user immediately calls MatMult() after assemblying 
    the matrix.
  */
  ierr = VecScatterPostRecvs(gvec,baij->lvec,INSERT_VALUES,SCATTER_FORWARD,baij->Mvctx);CHKERRQ(ierr);
  MPI_Barrier(mat->comm);

  PLogObjectParent(mat,baij->Mvctx);
  PLogObjectParent(mat,baij->lvec);
  PLogObjectParent(mat,from);
  PLogObjectParent(mat,to);
  baij->garray = garray;
  PLogObjectMemory(mat,(ec+1)*sizeof(int));
  ierr = ISDestroy(from); CHKERRQ(ierr);
  ierr = ISDestroy(to); CHKERRQ(ierr);
  ierr = VecDestroy(gvec);
  PetscFree(tmp);
  return 0;
}


/*
     Takes the local part of an already assembled MPIBAIJ matrix
   and disassembles it. This is to allow new nonzeros into the matrix
   that require more communication in the matrix vector multiply. 
   Thus certain data-structures must be rebuilt.

   Kind of slow! But that's what application programmers get when 
   they are sloppy.
*/
#undef __FUNC__  
#define __FUNC__ "DisAssemble_MPIBAIJ"
int DisAssemble_MPIBAIJ(Mat A)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) A->data;
  Mat        B = baij->B,Bnew;
  Mat_SeqBAIJ *Bbaij = (Mat_SeqBAIJ*)B->data;
  int        ierr,i,j,mbs=Bbaij->mbs,n = baij->N,col,*garray=baij->garray;
  int        k,bs=baij->bs,bs2=baij->bs2,*rvals,*nz,ec,m=Bbaij->m;
  Scalar     *a=Bbaij->a;

  /* free stuff related to matrix-vec multiply */
  ierr = VecGetSize(baij->lvec,&ec); /* needed for PLogObjectMemory below */
  ierr = VecDestroy(baij->lvec); CHKERRQ(ierr); baij->lvec = 0;
  ierr = VecScatterDestroy(baij->Mvctx); CHKERRQ(ierr); baij->Mvctx = 0;
  if (baij->colmap) {
    PetscFree(baij->colmap); baij->colmap = 0;
    PLogObjectMemory(A,-Bbaij->nbs*sizeof(int));
  }

  /* make sure that B is assembled so we can access its values */
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  /* invent new B and copy stuff over */
  nz = (int *) PetscMalloc( mbs*sizeof(int) ); CHKPTRQ(nz);
  for ( i=0; i<mbs; i++ ) {
    nz[i] = Bbaij->i[i+1]-Bbaij->i[i];
  }
  ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,baij->bs,m,n,0,nz,&Bnew); CHKERRQ(ierr);
  PetscFree(nz);
  
  rvals = (int *) PetscMalloc(bs*sizeof(int)); CHKPTRQ(rvals);
  for ( i=0; i<mbs; i++ ) {
    rvals[0] = bs*i;
    for ( j=1; j<bs; j++ ) { rvals[j] = rvals[j-1] + 1; }
    for ( j=Bbaij->i[i]; j<Bbaij->i[i+1]; j++ ) {
      col = garray[Bbaij->j[j]]*bs;
      for (k=0; k<bs; k++ ) {
        ierr = MatSetValues(Bnew,bs,rvals,1,&col,a+j*bs2,INSERT_VALUES);CHKERRQ(ierr);
        col++;
      }
    }
  }
  PetscFree(baij->garray); baij->garray = 0;
  PetscFree(rvals);
  PLogObjectMemory(A,-ec*sizeof(int));
  ierr = MatDestroy(B); CHKERRQ(ierr);
  PLogObjectParent(A,Bnew);
  baij->B = Bnew;
  A->was_assembled = PETSC_FALSE;
  return 0;
}


