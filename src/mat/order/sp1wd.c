#ifndef lint
static char vcid[] = "$Id: sp1wd.c,v 1.17 1997/01/01 03:38:11 bsmith Exp balay $";
#endif

#include "petsc.h"
#include "mat.h"
#include "src/mat/impls/order/order.h"

/*
    MatOrder_1WD - Find the 1-way dissection ordering of a given matrix.
*/    
#undef __FUNC__  
#define __FUNC__ "MatOrder_1WD"
int MatOrder_1WD( Mat mat, MatReordering type, IS *row, IS *col)
{
  int i,   *mask, *xls, nblks, *xblk, *ls, nrow, *perm, ierr,*ia,*ja;
  PetscTruth done;

  ierr = MatGetRowIJ(mat,1,PETSC_TRUE,&nrow,&ia,&ja,&done); CHKERRQ(ierr);
  if (!done) SETERRQ(1,0,"Cannot get rows for matrix");

  mask = (int *)PetscMalloc( (5*nrow+1) * sizeof(int) );     CHKPTRQ(mask);
  xls  = mask + nrow;
  ls   = xls + nrow + 1;
  xblk = ls + nrow;
  perm = xblk + nrow;
  gen1wd( &nrow, ia, ja, mask, &nblks, xblk, perm, xls, ls );
  ierr = MatRestoreRowIJ(mat,1,PETSC_TRUE,&nrow,&ia,&ja,&done); CHKERRQ(ierr);

  for (i=0; i<nrow; i++) perm[i]--;

  ierr = ISCreateGeneral(MPI_COMM_SELF,nrow,perm,row); CHKERRQ(ierr);
  ierr = ISCreateGeneral(MPI_COMM_SELF,nrow,perm,col); CHKERRQ(ierr);
  PetscFree(mask);

  return 0;
}

