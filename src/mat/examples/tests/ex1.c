#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.5 1997/10/19 03:26:38 bsmith Exp bsmith $";
#endif

static char help[] = "Tests LU and Cholesky factorization for a dense matrix.\n\n";

#include <math.h>
#include "mat.h"

int main(int argc,char **argv)
{
  Mat        mat, fact;
  MatType    type;
  MatInfo    info;
  int        m = 10, n = 10, i = 4, j = 6, ierr, rstart, rend;
  PetscTruth set;
  Scalar     value = 1.0;
  Vec        x, y, b;
  double     norm;

  PetscInitialize(&argc,&argv,(char*) 0,help);

  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,m,&y); CHKERRA(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,m,&x); CHKERRA(ierr);
  ierr = VecSet(&value,x); CHKERRA(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,n,&b); CHKERRA(ierr);

  ierr = MatGetTypeFromOptions(PETSC_COMM_WORLD,0,&type,&set); CHKERRQ(ierr);
  if (type == MATMPIDENSE) {
    ierr = MatCreateMPIDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,
           PETSC_NULL,&mat); CHKERRA(ierr);
  }
  else {
    ierr = MatCreateSeqDense(PETSC_COMM_WORLD,m,n,PETSC_NULL,&mat); CHKERRA(ierr);
  }

  ierr = MatGetOwnershipRange(mat,&rstart,&rend); CHKERRA(ierr);
  for (i=rstart; i<rend; i++ ) {
    value = (double) i+1;
    ierr = MatSetValues(mat,1,&i,1,&i,&value,INSERT_VALUES); CHKERRA(ierr);
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  ierr = MatGetInfo(mat,MAT_LOCAL,&info); CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"matrix nonzeros = %d, allocated nonzeros = %d\n",
    (int)info.nz_used,(int)info.nz_allocated); 

  if (type != MATMPIDENSE) {
    /* Cholesky factorization is not yet in place for this matrix format */
    ierr = MatMult(mat,x,b); CHKERRA(ierr);
    ierr = MatConvert(mat,MATSAME,&fact); CHKERRA(ierr);
    ierr = MatCholeskyFactor(fact,0,1.0); CHKERRA(ierr);
    ierr = MatSolve(fact,b,y); CHKERRA(ierr);
    ierr = MatDestroy(fact); CHKERRA(ierr);
    value = -1.0; ierr = VecAXPY(&value,x,y); CHKERRA(ierr);
    ierr = VecNorm(y,NORM_2,&norm); CHKERRA(ierr);
    if (norm > 1.e-12) 
      PetscPrintf(PETSC_COMM_WORLD,"Norm of error for Cholesky %g\n",norm);
    else 
      PetscPrintf(PETSC_COMM_WORLD,"Norm of error for Cholesky < 1.e-12\n");

    ierr = MatCholeskyFactorSymbolic(mat,0,1.0,&fact); CHKERRA(ierr);
    ierr = MatCholeskyFactorNumeric(mat,&fact); CHKERRA(ierr);
    ierr = MatSolve(fact,b,y); CHKERRA(ierr);
    value = -1.0; ierr = VecAXPY(&value,x,y); CHKERRA(ierr);
    ierr = VecNorm(y,NORM_2,&norm); CHKERRA(ierr);
    if (norm > 1.e-12) 
      PetscPrintf(PETSC_COMM_WORLD,"Norm of error for Cholesky %g\n",norm);
    else 
      PetscPrintf(PETSC_COMM_WORLD,"Norm of error for Cholesky < 1.e-12\n");
    ierr = MatDestroy(fact); CHKERRA(ierr);
  }

  i = m-1; j = 0; value = 1.0;
  ierr = MatSetValues(mat,1,&i,1,&i,&value,INSERT_VALUES); CHKERRA(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatMult(mat,x,b); CHKERRA(ierr);
  ierr = MatConvert(mat,MATSAME,&fact); CHKERRA(ierr);
  ierr = MatLUFactor(fact,0,0,1.0); CHKERRA(ierr);
  ierr = MatSolve(fact,b,y); CHKERRA(ierr);
  value = -1.0; ierr = VecAXPY(&value,x,y); CHKERRA(ierr);
  ierr = VecNorm(y,NORM_2,&norm); CHKERRA(ierr);
  if (norm > 1.e-12) 
    PetscPrintf(PETSC_COMM_WORLD,"Norm of error for LU %g\n",norm);
  else 
    PetscPrintf(PETSC_COMM_WORLD,"Norm of error for LU < 1.e-12\n");
  ierr = MatDestroy(fact);CHKERRA(ierr);

  ierr = MatLUFactorSymbolic(mat,0,0,1.0,&fact); CHKERRA(ierr);
  ierr = MatLUFactorNumeric(mat,&fact); CHKERRA(ierr);
  ierr = MatSolve(fact,b,y); CHKERRA(ierr);
  value = -1.0; ierr = VecAXPY(&value,x,y); CHKERRA(ierr);
  ierr = VecNorm(y,NORM_2,&norm); CHKERRA(ierr);
  if (norm > 1.e-12) 
    PetscPrintf(PETSC_COMM_WORLD,"Norm of error for LU %g\n",norm);
  else 
    PetscPrintf(PETSC_COMM_WORLD,"Norm of error for LU < 1.e-12\n");

  ierr = MatDestroy(fact);CHKERRA(ierr);
  ierr = MatDestroy(mat);CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
