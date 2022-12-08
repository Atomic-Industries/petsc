static char help[] = "Test of setting values in a matrix without preallocation\n\n";

#include <petscmat.h>

PetscErrorCode ex1_nonsquare_bs1(void)
{
  Mat      A;
  PetscInt M, N, m, n, bs;

  /*
     Create the Jacobian matrix
  */
  PetscFunctionBegin;
  M = 10;
  N = 8;
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, M, N));
  PetscCall(MatSetBlockSize(A, 1));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatXAIJSetNoPreallocation(A));
  PetscCall(MatSetUp(A));

  /*
     Get the sizes of the matrix
  */
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatGetBlockSize(A, &bs));

  /*
     Insert non-zero pattern (e.g. perform a sweep over the grid).
     You can use MatSetValues(), MatSetValuesBlocked() or MatSetValue().
  */
  {
    PetscInt    ii, jj;
    PetscScalar vv = 22.0;

    ii = 3;
    jj = 3;
    PetscCall(MatSetValues(A, 1, &ii, 1, &jj, &vv, INSERT_VALUES));

    ii = 7;
    jj = 4;
    PetscCall(MatSetValues(A, 1, &ii, 1, &jj, &vv, INSERT_VALUES));

    ii = 9;
    jj = 7;
    PetscCall(MatSetValue(A, ii, jj, vv, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  /*
     Insert same location non-zero values into A.
  */
  {
    PetscInt    ii, jj;
    PetscScalar vv;

    ii = 3;
    jj = 3;
    vv = 0.3;
    PetscCall(MatSetValue(A, ii, jj, vv, INSERT_VALUES));

    ii = 7;
    jj = 4;
    vv = 3.3;
    PetscCall(MatSetValues(A, 1, &ii, 1, &jj, &vv, INSERT_VALUES));

    ii = 9;
    jj = 7;
    vv = 4.3;
    PetscCall(MatSetValues(A, 1, &ii, 1, &jj, &vv, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(0);
}

int main(int argc, char **args)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(ex1_nonsquare_bs1());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

   test:
     suffix: 2
     nsize: 2

TEST*/
