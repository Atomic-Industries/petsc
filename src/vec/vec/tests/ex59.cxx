
static char help[] = "Test VecCreate{Seq|MPI}ViennaCLWithArrays.\n\n";

#include "petsc.h"
#include "petscviennacl.h"

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Vec            x,y;
  PetscMPIInt    size;
  PetscInt       n = 5;
  PetscScalar    xHost[5] = {0.,1.,2.,3.,4.};

  ierr = PetscInitialize(&argc, &argv, (char*)0, help); if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  if (size == 1) {
    CHKERRQ(VecCreateSeqViennaCLWithArrays(PETSC_COMM_WORLD,1,n,xHost,NULL,&x));
  } else {
    CHKERRQ(VecCreateMPIViennaCLWithArrays(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,xHost,NULL,&x));
  }
  /* print x should be equivalent too xHost */
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecSet(x,42.0));
  /* print x should be all 42 */
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  if (size == 1) {
    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_WORLD,1,n,xHost,&y));
  } else {
    CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,xHost,&y));
  }

  /* print y should be all 42 */
  CHKERRQ(VecView(y, PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&x));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: viennacl defined(PETSC_HAVE_VIENNACL_NO_CUDA)

   test:
      nsize: 1
      suffix: 1
      args: -viennacl_backend opencl

   test:
      nsize: 2
      suffix: 2
      args: -viennacl_backend opencl

TEST*/
