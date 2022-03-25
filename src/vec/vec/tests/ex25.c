
static char help[] = "Scatters from a parallel vector to a sequential vector.  In\n\
this case processor zero is as long as the entire parallel vector; rest are zero length.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n = 5,N,low,high,iglobal,i;
  PetscMPIInt    size,rank;
  PetscScalar    value,zero = 0.0;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* create two vectors */
  N    = size*n;
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&y));
  CHKERRQ(VecSetSizes(y,PETSC_DECIDE,N));
  CHKERRQ(VecSetFromOptions(y));
  if (rank == 0) {
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,N,&x));
  } else {
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,0,&x));
  }

  /* create two index sets */
  if (rank == 0) {
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,N,0,1,&is1));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,N,0,1,&is2));
  } else {
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,0,0,1,&is1));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,0,0,1,&is2));
  }

  CHKERRQ(VecSet(x,zero));
  CHKERRQ(VecGetOwnershipRange(y,&low,&high));
  for (i=0; i<n; i++) {
    iglobal = i + low; value = (PetscScalar) (i + 10*rank);
    CHKERRQ(VecSetValues(y,1,&iglobal,&value,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(y));
  CHKERRQ(VecAssemblyEnd(y));
  CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecScatterCreate(y,is2,x,is1,&ctx));
  CHKERRQ(VecScatterBegin(ctx,y,x,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx,y,x,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&ctx));

  if (rank == 0) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"----\n"));
    CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_SELF));
  }

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(ISDestroy(&is2));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 3

TEST*/
