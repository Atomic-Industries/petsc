static char help[] = "Test MATMFFD for the rectangular case\n\n";

#include <petscmat.h>

static PetscErrorCode myF(void* ctx,Vec x,Vec y)
{
  const PetscScalar *ax;
  PetscScalar       *ay;
  PetscInt          i,j,m,n;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(x,&ax));
  CHKERRQ(VecGetArray(y,&ay));
  CHKERRQ(VecGetLocalSize(y,&m));
  CHKERRQ(VecGetLocalSize(x,&n));
  for (i=0;i<m;i++) {
    PetscScalar xx,yy;

    yy = 0.0;
    for (j=0;j<n;j++) {
      xx = PetscPowScalarInt(ax[j],i+1);
      yy += xx;
    }
    ay[i] = yy;
  }
  CHKERRQ(VecRestoreArray(y,&ay));
  CHKERRQ(VecRestoreArrayRead(x,&ax));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            A,B;
  Vec            base;
  PetscInt       m = 3 ,n = 2;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(MatCreateMFFD(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,&A));
  CHKERRQ(MatCreateVecs(A,&base,NULL));
  CHKERRQ(VecSet(base,2.0));
  CHKERRQ(MatMFFDSetFunction(A,myF,NULL));
  CHKERRQ(MatMFFDSetBase(A,base,NULL));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatComputeOperator(A,NULL,&B));
  CHKERRQ(VecDestroy(&base));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

    test:
      nsize: {{1 2 3 4}}

TEST*/
