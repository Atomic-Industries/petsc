
static char help[] = "Tests external Clique direct solvers. Simplified from ex130.c\n\
Example: mpiexec -n <np> ./ex168 -f <matrix binary file> \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,F;
  Vec            u,x,b;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       m,n,nfact;
  PetscReal      norm,tol=1.e-12,Anorm;
  IS             perm,iperm;
  MatFactorInfo  info;
  PetscBool      flg,testMatSolve=PETSC_TRUE;
  PetscViewer    fd;              /* viewer */
  char           file[PETSC_MAX_PATH_LEN]; /* input file name */

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  /* Determine file from which we read the matrix A */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");

  /* Load matrix A */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&b));
  CHKERRQ(VecLoad(b,fd));
  CHKERRQ(PetscViewerDestroy(&fd));
  CHKERRQ(MatGetLocalSize(A,&m,&n));
  PetscCheckFalse(m != n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%d, %d)", m, n);
  CHKERRQ(MatNorm(A,NORM_INFINITY,&Anorm));

  /* Create vectors */
  CHKERRQ(VecDuplicate(b,&x));
  CHKERRQ(VecDuplicate(x,&u)); /* save the true solution */

  /* Test Cholesky Factorization */
  CHKERRQ(MatGetOrdering(A,MATORDERINGNATURAL,&perm,&iperm));

  if (rank == 0) printf(" Clique Cholesky:\n");
  CHKERRQ(MatGetFactor(A,MATSOLVERCLIQUE,MAT_FACTOR_CHOLESKY,&F));

  info.fill = 5.0;
  CHKERRQ(MatCholeskyFactorSymbolic(F,A,perm,&info));

  for (nfact = 0; nfact < 1; nfact++) {
    if (rank == 0) printf(" %d-the Cholesky numfactorization \n",nfact);
    CHKERRQ(MatCholeskyFactorNumeric(F,A,&info));

    /* Test MatSolve() */
    if (testMatSolve && nfact == 2) {
      CHKERRQ(MatSolve(F,b,x));

      /* Check the residual */
      CHKERRQ(MatMult(A,x,u));
      CHKERRQ(VecAXPY(u,-1.0,b));
      CHKERRQ(VecNorm(u,NORM_INFINITY,&norm));
      /* if (norm > tol) { */
      if (rank == 0) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"MatSolve: rel residual %g/%g = %g, LU numfact %d\n",norm,Anorm,norm/Anorm,nfact));
      }
      /*} */
    }
  }

  /* Free data structures */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&F));
  CHKERRQ(ISDestroy(&perm));
  CHKERRQ(ISDestroy(&iperm));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&u));
  ierr = PetscFinalize();
  return ierr;
}
