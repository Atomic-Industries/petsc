static char help[] = "Check the difference of the two matrices \n\
Reads PETSc matrix A and B, then check B=A-B \n\
Input parameters include\n\
  -fA <input_file> -fB <input_file> \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,B;
  PetscViewer    fd;
  char           file[2][PETSC_MAX_PATH_LEN];
  PetscBool      flg;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       ma,na,mb,nb;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  /* read the two matrices, A and B */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-fA",file[0],sizeof(file[0]),&flg));
  PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -fA options");
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-fB",file[1],sizeof(file[1]),&flg));
  PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -fP options");

  /* Load matrices */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(PetscViewerDestroy(&fd));
  printf("\n A:\n");
  printf("----------------------\n");
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatGetSize(A,&ma,&na));

  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[1],FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatLoad(B,fd));
  CHKERRQ(PetscViewerDestroy(&fd));
  printf("\n B:\n");
  printf("----------------------\n");
  CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatGetSize(B,&mb,&nb));

  /* Compute B = -A + B */
  PetscCheckFalse(ma != mb || na != nb,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"nonconforming matrix size");
  CHKERRQ(MatAXPY(B,-1.0,A,DIFFERENT_NONZERO_PATTERN));
  printf("\n B - A:\n");
  printf("----------------------\n");
  CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&A));
  ierr = PetscFinalize();
  return ierr;
}
