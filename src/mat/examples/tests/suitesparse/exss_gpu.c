static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
Input arguments are:\n\
  -A <input_file> : file to load.  For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\n";

#include <petscmat.h>
#include <petscksp.h>
#include <petsctime.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       m,n,mvec;
  PetscReal      norm;
  Vec            b,u;
  Mat            A,B;
  MatType        mattype;
  char           file[PETSC_MAX_PATH_LEN];
  char           file2[PETSC_MAX_PATH_LEN];
  char           groupname[PETSC_MAX_PATH_LEN],matname[PETSC_MAX_PATH_LEN],str1[PETSC_MAX_PATH_LEN],str2[PETSC_MAX_PATH_LEN];
  PetscViewer    fd;
  PetscBool      flg,test_sell = PETSC_FALSE;
  PetscLogDouble vstart,vend;
  PetscMPIInt    rank;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  /* Read matrix and RHS */
  ierr = PetscOptionsGetString(NULL,NULL,"-groupname",groupname,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-matname",matname,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-str1",str1,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-str2",str2,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-A",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -A option");
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_sell",&test_sell,NULL);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);

  /* Let the vec object trigger the first CUDA call, which takes a relatively long time to init CUDA */
  ierr = PetscOptionsGetString(NULL,NULL,"-b",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
    ierr = VecLoad(b,fd);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  } else {
    ierr = VecSetSizes(b,PETSC_DECIDE,m);CHKERRQ(ierr);
    ierr = VecSet(b,1.0);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(b,&u);CHKERRQ(ierr);

  if (test_sell) {
    /* two-step convert is much faster than the basic convert */
    ierr = MatConvert(A,MATSELL,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
    ierr = MatConvert(A,MATSELLCUDA,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  } else {
    ierr = MatConvert(A,MATAIJCUSPARSE,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  }

  /* Timing MatMult */
  ierr = PetscTime(&vstart);CHKERRQ(ierr);
  ierr = MatMult(A,b,u);CHKERRQ(ierr);
  ierr = PetscTime(&vend);CHKERRQ(ierr);

  /* Show result */
  ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
  /* ierr = PetscPrintf(PETSC_COMM_WORLD,"norm=%g\n",norm);CHKERRQ(ierr); */
  /*
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  if (!rank) {
    FILE *f;
    ierr = PetscSNPrintf(file2,sizeof(file2),"%s_%s_%s_%s.txt",groupname,matname,str1,str2);CHKERRQ(ierr);
    ierr = PetscFOpen(PETSC_COMM_SELF,file2,"w",&f);CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,f,"%s %s %s %s %lf\n",groupname,matname,str1,str2,(double)(vend-vstart));CHKERRQ(ierr);
    ierr = PetscFClose(PETSC_COMM_SELF,f);CHKERRQ(ierr);
  }
  */
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  {
    PetscLogEvent      event;
    PetscEventPerfInfo eventInfo;
    PetscReal          gpuflopRate;

    if (test_sell) {
      ierr = PetscLogEventGetId("MatCUDACopyTo",&event);CHKERRQ(ierr);
    } else {
      ierr = PetscLogEventGetId("MatCUSPARSCopyTo",&event);CHKERRQ(ierr);
    }
    ierr = PetscLogEventGetPerfInfo(PETSC_DETERMINE, event, &eventInfo);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%.4e ", eventInfo.time);CHKERRQ(ierr);

    ierr = PetscLogEventGetId("MatMult",&event);CHKERRQ(ierr);
    ierr = PetscLogEventGetPerfInfo(PETSC_DETERMINE, event, &eventInfo);CHKERRQ(ierr);
    gpuflopRate = eventInfo.GpuFlops/eventInfo.GpuTime;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%.2f %.4e %.4e\n", gpuflopRate/1.e6,eventInfo.GpuTime,eventInfo.time);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}
