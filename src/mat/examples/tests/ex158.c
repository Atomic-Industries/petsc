static char help[] = "Illustrate how to use mpi FFTW and PETSc-FFTW interface \n\n";

/*
 Usage:
   mpiexec -n <np> ./ex158 -use_FFTW_interface NO
   mpiexec -n <np> ./ex158 -use_FFTW_interface YES
*/

#include <petscmat.h>
#include <fftw3-mpi.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       N0=50,N1=20,N=N0*N1;
  PetscRandom    rdm;
  PetscScalar    a;
  PetscReal      enorm;
  Vec            x,y,z;
  PetscBool      view=PETSC_FALSE,use_interface=PETSC_TRUE;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP, "This example requires real numbers. Your current scalar type is complex");
#endif

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "FFTW Options", "ex158");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_FFTW_interface", "Use PETSc-FFTW interface", "ex158",use_interface, &use_interface, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD, &rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);

  if (!use_interface) {
    /* Use mpi FFTW without PETSc-FFTW interface, 2D case only */
    /*---------------------------------------------------------*/
    fftw_plan    fplan,bplan;
    fftw_complex *data_in,*data_out,*data_out2;
    ptrdiff_t    alloc_local,local_n0,local_0_start;

    if (!rank) printf("Use FFTW without PETSc-FFTW interface\n");
    fftw_mpi_init();
    N           = N0*N1;
    alloc_local = fftw_mpi_local_size_2d(N0,N1,PETSC_COMM_WORLD,&local_n0,&local_0_start);

    data_in   = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
    data_out  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
    data_out2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);

    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,(PetscInt)local_n0*N1,(PetscInt)N,(const PetscScalar*)data_in,&x);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) x, "Real Space vector");CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,(PetscInt)local_n0*N1,(PetscInt)N,(const PetscScalar*)data_out,&y);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) y, "Frequency space vector");CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,(PetscInt)local_n0*N1,(PetscInt)N,(const PetscScalar*)data_out2,&z);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) z, "Reconstructed vector");CHKERRQ(ierr);

    fplan = fftw_mpi_plan_dft_2d(N0,N1,data_in,data_out,PETSC_COMM_WORLD,FFTW_FORWARD,FFTW_ESTIMATE);
    bplan = fftw_mpi_plan_dft_2d(N0,N1,data_out,data_out2,PETSC_COMM_WORLD,FFTW_BACKWARD,FFTW_ESTIMATE);

    ierr = VecSetRandom(x, rdm);CHKERRQ(ierr);
    if (view) {ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}

    fftw_execute(fplan);
    if (view) {ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}

    fftw_execute(bplan);

    /* Compare x and z. FFTW computes an unnormalized DFT, thus z = N*x */
    a    = 1.0/(PetscReal)N;
    ierr = VecScale(z,a);CHKERRQ(ierr);
    if (view) {ierr = VecView(z, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
    ierr = VecAXPY(z,-1.0,x);CHKERRQ(ierr);
    ierr = VecNorm(z,NORM_1,&enorm);CHKERRQ(ierr);
    if (enorm > 1.e-11) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %g\n",(double)enorm);CHKERRQ(ierr);
    }

    /* Free spaces */
    fftw_destroy_plan(fplan);
    fftw_destroy_plan(bplan);
    fftw_free(data_in);  ierr = VecDestroy(&x);CHKERRQ(ierr);
    fftw_free(data_out); ierr = VecDestroy(&y);CHKERRQ(ierr);
    fftw_free(data_out2);ierr = VecDestroy(&z);CHKERRQ(ierr);

  } else {
    /* Use PETSc-FFTW interface                  */
    /*-------------------------------------------*/
    PetscInt i,*dim,k,DIM;
    Mat      A;
    Vec      input,output;

    N=30;
    for (i=2; i<3; i++) { /* (i=3,4: -- error in VecScatterPetscToFFTW(A,input,x); */
      DIM  = i;
      ierr = PetscMalloc1(i,&dim);CHKERRQ(ierr);
      for (k=0; k<i; k++) {
        dim[k]=30;
      }
      N *= dim[i-1];

      /* Create FFTW object */
      if (!rank) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"Use PETSc-FFTW interface...%d-DIM:%d \n",DIM,N);CHKERRQ(ierr);
      }
      ierr = MatCreateFFT(PETSC_COMM_WORLD,DIM,dim,MATFFTW,&A);CHKERRQ(ierr);

      /* Create FFTW vectors that are compatible with parallel layout of A */
      ierr = MatCreateVecsFFTW(A,&x,&y,&z);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) x, "Real space vector");CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) y, "Frequency space vector");CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) z, "Reconstructed vector");CHKERRQ(ierr);

      /* Create and set PETSc vector */
      ierr = VecCreate(PETSC_COMM_WORLD,&input);CHKERRQ(ierr);
      ierr = VecSetSizes(input,PETSC_DECIDE,N);CHKERRQ(ierr);
      ierr = VecSetFromOptions(input);CHKERRQ(ierr);
      ierr = VecSetRandom(input,rdm);CHKERRQ(ierr);
      ierr = VecDuplicate(input,&output);CHKERRQ(ierr);
      if (view) {ierr = VecView(input,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}

      /* Vector input is copied to another vector x using VecScatterPetscToFFTW. This is because the user data
         can have any parallel layout. But FFTW requires special parallel layout of the data. Hence the original
         data which is in the vector "input" here, needs to be copied to a vector x, which has the correct parallel
         layout for FFTW. Also, during parallel real transform, this pads extra zeros automatically
         at the end of last  dimension. This padding is required by FFTW to perform parallel real D.F.T.  */
      ierr = VecScatterPetscToFFTW(A,input,x);CHKERRQ(ierr);/* buggy for dim = 3, 4... */

      /* Apply FFTW_FORWARD and FFTW_BACKWARD */
      ierr = MatMult(A,x,y);CHKERRQ(ierr);
      if (view) {ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
      ierr = MatMultTranspose(A,y,z);CHKERRQ(ierr);

      /* Output from Backward DFT needs to be modified to obtain user readable data the routine VecScatterFFTWToPetsc
         performs the job. In some sense this is the reverse operation of VecScatterPetscToFFTW. This routine gets rid of
         the extra spaces that were artificially padded to perform real parallel transform.    */
      ierr = VecScatterFFTWToPetsc(A,z,output);CHKERRQ(ierr);

      /* Compare x and z. FFTW computes an unnormalized DFT, thus z = N*x */
      a    = 1.0/(PetscReal)N;
      ierr = VecScale(output,a);CHKERRQ(ierr);
      if (view) {ierr = VecView(output,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
      ierr = VecAXPY(output,-1.0,input);CHKERRQ(ierr);
      ierr = VecNorm(output,NORM_1,&enorm);CHKERRQ(ierr);
      if (enorm > 1.e-09 && !rank) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %e\n",enorm);CHKERRQ(ierr);
      }

      /* Free spaces */
      ierr = PetscFree(dim);CHKERRQ(ierr);
      ierr = VecDestroy(&input);CHKERRQ(ierr);
      ierr = VecDestroy(&output);CHKERRQ(ierr);
      ierr = VecDestroy(&x);CHKERRQ(ierr);
      ierr = VecDestroy(&y);CHKERRQ(ierr);
      ierr = VecDestroy(&z);CHKERRQ(ierr);
      ierr = MatDestroy(&A);CHKERRQ(ierr);
    }
  }
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   build:
      requires: fftw !complex

   test:
      output_file: output/ex158.out

   test:
      suffix: 2
      nsize: 3

TEST*/
