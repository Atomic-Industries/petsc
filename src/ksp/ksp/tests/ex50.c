
static char help[] = "Tests point block Jacobi and ILU for different block sizes\n\n";

#include <pthread.h>
#include <petscksp.h>

typedef struct {
  int      argc;
  char   **argv;
  MPI_Comm comm;
} ThreadInfo;

PetscErrorCode petsc_main(int argc, char **argv)
{
  Vec         x, b, u;
  Mat         A;    /* linear system matrix */
  KSP         ksp;  /* linear solver context */
  PetscRandom rctx; /* random number generator context */
  PetscReal   norm; /* norm of solution error */
  PetscInt    i, j, k, l, n = 27, its, bs = 2, Ii, J;
  PetscScalar v;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-bs", &bs, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, n * bs, n * bs, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetBlockSize(A, bs));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  /*
     Don't bother to preallocate matrix
  */
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rctx));
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      PetscCall(PetscRandomGetValue(rctx, &v));
      if (PetscRealPart(v) < .25 || i == j) {
        for (k = 0; k < bs; k++) {
          for (l = 0; l < bs; l++) {
            PetscCall(PetscRandomGetValue(rctx, &v));
            Ii = i * bs + k;
            J  = j * bs + l;
            if (Ii == J) v += 10.;
            PetscCall(MatSetValue(A, Ii, J, v, INSERT_VALUES));
          }
        }
      }
    }
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(A, &u, &b));
  PetscCall(VecDuplicate(u, &x));
  PetscCall(VecSet(u, 1.0));
  PetscCall(MatMult(A, u, b));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create linear solver context
  */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  PetscCall(KSPSetOperators(ksp, A, A));

  PetscCall(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(KSPSolve(ksp, b, x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Check the error
  */
  PetscCall(VecAXPY(x, -1.0, u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(KSPGetIterationNumber(ksp, &its));

  /*
     Print convergence information.  PetscPrintf() produces a single
     print statement from all processes that share a communicator.
     An alternative is PetscFPrintf(), which prints to a file.
  */
  if (norm > .1) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Norm of residual %g iterations %" PetscInt_FMT " bs %" PetscInt_FMT "\n", (double)norm, its, bs));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscRandomDestroy(&rctx));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  PetscCall(PetscFinalize());
  return 0;
}

void *thread_start(void *arg)
{
  ThreadInfo    *tinfo = (ThreadInfo *)arg;
  MPI_Comm       comm  = tinfo->comm;
  PetscErrorCode ierr;

  PETSC_COMM_WORLD = comm;
  MPIX_Threadcomm_start(comm);
  ierr = petsc_main(tinfo->argc, tinfo->argv);
  MPIX_Threadcomm_finish(comm);
  return (void *)(intptr_t)ierr;
}

int main(int argc, char **argv)
{
  int        rc, ierr, provided, nthreads = 4;
  MPI_Comm   comm;
  pthread_t  threads[16];
  ThreadInfo tinfo;
  void      *res;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  MPIX_Threadcomm_init(MPI_COMM_WORLD, nthreads, &comm);

  tinfo.argc = argc;
  tinfo.argv = argv;
  tinfo.comm = comm;

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  for (int i = 0; i < nthreads; i++) pthread_create(&threads[i], &attr, thread_start, &tinfo);
  pthread_attr_destroy(&attr);
  for (int i = 0; i < nthreads; i++) {
    rc = pthread_join(threads[i], &res);
    if (rc) ierr = rc;
    if (res) ierr = (intptr_t)res;
  }
  MPIX_Threadcomm_free(&comm);
  return ierr;
}

/*TEST

  testset:
    args: -bs {{1 2 3 4 5 6 7 8 11 15}} -pc_type {{pbjacobi ilu}}
    output_file: output/ex50_1.out

    test:
      args: -mat_type {{aij baij}}

    test:
      suffix: cuda
      requires: cuda
      args: -mat_type aijcusparse

    test:
      suffix: kok
      requires: kokkos_kernels
      args: -mat_type aijkokkos

TEST*/
