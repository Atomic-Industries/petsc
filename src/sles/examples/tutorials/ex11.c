#ifndef lint
static char vcid[] = "$Id: ex11.c,v 1.7 1996/10/03 03:30:24 curfman Exp curfman $";
#endif

static char help[] = "Solves a linear system in parallel with SLES.\n\n";

/*T
   Concepts: SLES^Solving a Helmholtz equation (basic parallel example);
   Concepts: Complex numbers;
   Routines: SLESCreate(); SLESSetOperators(); SLESSetFromOptions();
   Routines: SLESSolve(); PetscRandomCreate(); PetscRandomGetValue();
   Routines: PetscRandomDestroy();
   Processors: n
T*/

/*
   Description: Solves a complex linear system in parallel with SLES.

   The model problem:
      Solve Helmholtz equation on the unit square: (0,1) x (0,1)
          -delta u - sigma1*u + i*sigma2*u = f, 
           where delta = Laplace operator
      Dirichlet b.c.'s on all sides
      Use the 2-D, five-point finite difference stencil.

   Compiling the code:
      This code uses the complex numbers version of PETSc, so one of the
      following values of BOPT must be used for compiling the PETSc libraries
      and this example:
         BOPT=g_complex   - debugging version
         BOPT=O_complex   - optimized version
         BOPT=Opg_complex - profiling version
*/

/* 
  Include "sles.h" so that we can use SLES solvers.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines   vec.h - vectors
     sys.h    - system routines       mat.h - matrices
     is.h     - index sets            ksp.h - Krylov subspace methods
     viewer.h - viewers               pc.h  - preconditioners
*/
#include "sles.h"
#include <stdio.h>

int main(int argc,char **args)
{
  Vec         x, b, u;      /* approx solution, RHS, exact solution */
  Mat         A;            /* linear system matrix */
  SLES        sles;         /* linear solver context */
  double      norm;         /* norm of solution error */
  int         dim, i, j, I, J, Istart, Iend, ierr, n = 6, its, flg, use_random;
  Scalar      v, none = -1.0, sigma2, pfive = 0.5;
  PetscRandom rctx;
  double      h2, sigma1 = 100.0;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetDouble(PETSC_NULL,"-sigma1",&sigma1,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);
  dim = n*n;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* 
     Create parallel matrix, specifying only its global dimensions.
     When using MatCreate(), the matrix format can be specified at
     runtime. Also, the parallel partioning of the matrix is
     determined by PETSc at runtime.
  */
  ierr = MatCreate(MPI_COMM_WORLD,dim,dim,&A); CHKERRA(ierr);

  /* 
     Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned. 
  */
  ierr = MatGetOwnershipRange(A,&Istart,&Iend); CHKERRA(ierr);

  /* 
     Set matrix elements in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly). 
      - Always specify global rows and columns of matrix entries.
  */

  ierr = OptionsHasName(PETSC_NULL,"-norandom",&flg); CHKERRA(ierr);
  if (flg) use_random = 0;
  else     use_random = 1;
  if (use_random) {
    ierr = PetscRandomCreate(MPI_COMM_WORLD,RANDOM_DEFAULT_IMAGINARY,&rctx); CHKERRA(ierr);
  } else {
    sigma2 = 10.0*PETSC_i;
  }
  h2 = 1.0/((n+1)*(n+1));
  for ( I=Istart; I<Iend; I++ ) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if ( i>0 ) {
      J = I-n; ierr = MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES); CHKERRA(ierr);}
    if ( i<n-1 ) {
      J = I+n; ierr = MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES); CHKERRA(ierr);}
    if ( j>0 ) {
      J = I-1; ierr = MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES); CHKERRA(ierr);}
    if ( j<n-1 ) {
      J = I+1; ierr = MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES); CHKERRA(ierr);}
    if (use_random) {ierr = PetscRandomGetValue(rctx,&sigma2); CHKERRA(ierr);}
    v = 4.0 - sigma1*h2 + sigma2*h2;
    ierr = MatSetValues(A,1,&I,1,&I,&v,ADD_VALUES); CHKERRA(ierr);
  }
  if (use_random) {ierr = PetscRandomDestroy(rctx); CHKERRA(ierr);}

  /* 
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  /* 
     Create parallel vectors.
      - When using VecCreate(), we specify only the vector's global
        dimension; the parallel partitioning is determined at runtime. 
      - Note: We form 1 vector from scratch and then duplicate as needed.
  */
  ierr = VecCreate(MPI_COMM_WORLD,dim,&u); CHKERRA(ierr);
  ierr = VecDuplicate(u,&b); CHKERRA(ierr); 
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);

  /* 
     Set exact solution; then compute right-hand-side vector.
  */
  
  if (use_random) {
    ierr = PetscRandomCreate(MPI_COMM_WORLD,RANDOM_DEFAULT,&rctx); CHKERRA(ierr);
    ierr = VecSetRandom(rctx,u); CHKERRA(ierr);
  } else {
    ierr = VecSet(&pfive,u); CHKERRA(ierr);
  }
  ierr = MatMult(A,u,b); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create linear solver context
  */
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);

  /* 
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN); CHKERRA(ierr);

  /* 
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
  */
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Check the error
  */
  ierr = VecAXPY(&none,u,x); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  if (norm > 1.e-12)
    PetscPrintf(MPI_COMM_WORLD,"Norm of error %g iterations %d\n",norm,its);
  else 
    PetscPrintf(MPI_COMM_WORLD,"Norm of error < 1.e-12 Iterations %d\n",its);

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  if (use_random) {ierr = PetscRandomDestroy(rctx); CHKERRA(ierr);}
  ierr = VecDestroy(u); CHKERRA(ierr); ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr); ierr = MatDestroy(A); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
