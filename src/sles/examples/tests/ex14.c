#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex14.c,v 1.11 1998/04/16 04:02:44 curfman Exp bsmith $";
#endif

/* Program usage:  mpirun -np <procs> ex14 [-help] [all PETSc options] */

static char help[] = "Solves a nonlinear system in parallel with a user-defined\n\
Newton method that uses SLES to solve the linearized Newton sytems.  This solver\n\
is a very simplistic inexact Newton method.  The intent of this code is to\n\
demonstrate the repeated solution of linear sytems with the same nonzero pattern.\n\
\n\
This is NOT the recommended approach for solving nonlinear problems with PETSc!\n\
We urge users to employ the SNES component for solving nonlinear problems whenever\n\
possible, as it offers many advantages over coding nonlinear solvers independently.\n\
\n\
We solve the  Bratu (SFI - solid fuel ignition) problem in a 2D rectangular\n\
domain, using distributed arrays (DAs) to partition the parallel grid.\n\
The command line options include:\n\
  -par <parameter>, where <parameter> indicates the problem's nonlinearity\n\
     problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)\n\
  -mx <xg>, where <xg> = number of grid points in the x-direction\n\
  -my <yg>, where <yg> = number of grid points in the y-direction\n\
  -Nx <npx>, where <npx> = number of processors in the x-direction\n\
  -Ny <npy>, where <npy> = number of processors in the y-direction\n\n";

/*T
   Concepts: SLES^Writing a user-defined nonlinear solver (parallel Bratu example);
   Concepts: DA^Using distributed arrays;
   Routines: SLESCreate(); SLESSetOperators(); SLESSolve(); SLESSetFromOptions();
   Routines: DACreate2d(); DADestroy(); DACreateGlobalVector(); DACreateLocalVector();
   Routines: DAGetCorners(); DAGetGhostCorners(); DALocalToGlobal();
   Routines: DAGlobalToLocalBegin(); DAGlobalToLocalEnd(); DAGetGlobalIndices();
   Processors: n
T*/

/* ------------------------------------------------------------------------

    Solid Fuel Ignition (SFI) problem.  This problem is modeled by
    the partial differential equation
  
            -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1 ,
  
    with boundary conditions
   
             u = 0  for  x = 0, x = 1, y = 0, y = 1.
  
    A finite difference approximation with the usual 5-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear 
    system of equations.

    The SNES version of this problem is:  snes/examples/tutorials/ex5.c
    We urge users to employ the SNES component for solving nonlinear
    problems whenever possible, as it offers many advantages over coding 
    nonlinear solvers independently.

  ------------------------------------------------------------------------- */

/* 
   Include "da.h" so that we can use distributed arrays (DAs).
   Include "sles.h" so that we can use SLES solvers.  Note that this
   file automatically includes:
     petsc.h  - base PETSc routines   vec.h - vectors
     sys.h    - system routines       mat.h - matrices
     is.h     - index sets            ksp.h - Krylov subspace methods
     viewer.h - viewers               pc.h  - preconditioners
*/
#include "da.h"
#include "sles.h"
#include <math.h>

/* 
   User-defined application context - contains data needed by the 
   application-provided call-back routines, ComputeJacobian() and
   ComputeFunction().
*/
typedef struct {
   double      param;          /* test problem parameter */
   int         mx,my;          /* discretization in x, y directions */
   Vec         localX, localF; /* ghosted local vector */
   DA          da;             /* distributed array data structure */
   int         rank;           /* processor rank */
} AppCtx;

/* 
   User-defined routines
*/
int ComputeFunction(AppCtx*,Vec,Vec), FormInitialGuess(AppCtx*,Vec);
int ComputeJacobian(AppCtx*,Vec,Mat,MatStructure*);

int main( int argc, char **argv )
{
  /* -------------- Data to define application problem ---------------- */
  MPI_Comm comm;                /* communicator */
  SLES     sles;                /* linear solver */
  Vec      X, Y, F;             /* solution, update, residual vectors */
  Mat      J;                   /* Jacobian matrix */
  AppCtx   user;                /* user-defined work context */
  int      Nx, Ny;              /* number of preocessors in x- and y- directions */
  int      size;                /* number of processors */
  double   bratu_lambda_max = 6.81, bratu_lambda_min = 0.;
  int      m, flg, N, ierr;

  /* --------------- Data to define nonlinear solver -------------- */
  double   rtol = 1.e-8;        /* relative convergence tolerance */
  double   xtol = 1.e-8;        /* step convergence tolerance */
  double   ttol;                /* convergence tolerance */
  double   fnorm, ynorm, xnorm; /* various vector norms */
  int      max_nonlin_its = 10; /* maximum number of iterations for nonlinear solver */
  int      max_functions = 50;  /* maximum number of function evaluations */
  int      lin_its;             /* number of linear solver iterations for each step */
  MatStructure mat_flag;        /* flag indicating structure of preconditioner matrix */
  int      i;                   /* nonlinear solve iteration number */
  int      no_output;           /* flag indicating whether to surpress output */
  Scalar   mone = -1.0;       

  PetscInitialize( &argc, &argv,(char *)0,help );
  comm = PETSC_COMM_WORLD;
  MPI_Comm_rank(comm,&user.rank);
  ierr = OptionsHasName(PETSC_NULL,"-no_output",&no_output); CHKERRA(ierr);

  /*
     Initialize problem parameters
  */
  user.mx = 4; user.my = 4; user.param = 6.0;
  ierr = OptionsGetInt(PETSC_NULL,"-mx",&user.mx,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-my",&user.my,&flg); CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-par",&user.param,&flg); CHKERRA(ierr);
  if (user.param >= bratu_lambda_max || user.param <= bratu_lambda_min) {
    SETERRA(1,0,"Lambda is out of range");
  }
  N = user.mx*user.my;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create linear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SLESCreate(comm,&sles); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create distributed array (DA) to manage parallel grid and vectors
  */
  MPI_Comm_size(comm,&size);
  Nx = PETSC_DECIDE; Ny = PETSC_DECIDE;
  ierr = OptionsGetInt(PETSC_NULL,"-Nx",&Nx,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-Ny",&Ny,&flg); CHKERRA(ierr);
  if (Nx*Ny != size && (Nx != PETSC_DECIDE || Ny != PETSC_DECIDE))
    SETERRA(1,0,"Incompatible number of processors:  Nx * Ny != size");
  ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_STAR,user.mx,
                    user.my,Nx,Ny,1,1,PETSC_NULL,PETSC_NULL,&user.da); CHKERRA(ierr);

  /*
     Extract global and local vectors from DA; then duplicate for remaining
     vectors that are the same types
  */
  ierr = DACreateGlobalVector(user.da,&X); CHKERRA(ierr);
  ierr = DACreateLocalVector(user.da,&user.localX); CHKERRA(ierr);
  ierr = VecDuplicate(X,&F); CHKERRA(ierr);
  ierr = VecDuplicate(X,&Y); CHKERRA(ierr);
  ierr = VecDuplicate(user.localX,&user.localF); CHKERRA(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure for Jacobian
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Note:  For the parallel case, vectors and matrices MUST be partitioned
     accordingly.  When using distributed arrays (DAs) to create vectors,
     the DAs determine the problem partitioning.  We must explicitly
     specify the local matrix dimensions upon its creation for compatibility
     with the vector distribution.  Thus, the generic MatCreate() routine
     is NOT sufficient when working with distributed arrays.

     Note: Here we only approximately preallocate storage space for the
     Jacobian.  See the users manual for a discussion of better techniques
     for preallocating matrix memory.
  */
  if (size == 1) {
    ierr = MatCreateSeqAIJ(comm,N,N,5,PETSC_NULL,&J); CHKERRA(ierr);
  } else {
    ierr = VecGetLocalSize(X,&m); CHKERRA(ierr);
    ierr = MatCreateMPIAIJ(comm,m,m,N,N,5,PETSC_NULL,3,PETSC_NULL,&J); CHKERRA(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize linear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Set runtime options (e.g., -ksp_monitor -ksp_rtol <rtol> -ksp_type <type>)
  */
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = FormInitialGuess(&user,X); CHKERRA(ierr);
  ierr = ComputeFunction(&user,X,F); CHKERRA(ierr);   /* Compute F(X)    */
  ierr = VecNorm(F,NORM_2,&fnorm); CHKERRA(ierr);     /* fnorm = || F || */
  ttol = fnorm*rtol;
  if (!no_output) PetscPrintf(comm,"Initial function norm = %g\n",fnorm);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system with a user-defined method
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
      This solver is a very simplistic inexact Newton method, with no
      no damping strategies or bells and whistles. The intent of this code
      is  merely to demonstrate the repeated solution with SLES of linear
      sytems with the same nonzero structure.

      This is NOT the recommended approach for solving nonlinear problems
      with PETSc!  We urge users to employ the SNES component for solving
      nonlinear problems whenever possible with application codes, as it
      offers many advantages over coding nonlinear solvers independently.
   */

  for ( i=0; i<max_nonlin_its; i++ ) {

    /* 
        Compute the Jacobian matrix.  See the comments in this routine for
        important information about setting the flag mat_flag.
     */
    ierr = ComputeJacobian(&user,X,J,&mat_flag); CHKERRA(ierr);

    /* 
        Solve J Y = F, where J is the Jacobian matrix.
          - First, set the SLES linear operators.  Here the matrix that
            defines the linear system also serves as the preconditioning
            matrix.
          - Then solve the Newton system.
     */
    ierr = SLESSetOperators(sles,J,J,mat_flag); CHKERRA(ierr);
    ierr = SLESSolve(sles,F,Y,&lin_its); CHKERRA(ierr);

    /* 
       Compute updated iterate
     */
    ierr = VecNorm(Y,NORM_2,&ynorm); CHKERRA(ierr);       /* ynorm = || Y || */
    ierr = VecAYPX(&mone,X,Y); CHKERRA(ierr);             /* Y <- X - Y      */
    ierr = VecCopy(Y,X); CHKERRA(ierr);                   /* X <- Y          */
    ierr = VecNorm(X,NORM_2,&xnorm); CHKERRA(ierr);       /* xnorm = || X || */
    if (!no_output) PetscPrintf(comm,"   linear solve iterations = %d, xnorm=%g, ynorm=%g\n",
                                lin_its,xnorm,ynorm);

    /* 
       Evaluate new nonlinear function
     */
    ierr = ComputeFunction(&user,X,F); CHKERRA(ierr);     /* Compute F(X)    */
    ierr = VecNorm(F,NORM_2,&fnorm); CHKERRA(ierr);       /* fnorm = || F || */
    if (!no_output) PetscPrintf(comm,"Iteration %d, function norm = %g\n",i+1,fnorm);

    /*
       Test for convergence
     */
    if (fnorm <= ttol) {
      if (!no_output) PetscPrintf(comm,
         "Converged due to function norm %g < %g (relative tolerance)\n",fnorm,ttol);
      break;
    }
    if (ynorm < xtol*(xnorm)) {
      if (!no_output) PetscPrintf(comm,
         "Converged due to small update length: %g < %g * %g\n",ynorm,xtol,xnorm);
      break;
    }
    if (i > max_functions) {
      if (!no_output) PetscPrintf(comm,
         "Exceeded maximum number of function evaluations: %d > %d\n",i, max_functions );
      break;
    }  
  }
  PetscPrintf(comm,"Number of Newton iterations = %d\n",i+1);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatDestroy(J); CHKERRA(ierr);           ierr = VecDestroy(Y); CHKERRA(ierr);
  ierr = VecDestroy(user.localX); CHKERRA(ierr); ierr = VecDestroy(X); CHKERRA(ierr);
  ierr = VecDestroy(user.localF); CHKERRA(ierr); ierr = VecDestroy(F); CHKERRA(ierr);      
  ierr = SLESDestroy(sles); CHKERRA(ierr);  ierr = DADestroy(user.da); CHKERRA(ierr);
  PetscFinalize();

  return 0;
}
/* ------------------------------------------------------------------- */
/* 
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
int FormInitialGuess(AppCtx *user,Vec X)
{
  int     i, j, row, mx, my, ierr, xs, ys, xm, ym, gxm, gym, gxs, gys;
  double  one = 1.0, lambda, temp1, temp, hx, hy;
  Scalar  *x;
  Vec     localX = user->localX;

  mx = user->mx;            my = user->my;            lambda = user->param;
  hx = one/(double)(mx-1);  hy = one/(double)(my-1);
  temp1 = lambda/(lambda + one);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = VecGetArray(localX,&x); CHKERRQ(ierr);

  /*
     Get local grid boundaries (for 2-dimensional DA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)
       gxs, gys - starting grid indices (including ghost points)
       gxm, gym - widths of local grid (including ghost points)
  */
  ierr = DAGetCorners(user->da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL); CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&gxs,&gys,PETSC_NULL,&gxm,&gym,PETSC_NULL); CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    temp = (double)(PetscMin(j,my-j-1))*hy;
    for (i=xs; i<xs+xm; i++) {
      row = i - gxs + (j - gys)*gxm; 
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        x[row] = 0.0; 
        continue;
      }
      x[row] = temp1*sqrt( PetscMin( (double)(PetscMin(i,mx-i-1))*hx,temp) ); 
    }
  }

  /*
     Restore vector
  */
  ierr = VecRestoreArray(localX,&x); CHKERRQ(ierr);

  /*
     Insert values into global vector
  */
  ierr = DALocalToGlobal(user->da,localX,INSERT_VALUES,X); CHKERRQ(ierr);
  return 0;
} 
/* ------------------------------------------------------------------- */
/* 
   ComputeFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  X - input vector
.  user - user-defined application context

   Output Parameter:
.  F - function vector
 */
int ComputeFunction(AppCtx *user,Vec X,Vec F)
{
  int     ierr, i, j, row, mx, my, xs, ys, xm, ym, gxs, gys, gxm, gym;
  double  two = 2.0, one = 1.0, lambda,hx, hy, hxdhy, hydhx,sc;
  Scalar  u, uxx, uyy, *x,*f;
  Vec     localX = user->localX, localF = user->localF; 

  mx = user->mx;            my = user->my;            lambda = user->param;
  hx = one/(double)(mx-1);  hy = one/(double)(my-1);
  sc = hx*hy*lambda;        hxdhy = hx/hy;            hydhx = hy/hx;

  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DAGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX); CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX); CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = VecGetArray(localX,&x); CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f); CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DAGetCorners(user->da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL); CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&gxs,&gys,PETSC_NULL,&gxm,&gym,PETSC_NULL); CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    row = (j - gys)*gxm + xs - gxs - 1; 
    for (i=xs; i<xs+xm; i++) {
      row++;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        f[row] = x[row];
        continue;
      }
      u = x[row];
      uxx = (two*u - x[row-1] - x[row+1])*hydhx;
      uyy = (two*u - x[row-gxm] - x[row+gxm])*hxdhy;
      f[row] = uxx + uyy - sc*exp(u);
    }
  }

  /*
     Restore vectors
  */
  ierr = VecRestoreArray(localX,&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f); CHKERRQ(ierr);

  /*
     Insert values into global vector
  */
  ierr = DALocalToGlobal(user->da,localF,INSERT_VALUES,F); CHKERRQ(ierr);
  PLogFlops(11*ym*xm);
  return 0; 
} 
/* ------------------------------------------------------------------- */
/*
   ComputeJacobian - Evaluates Jacobian matrix.

   Input Parameters:
.  x - input vector
.  user - user-defined application context

   Output Parameters:
.  jac - Jacobian matrix
.  flag - flag indicating matrix structure

   Notes:
   Due to grid point reordering with DAs, we must always work
   with the local grid points, and then transform them to the new
   global numbering with the "ltog" mapping (via DAGetGlobalIndices()).
   We cannot work directly with the global numbers for the original
   uniprocessor grid!
*/
int ComputeJacobian(AppCtx *user,Vec X,Mat jac,MatStructure *flag)
{
  Vec     localX = user->localX;   /* local vector */
  int     *ltog;                   /* local-to-global mapping */
  int     ierr, i, j, row, mx, my, col[5];
  int     nloc, xs, ys, xm, ym, gxs, gys, gxm, gym, grow;
  Scalar  two = 2.0, one = 1.0, lambda, v[5], hx, hy, hxdhy, hydhx, sc, *x;

  mx = user->mx;            my = user->my;            lambda = user->param;
  hx = one/(double)(mx-1);  hy = one/(double)(my-1);
  sc = hx*hy;               hxdhy = hx/hy;            hydhx = hy/hx;

  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DAGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX); CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX); CHKERRQ(ierr);

  /*
     Get pointer to vector data
  */
  ierr = VecGetArray(localX,&x); CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DAGetCorners(user->da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL); CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&gxs,&gys,PETSC_NULL,&gxm,&gym,PETSC_NULL); CHKERRQ(ierr);

  /*
     Get the global node numbers for all local nodes, including ghost points
  */
  ierr = DAGetGlobalIndices(user->da,&nloc,&ltog); CHKERRQ(ierr);

  /* 
     Compute entries for the locally owned part of the Jacobian.
      - Currently, all PETSc parallel matrix formats are partitioned by
        contiguous chunks of rows across the processors. The "grow"
        parameter computed below specifies the global row number 
        corresponding to each local grid point.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly). 
      - Always specify global row and columns of matrix entries.
      - Here, we set all entries for a particular row at once.
  */
  for (j=ys; j<ys+ym; j++) {
    row = (j - gys)*gxm + xs - gxs - 1; 
    for (i=xs; i<xs+xm; i++) {
      row++;
      grow = ltog[row];
      /* boundary points */
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        ierr = MatSetValues(jac,1,&grow,1,&grow,&one,INSERT_VALUES); CHKERRQ(ierr);
        continue;
      }
      /* interior grid points */
      v[0] = -hxdhy; col[0] = ltog[row - gxm];
      v[1] = -hydhx; col[1] = ltog[row - 1];
      v[2] = two*(hydhx + hxdhy) - sc*lambda*exp(x[row]); col[2] = grow;
      v[3] = -hydhx; col[3] = ltog[row + 1];
      v[4] = -hxdhy; col[4] = ltog[row + gxm];
      ierr = MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES); CHKERRQ(ierr);
    }
  }

  /* 
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = VecRestoreArray(localX,&x); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  /*
     Set flag to indicate that the Jacobian matrix retains an identical
     nonzero structure throughout all nonlinear iterations (although the
     values of the entries change). Thus, we can save some work in setting
     up the preconditioner (e.g., no need to redo symbolic factorization for
     ILU/ICC preconditioners).
      - If the nonzero structure of the matrix is different during
        successive linear solves, then the flag DIFFERENT_NONZERO_PATTERN
        must be used instead.  If you are unsure whether the matrix
        structure has changed or not, use the flag DIFFERENT_NONZERO_PATTERN.
      - Caution:  If you specify SAME_NONZERO_PATTERN, PETSc
        believes your assertion and does not check the structure
        of the matrix.  If you erroneously claim that the structure
        is the same when it actually is not, the new preconditioner
        will not function correctly.  Thus, use this optimization
        feature with caution!
  */
  *flag = SAME_NONZERO_PATTERN;
  return 0;
}
