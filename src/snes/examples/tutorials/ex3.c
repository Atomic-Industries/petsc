#ifndef lint
static char vcid[] = "$Id: ex3.c,v 1.46 1997/02/06 17:34:43 curfman Exp bsmith $";
#endif

static char help[] = "Uses Newton-like methods to solve u'' + u^{2} = f in parallel.\n\
This example employs a user-defined monitoring routine.  This code also demonstrates\n\
use of the macro __FUNC__ to define routine names for use in error handling.\n\n";

/*T
   Concepts: SNES^Solving a system of nonlinear equations (basic parallel example);
   Concepts: SNES^Setting a user-defined monitoring routine;
   Concepts: Error Handling^Using the macro __FUNC__ to define routine names;
   Routines: SNESCreate(); SNESSetFunction(); SNESSetJacobian(); SNESSolve();
   Routines: SNESGetTolerances(); SNESSetFromOptions(); SNESSetMonitor();
   Routines: SNESGetSolution(); ViewerDrawOpenX(); PetscObjectSetName();
   Routines: DACreate1d(); DADestroy(); DAGetDistributedVector(); DAGetLocalVector();
   Routines: DAGetCorners(); DAGetGhostCorners();
   Routines: DAGlobalToLocalBegin(); DAGlobalToLocalEnd();
   Processors: n
T*/

/* 
   Include "draw.h" so that we can use distributed arrays (DAs).
   Include "draw.h" so that we can use PETSc drawing routines.
   Include "snes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petsc.h  - base PETSc routines   vec.h - vectors
     sys.h    - system routines       mat.h - matrices
     is.h     - index sets            ksp.h - Krylov subspace methods
     viewer.h - viewers               pc.h  - preconditioners
     sles.h   - linear solvers
*/

#include "da.h"
#include "snes.h"
#include <math.h>

/* 
   User-defined routines.  Note that immediately before each routine below,
   we define the macro __FUNC__ to be a string containing the routine name.
   If defined, this macro is used in the PETSc error handlers to provide a
   complete traceback of routine names.  All PETSc library routines use this
   macro, and users can optionally employ it as well in their application
   codes.  Note that users can get a traceback of PETSc errors regardless of
   whether they define __FUNC__ in application codes; this macro merely
   provides the added traceback detail of the application routine names.
*/
int FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
int FormFunction(SNES,Vec,Vec,void*);
int FormInitialGuess(Vec);
int Monitor(SNES,int,double,void *);

/* 
   User-defined application context
*/
typedef struct {
   DA     da;     /* distributed array */
   Vec    F;      /* right-hand-side of PDE */
   Vec    xlocal;     /* local work vector */
   int    rank;   /* rank of processor */
   int    size;   /* size of communicator */
   double h;      /* mesh spacing */
} ApplicationCtx;

/*
   User-defined context for monitoring
*/
typedef struct {
   Viewer viewer;
} MonitorCtx;

#undef __FUNC__
#define __FUNC__ "main"
int main( int argc, char **argv )
{
  SNES           snes;                 /* SNES context */
  Mat            J;                    /* Jacobian matrix */
  ApplicationCtx ctx;                  /* user-defined context */
  Vec            x, r, U, F;           /* vectors */
  MonitorCtx     monP;                 /* monitoring context */
  Scalar         xp, *FF, *UU, none = -1.0;
  int            ierr, its, N = 5, i, flg, maxit, maxf, xs, xm;
  double         atol, rtol, stol, norm;

  PetscInitialize( &argc, &argv,(char *)0,help );
  MPI_Comm_rank(MPI_COMM_WORLD,&ctx.rank);
  MPI_Comm_size(MPI_COMM_WORLD,&ctx.size);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&N,&flg); CHKERRA(ierr);
  ctx.h = 1.0/(N-1);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESCreate(MPI_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create distributed array (DA) to manage parallel grid and vectors
  */
  ierr = DACreate1d(MPI_COMM_WORLD,DA_NONPERIODIC,N,1,1,PETSC_NULL,&ctx.da);CHKERRA(ierr);

  /*
     Extract global and local vectors from DA; then duplicate for remaining
     vectors that are the same types
  */
  ierr = DAGetDistributedVector(ctx.da,&x); CHKERRA(ierr);
  ierr = DAGetLocalVector(ctx.da,&ctx.xlocal); CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);
  ierr = VecDuplicate(x,&F); CHKERRA(ierr); ctx.F = F;
  ierr = VecDuplicate(x,&U); CHKERRA(ierr); 

  /* 
     Set function evaluation routine and vector.  Whenever the nonlinear
     solver needs to compute the nonlinear function, it will call this
     routine.
      - Note that the final routine argument is the user-defined
        context that provides application-specific data for the
        function evaluation routine.
  */
  ierr = SNESSetFunction(snes,r,FormFunction,(void*)&ctx);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(MPI_COMM_WORLD,N,N,&J);CHKERRA(ierr);

  /* 
     Set Jacobian matrix data structure and default Jacobian evaluation
     routine.  Whenever the nonlinear solver needs to compute the
     Jacobian matrix, it will call this routine.
      - Note that the final routine argument is the user-defined
        context that provides application-specific data for the
        Jacobian evaluation routine.
  */
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,(void*)&ctx); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Set an optional user-defined monitoring routine
  */
  ierr = ViewerDrawOpenX(MPI_COMM_WORLD,0,0,0,0,400,400,&monP.viewer);CHKERRA(ierr);
  ierr = SNESSetMonitor(snes,Monitor,(void*)&monP); CHKERRA(ierr); 

  /*
     Set names for some vectors to facilitate monitoring (optional)
  */
  PetscObjectSetName((PetscObject)x,"Approximate Solution");
  PetscObjectSetName((PetscObject)U,"Exact Solution");

  /* 
     Set SNES/SLES/KSP/PC runtime options, e.g.,
         -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
  */
  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);

  /* 
     Print parameters used for convergence testing (optional) ... just
     to demonstrate this routine; this information is also printed with
     the option -snes_view
  */
  ierr = SNESGetTolerances(snes,&atol,&rtol,&stol,&maxit,&maxf); CHKERRA(ierr);
  PetscPrintf(MPI_COMM_WORLD,"atol=%g, rtol=%g, stol=%g, maxit=%d, maxf=%d\n",
     atol,rtol,stol,maxit,maxf);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize application:
     Store right-hand-side of PDE and exact solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Get local grid boundaries (for 1-dimensional DA):
       xs, xm - starting grid index, width of local grid (no ghost points)
  */
  ierr = DAGetCorners(ctx.da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = VecGetArray(F,&FF); CHKERRA(ierr);
  ierr = VecGetArray(U,&UU); CHKERRA(ierr);

  /*
     Compute local vector entries
  */
  xp = ctx.h*xs;
  for (i=0; i<xm; i++ ) {
    FF[i] = 6.0*xp + pow(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
    UU[i] = xp*xp*xp;
    xp += ctx.h;
  }

  /*
     Restore vectors
  */
  ierr = VecRestoreArray(F,&FF); CHKERRA(ierr);
  ierr = VecRestoreArray(U,&UU); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */
  ierr = FormInitialGuess(x); CHKERRA(ierr);
  ierr = SNESSolve(snes,x,&its); CHKERRA(ierr);
  PetscPrintf(MPI_COMM_WORLD,"Number of Newton iterations = %d\n\n", its );

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Check solution and clean up
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Check the error
  */
  ierr = VecAXPY(&none,U,x); CHKERRA(ierr);
  ierr  = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  if (norm > 1.e-12) 
    PetscPrintf(MPI_COMM_WORLD,"Norm of error %g, Iterations %d\n",norm,its);
  else 
    PetscPrintf(MPI_COMM_WORLD,"Norm of error < 1.e-12, Iterations %d\n",its);

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = ViewerDestroy(monP.viewer);  CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(ctx.xlocal); CHKERRA(ierr);
  ierr = VecDestroy(r); CHKERRA(ierr);
  ierr = VecDestroy(U); CHKERRA(ierr);
  ierr = VecDestroy(F); CHKERRA(ierr);
  ierr = MatDestroy(J); CHKERRA(ierr);
  ierr = SNESDestroy(snes); CHKERRA(ierr);
  ierr = DADestroy(ctx.da); CHKERRA(ierr);
  PetscFinalize();

  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "FormInitialGuess"
/*
   FormInitialGuess - Computes initial guess.

   Input/Output Parameter:
.  x - the solution vector
*/
int FormInitialGuess(Vec x)
{
   int    ierr;
   Scalar pfive = .50;
   ierr = VecSet(&pfive,x); CHKERRQ(ierr);
   return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "FormFunction"
/* 
   FormFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  ctx - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  f - function vector

   Note:
   The user-defined context can contain any application-specific
   data needed for the function evaluation.
*/
int FormFunction(SNES snes,Vec x,Vec f,void *ctx)
{
  ApplicationCtx *user = (ApplicationCtx*) ctx;
  DA             da = user->da;
  Scalar         *xx, *ff, *FF, d;
  int            i, ierr, N, xs, xm, gxs, gxm, xsi, xei, ilg, ilr;
  Vec            xlocal = user->xlocal;

  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
     By placing code between these two statements, computations can
     be done while messages are in transition.
  */
  ierr = DAGlobalToLocalBegin(da,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

  /*
     Get pointers to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
       - The vector xlocal includes ghost point; the vectors x and f do
         NOT include ghost points.
  */
  ierr = VecGetArray(xlocal,&xx); CHKERRQ(ierr);
  ierr = VecGetArray(f,&ff); CHKERRQ(ierr);
  ierr = VecGetArray(user->F,&FF); CHKERRQ(ierr);

  /*
     Get local grid boundaries (for 1-dimensional DA):
       xs, xm  - starting grid index, width of local grid (no ghost points)
       gxs, gxm - starting grid index, width of local grid (including ghost points)
  */
  ierr = DAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,PETSC_NULL,PETSC_NULL,&gxm,PETSC_NULL,
         PETSC_NULL); CHKERRQ(ierr);
  ierr = VecGetSize(f,&N); CHKERRQ(ierr);

  /*
     Set function values for boundary points; define local interior grid point range:
        xsi - starting interior grid index
        xei - ending interior grid index
  */
  if (xs == 0) { /* left boundary */
    ff[0] = xx[0];
    xsi = 1;
  } else {
    xsi = xs;
  }
  if (xs+xm == N) {  /* right boundary */
    ff[xs+xm-1-xs] = xx[xs+xm-1-gxs] - 1.0;
    xei = N-1;
  } else {
    xei = xs+xm;
  }

  /*
     Compute function over locally owned part of the grid (interior points only)
  */
  d = 1.0/(user->h*user->h);
  for ( i=xsi; i<xei; i++ ) {
    ilg = i-gxs;
    ilr = i-xs;
    ff[ilr] = d*(xx[ilg-1] - 2.0*xx[ilg] + xx[ilg+1]) + xx[ilg]*xx[ilg] - FF[ilr];
  }

  /*
     Restore vectors
  */
  ierr = VecRestoreArray(xlocal,&xx); CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&ff); CHKERRQ(ierr);
  ierr = VecRestoreArray(user->F,&FF); CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "FormJacobian"
/*
   FormJacobian - Evaluates Jacobian matrix.

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  dummy - optional user-defined context (not used here)

   Output Parameters:
.  jac - Jacobian matrix
.  B - optionally different preconditioning matrix
.  flag - flag indicating matrix structure
*/
int FormJacobian(SNES snes,Vec x,Mat *jac,Mat *B,MatStructure*flag,void *ctx)
{
  ApplicationCtx *user = (ApplicationCtx*) ctx;
  Scalar         *xx, d, A[3];
  int            i, j[3], ierr, start, end, M, N, istart, iend;

  /*
     Get pointer to vector data
  */
  ierr = VecGetArray(x,&xx); CHKERRQ(ierr);

  /*
    Get range of locally owned matrix
  */
  ierr = MatGetOwnershipRange(*jac,&start,&end); CHKERRQ(ierr);
  ierr = MatGetSize(*jac,&M,&N); CHKERRQ(ierr);

  /*
     Determine starting and ending local indices for interior grid points.
     Set Jacobian entries for boundary points.
  */

  if (start == 0) {  /* left boundary */
    i = 0; A[0] = 1.0; 
    ierr = MatSetValues(*jac,1,&i,1,&i,A,INSERT_VALUES); CHKERRQ(ierr);
    istart = 1;
  } else {
    istart = start;
  }
  if (end == M) { /* right boundary */
    i = N-1; A[0] = 1.0; 
    ierr = MatSetValues(*jac,1,&i,1,&i,A,INSERT_VALUES); CHKERRQ(ierr);
    iend = N-1;
  } else {
    iend = end;
  }

  /*
     Interior grid points
      - Note that in this case we set all elements for a particular
        row at once.
  */
  d = 1.0/(user->h*user->h);
  for ( i=istart; i<iend; i++ ) {
    j[0] = i - 1; j[1] = i; j[2] = i + 1; 
    A[0] = A[2] = d; A[1] = -2.0*d + 2.0*xx[i-start];
    ierr = MatSetValues(*jac,1,&i,3,j,A,INSERT_VALUES); CHKERRQ(ierr);
  }

  /* 
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.

     Also, restore vector.
  */

  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&xx); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "Monitor"
/*
   Monitor - User-defined monitoring routine that views the
   current iterate with an x-window plot.

   Input Parameters:
   snes - the SNES context
   its - iteration number
   norm - 2-norm function value (may be estimated)
   ctx - optional user-defined context for private data for the 
         monitor routine, as set by SNESSetMonitor()

   Note:
   See the manpage for ViewerDrawOpenX() for useful runtime options,
   such as -nox to deactivate all x-window output.
 */
int Monitor(SNES snes,int its,double fnorm,void *ctx)
{
  int        ierr;
  MonitorCtx *monP = (MonitorCtx*) ctx;
  Vec        x;

  PetscPrintf(MPI_COMM_WORLD,"iter = %d, SNES Function norm %g\n",its,fnorm);
  ierr = SNESGetSolution(snes,&x); CHKERRQ(ierr);
  ierr = VecView(x,monP->viewer); CHKERRQ(ierr);
  return 0;
}


