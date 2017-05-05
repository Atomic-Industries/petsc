
static char help[] ="Solves a simple time-dependent linear PDE (the heat equation).\n\
Input parameters include:\n\
  -m <points>, where <points> = number of grid points\n\
  -time_dependent_rhs : Treat the problem as having a time-dependent right-hand side\n\
  -debug              : Activate debugging printouts\n\
  -nox                : Deactivate x-window graphics\n\n";

/*
   Concepts: TS^time-dependent linear problems
   Concepts: TS^heat equation
   Concepts: TS^diffusion equation
   Processors: 1
*/
// Run with
//./heatEDMlinear -ts_monitor -ts_view -log_summary -ts_exact_final_time matchstep -ts_type beuler -ts_exact_final_time matchstep -ts_dt 1e-2 -Nl 20
/* ------------------------------------------------------------------------

   This program solves the one-dimensional heat equation (also called the
   diffusion equation),
       u_t = u_xx,
   on the domain 0 <= x <= 1, with the boundary conditions
       u(t,0) = 0, u(t,1) = 0,
   and the initial condition
       u(0,x) = sin(6*pi*x) + 3*sin(2*pi*x).
   This is a linear, second-order, parabolic equation.

   We discretize the right-hand side using the spectral element method
   We then demonstrate time evolution using the various TS methods by
   running the program via heatEDMlinear -ts_type <timestepping solver>

   Adjoints version 
  ------------------------------------------------------------------------- */

/*
   Include "petscts.h" so that we can use TS solvers.  Note that this file
   automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h  - vectors
     petscmat.h  - matrices
     petscis.h     - index sets            petscksp.h  - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h   - preconditioners
     petscksp.h   - linear solvers        petscsnes.h - nonlinear solvers
*/

#include <petscts.h>
#include "petscvec.h" 
#include "petscgll.h"
#include <petscdraw.h>
#include <petscdmda.h>
#include <petscmat.h>  
#include <petsctao.h>   
#include "petscdmlabel.h" 
/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/
typedef struct {
  Vec         obj;          /* desired end state */
  DM          da;                /* distributed array data structure */
  Vec         localwork;         /* local ghosted work vector */
  Vec         u_local;           /* local ghosted approximate solution vector */
  Vec         grid;              /* total grid */   
  Vec         mass;              /* mass matrix for total integration */
  Vec         grad;
  Vec         ic;
  Vec         curr_sol;
  Mat         stiff;             // stifness matrix
  Mat         adj;           //adjoint jacobian    
  PetscInt    Nl;                 /* total number of grid points */
  PetscInt    E;                 /* number of elements */
  PetscReal   *Z;                 /* mesh grid */
  PetscReal   *mult;                 /* multiplicity*/
  PetscScalar *W;                 /* weights */
  PetscBool   debug;             /* flag (1 indicates activation of debugging printouts) */
  PetscViewer viewer1,viewer2;  /* viewers for the solution and error */
  PetscReal   norm_2,norm_max,norm_L2;  /* error norms */
  PetscInt    steps;
  PetscReal   ftime;
  PetscReal   mu;
  PetscReal   dt;
  PetscReal   L;
  PetscReal   Le;
  PetscReal   Tadj;
} AppCtx;
/*
   User-defined routines
*/
extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);
extern PetscErrorCode RHSMatrixHeatgllDM(PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSAdjointgllDM(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSFunctionHeat(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode Objective(PetscReal,Vec,AppCtx*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  /*TS             ts;                */     /* timestepping context */
  Tao            tao;
  KSP            ksp;
  PC                 pc;
  Mat            A;                      /* matrix data structure */
  Vec            u,ic;                      /* approximate solution vector */
  PetscReal      time_total_max = 0.1, Tadj=0.2; /* default max total time, should be short since the decay is very fast, for slower decay higher viscosity needed */
  PetscInt       time_steps_max = 200;   /* default max timesteps */
  PetscErrorCode ierr;
  PetscInt       Nl=8,i, E=3, xs, xm, ind, j, lenglob; /*steps*/
  PetscMPIInt    size;
  PetscReal      dt=5e-4, x, *wrk_ptr1, *wrk_ptr2, L=1.0, Le;
  //PetscBool      flg;
  PetscGLLIP     gll;
  PetscViewer    viewfile;
   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBegin;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");
  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOCG);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(NULL,NULL,"-Nl",&Nl,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-E",&E,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-debug",&appctx.debug);CHKERRQ(ierr);

  appctx.Nl       = Nl;
  appctx.E        = E;
  appctx.L        = L;
  Le=L/appctx.E;
  appctx.Le       = appctx.L/appctx.E;
  appctx.norm_2   = 0.0;
  appctx.norm_max = 0.0;
  appctx.steps=time_steps_max;
  appctx.ftime=time_total_max; 
  appctx.mu       = 0.001;
  appctx.dt       = dt;
  appctx.Tadj     = Tadj;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Solving a linear TS problem on 1 processor\n");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscGLLIPCreate(Nl,PETSCGLLIP_VIA_LINEARALGEBRA,&gll);CHKERRQ(ierr);
  
  ierr= PetscMalloc1(Nl, &appctx.Z);
  ierr= PetscMalloc1(Nl, &appctx.W);
  ierr= PetscMalloc1(Nl, &appctx.mult);

  for(i=0; i<Nl; i++)
     { 
     appctx.Z[i]=(gll.nodes[i]+1.0);
     appctx.W[i]=gll.weights[i]; //not correct, needs weights, check Paul's book
     appctx.mult[i]=1.0;
      }

  appctx.mult[0]=0.5;
  appctx.mult[Nl-1]=0.5; 

  //lenloc   = E*Nl; only if I want to do it totally local for explicit
  lenglob  = E*(Nl-1)+1;

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
     and to set up the ghost point communication pattern.  There are E*(Nl-1)+1
     total grid values spread equally among all the processors, except first and last
  */

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,lenglob,1,1,NULL,&appctx.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(appctx.da);CHKERRQ(ierr);
  ierr = DMSetUp(appctx.da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(appctx.da,NULL,&E,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
 
  /*
     Extract global and local vectors from DMDA; we use these to store the
     approximate solution.  Then duplicate these for remaining vectors that
     have the same types.
  */
  ierr = DMCreateGlobalVector(appctx.da,&u);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(appctx.da,&ic);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(appctx.da,&appctx.u_local);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.ic);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.obj);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.grid);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.grad);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.mass);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.curr_sol);CHKERRQ(ierr);
 
  ierr = DMDAGetCorners(appctx.da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(appctx.da,appctx.grid,&wrk_ptr1);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx.da,appctx.mass,&wrk_ptr2);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
    xs=xs/(appctx.Nl-1);
    xm=xm/(appctx.Nl-1);
  
  /* 
     Build total grid and mass over entire mesh (multi-elemental) 
  */ 

   for (i=xs; i<xs+xm; i++) {
      for (j=0; j<appctx.Nl; j++)
      {
      x = (appctx.Le/2.0)*(appctx.Z[j])+appctx.Le*i; 
      ind=i*(appctx.Nl-1)+j;
      wrk_ptr1[ind]=x;
      wrk_ptr2[ind]=appctx.Le/2.0*appctx.W[j]*appctx.mult[j];
      } 
  }

   ierr = DMDAVecRestoreArray(appctx.da,appctx.grid,&wrk_ptr1);CHKERRQ(ierr);
   ierr = DMDAVecRestoreArray(appctx.da,appctx.mass,&wrk_ptr2);CHKERRQ(ierr);
//VecSetValues(Vec x,PetscInt ni,const PetscInt ix[],const PetscScalar y[],InsertMode iora)

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"grid.m",&viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    //ierr = MatView(appctx.stiff,viewfile);CHKERRQ(ierr);
    //ierr = MatView(appctx.adj,viewfile);CHKERRQ(ierr);
    ierr = VecView(appctx.grid,viewfile);CHKERRQ(ierr);
    ierr = VecView(appctx.mass,viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewfile);CHKERRQ(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  //ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  //ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);
  //ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
  //ierr = TSSetDM(ts,appctx.da);CHKERRQ(ierr);
 
 /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

     Create matrix data structure; set matrix evaluation routine.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = DMSetMatrixPreallocateOnly(appctx.da, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMCreateMatrix(appctx.da,&A);CHKERRQ(ierr);
    ierr = DMCreateMatrix(appctx.da,&appctx.stiff);CHKERRQ(ierr);
    
    /*
       For linear problems with a time-dependent f(u,t) in the equation
       u_t = f(u,t), the user provides the discretized right-hand-side
       as a time-dependent matrix.
    */
  ierr = RHSMatrixHeatgllDM(0.0,u,A,A,&appctx);CHKERRQ(ierr);
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&appctx.stiff);CHKERRQ(ierr);
  ierr = MatScale(A, -1.0);
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&appctx.adj);CHKERRQ(ierr);
  //ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,&appctx);CHKERRQ(ierr);
  //ierr = TSSetRHSJacobian(ts,appctx.stiff,appctx.stiff,TSComputeRHSJacobianConstant,&appctx);CHKERRQ(ierr);
    
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize timestepping solver:
       - Set the solution method to be the Backward Euler method.
       - Set timestepping duration info
     Then set runtime options, which can override these defaults.
     For example,
          -ts_max_steps <maxsteps> -ts_final_time <maxtime>
     to override the defaults set by TSSetDuration().
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  //ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  //ierr = TSSetDM(ts,appctx.da);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solution vector and initial timestep
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  //ierr = InitialConditions(u,&appctx);CHKERRQ(ierr);
  //ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);
  //ierr = VecView(u,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  /*
     Run the timestepping solver
  */
  //ierr = TSSolve(ts,u);CHKERRQ(ierr);
  //ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     View timestepping solver info
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  //ierr = PetscPrintf(PETSC_COMM_SELF,"avg. error (2 norm) = %g, avg. error (max norm) = %g\n",(double)(appctx.norm_2/steps),(double)(appctx.norm_L2/steps));CHKERRQ(ierr);
  //ierr = TSView(ts,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);



  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOCG);CHKERRQ(ierr);

  ierr = InitialConditions(ic,&appctx);CHKERRQ(ierr);
  ierr = VecDuplicate(ic,&appctx.ic);CHKERRQ(ierr);
  ierr = VecCopy(ic,appctx.ic);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,ic);CHKERRQ(ierr);

  /* Set routine for function and gradient evaluation */
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void *)&appctx);CHKERRQ(ierr);

  /* Check for any TAO command line options */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  ierr = TaoGetKSP(tao,&ksp);CHKERRQ(ierr);
  if (ksp) {
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  }

  ierr = TaoSetTolerances(tao,1e-5,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

  /* SOLVE THE APPLICATION */
  ierr = TaoSolve(tao); CHKERRQ(ierr);

  /* Free TAO data structures */
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     For matlab output
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"obbbj.m",&viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    //ierr = MatView(appctx.stiff,viewfile);CHKERRQ(ierr);
    //ierr = MatView(appctx.adj,viewfile);CHKERRQ(ierr);
    ierr = VecView(appctx.curr_sol,viewfile);CHKERRQ(ierr);
    ierr = VecView(appctx.ic,viewfile);CHKERRQ(ierr);
    ierr = VecView(appctx.obj,viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewfile);CHKERRQ(ierr);
 
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    //ierr = TSDestroy(&ts);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = VecDestroy(&u);CHKERRQ(ierr);
    ierr = VecDestroy(&appctx.obj);CHKERRQ(ierr);
    ierr = PetscGLLIPDestroy(&gll);CHKERRQ(ierr);
    ierr = DMDestroy(&appctx.da);CHKERRQ(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).
  */
    ierr = PetscFinalize();
    return ierr;
}
/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "InitialConditions"
/*
   InitialConditions - Computes the solution at the initial time.

   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode InitialConditions(Vec u,AppCtx *appctx)
{
  PetscScalar    *u_localptr;
  PetscErrorCode ierr;
  PetscInt       i,j,ind,xs,xm;
  PetscReal      x,xx,Le;

  /*
    Get a pointer to vector data.
    - For default PETSc vectors, VecGetArray() returns a pointer to
      the data array.  Otherwise, the routine is implementation dependent.
    - You MUST call VecRestoreArray() when you no longer need access to
      the array.
    - Note that the Fortran interface to VecGetArray() differs from the
      C version.  See the users manual for details.
  */
  
  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArray(appctx->da,u,&u_localptr);CHKERRQ(ierr);
  
  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */

    Le=appctx->Le;
    xs=xs/(appctx->Nl-1);
    xm=xm/(appctx->Nl-1);
    //I could also apply this to the entire grid as a function using PF

/*    old sharp grad
   for (i=xs; i<xs+xm; i++) {
      for (j=0; j<appctx->Nl; j++)
      {
      x = (Le/2.0)*(appctx->Z[j])+Le*i; 
      xx= 0.5*(PetscSinScalar(PETSC_PI*6.*x) + 3.*PetscSinScalar(PETSC_PI*2.*x));
      ind=i*(appctx->Nl-1)+j;
      u_localptr[ind]=xx;
      } 
     }
*/

   for (i=xs; i<xs+xm; i++) {
      for (j=0; j<appctx->Nl; j++)
      {
      x = (Le/2.0)*(appctx->Z[j])+Le*i; 
      xx= PetscSinScalar(2.0*PETSC_PI*x) ;
      ind=i*(appctx->Nl-1)+j;
      u_localptr[ind]=xx;
      } 
     }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(appctx->da,u,&u_localptr);CHKERRQ(ierr);

  //minor test.. to be removed
  //ierr = PetscPrintf(PETSC_COMM_SELF,"Initial cond inside routine \n");CHKERRQ(ierr);
  //ierr = VecView(u,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  /*
     Print debugging information if desired
  */
  if (appctx->debug) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Initial guess vector\n");CHKERRQ(ierr);
    ierr = VecView(u,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  return 0;
}
/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "Objective"
/*
   Sets the profile at end time

   Input Parameters:
   t - current time
   obj - vector storing the end function
   appctx - user-defined application context

   Output Parameter:
   solution - vector with the newly computed exact solution
*/
PetscErrorCode Objective(PetscReal t,Vec obj,AppCtx *appctx)
{
  PetscScalar    *s_localptr,ex1,ex2,sc1,sc2,tc = t;
  PetscErrorCode ierr;
  PetscReal      xx, x, Le;
  PetscInt       i, xs, xm, j, ind;

  
  /*
     Simply write the solution directly into the array locations.
     Alternatively, we culd use VecSetValues() or VecSetValuesLocal().
  */
  ex1 = PetscExpScalar(-4.*PETSC_PI*tc);
  ex2 = PetscExpScalar(-4.*tc);
  sc1 = PETSC_PI*6.0;                 
  sc2 = PETSC_PI*2.0;
 

  ierr = DMDAVecGetArray(appctx->da,obj,&s_localptr);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
    xs=xs/(appctx->Nl-1);
    xm=xm/(appctx->Nl-1);

    Le=appctx->Le;
  /* old sharp grad  
   for (i=xs; i<xs+xm; i++) {
      for (j=0; j<appctx->Nl; j++)
      {
      x = (Le/2.0)*(appctx->Z[j])+Le*i; 
      xx= PetscSinScalar(sc1*x)*ex1 + 3.*PetscSinScalar(sc2*x)*ex2;
      ind=i*(appctx->Nl-1)+j;
      s_localptr[ind]=xx;
      } 
  }
*/

  for (i=xs; i<xs+xm; i++) {
      for (j=0; j<appctx->Nl; j++)
      {
      x = (Le/2.0)*(appctx->Z[j])+Le*i; 
      xx= PetscSinScalar(2.0*PETSC_PI*x)*PetscExpScalar(-0.4*tc);
      ind=i*(appctx->Nl-1)+j;
      s_localptr[ind]=xx;
      } 
   }



  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(appctx->da,obj,&s_localptr);CHKERRQ(ierr);
  
  return 0;
}


/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "RHSMatrixHeatgllDM"

/*
   RHSMatrixHeat - User-provided routine to compute the right-hand-side
   matrix for the heat equation.

   Input Parameters:
ss   ts - the TS context
   t - current time
   global_in - global input vector
   dummy - optional user-defined context, as set by TSetRHSJacobian()

   Output Parameters:
   AA - Jacobian matrix
   BB - optionally different preconditioning matrix
   str - flag indicating matrix structure

*/
PetscErrorCode RHSMatrixHeatgllDM(PetscReal t,Vec X,Mat A,Mat BB,void *ctx)
{
  //Mat            A=AA;                /* Jacobian matrix */
  PetscReal      **temp, Le, init;
  PetscReal      vv;
  PetscGLLIP     gll;
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  PetscInt       N=appctx->Nl;
  /*  PetscInt       E=appctx->E;*/
  PetscErrorCode ierr;
  PetscInt       i,xs,xn,l,j,id;
  PetscInt       *rowsDM; /*rows[2], */
  PetscViewer    viewfile;
  
   Le=appctx->Le; // this should be in the appctx, but I think I need a new struct only for grid info

    /*
       Creates the element stiffness matrix for the given gll
    */
   ierr = PetscGLLIPCreate(N,PETSCGLLIP_VIA_LINEARALGEBRA,&gll);CHKERRQ(ierr);
   ierr = PetscGLLIPElementStiffnessCreate(&gll,&temp);CHKERRQ(ierr);

    /*
        Create the global stiffness matrix and add the element stiffness for each local element
    */
    
    // scale by the mass matrix
    for (i=0; i<N; i++) 
     {
       vv=-appctx->mu*4.0/(Le*Le)*(appctx->mult[i]/appctx->W[i]); //note here I took the multiplicities in
       for (j=0; j<N; j++)
               {
                temp[i][j]=temp[i][j]*vv; 
                //printf("prefactor %f \n",temp[i][j]);
               }
      } 
    init=temp[0][0];
    
    ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL);CHKERRQ(ierr);

    xs   = xs/(N-1);
    xn   = xn/(N-1);

    ierr = PetscMalloc1(N,&rowsDM);CHKERRQ(ierr);

    /*
        loop over local elements
    */
    for (j=xs; j<xs+xn; j++) {
      for (l=0; l<N; l++) 
          {rowsDM[l] = j*(N-1)+l;
           
           }
      ierr = MatSetValues(A,N,rowsDM,N,rowsDM,&temp[0][0],ADD_VALUES);CHKERRQ(ierr);
    }

   id=0;
   ierr = MatSetValues(A,1,&id,1,&id,&init,ADD_VALUES);CHKERRQ(ierr);
   id=appctx->E*(appctx->Nl-1);
   ierr = MatSetValues(A,1,&id,1,&id,&init,ADD_VALUES);CHKERRQ(ierr);

   MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
   MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

   //Set BCs 
   //rows[0] = 0;
   //rows[1] = E*(N-1);
   //ierr = MatZeroRowsColumns(A,2,rows,0.0,appctx->grid,appctx->grid);CHKERRQ(ierr);

  
   // Output only for testing
   ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"inside.m",&viewfile);CHKERRQ(ierr);
   ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
   ierr = MatView(A,viewfile);CHKERRQ(ierr);
   ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);
   ierr = PetscViewerDestroy(&viewfile);CHKERRQ(ierr);

   ierr = PetscGLLIPElementStiffnessDestroy(&gll,&temp);CHKERRQ(ierr);
   
     /*
     Set and option to indicate that we will never add a new nonzero location
     to the matrix. If we do, it will generate an error.
  */
  //ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  return 0;
}

/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "RHSAdjointgllDM"

/*
   RHSMatrixHeat - User-provided routine to compute the right-hand-side
   matrix for the heat equation.

   Input Parameters:
   ts - the TS context
   t - current time
   global_in - global input vector
   dummy - optional user-defined context, as set by TSetRHSJacobian()

   Output Parameters:
   AA - Jacobian matrix
   BB - optionally different preconditioning matrix
   str - flag indicating matrix structure

*/
PetscErrorCode RHSAdjointgllDM(TS ts,PetscReal t,Vec X,Mat A,Mat BB,void *ctx)
{
  //Mat            A=AA;                /* Jacobian matrix */
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  /*PetscErrorCode ierr;*/
  /*PetscViewer    viewfile;*/
  
  //ierr=MatCopy(A,appctx->stiff,SAME_NONZERO_PATTERN);
  //ierr=MatScale(A,-1.0);
  A=appctx->adj;
  
  return 0;
}
/* ------------------------------------------------------------------ */
#undef __FUNCT__
#define __FUNCT__ "FormFunctionGradient"
/*
   FormFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetObjectiveAndGradientRoutine()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient
*/
PetscErrorCode FormFunctionGradient(Tao tao,Vec IC,PetscReal *f,Vec G,void *ctx)
{
  AppCtx           *appctx = (AppCtx*)ctx;     /* user-defined application context */
  TS                ts;
  PetscErrorCode    ierr;
  /*Mat               A;*/                      /* matrix data structure */
  Vec               temp, temp2; /*, u;*/
  PetscInt          its;
  PetscReal         ff, gnorm, cnorm, xdiff; 
  TaoConvergedReason reason;      
  PetscViewer    viewfile;

  ierr = VecCopy(IC,appctx->curr_sol);CHKERRQ(ierr);
  

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);
  ierr = TSSetDM(ts,appctx->da);CHKERRQ(ierr);
  //ierr = TSSetRHSFunction(ts,NULL,RHSFunction,user);CHKERRQ(ierr);

  //ierr = RHSMatrixHeatgllDM(ts,0.0,appctx->ic,A,A,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,appctx->stiff,appctx->stiff,TSComputeRHSJacobianConstant,&appctx);CHKERRQ(ierr);
   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set time
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,appctx->dt);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,appctx->steps,appctx->ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  ierr = TSSetTolerances(ts,1e-7,NULL,1e-7,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Save trajectory of solution so that TSAdjointSolve() may be used
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSolve(ts,appctx->curr_sol);CHKERRQ(ierr);
  //ierr = TSGetSolveTime(ts,&appctx->ftime);CHKERRQ(ierr);
  //ierr = TSGetTimeStepNumber(ts,&appctx->steps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"steps %D, ftime %g\n",appctx->steps,appctx->ftime);CHKERRQ(ierr);

 /*
     Compute the L2-norm of the objective function, cost function is f
  */
  ierr = Objective(appctx->Tadj,appctx->obj,appctx);CHKERRQ(ierr);
  //ierr = VecView(appctx->obj,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = VecDuplicate(appctx->obj,&temp);CHKERRQ(ierr);
  ierr = VecCopy(appctx->obj,temp);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Temp\n");CHKERRQ(ierr);
  //ierr = VecView(temp,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr   = VecAXPY(temp,-1.0,appctx->curr_sol);CHKERRQ(ierr);

  ierr   = VecDuplicate(appctx->obj,&temp2);CHKERRQ(ierr);
  ierr   = VecPointwiseMult(temp2,temp,temp);CHKERRQ(ierr);
  ierr   = VecDot(temp2,appctx->mass,f);CHKERRQ(ierr);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*  
   Initial conditions for the adjoint integration, given by 2*obj'=temp (rewrite)
   */
  
  ierr = VecScale(temp, -2.0);
  ierr = VecCopy(temp,appctx->grad);CHKERRQ(ierr);

  
  ierr = TSSetCostGradients(ts,1,&appctx->grad,NULL);CHKERRQ(ierr);
  //ierr = TSAdjointSetUp(ts);CHKERRQ(ierr);
  //ierr = TSSetDM(ts,appctx->da);CHKERRQ(ierr); 
  /* Set RHS Jacobian  for the adjoint integration */
  // instead of A you should have appctx->A (but you gotta store it first)
  
  //ierr = VecScale(A, -1.0);
  //ierr = MatDuplicate(A,MAT_COPY_VALUES,appctx->stiff);CHKERRQ(ierr);
  //ierr = VecScale(A, -1.0);
  ierr = TSSetRHSJacobian(ts,appctx->adj,appctx->adj,TSComputeRHSJacobianConstant,appctx);CHKERRQ(ierr);

  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);
  //ierr = TSGetSolution(ts,&temp); //does not work for adjoints
  //ierr = TSGetRHSJacobian(ts,&A,NULL,NULL,NULL);
  
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"inside.m",&viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    //ierr = MatView(appctx.stiff,viewfile);CHKERRQ(ierr);
    //ierr = MatView(appctx.adj,viewfile);CHKERRQ(ierr);
    ierr = VecView(appctx->grad,viewfile);CHKERRQ(ierr);
    ierr = VecView(temp,viewfile);CHKERRQ(ierr);
    ierr = VecView(appctx->ic,viewfile);CHKERRQ(ierr);
    ierr = VecView(appctx->curr_sol,viewfile);CHKERRQ(ierr);
    ierr = VecView(appctx->obj,viewfile);CHKERRQ(ierr);
    ierr = MatView(appctx->adj,viewfile);CHKERRQ(ierr);
    ierr = MatView(appctx->stiff,viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewfile);CHKERRQ(ierr);
    //exit(1);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Cost function f=%f\n",(double)(*f));CHKERRQ(ierr);
    ierr = VecCopy(appctx->grad,G);CHKERRQ(ierr);
    ierr= TaoGetSolutionStatus(tao, &its, &ff, &gnorm, &cnorm, &xdiff, &reason);
    PetscPrintf(PETSC_COMM_WORLD,"iteration=%D\tf=%g, cnorm %f\n",its,(double)ff,f);
    //VecView(G,PETSC_VIEWER_STDOUT_SELF);
    ierr = TSDestroy(&ts);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "RHSFunctionHeatgllDM"
PetscErrorCode RHSFunctionHeatgllDM(TS ts,PetscReal t,Vec globalin,Vec globalout,void *ctx)
{
  PetscErrorCode ierr;
  Mat            A;

  PetscFunctionBeginUser;
  ierr = TSGetRHSJacobian(ts,&A,NULL,NULL,&ctx);CHKERRQ(ierr);
  ierr = RHSMatrixHeatgllDM(t,globalin,A,NULL,ctx);CHKERRQ(ierr);
  /* ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  ierr = MatMult(A,globalin,globalout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

