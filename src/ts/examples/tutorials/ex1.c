#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.6 1997/01/01 03:40:07 bsmith Exp bsmith $";
#endif

static char help[] ="Solves the time dependent Bratu problem using pseudo-timestepping";

/*
   Concepts: TS,Pseudo-timestepping,nonlinear problems
   Routines: TSCreate(),TSSetSolution(),TSSetRHSFunction(),TSSetRHSJacobian()
   Routines: TSSetType(), TS_PSEUDO, TSSetInitialTimeStep(),TSSetDuration()
   Routines: TSPseudoSetTimeStep(),TSSetFromOptions(),TSStep(),TSDestroy()
   Processors: 1

*/

/*
     Demonstrates how one may solve a nonlinear problem 
   with pseudo timestepping. In this simple example, the pseudo-time
   step is the same for all grid points, i.e. this is equivalent to 
   a backward Euler with variable timestep.

     Note: this example does not require pseudo-timestepping since it
   is an easy nonlinear problem; but it is included to demonstrate how
   the pseudo-timestepping may be done.

     See snes/examples/tutorials/ex4.c[ex4f.F] and 
   snes/examples/tutorials/ex5.c[ex5f.F] where the problem is described
   and solved using Newton's method alone.

*/
/* ------------------------------------------------------------------------------*/
/*
    Include "ts.h" to use the PETSc timestepping routines. Note that this
    file automatically includes: "petsc.h" and other lower level PETSc include files
*/
#include "ts.h"
#include <math.h>

/*
  Create an application context to contain data needed by the 
  application-provided call-back routines, FormJacobian() and
  FormFunction().
*/
typedef struct {
  double      param;        /* test problem parameter */
  int         mx;           /* Discretization in x-direction */
  int         my;           /* Discretization in y-direction */
} AppCtx;

/* 
   User-defined routines
*/
int  FormJacobian(TS,double,Vec,Mat*,Mat*,MatStructure*,void*),
     FormFunction(TS,double,Vec,Vec,void*),
     FormInitialGuess(Vec,AppCtx*);

int main( int argc, char **argv )
{
  TS     ts;                 /* timestepping context */
  Vec    x, r;               /* solution, residual vectors */
  Mat    J;                  /* Jacobian matrix */
  AppCtx user;               /* user-defined work context */
  int    its;                /* iterations for convergence */
  int    ierr, N, flg; 
  double param_max = 6.81, param_min = 0., dt;
  double ftime;

  PetscInitialize( &argc, &argv, PETSC_NULL,help );
  user.mx        = 4;
  user.my        = 4;
  user.param     = 6.0;
  
  /*
     Allow user to set the grid dimensions and nonlinearity parameter at run-time
  */
  OptionsGetInt(0,"-mx",&user.mx,&flg);
  OptionsGetInt(0,"-my",&user.my,&flg);
  OptionsGetDouble(0,"-param",&user.param,&flg);
  if (user.param >= param_max || user.param <= param_min) {
    SETERRQ(1,0,"Parameter is out of range");
  }
  dt = .5/PetscMax(user.mx,user.my);
  ierr = OptionsGetDouble(0,"-dt",&dt,&flg); CHKERRA(ierr);
  N          = user.mx*user.my;
  
  /* 
      Create vectors to hold the solution and function value
  */
  ierr = VecCreateSeq(MPI_COMM_SELF,N,&x); CHKERRA(ierr);
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);

  /*
    Create matrix to hold Jacobian. Preallocate 5 nonzeros per row
    in the sparse matrix. Note that this is not the optimal strategy; see
    the Performance chapter of the users manual for information on 
    preallocating memory in sparse matrices.
  */
  ierr = MatCreateSeqAIJ(MPI_COMM_SELF,N,N,5,0,&J); CHKERRA(ierr);

  /* 
     Create timestepper context 
  */
  ierr = TSCreate(MPI_COMM_WORLD,TS_NONLINEAR,&ts); CHKERRA(ierr);

  /*
     Tell the timestepper context where to compute solutions
  */
  ierr = TSSetSolution(ts,x); CHKERRA(ierr);

  /*
     Provide the call-back for the nonlinear function we are 
     evaluating. Thus whenever the timestepping routines need the
     function they will call this routine. Note the final argument
     is the application context used by the call-back functions.
  */
  ierr = TSSetRHSFunction(ts,FormFunction,&user); CHKERRA(ierr);

  /*
     Set the Jacobian matrix and the function used to compute 
     Jacobians.
  */
  ierr = TSSetRHSJacobian(ts,J,J,FormJacobian,&user);CHKERRA(ierr);

  /*
       For the initial guess for the problem
  */
  ierr = FormInitialGuess(x,&user);

  /*
       This indicates that we are using pseudo timestepping to 
     find a steady state solution to the nonlinear problem.
  */
  ierr = TSSetType(ts,TS_PSEUDO); CHKERRA(ierr);

  /*
       Set the initial time to start at (this is arbitrary for 
     steady state problems; and the initial timestep given above
  */
  ierr = TSSetInitialTimeStep(ts,0.0,dt); CHKERRA(ierr);

  /*
      Set a large number of timesteps and final duration time
     to insure convergence to steady state.
  */
  ierr = TSSetDuration(ts,1000,1.e12);

  /*
      Use the default strategy for increasing the timestep
  */
  ierr = TSPseudoSetTimeStep(ts,TSPseudoDefaultTimeStep,0);CHKERRA(ierr);

  /*
      Set any additional options from the options database. This
     includes all options for the nonlinear and linear solvers used
     internally the the timestepping routines.
  */
  ierr = TSSetFromOptions(ts); CHKERRA(ierr);

  ierr = TSSetUp(ts); CHKERRA(ierr);

  /*
      Perform the solve. This is where the timestepping takes place.
  */
  ierr = TSStep(ts,&its,&ftime); CHKERRA(ierr);
  
  printf( "Number of pseudo timesteps = %d final time %4.2e\n", its,ftime );

  /* 
     Free the data structures constructed above
  */
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(r); CHKERRA(ierr);
  ierr = MatDestroy(J); CHKERRA(ierr);
  ierr = TSDestroy(ts); CHKERRA(ierr);
  PetscFinalize();

  return 0;
}
/* ------------------------------------------------------------------ */
/*           Bratu (Solid Fuel Ignition) Test Problem                 */
/* ------------------------------------------------------------------ */

/* --------------------  Form initial approximation ----------------- */

int FormInitialGuess(Vec X,AppCtx *user)
{
  int     i, j, row, mx, my, ierr;
  double  one = 1.0, lambda;
  double  temp1, temp, hx, hy, hxdhy, hydhx;
  double  sc;
  Scalar  *x;

  mx	 = user->mx; 
  my	 = user->my;
  lambda = user->param;

  hx    = one / (double)(mx-1);
  hy    = one / (double)(my-1);
  sc    = hx*hy;
  hxdhy = hx/hy;
  hydhx = hy/hx;

  ierr = VecGetArray(X,&x); CHKERRQ(ierr);
  temp1 = lambda/(lambda + one);
  for (j=0; j<my; j++) {
    temp = (double)(PetscMin(j,my-j-1))*hy;
    for (i=0; i<mx; i++) {
      row = i + j*mx;  
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        x[row] = 0.0; 
        continue;
      }
      x[row] = temp1*sqrt( PetscMin( (double)(PetscMin(i,mx-i-1))*hx,temp) ); 
    }
  }
  ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);
  return 0;
}
/* --------------------  Evaluate Function F(x) --------------------- */

int FormFunction(TS ts,double t,Vec X,Vec F,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  int     ierr, i, j, row, mx, my;
  double  two = 2.0, one = 1.0, lambda;
  double  hx, hy, hxdhy, hydhx;
  Scalar  ut, ub, ul, ur, u, uxx, uyy, sc,*x,*f;

  mx	 = user->mx; 
  my	 = user->my;
  lambda = user->param;

  hx    = one / (double)(mx-1);
  hy    = one / (double)(my-1);
  sc    = hx*hy;
  hxdhy = hx/hy;
  hydhx = hy/hx;

  ierr = VecGetArray(X,&x); CHKERRQ(ierr);
  ierr = VecGetArray(F,&f); CHKERRQ(ierr);
  for (j=0; j<my; j++) {
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        f[row] = x[row];
        continue;
      }
      u = x[row];
      ub = x[row - mx];
      ul = x[row - 1];
      ut = x[row + mx];
      ur = x[row + 1];
      uxx = (-ur + two*u - ul)*hydhx;
      uyy = (-ut + two*u - ub)*hxdhy;
      f[row] = -uxx + -uyy + sc*lambda*exp(u);
    }
  }
  ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f); CHKERRQ(ierr);
  return 0; 
}
/* --------------------  Evaluate Jacobian F'(x) -------------------- */

int FormJacobian(TS ts,double t,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  Mat     jac = *B;
  int     i, j, row, mx, my, col[5], ierr;
  Scalar  two = 2.0, one = 1.0, lambda, v[5],sc, *x;
  double  hx, hy, hxdhy, hydhx;


  mx	 = user->mx; 
  my	 = user->my;
  lambda = user->param;

  hx    = 1.0 / (double)(mx-1);
  hy    = 1.0 / (double)(my-1);
  sc    = hx*hy;
  hxdhy = hx/hy;
  hydhx = hy/hx;

  ierr = VecGetArray(X,&x); CHKERRQ(ierr);
  for (j=0; j<my; j++) {
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        ierr = MatSetValues(jac,1,&row,1,&row,&one,INSERT_VALUES); CHKERRQ(ierr);
        continue;
      }
      v[0] = hxdhy; col[0] = row - mx;
      v[1] = hydhx; col[1] = row - 1;
      v[2] = -two*(hydhx + hxdhy) + sc*lambda*exp(x[row]); col[2] = row;
      v[3] = hydhx; col[3] = row + 1;
      v[4] = hxdhy; col[4] = row + mx;
      ierr = MatSetValues(jac,1,&row,5,col,v,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  return 0;
}




