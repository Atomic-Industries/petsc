#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: util2.c,v 1.4 1997/08/07 14:41:33 bsmith Exp bsmith $";
#endif

/*
   This file contains utility routines for finite difference
   approximation of Jacobian matrices.  This functionality for
   the TS component will eventually be incorporated as part of
   the base PETSc libraries.
*/
#include "src/ts/tsimpl.h"
#include "src/snes/snesimpl.h"
#include "src/fortran/custom/zpetsc.h"

int RHSFunction(TS,double,Vec,Vec,void*);
int RHSJacobianFD(TS,double,Vec,Mat*,Mat*,MatStructure *,void*);

/* -------------------------------------------------------------------*/

/* Temporary interface routine; this will be eliminated soon! */
#ifdef HAVE_FORTRAN_CAPS
#define setcroutinefromfortran_ SETCROUTINEFROMFORTRAN
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define setcroutinefromfortran_ setcroutinefromfortran
#endif

void setcroutinefromfortran_(TS ts,Mat A,Mat B,int *__ierr )
{
    *__ierr = TSSetRHSJacobian((TS)PetscToPointer( *(int*)(ts) ),
	                       (Mat)PetscToPointer( *(int*)(A) ),
	                       (Mat)PetscToPointer( *(int*)(B) ),RHSJacobianFD,PETSC_NULL);
}

/* -------------------------------------------------------------------*/
/*
   RHSJacobianFD - Computes the Jacobian using finite differences.

   Input Parameters:
.  ts - TS context
.  xx1 - compute Jacobian at this point
.  ctx - application's function context, as set with SNESSetFunction()

   Output Parameters:
.  J - Jacobian
.  B - preconditioner, same as Jacobian
.  flag - matrix flag

   Notes:
   This routine is slow and expensive, and is not currently optimized
   to take advantage of sparsity in the problem.

   Sparse approximations using colorings are also available and
   would be a much better alternative!
*/
int RHSJacobianFD(TS ts,double t,Vec xx1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  Vec      jj1,jj2,xx2;
  int      i,ierr,N,start,end,j;
  Scalar   dx, mone = -1.0,*y,scale,*xx,wscale;
  double   amax, epsilon = 1.e-8; /* assumes double precision */
  double   dx_min = 1.e-16, dx_par = 1.e-1;
  MPI_Comm comm;

  ierr = VecDuplicate(xx1,&jj1); CHKERRQ(ierr);
  ierr = VecDuplicate(xx1,&jj2); CHKERRQ(ierr);
  ierr = VecDuplicate(xx1,&xx2); CHKERRQ(ierr);

  PetscObjectGetComm((PetscObject)xx1,&comm);
  MatZeroEntries(*J);

  ierr = VecGetSize(xx1,&N); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(xx1,&start,&end); CHKERRQ(ierr);
  VecGetArray(xx1,&xx);
  ierr = TSComputeRHSFunction(ts,ts->ptime,xx1,jj1); CHKERRQ(ierr);

  /* Compute Jacobian approximation, 1 column at a time.
      xx1 = current iterate, jj1 = F(xx1)
      xx2 = perturbed iterate, jj2 = F(xx2)
   */
  for ( i=0; i<N; i++ ) {
    ierr = VecCopy(xx1,xx2); CHKERRQ(ierr);
    if ( i>= start && i<end) {
      dx = xx[i-start];
#if !defined(USE_PETSC_COMPLEX)
      if (dx < dx_min && dx >= 0.0) dx = dx_par;
      else if (dx < 0.0 && dx > -dx_min) dx = -dx_par;
#else
      if (abs(dx) < dx_min && real(dx) >= 0.0) dx = dx_par;
      else if (real(dx) < 0.0 && abs(dx) < dx_min) dx = -dx_par;
#endif
      dx *= epsilon;
      wscale = 1.0/dx;
      VecSetValues(xx2,1,&i,&dx,ADD_VALUES);
    }
    else {
      wscale = 0.0;
    }
    ierr = TSComputeRHSFunction(ts,t,xx2,jj2); CHKERRQ(ierr);
    ierr = VecAXPY(&mone,jj1,jj2); CHKERRQ(ierr);
    /* Communicate scale to all processors */
#if !defined(USE_PETSC_COMPLEX)
    MPI_Allreduce(&wscale,&scale,1,MPI_DOUBLE,MPI_SUM,comm);
#else
#endif
    VecScale(&scale,jj2);
    VecGetArray(jj2,&y);
    VecNorm(jj2,NORM_INFINITY,&amax); amax *= 1.e-14;
    for ( j=start; j<end; j++ ) {
      if (PetscAbsScalar(y[j-start]) > amax) {
        ierr = MatSetValues(*J,1,&j,1,&i,y+j-start,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
    VecRestoreArray(jj2,&y);
  }
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  *flag =  DIFFERENT_NONZERO_PATTERN;

  ierr = VecDestroy(jj1); CHKERRQ(ierr);
  ierr = VecDestroy(jj2); CHKERRQ(ierr);
  ierr = VecDestroy(xx2); CHKERRQ(ierr);

  return 0;
}
