#ifndef lint
static char vcid[] = "$Id: jacobi.c,v 1.29 1997/01/06 20:23:31 balay Exp bsmith $";
#endif
/*
   Defines a  Jacobi preconditioner for any Mat implementation
*/
#include "src/pc/pcimpl.h"   /*I "pc.h" I*/
#include <math.h>

typedef struct {
  Vec diag;
  Vec diagsqrt;
} PC_Jacobi;

#undef __FUNC__  
#define __FUNC__ "PCSetUp_Jacobi"
static int PCSetUp_Jacobi(PC pc)
{
  int        ierr, i, n;
  PC_Jacobi  *jac = (PC_Jacobi *) pc->data;
  Vec        diag, diagsqrt;
  Scalar     *x;

  /* We set up both regular and symmetric preconditioning. Perhaps there
     actually should be an option to use only one or the other? */
  if (pc->setupcalled == 0) {
    ierr = VecDuplicate(pc->vec,&diag); CHKERRQ(ierr);
    PLogObjectParent(pc,diag);
    ierr = VecDuplicate(pc->vec,&diagsqrt); CHKERRQ(ierr);
    PLogObjectParent(pc,diagsqrt);
  }
  else {
    diag = jac->diag;
    diagsqrt = jac->diagsqrt;
  }
  ierr = MatGetDiagonal(pc->pmat,diag); CHKERRQ(ierr);
  ierr = MatGetDiagonal(pc->pmat,diagsqrt); CHKERRQ(ierr);
  ierr = VecReciprocal(diag); CHKERRQ(ierr);
  ierr = VecGetLocalSize(diagsqrt,&n); CHKERRQ(ierr);
  ierr = VecGetArray(diagsqrt,&x); CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    if (x[i] != 0.0) x[i] = 1.0/sqrt(PetscAbsScalar(x[i]));
  }
  jac->diag = diag;
  jac->diagsqrt = diagsqrt;

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCApply_Jacobi"
static int PCApply_Jacobi(PC pc,Vec x,Vec y)
{
  PC_Jacobi *jac = (PC_Jacobi *) pc->data;
  int       ierr;
  ierr = VecPointwiseMult(x,jac->diag,y); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCApplySymmetricLeftOrRight_Jacobi"
static int PCApplySymmetricLeftOrRight_Jacobi(PC pc,Vec x,Vec y)
{
  PC_Jacobi *jac = (PC_Jacobi *) pc->data;
  VecPointwiseMult(x,jac->diagsqrt,y);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCDestroy_Jacobi" /* ADIC Ignore */
static int PCDestroy_Jacobi(PetscObject obj)
{
  PC pc = (PC) obj;
  PC_Jacobi *jac = (PC_Jacobi *) pc->data;
  if (jac->diag)     VecDestroy(jac->diag);
  if (jac->diagsqrt) VecDestroy(jac->diagsqrt);
  PetscFree(jac);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCCreate_Jacobi"
int PCCreate_Jacobi(PC pc)
{
  PC_Jacobi *jac = PetscNew(PC_Jacobi); CHKPTRQ(jac);
  jac->diag          = 0;
  pc->apply          = PCApply_Jacobi;
  pc->setup          = PCSetUp_Jacobi;
  pc->destroy        = PCDestroy_Jacobi;
  pc->type           = PCJACOBI;
  pc->data           = (void *) jac;
  pc->view           = 0;
  pc->applyrich      = 0;
  pc->applysymmetricleft  = PCApplySymmetricLeftOrRight_Jacobi;
  pc->applysymmetricright = PCApplySymmetricLeftOrRight_Jacobi;
  return 0;
}


