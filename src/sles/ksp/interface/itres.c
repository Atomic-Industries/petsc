#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: itres.c,v 1.38 1999/01/31 16:08:34 bsmith Exp curfman $";
#endif

#include "src/sles/ksp/kspimpl.h"   /*I "ksp.h" I*/

#undef __FUNC__  
#define __FUNC__ "KSPResidual"
/*@
   KSPResidual - Computes the residual.

   Collective on KSP

   Input Parameters:
+  vsoln    - solution to use in computing residual
.  vt1, vt2 - temporary work vectors
.  vres     - calculated residual
.  vbinvf   - the result of binv^{-1} b.  If null, don't do it.
-  vb       - right-hand-side vector

   Notes:
   This routine assumes that an iterative method, designed for
$     A x = b
   will be used with a preconditioner, C, such that the actual problem is
$     M u = f    
   where M = AC (right preconditioning) or CA (left preconditioning).

   Level: intermediate

.keywords: KSP, residual

.seealso:  KSPMonitor()
@*/
int KSPResidual(KSP ksp,Vec vsoln,Vec vt1,Vec vt2,Vec vres, Vec vbinvf,Vec vb)
{
  Scalar        one = -1.0;
  MatStructure  pflag;
  Mat           Amat, Pmat;
  int           ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PCGetOperators(ksp->B,&Amat,&Pmat,&pflag);
  if (ksp->pc_side == PC_RIGHT) {
    if (vbinvf) {ierr = VecCopy(vb,vbinvf); CHKERRQ(ierr);}
    vbinvf = vb;
  } else if (ksp->pc_side == PC_LEFT) {
    ierr = PCApply(ksp->B,vb,vbinvf); CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"Only right and left preconditioning are currently supported");
  }
  if (!ksp->guess_zero) {
    /* compute initial residual: f - M*x */
    /* (inv(b)*a)*x or (a*inv(b)*b)*x into dest */
    if (ksp->pc_side == PC_RIGHT) {
      /* we want a * binv * b * x, or just a * x for the first step */
      /* a*x into temp */
      ierr = MatMult(Amat,vsoln,vt1); CHKERRQ(ierr);
    } else {
      /* else we do binv * a * x */
      ierr = PCApplyBAorAB(ksp->B,ksp->pc_side,vsoln,vt1,vt2); CHKERRQ(ierr);
    }
    /* This is an extra copy for the right-inverse case */
    ierr = VecCopy(vbinvf,vres); CHKERRQ(ierr);
    ierr = VecAXPY(&one,vt1,vres); CHKERRQ(ierr);
          /* inv(b)(f - a*x) into dest */
  } else {
    ierr = VecCopy(vbinvf,vres); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPUnwindPreconditioner"
/*@
   KSPUnwindPreconditioner - Unwinds the preconditioning in the solution.

   Collective on KSP

   Input Parameters:
+  ksp  - iterative context
.  vsoln - solution vector 
-  vt1   - temporary work vector

   Output Parameter:
.  vsoln - contains solution on output  

   Notes:
   If preconditioning either symmetrically or on the right, this routine solves 
   for the correction to the unpreconditioned problem.  If preconditioning on 
   the left, nothing is done.

   Level: advanced

.keywords: KSP, unwind, preconditioner

.seealso: KSPSetPreconditionerSide()
@*/
int KSPUnwindPreconditioner(KSP ksp,Vec vsoln,Vec vt1)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (ksp->pc_side == PC_RIGHT) {
    ierr = PCApply(ksp->B,vsoln,vt1); CHKERRQ(ierr);
    ierr = VecCopy(vt1,vsoln); CHKERRQ(ierr);
  } else if (ksp->pc_side == PC_SYMMETRIC) {
    ierr = PCApplySymmetricRight(ksp->B,vsoln,vt1); CHKERRQ(ierr);
    ierr = VecCopy(vt1,vsoln); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
