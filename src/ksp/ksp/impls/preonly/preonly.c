
#include <petsc/private/kspimpl.h>

static PetscErrorCode KSPSetUp_PREONLY(KSP ksp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPSolve_PREONLY(KSP ksp)
{
  PetscBool      diagonalscale;
  PCFailedReason pcreason;

  PetscFunctionBegin;
  CHKERRQ(PCGetDiagonalScale(ksp->pc,&diagonalscale));
  PetscCheck(!diagonalscale,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);
  PetscCheck(ksp->guess_zero,PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Running KSP of preonly doesn't make sense with nonzero initial guess\n\
               you probably want a KSP type of Richardson");
  ksp->its = 0;
  CHKERRQ(KSP_PCApply(ksp,ksp->vec_rhs,ksp->vec_sol));
  CHKERRQ(PCGetFailedReasonRank(ksp->pc,&pcreason));
  /* Note: only some ranks may have this set; this may lead to problems if the caller assumes ksp->reason is set on all processes or just uses the result */
  if (pcreason) {
    CHKERRQ(VecSetInf(ksp->vec_sol));
    ksp->reason = KSP_DIVERGED_PC_FAILED;
  } else {
    ksp->its    = 1;
    ksp->reason = KSP_CONVERGED_ITS;
  }
  if (ksp->numbermonitors) {
    Vec       v;
    PetscReal norm;
    Mat       A;

    CHKERRQ(VecNorm(ksp->vec_rhs,NORM_2,&norm));
    CHKERRQ(KSPMonitor(ksp,0,norm));
    CHKERRQ(VecDuplicate(ksp->vec_rhs,&v));
    CHKERRQ(PCGetOperators(ksp->pc,&A,NULL));
    CHKERRQ(KSP_MatMult(ksp,A,ksp->vec_sol,v));
    CHKERRQ(VecAYPX(v,-1.0,ksp->vec_rhs));
    CHKERRQ(VecNorm(v,NORM_2,&norm));
    CHKERRQ(VecDestroy(&v));
    CHKERRQ(KSPMonitor(ksp,1,norm));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPMatSolve_PREONLY(KSP ksp, Mat B, Mat X)
{
  PetscBool      diagonalscale;
  PCFailedReason pcreason;

  PetscFunctionBegin;
  CHKERRQ(PCGetDiagonalScale(ksp->pc,&diagonalscale));
  PetscCheck(!diagonalscale,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);
  PetscCheck(ksp->guess_zero,PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Running KSP of preonly doesn't make sense with nonzero initial guess\n\
               you probably want a KSP type of Richardson");
  ksp->its = 0;
  CHKERRQ(PCMatApply(ksp->pc,B,X));
  CHKERRQ(PCGetFailedReason(ksp->pc,&pcreason));
  /* Note: only some ranks may have this set; this may lead to problems if the caller assumes ksp->reason is set on all processes or just uses the result */
  if (pcreason) {
    CHKERRQ(MatSetInf(X));
    ksp->reason = KSP_DIVERGED_PC_FAILED;
  } else {
    ksp->its    = 1;
    ksp->reason = KSP_CONVERGED_ITS;
  }
  PetscFunctionReturn(0);
}

/*MC
     KSPPREONLY - This implements a method that applies ONLY the preconditioner exactly once.
                  This may be used in inner iterations, where it is desired to
                  allow multiple iterations as well as the "0-iteration" case. It is
                  commonly used with the direct solver preconditioners like PCLU and PCCHOLESKY

   Options Database Keys:
.   -ksp_type preonly - use preconditioner only

   Level: beginner

   Notes:
    Since this does not involve an iteration the basic KSP parameters such as tolerances and iteration counts
          do not apply

    To apply multiple preconditioners in a simple iteration use KSPRICHARDSON

   Developer Notes:
    Even though this method does not use any norms, the user is allowed to set the KSPNormType to any value.
    This is so the users does not have to change KSPNormType options when they switch from other KSP methods to this one.

.seealso:  KSPCreate(), KSPSetType(), KSPType, KSP, KSPRICHARDSON, KSPCHEBYSHEV

M*/

PETSC_EXTERN PetscErrorCode KSPCreate_PREONLY(KSP ksp)
{
  PetscFunctionBegin;
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,3));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_RIGHT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2));

  ksp->data                = NULL;
  ksp->ops->setup          = KSPSetUp_PREONLY;
  ksp->ops->solve          = KSPSolve_PREONLY;
  ksp->ops->matsolve       = KSPMatSolve_PREONLY;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->view           = NULL;
  PetscFunctionReturn(0);
}
