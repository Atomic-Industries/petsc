#include <petsc/private/kspimpl.h>  /*I "petscksp.h" I*/
#include <petscviewersaws.h>

typedef struct {
  PetscViewer    viewer;
  PetscInt       neigs;
  PetscReal      *eigi;
  PetscReal      *eigr;
} KSPMonitor_SAWs;

/*@C
   KSPMonitorSAWsCreate - create an SAWs monitor context

   Collective

   Input Parameter:
.  ksp - KSP to monitor

   Output Parameter:
.  ctx - context for monitor

   Level: developer

.seealso: KSPMonitorSAWs(), KSPMonitorSAWsDestroy()
@*/
PetscErrorCode KSPMonitorSAWsCreate(KSP ksp,void **ctx)
{
  KSPMonitor_SAWs *mon;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(ksp,&mon));
  mon->viewer = PETSC_VIEWER_SAWS_(PetscObjectComm((PetscObject)ksp));
  PetscCheckFalse(!mon->viewer,PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Cannot create SAWs default viewer");
  *ctx = (void*)mon;
  PetscFunctionReturn(0);
}

/*@C
   KSPMonitorSAWsDestroy - destroy a monitor context created with KSPMonitorSAWsCreate()

   Collective

   Input Parameter:
.  ctx - monitor context

   Level: developer

.seealso: KSPMonitorSAWsCreate()
@*/
PetscErrorCode KSPMonitorSAWsDestroy(void **ctx)
{
  KSPMonitor_SAWs *mon = (KSPMonitor_SAWs*)*ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscFree2(mon->eigr,mon->eigi));
  CHKERRQ(PetscFree(*ctx));
  PetscFunctionReturn(0);
}

/*@C
   KSPMonitorSAWs - monitor solution using SAWs

   Logically Collective on ksp

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm (preconditioned) residual value (may be estimated).
-  ctx -  PetscViewer of type SAWs

   Level: advanced

.seealso: KSPMonitorSingularValue(), KSPComputeExtremeSingularValues(), PetscViewerSAWsOpen()
@*/
PetscErrorCode KSPMonitorSAWs(KSP ksp,PetscInt n,PetscReal rnorm,void *ctx)
{
  KSPMonitor_SAWs *mon   = (KSPMonitor_SAWs*)ctx;
  PetscReal       emax,emin;
  PetscMPIInt     rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  CHKERRQ(KSPComputeExtremeSingularValues(ksp,&emax,&emin));

  CHKERRQ(PetscFree2(mon->eigr,mon->eigi));
  CHKERRQ(PetscMalloc2(n,&mon->eigr,n,&mon->eigi));
  if (n) {
    CHKERRQ(KSPComputeEigenvalues(ksp,n,mon->eigr,mon->eigi,&mon->neigs));

    CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
    if (rank == 0) {
      SAWs_Delete("/PETSc/ksp_monitor_saws/eigr");
      SAWs_Delete("/PETSc/ksp_monitor_saws/eigi");

      PetscStackCallSAWs(SAWs_Register,("/PETSc/ksp_monitor_saws/rnorm",&ksp->rnorm,1,SAWs_READ,SAWs_DOUBLE));
      PetscStackCallSAWs(SAWs_Register,("/PETSc/ksp_monitor_saws/neigs",&mon->neigs,1,SAWs_READ,SAWs_INT));
      if (mon->neigs > 0) {
        PetscStackCallSAWs(SAWs_Register,("/PETSc/ksp_monitor_saws/eigr",mon->eigr,mon->neigs,SAWs_READ,SAWs_DOUBLE));
        PetscStackCallSAWs(SAWs_Register,("/PETSc/ksp_monitor_saws/eigi",mon->eigi,mon->neigs,SAWs_READ,SAWs_DOUBLE));
      }
      CHKERRQ(PetscInfo(ksp,"KSP extreme singular values min=%g max=%g\n",(double)emin,(double)emax));
      CHKERRQ(PetscSAWsBlock());
    }
  }
  PetscFunctionReturn(0);
}
