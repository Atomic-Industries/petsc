#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: gmpre.c,v 1.12 1998/04/27 14:05:46 curfman Exp bsmith $";
#endif

#include "src/ksp/impls/gmres/gmresp.h"       /*I  "ksp.h"  I*/

#undef __FUNC__  
#define __FUNC__ "KSPGMRESSetPreAllocateVectors" 
/*@
    KSPGMRESSetPreAllocateVectors - Causes GMRES to preallocate all its
    needed work vectors at initial setup rather than the default, which 
    is to allocate them in chunks when needed.

    Collective on KSP

    Input Parameter:
.   ksp   - iterative context obtained from KSPCreate

    Options Database Key:
.   -ksp_gmres_preallocate - Activates KSPGmresSetPreAllocateVectors()

.keywords: GMRES, preallocate, vectors

.seealso: KSPGMRESSetRestart(), KSPGMRESSetOrthogonalization()
@*/
int KSPGMRESSetPreAllocateVectors(KSP ksp)
{
  int ierr, (*f)(KSP);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPGMRESSetPreAllocateVectors_C",(void **)&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

