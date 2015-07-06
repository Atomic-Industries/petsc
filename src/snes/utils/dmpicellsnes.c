#include <petsc/private/dmpicellimpl.h>    /*I   "petscdmpicell.h"   I*/

/************************** DMPICellSolve *******************************/

#undef __FUNCT__
#define __FUNCT__ "DMPICellSolve"
PetscErrorCode DMPICellSolve(DM dm)
{
  DM_PICell      *mesh = (DM_PICell *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);

  /* solve for potential and zero density for next solve */


  
  PetscFunctionReturn(0);
}
