#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: vmpicr.c,v 1.1 1999/01/11 01:45:00 bsmith Exp bsmith $";
#endif

/*
   This file contains routines for Parallel vector operations.
 */

#include "src/vec/impls/mpi/pvecimpl.h"   /*I  "vec.h"   I*/

#undef __FUNC__  
#define __FUNC__ "VecCreateMPI"
/*@C
   VecCreateMPI - Creates a parallel vector.

   Collective on MPI_Comm
 
   Input Parameters:
+  comm - the MPI communicator to use 
.  n - local vector length (or PETSC_DECIDE to have calculated if N is given)
-  N - global vector length (or PETSC_DECIDE to have calculated if n is given)

   Output Parameter:
.  vv - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

.keywords: vector, create, MPI

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
          VecCreateMPIWithArray(), VecCreateGhostWithArray()

@*/ 
int VecCreateMPI(MPI_Comm comm, int n, int N, Vec *v)
{
  int ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,n,N,v); CHKERRQ(ierr);
  ierr = VecSetType(*v,"PETSc#VecMPI");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
