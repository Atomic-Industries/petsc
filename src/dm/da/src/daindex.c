#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: daindex.c,v 1.21 1998/06/15 20:10:15 curfman Exp bsmith $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/dm/da/daimpl.h"    /*I   "da.h"   I*/

#undef __FUNC__  
#define __FUNC__ "DAGetGlobalIndices"
/*@C
   DAGetGlobalIndices - Returns the global node number of all local nodes,
   including ghost nodes.

   Not Collective

   Input Parameter:
.  da - the distributed array

   Output Parameters:
+  n - the number of local elements, including ghost nodes (or PETSC_NULL)
-  idx - the global indices

   Note: 
   For DA_STENCIL_STAR stencils the inactive corner ghost nodes are also included
   in the list of local indices (even though those nodes are not updated 
   during calls to DAXXXToXXX().

   Fortran Note:
   This routine is used differently from Fortran
$    DA          da
$    integer     da_array(1)
$    PetscOffset i_da
$    int         ierr
$       call DAGetGlobalIndices(da,da_array,i_da,ierr)
$
$   Access first local entry in list
$      value = da_array(i_da + 1)
$

   See the Fortran chapter of the users manual for details

.keywords: distributed array, get, global, indices, local-to-global

.seealso: DACreate2d(), DAGetGhostCorners(), DAGetCorners(), DALocalToGlobal()
          DAGlobalToLocal(), DALocalToLocal(), DAGetAO(), DAGetGlobalIndicesF90()
@*/
int DAGetGlobalIndices(DA da, int *n,int **idx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (n) *n   = da->Nl;
  *idx = da->idx;
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "DAGetAO"
/*@C
   DAGetAO - Gets the application ordering context for a distributed array.

   Not Collective, but AO is parallel if DA is parallel

   Input Parameter:
.  da - the distributed array

   Output Parameters:
.  ao - the application ordering context for DAs

   Notes:
   In this case, the AO maps to the natural grid ordering that would be used
   for the DA if only 1 processor were employed (ordering most rapidly in the
   x-direction, then y, then z).  Multiple degrees of freedom are numbered
   for each node (rather than 1 component for the whole grid, then the next
   component, etc.)

.keywords: distributed array, get, global, indices, local-to-global

.seealso: DACreate2d(), DAGetGhostCorners(), DAGetCorners(), DALocalToGlocal()
          DAGlobalToLocal(), DALocalToLocal(), DAGetGlobalIndices()
@*/
int DAGetAO(DA da, AO *ao)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  *ao = da->ao;
  PetscFunctionReturn(0);
}

/*MC
    DAGetGlobalIndicesF90 - Returns a Fortran90 pointer to the list of 
    global indices (global node number of all local nodes, including
    ghost nodes).

    Synopsis:
    DAGetGlobalIndicesF90(DA da,integer n,{Scalar, pointer :: idx(:)},integer ierr)

    Input Parameter:
.   da - the distributed array

    Output Parameters:
+   n - the number of local elements, including ghost nodes (or PETSC_NULL)
.   idx - the Fortran90 pointer to the global indices
-   ierr - error code

    Notes:
     Not yet supported for all F90 compilers

.keywords: distributed array, get, global, indices, local-to-global, f90

.seealso: DAGetGlobalIndices()
M*/
