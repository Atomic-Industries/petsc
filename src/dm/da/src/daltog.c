#ifndef lint
static char vcid[] = "$Id: daltog.c,v 1.5 1996/12/16 21:10:43 balay Exp balay $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/da/daimpl.h"    /*I   "da.h"   I*/

#undef __FUNC__  
#define __FUNC__ "DALocalToGlobal"
/*@
   DALocalToGlobal - Maps values from the local patch back to the 
   global vector. The ghost points are discarded.

   Input Parameters:
.  da - the distributed array context
.  l  - the local values
.  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  g - the global vector

.keywords: distributed array, local to global

.seealso: DAGlobalToLocalBegin(), DACreate2d()
@*/
int DALocalToGlobal(DA da,Vec l, InsertMode mode,Vec g)
{
  int ierr;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  ierr = VecScatterBegin(l,g,mode,SCATTER_FORWARD,da->ltog); CHKERRQ(ierr);
  ierr = VecScatterEnd(l,g,mode,SCATTER_FORWARD,da->ltog); CHKERRQ(ierr);
  return 0;
}








