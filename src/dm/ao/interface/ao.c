#ifndef lint
static char vcid[] = "$Id: ao.c,v 1.7 1996/12/16 17:50:48 balay Exp balay $";
#endif
/*  
   Defines the abstract operations on AO (application orderings) 
*/
#include "src/ao/aoimpl.h"      /*I "ao.h" I*/

#undef __FUNC__  
#define __FUNC__ "AOView"
/*@
   AOView - Displays an application ordering.

   Input Parameters:
.  ao - the application ordering context
.  viewer - viewer used to display the set, for example VIEWER_STDOUT_SELF.

.keywords:application ordering

.seealso: ViewerFileOpenASCII()
@*/
int AOView(AO ao, Viewer viewer)
{
  PetscValidHeaderSpecific(ao,AO_COOKIE);
  return (*ao->view)((PetscObject)ao,viewer);
}

#undef __FUNC__  
#define __FUNC__ "AODestroy"
/*@
   AODestroy - Destroys an application ordering set.

   Input Parameters:
.  ao - the application ordering context

.keywords: destroy, application ordering

.seealso: AOCreateDebug(), AOCreateBasic()
@*/
int AODestroy(AO ao)
{
  if (!ao) return 0;
  PetscValidHeaderSpecific(ao,AO_COOKIE);
  return (*ao->destroy)((PetscObject)ao);
}


/* ---------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "AOPetscToApplicationIS"
/*@
   AOPetscToApplicationIS - Maps an index set in the PETSc ordering to 
   the application-defined ordering.

   Input Parameters:
.  ao - the application ordering context
.  is - the index set

.keywords: application ordering, mapping

.seealso: AOCreateDebug(), AOCreateBasic(), AOView(),AOApplicationToPetsc(),
          AOApplicationToPetscIS(),AOPetscToApplication()
@*/
int AOPetscToApplicationIS(AO ao,IS is)
{
  int n,*ia,ierr;
  PetscValidHeaderSpecific(ao,AO_COOKIE);
  ierr = ISGetSize(is,&n); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&ia); CHKERRQ(ierr);
  ierr = (*ao->ops.petsctoapplication)(ao,n,ia); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&ia); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "AOApplicationToPetscIS"
/*@
   AOApplicationToPetscIS - Maps an index set in the application-defined
   ordering to the PETSc ordering.

   Input Parameters:
.  ao - the application ordering context
.  is - the index set

.keywords: application ordering, mapping

.seealso: AOCreateDebug(), AOCreateBasic(), AOView(), AOPetscToApplication(),
          AOPetscToApplicationIS(), AOApplicationToPetsc()
@*/
int AOApplicationToPetscIS(AO ao,IS is)
{
  int n,*ia,ierr;
  PetscValidHeaderSpecific(ao,AO_COOKIE);
  ierr = ISGetSize(is,&n); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&ia); CHKERRQ(ierr);
  ierr = (*ao->ops.applicationtopetsc)(ao,n,ia); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&ia); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "AOPetscToApplication"
/*@
   AOPetscToApplication - Maps a set of integers in the PETSc ordering to 
   the application-defined ordering.

   Input Parameters:
.  ao - the application ordering context
.  n - the number of integers
.  ia - the integers

.keywords: application ordering, mapping

.seealso: AOCreateDebug(), AOCreateBasic(), AOView(),AOApplicationToPetsc(),
          AOPetscToApplicationIS(), AOApplicationToPetsc()
@*/
int AOPetscToApplication(AO ao,int n,int *ia)
{
  PetscValidHeaderSpecific(ao,AO_COOKIE);
  return (*ao->ops.petsctoapplication)(ao,n,ia);
}

#undef __FUNC__  
#define __FUNC__ "AOApplicationToPetsc"
/*@
   AOApplicationToPetsc - Maps a set of integers in the application-defined
   ordering to the PETSc ordering.

   Input Parameters:
.  ao - the application ordering context
.  n - the number of integers
.  ia - the integers

.keywords: application ordering, mapping

.seealso: AOCreateDebug(), AOCreateBasic(), AOView(), AOPetscToApplication(),
          AOPetscToApplicationIS(), AOApplicationToPetsc()
@*/
int AOApplicationToPetsc(AO ao,int n,int *ia)
{
  PetscValidHeaderSpecific(ao,AO_COOKIE);
  return (*ao->ops.applicationtopetsc)(ao,n,ia);
}





