
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: aopart.c,v 1.1 1997/11/05 23:24:08 bsmith Exp bsmith $";
#endif

#include "ao.h"       /*I  "ao.h"  I*/

#undef __FUNC__
#define __FUNC__ "AODataKeyPartition"
/*@C
     AODataKeyPartition - Partition a key across the processors to reduce
        communication costs.

    Input Parameters:
.    aodata - the database
.    key - the key you wish partitioned and renumbered

.seealso: AODataSegmentPartition()
@*/
int AODataKeyPartition(AOData aodata,char *key)
{
  AO              ao;
  Mat             adj;
  Partitioning    part;
  IS              is,isg;
  int             ierr;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  PetscObjectGetComm((PetscObject) aodata,&comm);

  ierr = AODataKeyGetAdjacency(aodata,key,&adj);CHKERRA(ierr);
  ierr = PartitioningCreate(comm,&part);CHKERRA(ierr);
  ierr = PartitioningSetAdjacency(part,adj);CHKERRA(ierr);
  ierr = PartitioningSetFromOptions(part);CHKERRA(ierr);
  ierr = PartitioningApply(part,&is);CHKERRA(ierr);
  ierr = PartitioningDestroy(part); CHKERRA(ierr);
  ierr = MatDestroy(adj);CHKERRQ(ierr);
  ierr = ISPartitioningToNumbering(is,&isg);CHKERRA(ierr);
  ierr = ISDestroy(is);CHKERRQ(ierr);

  ierr = AOCreateBasicIS(isg,PETSC_NULL,&ao);CHKERRA(ierr);
  ierr = ISDestroy(isg);CHKERRA(ierr);

  ierr = AODataKeyRemap(aodata,key,ao);CHKERRA(ierr);
  ierr = AODestroy(ao);CHKERRA(ierr);
  PetscFunctionReturn(0);
}
