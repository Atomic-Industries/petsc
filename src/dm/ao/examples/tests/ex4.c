#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex4.c,v 1.2 1997/10/01 22:47:40 bsmith Exp bsmith $";
#endif

static char help[] = "Tests AOData loading\n\n";

#include "petsc.h"
#include "ao.h"
#include <math.h>

int main(int argc,char **argv)
{
  AOData      aodata;
  Viewer      binary;
  int         ierr,indices[4],*intv,i,rank;

  PetscInitialize(&argc,&argv,(char*)0,help);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  /*
        Load the database from the file
  */
  ierr = ViewerFileOpenBinary(PETSC_COMM_WORLD,"dataoutput",BINARY_RDONLY,&binary);CHKERRA(ierr);
  ierr = AODataLoadBasic(binary,&aodata);CHKERRA(ierr);
  ierr = ViewerDestroy(binary); CHKERRA(ierr);

  /*
        Access part of the data 
  */
  indices[0] = 0; indices[1] = 2; indices[2] = 1; indices[3] = 5;
  ierr = AODataGetSegment(aodata,"key1","seg1",4,indices,(void **)&intv);CHKERRQ(ierr);
  for (i=0; i<4; i++ ) {
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] %d %d\n",rank,i,intv[i]);
  }
  PetscSynchronizedFlush(PETSC_COMM_WORLD);
  ierr = AODataRestoreSegment(aodata,"key1","seg1",4,indices,(void **)&intv);CHKERRQ(ierr);
 
  ierr = AODataDestroy(aodata); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 


