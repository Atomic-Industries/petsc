#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex16.c,v 1.1 1997/12/06 18:39:36 bsmith Exp bsmith $";
#endif

static char help[] = "Tests VecSetValuesBlocked() on MPI vectors\n\n";

#include "vec.h"
#include "sys.h"

int main(int argc,char **argv)
{
  int          i,n = 8, ierr, size,rank,bs = 2,indices[2];
  Scalar       values[4];
  Vec          x;

  PetscInitialize(&argc,&argv,(char*)0,help);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  if (size != 2) SETERRA(1,0,"Must be run with two processors");

  /* create vector */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,n,&x); CHKERRA(ierr);
  ierr = VecSetBlockSize(x,bs);CHKERRA(ierr);

  if (!rank) {
    for ( i=0; i<4; i++ ) values[i] = i+1;
    indices[0] = 0; indices[1] = 2;
    ierr = VecSetValuesBlocked(x,2,indices,values,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(x); CHKERRA(ierr);
  ierr = VecAssemblyEnd(x); CHKERRA(ierr);

  /* 
      Resulting vector should be 1 2 0 0 3 4 0 0
  */
  ierr = VecView(x,VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  ierr = VecDestroy(x); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
