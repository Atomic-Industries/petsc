#ifndef lint
static char vcid[] = "$Id: ex3.c,v 1.31 1995/12/21 18:29:41 bsmith Exp bsmith $";
#endif

static char help[] = "Tests parallel vector assembly.  Input arguments are\n\
  -n <length> : local vector length\n\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int          n = 5, ierr, size,rank,flg;
  Scalar       one = 1.0, two = 2.0, three = 3.0;
  Vec          x,y;
  int          idx;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0,help);
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg); if (n < 5) n = 5;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); 

  if (size < 2) SETERRA(1,"Must be run with at least two processors");

  /* create two vector */
  ierr = VecCreateSeq(MPI_COMM_SELF,n,&x); CHKERRA(ierr);
  ierr = VecCreateMPI(MPI_COMM_WORLD,n,PETSC_DECIDE,&y); CHKERRA(ierr);
  ierr = VecSet(&one,x); CHKERRA(ierr);
  ierr = VecSet(&two,y); CHKERRA(ierr);

  if (rank == 1) {
    idx = 2; ierr = VecSetValues(y,1,&idx,&three,INSERT_VALUES); CHKERRA(ierr);
    idx = 0; ierr = VecSetValues(y,1,&idx,&two,INSERT_VALUES); CHKERRA(ierr); 
    idx = 0; ierr = VecSetValues(y,1,&idx,&one,INSERT_VALUES); CHKERRA(ierr); 
  }
  else {
    idx = 7; ierr = VecSetValues(y,1,&idx,&three,INSERT_VALUES); CHKERRA(ierr); 
  } 
  ierr = VecAssemblyBegin(y); CHKERRA(ierr);
  ierr = VecAssemblyEnd(y); CHKERRA(ierr);

  ierr = VecView(y,STDOUT_VIEWER_WORLD); CHKERRA(ierr);

  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
