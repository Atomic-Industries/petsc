#ifndef lint
static char vcid[] = "$Id: ex8.c,v 1.29 1996/07/08 22:16:40 bsmith Exp bsmith $";
#endif

static char help[] = "Demonstrates scattering with strided index sets.\n\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include <math.h>

int main(int argc,char **argv)
{
  int           n = 6, ierr, loc[6] = {0,1,2,3,4,5};
  Scalar        two = 2.0, vals[6] = {10,11,12,13,14,15};
  Vec           x,y;
  IS            is1,is2;
  VecScatter    ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,help);

  /* create two vectors */
  ierr = VecCreateSeq(MPI_COMM_SELF,n,&x); CHKERRA(ierr);
  ierr = VecDuplicate(x,&y); CHKERRA(ierr);

  /* create two index sets */
  ierr = ISCreateStride(MPI_COMM_SELF,3,0,2,&is1); CHKERRA(ierr);
  ierr = ISCreateStride(MPI_COMM_SELF,3,1,2,&is2); CHKERRA(ierr);

  ierr = VecSetValues(x,6,loc,vals,INSERT_VALUES); CHKERRA(ierr);
  VecView(x,VIEWER_STDOUT_SELF); PetscPrintf(MPI_COMM_SELF,"----\n");
  ierr = VecSet(&two,y); CHKERRA(ierr);
  ierr = VecScatterCreate(x,is1,y,is2,&ctx); CHKERRA(ierr);
  ierr = VecScatterBegin(x,y,INSERT_VALUES,SCATTER_ALL,ctx);CHKERRA(ierr);
  ierr = VecScatterEnd(x,y,INSERT_VALUES,SCATTER_ALL,ctx); CHKERRA(ierr);
  ierr = VecScatterDestroy(ctx); CHKERRA(ierr);
  
  ierr = VecView(y,VIEWER_STDOUT_SELF); CHKERRA(ierr);

  ierr = ISDestroy(is1); CHKERRA(ierr);
  ierr = ISDestroy(is2); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
