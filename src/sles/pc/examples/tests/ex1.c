#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.26 1997/07/09 20:53:11 balay Exp balay $";
#endif

static char help[] = "Tests the creation of a PC context.\n\n";

#include "pc.h"
#include "petsc.h"
#include <stdio.h>

int main(int argc,char **args)
{
  PC  pc;
  int ierr, n = 5;
  Vec u;
  Mat mat;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PCCreate(PETSC_COMM_WORLD,&pc); CHKERRA(ierr);
  ierr = PCSetType(pc,PCNONE); CHKERRA(ierr);

  /* Vector and matrix must be set before calling PCSetUp */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&u);CHKERRA(ierr);
  ierr = PCSetVector(pc,u); CHKERRA(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,3,PETSC_NULL,&mat); CHKERRA(ierr);
  ierr = PCSetOperators(pc,mat,mat,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = PCSetUp(pc); CHKERRA(ierr);

  ierr = VecDestroy(u);	CHKERRA(ierr);
  ierr = MatDestroy(mat); CHKERRA(ierr);
  ierr = PCDestroy(pc);	CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
    


