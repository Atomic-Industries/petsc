#ifndef lint
static char vcid[] = "$Id: ex38.c,v 1.8 1996/07/08 22:20:09 bsmith Exp $";
#endif

static char help[] = "Tests MatSetValues() for column oriented storage.\n\n"; 

#include "mat.h"
#include <stdio.h>

int main(int argc,char **args)
{
  Mat         C; 
  int         i,  n = 5, midx[3], nidx[2], ierr,flg;
  Scalar      v[6];

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);

  ierr = MatCreate(MPI_COMM_WORLD,n,n,&C); CHKERRA(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-column_oriented",&flg); CHKERRA(ierr);
  if (flg) {ierr = MatSetOption(C,MAT_COLUMN_ORIENTED); CHKERRA(ierr);}
  for ( i=0; i<6; i++ ) v[i] = (double) i;
  midx[0] = 0; midx[1] = 2; midx[2] = 3;
  nidx[0] = 1; nidx[1] = 3;
  ierr = MatSetValues(C,3,midx,2,nidx,v,ADD_VALUES); CHKERRA(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  ierr = MatView(C,VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  ierr = MatDestroy(C); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}

 
