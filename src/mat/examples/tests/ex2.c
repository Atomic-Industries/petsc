#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex2.c,v 1.3 1997/07/09 20:55:45 balay Exp balay $";
#endif

static char help[] = "Tests MatTranspose(), MatNorm(), MatValid(), and MatAXPY().\n\n";

#include <stdio.h>
#include <math.h>
#include "mat.h"

int main(int argc,char **argv)
{
  Mat     mat, tmat = 0;
  int     m = 7, n, i, j, ierr, size, rank, rstart, rend, rect = 0, flg;
  Scalar  v;
  double  normf, normi, norm1;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = ViewerSetFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_ASCII_COMMON,0); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  n = m;
  ierr = OptionsHasName(PETSC_NULL,"-rectA",&flg); CHKERRA(ierr);
  if (flg) {n += 2; rect = 1;}
  ierr = OptionsHasName(PETSC_NULL,"-rectB",&flg); CHKERRA(ierr);
  if (flg) {n -= 2; rect = 1;}

  /* ------- Assemble matrix, test MatValid() --------- */

  ierr = MatCreate(PETSC_COMM_WORLD,m,n,&mat); CHKERRA(ierr);
  ierr = MatGetOwnershipRange(mat,&rstart,&rend); CHKERRA(ierr);
  for ( i=rstart; i<rend; i++ ) { 
    for ( j=0; j<n; j++ ) { 
      v=10*i+j; 
      ierr = MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES); CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  /* Test whether matrix has been corrupted (just to demonstrate this
     routine) not needed in most application codes. */
  ierr = MatValid(mat,(PetscTruth*)&flg); CHKERRA(ierr);
  if (!flg) SETERRA(1,0,"Corrupted matrix.");

  /* ----------------- Test MatNorm()  ----------------- */

  ierr = MatNorm(mat,NORM_FROBENIUS,&normf); CHKERRA(ierr);
  ierr = MatNorm(mat,NORM_1,&norm1); CHKERRA(ierr);
  ierr = MatNorm(mat,NORM_INFINITY,&normi); CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_WORLD,
    "original: Frobenious norm = %g, one norm = %g, infinity norm = %g\n",
    normf,norm1,normi);
  ierr = MatView(mat,VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  /* --------------- Test MatTranspose()  -------------- */

  ierr = OptionsHasName(PETSC_NULL,"-in_place",&flg); CHKERRA(ierr);
  if (!rect && flg) {
    ierr = MatTranspose(mat,0); CHKERRA(ierr);   /* in-place transpose */
    tmat = mat; mat = 0;
  } else {      /* out-of-place transpose */
    ierr = MatTranspose(mat,&tmat); CHKERRA(ierr); 
  }

  /* ----------------- Test MatNorm()  ----------------- */

  /* Print info about transpose matrix */
  ierr = MatNorm(tmat,NORM_FROBENIUS,&normf); CHKERRA(ierr);
  ierr = MatNorm(tmat,NORM_1,&norm1); CHKERRA(ierr);
  ierr = MatNorm(tmat,NORM_INFINITY,&normi); CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_WORLD,
    "transpose: Frobenious norm = %g, one norm = %g, infinity norm = %g\n",
    normf,norm1,normi);
  ierr = MatView(tmat,VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  /* ----------------- Test MatAXPY()  ----------------- */

  if (mat && !rect) {
    Scalar alpha = 1.0;
    ierr = OptionsGetScalar(PETSC_NULL,"-alpha",&alpha,&flg); CHKERRA(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"matrix addition:  B = B + alpha * A\n");
    ierr = MatAXPY(&alpha,mat,tmat); CHKERRA(ierr); 
    ierr = MatView(tmat,VIEWER_STDOUT_WORLD); CHKERRA(ierr);
  }

  /* Free data structures */  
  ierr = MatDestroy(tmat); CHKERRA(ierr);
  if (mat) {ierr = MatDestroy(mat); CHKERRA(ierr);}

  PetscFinalize();
  return 0;
}
 
