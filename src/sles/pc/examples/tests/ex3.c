#ifndef lint
static char vcid[] = "$Id: ex3.c,v 1.31 1996/01/03 14:42:17 curfman Exp bsmith $";
#endif

static char help[] = "Demonstrates the use of fast Richardson for SOR, and\n\
also tests the MatRelax() routines.  Input parameters are:\n\
 -n <n> : problem dimension\n\n";

#include "ksp.h"
#include "pc.h"
#include "petsc.h"
#include <stdio.h>

int main(int argc,char **args)
{
  Mat    mat;          /* matrix */
  Vec    b, ustar, u;  /* vectors (RHS, exact solution, approx solution) */
  PC     pc;           /* PC context */
  KSP    ksp;          /* KSP context */
  int    ierr, n = 10, i, its, col[3],flg;
  Scalar value[3], one = 1.0, zero = 0.0;
  char   *kspname, *pcname;

  PetscInitialize(&argc,&args,0,0,help);
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);

  /* Create and initialize vectors */
  ierr = VecCreateSeq(MPI_COMM_SELF,n,&b);     CHKERRA(ierr);
  ierr = VecCreateSeq(MPI_COMM_SELF,n,&ustar); CHKERRA(ierr);
  ierr = VecCreateSeq(MPI_COMM_SELF,n,&u);     CHKERRA(ierr);
  ierr = VecSet(&one,ustar); CHKERRA(ierr);
  ierr = VecSet(&zero,u); CHKERRA(ierr);

  /* Create and assemble matrix */
  ierr = MatCreateSeqAIJ(MPI_COMM_SELF,n,n,3,PETSC_NULL,&mat); CHKERRA(ierr);
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++ ) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr = MatSetValues(mat,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);
  }
  i = n - 1; col[0] = n - 2; col[1] = n - 1;
  ierr = MatSetValues(mat,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr);
  i = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  ierr = MatSetValues(mat,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr);
  ierr = MatAssemblyBegin(mat,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(mat,FINAL_ASSEMBLY); CHKERRA(ierr);

  /* Compute right-hand-side vector */
  ierr = MatMult(mat,ustar,b); CHKERRA(ierr);

  /* Create PC context and set up data structures */
  ierr = PCCreate(MPI_COMM_WORLD,&pc); CHKERRA(ierr);
  ierr = PCSetType(pc,PCNONE); CHKERRA(ierr);
  ierr = PCSetFromOptions(pc); CHKERRA(ierr);
  ierr = PCSetOperators(pc,mat,mat, ALLMAT_DIFFERENT_NONZERO_PATTERN);
         CHKERRA(ierr);
  ierr = PCSetVector(pc,u);   CHKERRA(ierr);
  ierr = PCSetUp(pc); CHKERRA(ierr);

  /* Create KSP context and set up data structures */
  ierr = KSPCreate(MPI_COMM_WORLD,&ksp); CHKERRA(ierr);
  ierr = KSPSetType(ksp,KSPRICHARDSON); CHKERRA(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRA(ierr);
  ierr = KSPSetSolution(ksp,u); CHKERRA(ierr);
  ierr = KSPSetRhs(ksp,b); CHKERRA(ierr);
  ierr = PCSetOperators(pc,mat,mat, ALLMAT_DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = KSPSetBinv(ksp,pc); CHKERRA(ierr);
  ierr = KSPSetUp(ksp); CHKERRA(ierr);

  /* Solve the problem */
  ierr = KSPGetType(ksp,PETSC_NULL,&kspname); CHKERRA(ierr);
  ierr = PCGetType(pc,PETSC_NULL,&pcname); CHKERRA(ierr);
  MPIU_printf(MPI_COMM_SELF,"Running %s with %s preconditioning\n",kspname,pcname);
  ierr = KSPSolve(ksp,&its); CHKERRA(ierr);
  fprintf(stdout,"Number of iterations %d\n",its);

  /* Free data structures */
  ierr = KSPDestroy(ksp); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = VecDestroy(ustar); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = MatDestroy(mat); CHKERRA(ierr);
  ierr = PCDestroy(pc); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
    


