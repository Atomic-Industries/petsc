
#ifndef lint
static char vcid[] = "$Id: snestest.c,v 1.29 1996/12/19 00:15:18 balay Exp bsmith $";
#endif

#include "draw.h"
#include "src/snes/snesimpl.h"

typedef struct {
  int complete_print;
} SNES_Test;

/*
     SNESSolve_Test - Tests whether a hand computed Jacobian 
     matches one compute via finite differences.
*/
#undef __FUNCTION__  
#define __FUNCTION__ "SNESSolve_Test"
int SNESSolve_Test(SNES snes,int *its)
{
  Mat          A = snes->jacobian,B;
  Vec          x = snes->vec_sol;
  int          ierr,i;
  MatStructure flg;
  Scalar       mone = -1.0,one = 1.0;
  double       norm,gnorm;
  SNES_Test    *neP = (SNES_Test*) snes->data;

  *its = 0;

  if (A != snes->jacobian_pre) 
    SETERRQ(1,0,"Cannot test with alternative preconditioner");

  PetscPrintf(snes->comm,"Testing hand-coded Jacobian, if the ratio is\n");
  PetscPrintf(snes->comm,"O(1.e-8), the hand-coded Jacobian is probably correct.\n");
  if (!neP->complete_print) {
    PetscPrintf(snes->comm,"Run with -snes_test_display to show difference\n");
    PetscPrintf(snes->comm,"of hand-coded and finite difference Jacobian.\n");
  }

  for ( i=0; i<3; i++ ) {
    if (i == 1) {ierr = VecSet(&mone,x); CHKERRQ(ierr);}
    else if (i == 2) {ierr = VecSet(&one,x); CHKERRQ(ierr);}
 
    /* compute both versions of Jacobian */
    ierr = SNESComputeJacobian(snes,x,&A,&A,&flg);CHKERRQ(ierr);
    if (i == 0) {ierr = MatConvert(A,MATSAME,&B); CHKERRQ(ierr);}
    ierr = SNESDefaultComputeJacobian(snes,x,&B,&B,&flg,snes->funP);CHKERRQ(ierr);
    if (neP->complete_print) {
      PetscPrintf(snes->comm,"Finite difference Jacobian\n");
      ierr = MatView(B,VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    }
    /* compare */
    ierr = MatAXPY(&mone,A,B); CHKERRQ(ierr);
    ierr = MatNorm(B,NORM_FROBENIUS,&norm); CHKERRQ(ierr);
    ierr = MatNorm(A,NORM_FROBENIUS,&gnorm); CHKERRQ(ierr);
    if (neP->complete_print) {
      PetscPrintf(snes->comm,"Hand-coded Jacobian\n");
      ierr = MatView(A,VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    }
    PetscPrintf(snes->comm,"Norm of matrix ratio %g difference %g\n",norm/gnorm,norm);
  }
  ierr = MatDestroy(B); CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------ */
#undef __FUNCTION__  
#define __FUNCTION__ "SNESDestroy_Test"
int SNESDestroy_Test(PetscObject obj)
{
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "SNESPrintHelp_Test"
static int SNESPrintHelp_Test(SNES snes,char *p)
{
  PetscPrintf(snes->comm,"Test code to compute Jacobian\n");
  PetscPrintf(snes->comm,"-snes_test_display - display difference between\n");
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "SNESSetFromOptions_Test"
static int SNESSetFromOptions_Test(SNES snes)
{
  SNES_Test *ls = (SNES_Test *)snes->data;
  int       ierr,flg;

  ierr = OptionsHasName(PETSC_NULL,"-snes_test_display",&flg); CHKERRQ(ierr);
  if (flg) {
    ls->complete_print = 1;
  }
  return 0;
}

/* ------------------------------------------------------------ */
#undef __FUNCTION__  
#define __FUNCTION__ "SNESCreate_Test"
int SNESCreate_Test(SNES  snes )
{
  SNES_Test *neP;

  if (snes->method_class != SNES_NONLINEAR_EQUATIONS)
    SETERRQ(1,0,"For SNES_NONLINEAR_EQUATIONS only");
  snes->type		= SNES_EQ_TEST;
  snes->setup		= 0;
  snes->solve		= SNESSolve_Test;
  snes->destroy		= SNESDestroy_Test;
  snes->converged	= SNESConverged_EQ_LS;
  snes->printhelp       = SNESPrintHelp_Test;
  snes->setfromoptions  = SNESSetFromOptions_Test;

  neP			= PetscNew(SNES_Test);   CHKPTRQ(neP);
  PLogObjectMemory(snes,sizeof(SNES_Test));
  snes->data    	= (void *) neP;
  neP->complete_print   = 0;
  return 0;
}




