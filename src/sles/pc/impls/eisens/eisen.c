#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: eisen.c,v 1.81 1999/01/13 23:47:07 curfman Exp bsmith $";
#endif

/*
   Defines a  Eisenstat trick SSOR  preconditioner. This uses about 
 %50 of the usual amount of floating point ops used for SSOR + Krylov 
 method. But it requires actually solving the preconditioned problem 
 with both left and right preconditioning. 
*/
#include "src/pc/pcimpl.h"           /*I "pc.h" I*/

typedef struct {
  Mat    shell,A;
  Vec    b,diag;     /* temporary storage for true right hand side */
  double omega;
  int    usediag;    /* indicates preconditioner should include diagonal scaling*/
} PC_Eisenstat;


#undef __FUNC__  
#define __FUNC__ "PCMult_Eisenstat"
static int PCMult_Eisenstat(Mat mat,Vec b,Vec x)
{
  int          ierr;
  PC           pc;
  PC_Eisenstat *eis;

  PetscFunctionBegin;
  MatShellGetContext(mat,(void **)&pc);
  eis = (PC_Eisenstat *) pc->data;
  ierr = MatRelax(eis->A,b,eis->omega,SOR_EISENSTAT,0.0,1,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCApply_Eisenstat"
static int PCApply_Eisenstat(PC pc,Vec x,Vec y)
{
  PC_Eisenstat *eis = (PC_Eisenstat *) pc->data;
  int          ierr;

  PetscFunctionBegin;
  if (eis->usediag)  {ierr = VecPointwiseMult(x,eis->diag,y);CHKERRQ(ierr);}
  else               {ierr = VecCopy(x,y);  CHKERRQ(ierr);}
  PetscFunctionReturn(0); 
}

/* this cheats and looks inside KSP to determine if nonzero initial guess*/
#include "src/ksp/kspimpl.h"

#undef __FUNC__  
#define __FUNC__ "PCPre_Eisenstat"
static int PCPre_Eisenstat(PC pc,KSP ksp,Vec x, Vec b)
{
  PC_Eisenstat *eis = (PC_Eisenstat *) pc->data;
  int          ierr;

  PetscFunctionBegin;
  if (pc->mat != pc->pmat) SETERRQ(PETSC_ERR_SUP,0,"Cannot have different mat and pmat"); 
 
  /* swap shell matrix and true matrix */
  eis->A    = pc->mat;
  pc->mat   = eis->shell;

  if (!eis->b) {
    ierr = VecDuplicate(b,&eis->b); CHKERRQ(ierr);
    PLogObjectParent(pc,eis->b);
  }
  
  /* save true b, other option is to swap pointers */
  ierr = VecCopy(b,eis->b); CHKERRQ(ierr);

  /* if nonzero initial guess, modify x */
  if (!ksp->guess_zero) {
    ierr = MatRelax(eis->A,x,eis->omega,SOR_APPLY_UPPER,0.0,1,x);CHKERRQ(ierr);
  }

  /* modify b by (L + D)^{-1} */
  ierr =   MatRelax(eis->A,b,eis->omega,(MatSORType)(SOR_ZERO_INITIAL_GUESS | 
                                        SOR_FORWARD_SWEEP),0.0,1,b); CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCPost_Eisenstat"
static int PCPost_Eisenstat(PC pc,KSP ksp,Vec x,Vec b)
{
  PC_Eisenstat *eis = (PC_Eisenstat *) pc->data;
  int          ierr;

  PetscFunctionBegin;
  ierr =   MatRelax(eis->A,x,eis->omega,(MatSORType)(SOR_ZERO_INITIAL_GUESS | 
                                 SOR_BACKWARD_SWEEP),0.0,1,x); CHKERRQ(ierr);
  pc->mat = eis->A;
  /* get back true b */
  VecCopy(eis->b,b);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCDestroy_Eisenstat"
static int PCDestroy_Eisenstat(PC pc)
{
  PC_Eisenstat *eis = ( PC_Eisenstat  *) pc->data; 
  int          ierr;

  PetscFunctionBegin;
  if (eis->b)     {ierr = VecDestroy(eis->b);CHKERRQ(ierr);}
  if (eis->shell) {ierr = MatDestroy(eis->shell);CHKERRQ(ierr);}
  if (eis->diag)  {ierr = VecDestroy(eis->diag);CHKERRQ(ierr);}
  PetscFree(eis);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetFromOptions_Eisenstat"
static int PCSetFromOptions_Eisenstat(PC pc)
{
  double  omega;
  int     ierr,flg;

  PetscFunctionBegin;
  ierr = OptionsGetDouble(pc->prefix,"-pc_eisenstat_omega",&omega,&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = PCEisenstatSetOmega(pc,omega); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(pc->prefix,"-pc_eisenstat_diagonal_scaling",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCEisenstatUseDiagonalScaling(pc); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCPrintHelp_Eisenstat"
static int PCPrintHelp_Eisenstat(PC pc,char *p)
{
  PetscFunctionBegin;
  (*PetscHelpPrintf)(pc->comm," Options for PCEisenstat preconditioner:\n");
  (*PetscHelpPrintf)(pc->comm," %spc_eisenstat_omega omega: relaxation factor (0<omega<2)\n",p);
  (*PetscHelpPrintf)(pc->comm," %spc_eisenstat_diagonal_scaling\n",p);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCView_Eisenstat"
static int PCView_Eisenstat(PC pc,Viewer viewer)
{
  PC_Eisenstat  *eis = ( PC_Eisenstat  *) pc->data; 
  int           ierr;
  ViewerType    vtype;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (PetscTypeCompare(vtype,ASCII_VIEWER)) {
    ViewerASCIIPrintf(viewer,"  Eisenstat: omega = %g\n",eis->omega);
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetUp_Eisenstat"
static int PCSetUp_Eisenstat(PC pc)
{
  int          ierr, M, N, m, n;
  PC_Eisenstat *eis = (PC_Eisenstat *) pc->data;
  Vec          diag;

  PetscFunctionBegin;
  if (pc->setupcalled == 0) {
    ierr = MatGetSize(pc->mat,&M,&N); CHKERRA(ierr);
    ierr = MatGetLocalSize(pc->mat,&m,&n); CHKERRA(ierr);
    ierr = MatCreateShell(pc->comm,m,N,M,N,(void*)pc,&eis->shell); CHKERRQ(ierr);
    PLogObjectParent(pc,eis->shell);
    ierr = MatShellSetOperation(eis->shell,MATOP_MULT,(void*)PCMult_Eisenstat);CHKERRQ(ierr);
  }
  if (!eis->usediag) PetscFunctionReturn(0);
  if (pc->setupcalled == 0) {
    ierr = VecDuplicate(pc->vec,&diag); CHKERRQ(ierr);
    PLogObjectParent(pc,diag);
  } else {
    diag = eis->diag;
  }
  ierr = MatGetDiagonal(pc->pmat,diag); CHKERRQ(ierr);
  ierr = VecReciprocal(diag); CHKERRQ(ierr);
  eis->diag = diag;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCEisenstatSetOmega_Eisenstat"
int PCEisenstatSetOmega_Eisenstat(PC pc,double omega)
{
  PC_Eisenstat  *eis;

  PetscFunctionBegin;
  if (omega >= 2.0 || omega <= 0.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Relaxation out of range");
  eis = (PC_Eisenstat *) pc->data;
  eis->omega = omega;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCEisenstatUseDiagonalScaling_Eisenstat"
int PCEisenstatUseDiagonalScaling_Eisenstat(PC pc)
{
  PC_Eisenstat *eis;

  PetscFunctionBegin;
  eis = (PC_Eisenstat *) pc->data;
  eis->usediag = 1;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "PCEisenstatSetOmega"
/*@ 
   PCEisenstatSetOmega - Sets the SSOR relaxation coefficient, omega,
   to use with Eisenstat's trick (where omega = 1.0 by default).

   Input Parameters:
+  pc - the preconditioner context
-  omega - relaxation coefficient (0 < omega < 2)

   Collective on PC

   Options Database Key:
.  -pc_eisenstat_omega <omega> - Sets omega

   Notes: 
   The Eisenstat trick implementation of SSOR requires about 50% of the
   usual amount of floating point operations used for SSOR + Krylov method;
   however, the preconditioned problem must be solved with both left 
   and right preconditioning.

   To use SSOR without the Eisenstat trick, employ the PCSOR preconditioner, 
   which can be chosen with the database options
$    -pc_type  sor  -pc_sor_symmetric

.keywords: PC, Eisenstat, set, SOR, SSOR, relaxation, omega

.seealso: PCSORSetOmega()
@*/
int PCEisenstatSetOmega(PC pc,double omega)
{
  int ierr, (*f)(PC,double);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCEisenstatSetOmega_C",(void **)&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,omega);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCEisenstatUseDiagonalScaling"
/*@
   PCEisenstatUseDiagonalScaling - Causes the Eisenstat preconditioner
   to do an additional diagonal preconditioning. For matrices with very 
   different values along the diagonal, this may improve convergence.

   Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Level: intermediate

   Options Database Key:
.  -pc_eisenstat_diagonal_scaling - Activates PCEisenstatUseDiagonalScaling()

.keywords: PC, Eisenstat, use, diagonal, scaling, SSOR

.seealso: PCEisenstatSetOmega()
@*/
int PCEisenstatUseDiagonalScaling(PC pc)
{
  int ierr, (*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCEisenstatUseDiagonalScaling_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCCreate_Eisenstat"
int PCCreate_Eisenstat(PC pc)
{
  int          ierr;
  PC_Eisenstat *eis = PetscNew(PC_Eisenstat); CHKPTRQ(eis);

  PetscFunctionBegin;
  PLogObjectMemory(pc,sizeof(PC_Eisenstat));

  pc->apply          = PCApply_Eisenstat;
  pc->presolve       = PCPre_Eisenstat;
  pc->postsolve      = PCPost_Eisenstat;
  pc->applyrich      = 0;
  pc->setfromoptions = PCSetFromOptions_Eisenstat;
  pc->printhelp      = PCPrintHelp_Eisenstat ;
  pc->destroy        = PCDestroy_Eisenstat;
  pc->view           = PCView_Eisenstat;
  pc->data           = (void *) eis;
  pc->setup          = PCSetUp_Eisenstat;
  eis->omega         = 1.0;
  eis->b             = 0;
  eis->diag          = 0;
  eis->usediag       = 0;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCEisenstatSetOmega_C","PCEisenstatSetOmega_Eisenstat",
                    (void*)PCEisenstatSetOmega_Eisenstat);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCEisenstatUseDiagonalScaling_C",
                    "PCEisenstatUseDiagonalScaling_Eisenstat",
                    (void*)PCEisenstatUseDiagonalScaling_Eisenstat);CHKERRQ(ierr);
 PetscFunctionReturn(0);
}
EXTERN_C_END
