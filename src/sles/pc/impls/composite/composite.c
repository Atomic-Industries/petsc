#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: composite.c,v 1.17 1998/12/21 00:59:35 bsmith Exp bsmith $";
#endif
/*
      Defines a preconditioner that can consist of a collection of PCs
*/
#include "src/sles/pc/pcimpl.h"   /*I "pc.h" I*/
#include "sles.h"            /*I "sles.h" I*/

typedef struct _PC_CompositeLink *PC_CompositeLink;
struct _PC_CompositeLink {
  PC               pc;
  PC_CompositeLink next;
};
  
typedef struct {
  PC_CompositeLink head;
  PCCompositeType  type;
  Vec              work1;
  Vec              work2;
  PetscTruth       use_true_matrix;
} PC_Composite;

#undef __FUNC__  
#define __FUNC__ "PCApply_Composite_Multiplicative"
static int PCApply_Composite_Multiplicative(PC pc,Vec x,Vec y)
{
  int              ierr;
  PC_Composite     *jac = (PC_Composite *) pc->data;
  PC_CompositeLink next = jac->head;
  Scalar           one = 1.0,mone = -1.0;
  Mat              mat = pc->pmat;

  PetscFunctionBegin;
  if (!next) {
    SETERRQ(1,1,"No composite preconditioners supplied via PCCompositeAddPC()");
  }
  if (next->next && !jac->work2) { /* allocate second work vector */
    ierr = VecDuplicate(jac->work1,&jac->work2);CHKERRQ(ierr);
  }
  ierr = PCApply(next->pc,x,y); CHKERRQ(ierr);
  if (jac->use_true_matrix) mat = pc->mat;
  while (next->next) {
    next = next->next;
    ierr = MatMult(mat,y,jac->work1);CHKERRQ(ierr);
    ierr = VecWAXPY(&mone,jac->work1,x,jac->work2);CHKERRQ(ierr);
    ierr = PCApply(next->pc,jac->work2,jac->work1); CHKERRQ(ierr);
    ierr = VecAXPY(&one,jac->work1,y);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCApply_Composite_Additive"
static int PCApply_Composite_Additive(PC pc,Vec x,Vec y)
{
  int              ierr;
  PC_Composite     *jac = (PC_Composite *) pc->data;
  PC_CompositeLink next = jac->head;
  Scalar           one = 1.0;

  PetscFunctionBegin;
  if (!next) {
    SETERRQ(1,1,"No composite preconditioners supplied via PCCompositeAddPC()");
  }
  ierr = PCApply(next->pc,x,y); CHKERRQ(ierr);
  while (next->next) {
    next = next->next;
    ierr = PCApply(next->pc,x,jac->work1); CHKERRQ(ierr);
    ierr = VecAXPY(&one,jac->work1,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetUp_Composite"
static int PCSetUp_Composite(PC pc)
{
  int              ierr;
  PC_Composite     *jac = (PC_Composite *) pc->data;
  PC_CompositeLink next = jac->head;

  PetscFunctionBegin;
  if (!jac->work1) {
     ierr = VecDuplicate(pc->vec,&jac->work1);CHKERRQ(ierr);
  }
  while (next) {
    ierr = PCSetOperators(next->pc,pc->mat,pc->pmat,pc->flag);CHKERRQ(ierr);
    ierr = PCSetVector(next->pc,jac->work1);CHKERRQ(ierr);
    next = next->next;
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCDestroy_Composite"
static int PCDestroy_Composite(PC pc)
{
  PC_Composite     *jac = (PC_Composite *) pc->data;
  int              ierr;
  PC_CompositeLink next = jac->head;

  PetscFunctionBegin;
  while (next) {
    ierr = PCDestroy(next->pc);CHKERRQ(ierr);
    next = next->next;
  }

  if (jac->work1) {ierr = VecDestroy(jac->work1);CHKERRQ(ierr);}
  if (jac->work2) {ierr = VecDestroy(jac->work2);CHKERRQ(ierr);}
  PetscFree(jac);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetFromOptions_Composite"
static int PCSetFromOptions_Composite(PC pc)
{
  PC_Composite     *jac = (PC_Composite *) pc->data;
  int              ierr,flg,nmax = 8,i;
  PCCompositeType  type=PC_COMPOSITE_ADDITIVE;
  PC_CompositeLink next;
  char             *pcs[8];
  char             stype[16];

  PetscFunctionBegin;
  ierr = OptionsGetString(pc->prefix,"-pc_composite_type",stype,16,&flg); CHKERRQ(ierr);
  if (flg) {
    if (!PetscStrcmp(stype,"multiplicative")) type = PC_COMPOSITE_MULTIPLICATIVE;
    else if (!PetscStrcmp(stype,"additive"))  type = PC_COMPOSITE_ADDITIVE;
    else SETERRQ(1,1,"Unknown composite type given");

    ierr = PCCompositeSetType(pc,type); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(pc->prefix,"-pc_composite_true",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCCompositeSetUseTrue(pc);CHKERRQ(ierr);
  }
  ierr = OptionsGetStringArray(pc->prefix,"-pc_composite_pcs",pcs,&nmax,&flg);CHKERRQ(ierr);
  if (flg) {
    for ( i=0; i<nmax; i++ ) {
      ierr = PCCompositeAddPC(pc,pcs[i]);CHKERRQ(ierr);
    }
  }

  next = jac->head;
  while (next) {
    ierr = PCSetFromOptions(next->pc);CHKERRQ(ierr);
    next = next->next;
  }
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "PCPrintHelp_Composite"
static int PCPrintHelp_Composite(PC pc,char *p)
{
  PC_Composite     *jac = (PC_Composite *) pc->data;
  PC_CompositeLink next = jac->head;
  int              ierr;

  PetscFunctionBegin;
  (*PetscHelpPrintf)(pc->comm," Options for PCComposite preconditioner:\n"); 
  (*PetscHelpPrintf)(pc->comm," %spc_composite_type [additive,multiplicative]\n",p);
  (*PetscHelpPrintf)(pc->comm," %spc_composite_true\n",p);
  (*PetscHelpPrintf)(pc->comm," %spc_composite_pcs pc1,[pc2,pc3] preconditioner types to compose\n",p);

  (*PetscHelpPrintf)(pc->comm," ---------------------------------\n");
  while (next) {
    ierr = PCPrintHelp(next->pc); CHKERRQ(ierr);
    next = next->next;
  }
  (*PetscHelpPrintf)(pc->comm," ---------------------------------\n");

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCView_Composite"
static int PCView_Composite(PC pc,Viewer viewer)
{
  PC_Composite     *jac = (PC_Composite *) pc->data;
  int              ierr;
  ViewerType       vtype;
  PC_CompositeLink next = jac->head;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (PetscTypeCompare(vtype,ASCII_VIEWER)) {
    ViewerASCIIPrintf(viewer,"PCs on composite preconditioner follow\n");
    ViewerASCIIPrintf(viewer,"---------------------------------\n");
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
  }
  ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  while (next) {
    ierr = PCView(next->pc,viewer); CHKERRQ(ierr);
    next = next->next;
  }
  ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  if (PetscTypeCompare(vtype,ASCII_VIEWER)) {
    ViewerASCIIPrintf(viewer,"---------------------------------\n");
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCCompositeSetType_Composite"
int PCCompositeSetType_Composite(PC pc,PCCompositeType type)
{
  PetscFunctionBegin;
  if (type == PC_COMPOSITE_ADDITIVE) {
    pc->apply = PCApply_Composite_Additive;
  } else if (type ==  PC_COMPOSITE_MULTIPLICATIVE) {
    pc->apply = PCApply_Composite_Multiplicative;
  } else {
    SETERRQ(1,1,"Unkown composite preconditioner type");
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCCompositeAddPC_Composite"
int PCCompositeAddPC_Composite(PC pc,PCType type)
{
  PC_Composite     *jac;
  PC_CompositeLink next,link;
  int              ierr,cnt = 0;
  char             *prefix,newprefix[8];

  PetscFunctionBegin;
  link       = PetscNew(struct _PC_CompositeLink);CHKPTRQ(link);
  link->next = 0;
  ierr = PCCreate(pc->comm,&link->pc);CHKERRQ(ierr);

  jac  = (PC_Composite *) pc->data;
  next = jac->head;
  if (!next) {
    jac->head = link;
  } else {
    cnt++;
    while (next->next) {
      next = next->next;
      cnt++;
    }
    next->next = link;
  }
  ierr = PCGetOptionsPrefix(pc,&prefix); CHKERRQ(ierr);
  ierr = PCSetOptionsPrefix(link->pc,prefix); CHKERRQ(ierr);
  sprintf(newprefix,"sub_%d_",cnt);
  ierr = PCAppendOptionsPrefix(link->pc,newprefix); CHKERRQ(ierr);
  /* type is set after prefix, because some methods may modify prefix, e.g. pcsles */
  ierr = PCSetType(link->pc,type);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCCompositeGetPC_Composite"
int PCCompositeGetPC_Composite(PC pc,int n,PC *subpc)
{
  PC_Composite     *jac;
  PC_CompositeLink next;
  int              i;

  PetscFunctionBegin;
  jac  = (PC_Composite *) pc->data;
  next = jac->head;
  i    = 0;
  for ( i=0; i<n; i++ ) {
    if (!next->next) {
      SETERRQ(1,1,"Not enough PCs in composite preconditioner");
    }
    next = next->next;
  }
  *subpc = next->pc;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCCompositeSetUseTrue_Composite"
int PCCompositeSetUseTrue_Composite(PC pc)
{
  PC_Composite   *jac;

  PetscFunctionBegin;
  jac                  = (PC_Composite *) pc->data;
  jac->use_true_matrix = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "PCCompositeSetType"
/*@C
   PCCompositeSetType
   
   Input Parameter:
.  pc - the preconditioner context
.  type - PC_COMPOSITE_ADDITIVE (default) or PC_COMPOSITE_MULTIPLICATIVE

   Collective on PC

.keywords:  set,  composite preconditioner, additive, multiplicative

@*/
int PCCompositeSetType(PC pc,PCCompositeType type)
{
  int ierr, (*f)(PC,PCCompositeType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCompositeSetType_C",(void **)&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,type);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCCompositeAddPC"
/*@C
   PCCompositeAddPC - Adds another PC to the composite PC.
   
   Input Parameter:
.  pc - the preconditioner context
.  type - the type of the new preconditioner

   Collective on PC

.keywords:  composite preconditioner

@*/
int PCCompositeAddPC(PC pc,PCType type)
{
  int ierr, (*f)(PC,PCType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCompositeAddPC_C",(void **)&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,type);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCCompositeGetPC"
/*@C
   PCCompositeGetPC - Gets one of the PC objects in the composite PC.
   
   Input Parameter:
.  pc - the preconditioner context
.  n - the number of the pc requested

   Output Parameters:
.  subpc - the PC requested

   Not Collective

.keywords:  get, composite preconditioner, sub preconditioner

.seealso: PCCompositeAddPC()

@*/
int PCCompositeGetPC(PC pc,int n,PC *subpc)
{
  int ierr, (*f)(PC,int,PC *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCompositeGetPC_C",(void **)&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,n,subpc);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Cannot get pc, not composite type");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCCompositeSetUseTrue"
/*@
   PCCompositeSetUseTrue - Sets a flag to indicate that the true matrix (rather than
                      the matrix used to define the preconditioner) is used to compute
                      the residual when the multiplicative scheme is used.

   Input Parameters:
.  pc - the preconditioner context

   Collective on PC

   Options Database Key:
$  -pc_composite_true

   Note:
   For the common case in which the preconditioning and linear 
   system matrices are identical, this routine is unnecessary.

.keywords:  block, set, true, flag

.seealso: PCSetOperators(), PCBJacobiSetUseTrueLocal(), PCSLESSetUseTrue()
@*/
int PCCompositeSetUseTrue(PC pc)
{
  int ierr, (*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCompositeSetUseTrue_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCCreate_Composite"
int PCCreate_Composite(PC pc)
{
  int            rank,size,ierr;
  PC_Composite   *jac = PetscNew(PC_Composite); CHKPTRQ(jac);

  PetscFunctionBegin;
  PLogObjectMemory(pc,sizeof(PC_Composite));
  MPI_Comm_rank(pc->comm,&rank);
  MPI_Comm_size(pc->comm,&size);
  pc->apply              = PCApply_Composite_Additive;
  pc->setup              = PCSetUp_Composite;
  pc->destroy            = PCDestroy_Composite;
  pc->setfromoptions     = PCSetFromOptions_Composite;
  pc->printhelp          = PCPrintHelp_Composite;
  pc->view               = PCView_Composite;
  pc->applyrich          = 0;
  pc->data               = (void *) jac;

  jac->type              = PC_COMPOSITE_ADDITIVE;
  jac->work1             = 0;
  jac->work2             = 0;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCompositeSetType_C","PCCompositeSetType_Composite",
                    (void*)PCCompositeSetType_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCompositeAddPC_C","PCCompositeAddPC_Composite",
                    (void*)PCCompositeAddPC_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCompositeGetPC_C","PCCompositeGetPC_Composite",
                    (void*)PCCompositeGetPC_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCompositeSetUseTrue_C","PCCompositeSetUseTrue_Composite",
                    (void*)PCCompositeSetUseTrue_Composite);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

