
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pcset.c,v 1.69 1998/04/24 21:21:12 curfman Exp bsmith $";
#endif
/*
    Routines to set PC methods and options.
*/

#include "petsc.h"
#include "src/pc/pcimpl.h"      /*I "pc.h" I*/
#include "src/sys/nreg.h"
#include "sys.h"

int  PCRegisterAllCalled = 0;
/*
   Contains the list of registered KSP routines
*/
DLList PCList = 0;

#undef __FUNC__  
#define __FUNC__ "PCSetType"
/*@C
   PCSetType - Builds PC for a particular preconditioner.

   Collective on PC

   Input Parameter:
+  pc - the preconditioner context.
-  type - a known method

   Options Database Command:
.  -pc_type <type> - Sets PC type

   Use -help for a list of available methods (for instance,
   jacobi or bjacobi)

  Notes:
  See "petsc/include/pc.h" for available methods (for instance,
  PCJACOBI, PCILU, or PCBJACOBI).

  Normally, it is best to use the SLESSetFromOptions() command and
  then set the PC type from the options database rather than by using
  this routine.  Using the options database provides the user with
  maximum flexibility in evaluating the many different preconditioners. 
  The PCSetType() routine is provided for those situations where it
  is necessary to set the preconditioner independently of the command
  line or options database.  This might be the case, for example, when
  the choice of preconditioner changes during the execution of the
  program, and the user's application is taking responsibility for
  choosing the appropriate preconditioner.  In other words, this
  routine is for the advanced user.

.keywords: PC, set, method, type
@*/
int PCSetType(PC ctx,PCType type)
{
  int ierr,(*r)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ctx,PC_COOKIE);
  if (!PetscStrcmp(ctx->type_name,type)) PetscFunctionReturn(0);

  if (ctx->setupcalled) {
    if (ctx->destroy) ierr =  (*ctx->destroy)(ctx);
    else {if (ctx->data) PetscFree(ctx->data);}
    ctx->data        = 0;
    ctx->setupcalled = 0;
  }
  /* Get the function pointers for the method requested */
  if (!PCRegisterAllCalled) {ierr = PCRegisterAll(0); CHKERRQ(ierr);}
  ierr =  DLRegisterFind(ctx->comm, PCList, type,(int (**)(void *)) &r );CHKERRQ(ierr);
  if (!r) SETERRQ(1,1,"Unable to find requested PC type");
  if (ctx->data) PetscFree(ctx->data);

  ctx->destroy         = ( int (*)(PC )) 0;
  ctx->view            = ( int (*)(PC,Viewer) ) 0;
  ctx->apply           = ( int (*)(PC,Vec,Vec) ) 0;
  ctx->setup           = ( int (*)(PC) ) 0;
  ctx->applyrich       = ( int (*)(PC,Vec,Vec,Vec,int) ) 0;
  ctx->applyBA         = ( int (*)(PC,int,Vec,Vec,Vec) ) 0;
  ctx->setfromoptions  = ( int (*)(PC) ) 0;
  ctx->printhelp       = ( int (*)(PC,char*) ) 0;
  ctx->applytrans      = ( int (*)(PC,Vec,Vec) ) 0;
  ctx->applyBAtrans    = ( int (*)(PC,int,Vec,Vec,Vec) ) 0;
  ctx->presolve        = ( int (*)(PC,KSP) ) 0;
  ctx->postsolve       = ( int (*)(PC,KSP) ) 0;
  ctx->getfactoredmatrix   = ( int (*)(PC,Mat*) ) 0;
  ctx->applysymmetricleft  = ( int (*)(PC,Vec,Vec) ) 0;
  ctx->applysymmetricright = ( int (*)(PC,Vec,Vec) ) 0;
  ctx->setuponblocks       = ( int (*)(PC) ) 0;
  ctx->modifysubmatrices   = ( int (*)(PC,int,IS*,IS*,Mat*,void*) ) 0;
  ierr = (*r)(ctx);CHKERRQ(ierr);

  if (ctx->type_name) PetscFree(ctx->type_name);
  ctx->type_name = (char *) PetscMalloc((PetscStrlen(type)+1)*sizeof(char));CHKPTRQ(ctx->type_name);
  PetscStrcpy(ctx->type_name,type);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCRegisterDestroy"
/*@C
   PCRegisterDestroy - Frees the list of preconditioners that were
   registered by PCRegister().

   Not Collective

.keywords: PC, register, destroy

.seealso: PCRegisterAll(), PCRegisterAll()
@*/
int PCRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (PCList) {
    ierr = DLRegisterDestroy( PCList );CHKERRQ(ierr);
    PCList = 0;
  }
  PCRegisterAllCalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCPrintHelp"
/*@
   PCPrintHelp - Prints all the options for the PC component.

   Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Options Database Keys:
+  -help - Prints PC options
-  -h - Prints PC options

.keywords: PC, help

.seealso: PCSetFromOptions()
@*/
int PCPrintHelp(PC pc)
{
  char p[64]; 
  int  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  PetscStrcpy(p,"-");
  if (pc->prefix) PetscStrcat(p,pc->prefix);
  (*PetscHelpPrintf)(pc->comm,"PC options --------------------------------------------------\n");
  ierr = DLRegisterPrintTypes(pc->comm,stdout,pc->prefix,"pc_type",PCList);CHKERRQ(ierr);
  (*PetscHelpPrintf)(pc->comm,"Run program with -help %spc_type <method> for help on ",p);
  (*PetscHelpPrintf)(pc->comm,"a particular method\n");
  if (pc->printhelp) {
    ierr = (*pc->printhelp)(pc,p);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "PCGetType"
/*@C
   PCGetType - Gets the PC method type and name (as a string) from the PC
   context.

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  name - name of preconditioner 

.keywords: PC, get, method, name, type
@*/
int PCGetType(PC pc,PCType *meth)
{
  int ierr;

  PetscFunctionBegin;
  if (!PCList) {ierr = PCRegisterAll(0); CHKERRQ(ierr);}
  if (meth)  *meth = (PCType) pc->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetFromOptions"
/*@
   PCSetFromOptions - Sets PC options from the options database.
   This routine must be called before PCSetUp() if the user is to be
   allowed to set the preconditioner method. 

   Collective on PC

   Input Parameters:
.  pc - the preconditioner context

.keywords: PC, set, from, options, database

.seealso: PCPrintHelp()
@*/
int PCSetFromOptions(PC pc)
{
  char   method[256];
  int    ierr,flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);

  if (!PCList) {ierr = PCRegisterAll(0);CHKERRQ(ierr);}
  ierr = OptionsGetString(pc->prefix,"-pc_type",method,256,&flg);
  if (flg) {
    ierr = PCSetType(pc,method); CHKERRQ(ierr);
  }

  /*
        Since the private setfromoptions requires the type to all ready have 
      been set we make sure a type is set by this time
  */
  if (!pc->type_name) {
    int size;

    MPI_Comm_size(pc->comm,&size);
    if (size == 1) {
      ierr = PCSetType(pc,PCILU);CHKERRQ(ierr);
    } else {
      ierr = PCSetType(pc,PCBJACOBI);CHKERRQ(ierr);
    }
  }
  if (pc->setfromoptions) {
    ierr = (*pc->setfromoptions)(pc);CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); 
  if (flg){
    ierr = PCPrintHelp(pc); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}




