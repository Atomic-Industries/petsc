#ifndef lint
static char vcid[] = "$Id: precon.c,v 1.125 1997/05/23 15:46:34 balay Exp balay $";
#endif
/*
    The PC (preconditioner) interface routines, callable by users.
*/
#include "src/pc/pcimpl.h"            /*I "pc.h" I*/
#include "pinclude/pviewer.h"         /*I "ksp.h" I*/

extern int PCPrintTypes_Private(MPI_Comm,char*,char*);

#undef __FUNC__  
#define __FUNC__ "PCPrintHelp" /* ADIC Ignore */
/*@
   PCPrintHelp - Prints all the options for the PC component.

   Input Parameter:
.  pc - the preconditioner context

   Options Database Keys:
$  -help, -h

.keywords: PC, help

.seealso: PCSetFromOptions()
@*/
int PCPrintHelp(PC pc)
{
  char p[64]; 

  PetscValidHeaderSpecific(pc,PC_COOKIE);
  PetscStrcpy(p,"-");
  if (pc->prefix) PetscStrcat(p,pc->prefix);
  PetscPrintf(pc->comm,"PC options --------------------------------------------------\n");
  PCPrintTypes_Private(pc->comm,p,"pc_type");
  PetscPrintf(pc->comm,"Run program with -help %spc_type <method> for help on ",p);
  PetscPrintf(pc->comm,"a particular method\n");
  if (pc->printhelp) (*pc->printhelp)(pc,p);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCDestroy" /* ADIC Ignore */
/*@C
   PCDestroy - Destroys PC context that was created with PCCreate().

   Input Parameter:
.  pc - the preconditioner context

.keywords: PC, destroy

.seealso: PCCreate(), PCSetUp()
@*/
int PCDestroy(PC pc)
{
  int ierr = 0;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->destroy) ierr =  (*pc->destroy)((PetscObject)pc);
  else {if (pc->data) PetscFree(pc->data);}
  PLogObjectDestroy(pc);
  PetscHeaderDestroy(pc);
  return ierr;
}

#undef __FUNC__  
#define __FUNC__ "PCCreate"
/*@C
   PCCreate - Creates a preconditioner context.

   Input Parameter:
.   comm - MPI communicator 

   Output Parameter:
.  pc - location to put the preconditioner context

   Notes:
   The default preconditioner on one processor is PCILU on more 
   then one it is PCBJACOBI.

.keywords: PC, create, context

.seealso: PCSetUp(), PCApply(), PCDestroy()
@*/
int PCCreate(MPI_Comm comm,PC *newpc)
{
  PC     pc;
  PCType initialtype = PCJACOBI;
  int    size;

  *newpc          = 0;
  MPI_Comm_size(comm,&size);
  if (size == 1) initialtype = PCILU;
  else           initialtype = PCBJACOBI;

  PetscHeaderCreate(pc,_p_PC,PC_COOKIE,initialtype,comm);
  PLogObjectCreate(pc);
  pc->type               = -1;
  pc->vec                = 0;
  pc->mat                = 0;
  pc->setupcalled        = 0;
  pc->destroy            = 0;
  pc->data               = 0;
  pc->apply              = 0;
  pc->applytrans         = 0;
  pc->applyBA            = 0;
  pc->applyBAtrans       = 0;
  pc->applyrich          = 0;
  pc->view               = 0;
  pc->getfactoredmatrix  = 0;
  pc->nullsp             = 0;
  pc->applysymmetricright = 0;
  pc->applysymmetricleft  = 0;
  pc->setuponblocks       = 0;
  pc->modifysubmatrices   = 0;
  pc->modifysubmatricesP  = 0;
  *newpc                  = pc;
  /* this violates rule about seperating abstract from implementions*/
  return PCSetType(pc,initialtype);
}

#undef __FUNC__  
#define __FUNC__ "PCApply"
/*@
   PCApply - Applies the preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

.keywords: PC, apply

.seealso: PCApplyTrans(), PCApplyBAorAB()
@*/
int PCApply(PC pc,Vec x,Vec y)
{
  int ierr;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,0,"x and y must be different vectors");
  PLogEventBegin(PC_Apply,pc,x,y,0);
  ierr = (*pc->apply)(pc,x,y); CHKERRQ(ierr);
  PLogEventEnd(PC_Apply,pc,x,y,0);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCApplySymmetricLeft"
/*@
   PCApplySymmetricLeft - Applies the left part of a symmetric preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Notes:
   Currently, this routine is implemented only for PCICC and PCJACOBI preconditioners.

.keywords: PC, apply

.seealso: PCApply(), PCApplySymmetricRight()
@*/
int PCApplySymmetricLeft(PC pc,Vec x,Vec y)
{
  int ierr;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PLogEventBegin(PC_ApplySymmetricLeft,pc,x,y,0);
  ierr = (*pc->applysymmetricleft)(pc,x,y); CHKERRQ(ierr);
  PLogEventEnd(PC_ApplySymmetricLeft,pc,x,y,0);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCApplySymmetricRight"
/*@
   PCApplySymmetricRight - Applies the right part of a symmetric preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Notes:
   Currently, this routine is implemented only for PCICC and PCJACOBI preconditioners.

.keywords: PC, apply

.seealso: PCApply(), PCApplySymmetricLeft()
@*/
int PCApplySymmetricRight(PC pc,Vec x,Vec y)
{
  int ierr;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PLogEventBegin(PC_ApplySymmetricRight,pc,x,y,0);
  ierr = (*pc->applysymmetricright)(pc,x,y); CHKERRQ(ierr);
  PLogEventEnd(PC_ApplySymmetricRight,pc,x,y,0);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCApplyTrans"
/*@
   PCApplyTrans - Applies the transpose of preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

.keywords: PC, apply, transpose

.seealso: PCApplyTrans(), PCApplyBAorAB(), PCApplyBAorABTrans()
@*/
int PCApplyTrans(PC pc,Vec x,Vec y)
{
  int ierr;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,0,"x and y must be different vectors");
  if (!pc->applytrans) SETERRQ(PETSC_ERR_SUP,0,"");
  ierr = (*pc->applytrans)(pc,x,y); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCApplyBAorAB"
/*@
   PCApplyBAorAB - Applies the preconditioner and operator to a vector. 

   Input Parameters:
.  pc - the preconditioner context
.  side - indicates the preconditioner side, one of
$   PC_LEFT, PC_RIGHT, or PC_SYMMETRIC
.  x - input vector
.  work - work vector

   Output Parameter:
.  y - output vector

.keywords: PC, apply, operator

.seealso: PCApply(), PCApplyTrans(), PCApplyBAorABTrans()
@*/
int PCApplyBAorAB(PC pc, PCSide side,Vec x,Vec y,Vec work)
{
  int ierr;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(work,VEC_COOKIE);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,0,"x and y must be different vectors");
  if (side != PC_LEFT && side != PC_SYMMETRIC && side != PC_RIGHT) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Side must be right, left, or symmetric");
  }
  if (pc->applyBA)  return (*pc->applyBA)(pc,side,x,y,work);
  if (side == PC_RIGHT) {
    ierr = PCApply(pc,x,work); CHKERRQ(ierr);
    return MatMult(pc->mat,work,y); 
  }
  else if (side == PC_LEFT) {
    ierr = MatMult(pc->mat,x,work); CHKERRQ(ierr);
    return PCApply(pc,work,y);
  }
  else if (side == PC_SYMMETRIC) {
    /* There's an extra copy here; maybe should provide 2 work vectors instead? */
    ierr = PCApplySymmetricRight(pc,x,work); CHKERRQ(ierr);
    ierr = MatMult(pc->mat,work,y); CHKERRQ(ierr);
    ierr = VecCopy(y,work); CHKERRQ(ierr);
    return PCApplySymmetricLeft(pc,work,y);
  }
  return 0;  /* note: actually will never get here */
}

#undef __FUNC__  
#define __FUNC__ "PCApplyBAorABTrans"
/*@ 
   PCApplyBAorABTrans - Applies the transpose of the preconditioner
   and operator to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  side - indicates the preconditioner side, one of
$   PC_LEFT, PC_RIGHT, or PC_SYMMETRIC
.  x - input vector
.  work - work vector

   Output Parameter:
.  y - output vector

.keywords: PC, apply, operator, transpose

.seealso: PCApply(), PCApplyTrans(), PCApplyBAorAB()
@*/
int PCApplyBAorABTrans(PC pc,PCSide side,Vec x,Vec y,Vec work)
{
  int ierr;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(work,VEC_COOKIE);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,0,"x and y must be different vectors");
  if (pc->applyBAtrans)  return (*pc->applyBAtrans)(pc,side,x,y,work);
  if (side != PC_LEFT && side != PC_RIGHT) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Side must be right or left");
  }
  if (side == PC_RIGHT) {
    ierr = MatMultTrans(pc->mat,x,work); CHKERRQ(ierr);
    return PCApplyTrans(pc,work,y);
  }
  else if (side == PC_LEFT) {
    ierr = PCApplyTrans(pc,x,work); CHKERRQ(ierr);
    return MatMultTrans(pc->mat,work,y); 
  }
  /* add support for PC_SYMMETRIC */
  return 0; /* actually will never get here */
}

#undef __FUNC__  
#define __FUNC__ "PCApplyRichardsonExists"
/*@
   PCApplyRichardsonExists - Determines whether a particular preconditioner has a 
   built-in fast application of Richardson's method.

   Input Parameter:
.  pc - the preconditioner

   Output Parameter:
.  exists - PETSC_TRUE or PETSC_FALSE

.keywords: PC, apply, Richardson, exists

.seealso: PCApplyRichardson()
@*/
int PCApplyRichardsonExists(PC pc, PetscTruth *exists)
{
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  PetscValidIntPointer(exists);
  if (pc->applyrich) *exists = PETSC_TRUE; 
  else               *exists = PETSC_FALSE;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCApplyRichardson"
/*@
   PCApplyRichardson - Applies several steps of Richardson iteration with 
   the particular preconditioner. This routine is usually used by the 
   Krylov solvers and not the application code directly.

   Input Parameters:
.  pc  - the preconditioner context
.  x   - the initial guess 
.  w   - one work vector
.  its - the number of iterations to apply.

   Output Parameter:
.  y - the solution

   Notes: 
   Most preconditioners do not support this function. Use the command
   PCApplyRichardsonExists() to determine if one does.

.keywords: PC, apply, Richardson

.seealso: PCApplyRichardsonExists()
@*/
int PCApplyRichardson(PC pc,Vec x,Vec y,Vec w,int its)
{
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(w,VEC_COOKIE);
  if (!pc->applyrich) SETERRQ(PETSC_ERR_SUP,0,"");
  return (*pc->applyrich)(pc,x,y,w,its);
}

/* 
      a setupcall of 0 indicates never setup, 
                     1 needs to be resetup,
                     2 does not need any changes.
*/
#undef __FUNC__  
#define __FUNC__ "PCSetUp"
/*@
   PCSetUp - Prepares for the use of a preconditioner.

   Input parameters:
.  pc - the preconditioner context

.keywords: PC, setup

.seealso: PCCreate(), PCApply(), PCDestroy()
@*/
int PCSetUp(PC pc)
{
  int ierr;
  PetscValidHeaderSpecific(pc,PC_COOKIE);

  if (pc->setupcalled > 1) {
    PLogInfo(pc,"Setting PC with identical preconditioner\n");
  }
  else if (pc->setupcalled == 0) {
    PLogInfo(pc,"Setting up new PC\n");
  }
  else if (pc->flag == SAME_NONZERO_PATTERN) {
    PLogInfo(pc,"Setting up PC with same nonzero pattern\n");
  }
  else {
    PLogInfo(pc,"Setting up PC with different nonzero pattern\n");
  }
  if (pc->setupcalled > 1) return 0;
  PLogEventBegin(PC_SetUp,pc,0,0,0);
  if (!pc->vec) {SETERRQ(1,0,"Vector must be set first");}
  if (!pc->mat) {SETERRQ(1,0,"Matrix must be set first");}
  if (pc->setup) { ierr = (*pc->setup)(pc); CHKERRQ(ierr);}
  pc->setupcalled = 2;
  PLogEventEnd(PC_SetUp,pc,0,0,0);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCSetUpOnBlocks"
/*@
   PCSetUpOnBlocks - Sets up the preconditioner for each block in
   the block Jacobi, block Gauss-Seidel, and overlapping Schwarz 
   methods.

   Input Parameters:
.  pc - the preconditioner context

.keywords: PC, setup, blocks

.seealso: PCCreate(), PCApply(), PCDestroy(), PCSetUp()
@*/
int PCSetUpOnBlocks(PC pc)
{
  int ierr;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (!pc->setuponblocks) return 0;
  PLogEventBegin(PC_SetUpOnBlocks,pc,0,0,0);
  ierr = (*pc->setuponblocks)(pc); CHKERRQ(ierr);
  PLogEventEnd(PC_SetUpOnBlocks,pc,0,0,0);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCSetModifySubMatrices"
/*@
   PCSetModifySubMatrices - Sets a user-defined routine for modifying the
   submatrices that arise within certain subdomain-based preconditioners.
   The basic submatrices are extracted from the preconditioner matrix as
   usual; the user can then alter these (for example, to set different boundary
   conditions for each submatrix) before they are used for the local solves.

   Input Parameters:
.  pc - the preconditioner context
.  func - routine for modifying the submatrices
.  ctx - optional user-defined context (may be null)

   Calling sequence of func:
.  func (PC pc,int nsub,IS *row,IS *col,Mat *submat,void *ctx)

.  nsub - the number of local submatrices
.  row - an array of index sets that contain the global row numbers
         that comprise each local submatrix
.  col - an array of index sets that contain the global column numbers
         that comprise each local submatrix
.  submat - array of local submatrices
.  ctx - optional user-defined context for private data for the 
         user-defined func routine (may be null)

   Notes:
   PCSetModifySubMatrices() MUST be called before SLESSetUp() and
   SLESSolve().

   A routine set by PCSetModifySubMatrices() is currently called within
   the block Jacobi (PCBJACOBI), additive Schwarz (PCASM), and block
   Gauss-Seidel (PCBGS) preconditioners.  All other preconditioners 
   ignore this routine.

.keywords: PC, set, modify, submatrices

.seealso: PCModifySubMatrices()
@*/
int PCSetModifySubMatrices(PC pc,int(*func)(PC,int,IS*,IS*,Mat*,void*),void *ctx)
{
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  pc->modifysubmatrices  = func;
  pc->modifysubmatricesP = ctx;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCModifySubMatrices"
/*@
   PCModifySubMatrices - Calls an optional user-defined routine within 
   certain preconditioners if one has been set with PCSetModifySubMarices().

   Input Parameters:
.  pc - the preconditioner context
.  nsub - the number of local submatrices
.  row - an array of index sets that contain the global row numbers
         that comprise each local submatrix
.  col - an array of index sets that contain the global column numbers
         that comprise each local submatrix
.  submat - array of local submatrices
.  ctx - optional user-defined context for private data for the 
         user-defined routine (may be null)

   Output Parameter:
.  submat - array of local submatrices (the entries of which may
            have been modified)

   Notes:
   The user should NOT generally call this routine, as it will
   automatically be called within certain preconditioners (currently
   block Jacobi, additive Schwarz, and block Gauss-Seidel) if set.

   The basic submatrices are extracted from the preconditioner matrix
   as usual; the user can then alter these (for example, to set different
   boundary conditions for each submatrix) before they are used for the
   local solves.

.keywords: PC, modify, submatrices

.seealso: PCSetModifySubMatrices()
@*/
int PCModifySubMatrices(PC pc,int nsub,IS *row,IS *col,Mat *submat,void *ctx)
{
  int ierr;
  if (!pc->modifysubmatrices) return 0;
  PLogEventBegin(PC_ModifySubMatrices,pc,0,0,0);
  ierr = (*pc->modifysubmatrices)(pc,nsub,row,col,submat,ctx); CHKERRQ(ierr);
  PLogEventEnd(PC_ModifySubMatrices,pc,0,0,0);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCSetOperators"
/*@
   PCSetOperators - Sets the matrix associated with the linear system and 
   a (possibly) different one associated with the preconditioner.

   Input Parameters:
.  pc - the preconditioner context
.  Amat - the matrix associated with the linear system
.  Pmat - matrix to be used in constructing preconditioner, usually the same
          as Amat. 
.  flag - flag indicating information about the preconditioner matrix structure
   during successive linear solves.  This flag is ignored the first time a
   linear system is solved, and thus is irrelevant when solving just one linear
   system.

   Notes: 
   The flag can be used to eliminate unnecessary work in the preconditioner 
   during the repeated solution of linear systems of the same size.  The
   available options are
$    SAME_PRECONDITIONER -
$      Pmat is identical during successive linear solves.
$      This option is intended for folks who are using
$      different Amat and Pmat matrices and wish to reuse the
$      same preconditioner matrix.  For example, this option
$      saves work by not recomputing incomplete factorization
$      for ILU/ICC preconditioners.
$    SAME_NONZERO_PATTERN -
$      Pmat has the same nonzero structure during
$      successive linear solves. 
$    DIFFERENT_NONZERO_PATTERN -
$      Pmat does not have the same nonzero structure.

   Caution:
   If you specify SAME_NONZERO_PATTERN, PETSc believes your assertion
   and does not check the structure of the matrix.  If you erroneously
   claim that the structure is the same when it actually is not, the new
   preconditioner will not function correctly.  Thus, use this optimization
   feature carefully!

   If in doubt about whether your preconditioner matrix has changed
   structure or not, use the flag DIFFERENT_NONZERO_PATTERN.

   More Notes about Repeated Solution of Linear Systems:
   PETSc does NOT reset the matrix entries of either Amat or Pmat
   to zero after a linear solve; the user is completely responsible for
   matrix assembly.  See the routine MatZeroEntries() if desiring to
   zero all elements of a matrix.
    
.keywords: PC, set, operators, matrix, linear system

.seealso: PCGetOperators(), MatZeroEntries()
 @*/
int PCSetOperators(PC pc,Mat Amat,Mat Pmat,MatStructure flag)
{
  MatType type;
  int     ierr;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  PetscValidHeaderSpecific(Amat,MAT_COOKIE);
  PetscValidHeaderSpecific(Pmat,MAT_COOKIE);

  /*
      BlockSolve95 cannot use default BJacobi preconditioning
  */
  ierr = MatGetType(Amat,&type,PETSC_NULL); CHKERRQ(ierr);
  if (type == MATMPIROWBS) {
    if (pc->type == PCBJACOBI) {
      ierr = PCSetType(pc,PCILU); CHKERRQ(ierr);
      PLogInfo(pc,"PCSetOperators:Switching default PC to PCILU since BS95 doesn't support PCBJACOBI\n");
    }
  }
  /*
      Shell matrix (probably) cannot support a preconditioner
  */
  ierr = MatGetType(Pmat,&type,PETSC_NULL); CHKERRQ(ierr);
  if (type == MATSHELL && pc->type != PCSHELL && pc->type != PCMG) {
    ierr = PCSetType(pc,PCNONE); CHKERRQ(ierr);
    PLogInfo(pc,"PCSetOperators:Setting default PC to PCNONE since MATSHELL doesn't support\n\
    preconditioners (unless defined by the user)\n");
  }

  pc->mat  = Amat;
  pc->pmat = Pmat;
  if (pc->setupcalled == 2 && flag != SAME_PRECONDITIONER) {
    pc->setupcalled = 1;
  }
  pc->flag = flag;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCGetOperators" /* ADIC Ignore */
/*@C
   PCGetOperators - Gets the matrix associated with the linear system and
   possibly a different one associated with the preconditioner.

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
.  mat - the matrix associated with the linear system
.  pmat - matrix associated with the preconditioner, usually the same
          as mat. 
.  flag - flag indicating information about the preconditioner
          matrix structure.  See PCSetOperators() for details.

.keywords: PC, get, operators, matrix, linear system

.seealso: PCSetOperators()
@*/
int PCGetOperators(PC pc,Mat *mat,Mat *pmat,MatStructure *flag)
{
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (mat) *mat  = pc->mat;
  if (pmat) *pmat = pc->pmat;
  if (flag) *flag = pc->flag;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCSetVector" /* ADIC Ignore */
/*@
   PCSetVector - Sets a vector associated with the preconditioner.

   Input Parameters:
.  pc - the preconditioner context
.  vec - the vector

   Notes:
   The vector must be set so that the preconditioner knows what type
   of vector to allocate if necessary.

.keywords: PC, set, vector
@*/
int PCSetVector(PC pc,Vec vec)
{
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  PetscValidHeaderSpecific(vec,VEC_COOKIE);
  pc->vec = vec;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCGetFactoredMatrix" /* ADIC Ignore */
/*@C 
   PCGetFactoredMatrix - Gets the factored matrix from the
   preconditioner context.  This routine is valid only for the LU, 
   incomplete LU, Cholesky, and incomplete Cholesky methods.

   Input Parameters:
.  pc - the preconditioner context

   Output parameters:
.  mat - the factored matrix

.keywords: PC, get, factored, matrix
@*/
int PCGetFactoredMatrix(PC pc,Mat *mat)
{
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->getfactoredmatrix) return (*pc->getfactoredmatrix)(pc,mat);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCSetOptionsPrefix" /* ADIC Ignore */
/*@C
   PCSetOptionsPrefix - Sets the prefix used for searching for all 
   PC options in the database.

   Input Parameters:
.  pc - the preconditioner context
.  prefix - the prefix string to prepend to all PC option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

.keywords: PC, set, options, prefix, database

.seealso: PCAppendOptionsPrefix(), PCGetOptionsPrefix()
@*/
int PCSetOptionsPrefix(PC pc,char *prefix)
{
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  return PetscObjectSetOptionsPrefix((PetscObject)pc, prefix);
}

#undef __FUNC__  
#define __FUNC__ "PCAppendOptionsPrefix" /* ADIC Ignore */
/*@C
   PCAppendOptionsPrefix - Appends to the prefix used for searching for all 
   PC options in the database.

   Input Parameters:
.  pc - the preconditioner context
.  prefix - the prefix string to prepend to all PC option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

.keywords: PC, append, options, prefix, database

.seealso: PCSetOptionsPrefix(), PCGetOptionsPrefix()
@*/
int PCAppendOptionsPrefix(PC pc,char *prefix)
{
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  return PetscObjectAppendOptionsPrefix((PetscObject)pc, prefix);
}

#undef __FUNC__  
#define __FUNC__ "PCGetOptionsPrefix" /* ADIC Ignore */
/*@
   PCGetOptionsPrefix - Gets the prefix used for searching for all 
   PC options in the database.

   Input Parameters:
.  pc - the preconditioner context

   Output Parameters:
.  prefix - pointer to the prefix string used, is returned

.keywords: PC, get, options, prefix, database

.seealso: PCSetOptionsPrefix(), PCAppendOptionsPrefix()
@*/
int PCGetOptionsPrefix(PC pc,char **prefix)
{
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  return PetscObjectGetOptionsPrefix((PetscObject)pc, prefix);
}

#undef __FUNC__  
#define __FUNC__ "PCPreSolve"
/*@
   PCPreSolve - Optional pre-solve phase, intended for any
   preconditioner-specific actions that must be performed before 
   the iterative solve itself.

   Input Parameters:
.  pc - the preconditioner context
.  ksp - the Krylov subspace context

   Sample of Usage:
$    PCPreSolve(pc,ksp);
$    KSPSolve(ksp,its);
$    PCPostSolve(pc,ksp);

   Note:
   The pre-solve phase is distinct from the PCSetUp() phase.

.keywords: PC, pre-solve

.seealso: PCPostSolve()
@*/
int PCPreSolve(PC pc,KSP ksp)
{
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->presolve) return (*pc->presolve)(pc,ksp);
  else return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCPostSolve"
/*@
   PCPostSolve - Optional post-solve phase, intended for any
   preconditioner-specific actions that must be performed after
   the iterative solve itself.

   Input Parameters:
.  pc - the preconditioner context
.  ksp - the Krylov subspace context

   Sample of Usage:
$    PCPreSolve(pc,ksp);
$    KSPSolve(ksp,its);
$    PCPostSolve(pc,ksp);

.keywords: PC, post-solve

.seealso: PCPreSolve()
@*/
int PCPostSolve(PC pc,KSP ksp)
{
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->postsolve) return (*pc->postsolve)(pc,ksp);
  else return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCView" /* ADIC Ignore */
/*@ 
   PCView - Prints the PC data structure.

   Input Parameters:
.  PC - the PC context
.  viewer - optional visualization context

   Note:
   The available visualization contexts include
$     VIEWER_STDOUT_SELF - standard output (default)
$     VIEWER_STDOUT_WORLD - synchronized standard
$       output where only the first processor opens
$       the file.  All other processors send their 
$       data to the first processor to print. 

   The user can open alternative vistualization contexts with
$    ViewerFileOpenASCII() - output to a specified file

.keywords: PC, view

.seealso: KSPView(), ViewerFileOpenASCII()
@*/
int PCView(PC pc,Viewer viewer)
{
  FILE        *fd;
  char        *cstr;
  int         fmt, ierr, mat_exists;
  ViewerType  vtype;

  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (viewer) {PetscValidHeader(viewer);} 
  else { viewer = VIEWER_STDOUT_SELF;}

  ViewerGetType(viewer,&vtype);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    ierr = ViewerGetFormat(viewer,&fmt); CHKERRQ(ierr);
    PetscFPrintf(pc->comm,fd,"PC Object:\n");
    PCGetType(pc,PETSC_NULL,&cstr);
    PetscFPrintf(pc->comm,fd,"  method: %s\n",cstr);
    if (pc->view) (*pc->view)((PetscObject)pc,viewer);
    PetscObjectExists((PetscObject)pc->mat,&mat_exists);
    if (mat_exists) {
      ViewerPushFormat(viewer,VIEWER_FORMAT_ASCII_INFO,0);
      if (pc->pmat == pc->mat) {
        PetscFPrintf(pc->comm,fd,"  linear system matrix = precond matrix:\n");
        ierr = MatView(pc->mat,viewer); CHKERRQ(ierr);
      } else {
        PetscObjectExists((PetscObject)pc->pmat,&mat_exists);
        if (mat_exists) {
          PetscFPrintf(pc->comm,fd,"  linear system matrix followed by preconditioner matrix:\n");
        }
        else {
          PetscFPrintf(pc->comm,fd,"  linear system matrix:\n");
        }
        ierr = MatView(pc->mat,viewer); CHKERRQ(ierr);
        if (mat_exists) {ierr = MatView(pc->pmat,viewer); CHKERRQ(ierr);}
      }
      ViewerPopFormat(viewer);
    }
  }
  else if (vtype == STRING_VIEWER) {
    PCGetType(pc,PETSC_NULL,&cstr);
    ViewerStringSPrintf(viewer," %-7.7s",cstr);
    if (pc->view) (*pc->view)((PetscObject)pc,viewer);
  }
  return 0;
}

