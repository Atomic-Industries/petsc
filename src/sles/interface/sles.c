#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: sles.c,v 1.106 1998/12/03 04:02:32 bsmith Exp bsmith $";
#endif

#include "src/sles/slesimpl.h"     /*I  "sles.h"    I*/

#undef __FUNC__  
#define __FUNC__ "SLESPublish_Petsc"
static int SLESPublish_Petsc(PetscObject object)
{
  int          ierr;
  
  PetscFunctionBegin;
  ierr = PetscObjectPublishBaseBegin(object);CHKERRQ(ierr);
  ierr = PetscObjectPublishBaseEnd(object);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SLESView"
/*@ 
   SLESView - Prints the SLES data structure.

   Collective on SLES unless Viewer is VIEWER_STDOUT_SELF

   Input Parameters:
+  SLES - the SLES context
-  viewer - optional visualization context

   Options Database Key:
.  -sles_view -  Calls SLESView() at end of SLESSolve()

   Note:
   The available visualization contexts include
+     VIEWER_STDOUT_SELF - standard output (default)
-     VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   The user can open alternative visualization contexts with
.    ViewerASCIIOpen() - output to a specified file

.keywords: SLES, view

.seealso: ViewerASCIIOpen()
@*/
int SLESView(SLES sles,Viewer viewer)
{
  KSP         ksp;
  PC          pc;
  int         ierr;

  PetscFunctionBegin;
  if (!viewer) {viewer = VIEWER_STDOUT_SELF;}

  SLESGetPC(sles,&pc);
  SLESGetKSP(sles,&ksp);
  ierr = KSPView(ksp,viewer); CHKERRQ(ierr);
  ierr = PCView(pc,viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SLESPrintHelp"
/*@
   SLESPrintHelp - Prints SLES options.

   Collective on SLES

   Input Parameter:
.  sles - the SLES context

.keywords: SLES, help

.seealso: SLESSetFromOptions()
@*/
int SLESPrintHelp(SLES sles)
{
  char    *prefix = "-";
  int     ierr;

  PetscFunctionBegin;
  if (sles->prefix) prefix = sles->prefix;
  PetscValidHeaderSpecific(sles,SLES_COOKIE);
  (*PetscHelpPrintf)(sles->comm,"SLES options:\n");
  (*PetscHelpPrintf)(sles->comm," %ssles_view: view SLES info after each linear solve\n",prefix);
  ierr = KSPPrintHelp(sles->ksp);CHKERRQ(ierr);
  ierr = PCPrintHelp(sles->pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SLESSetOptionsPrefix"
/*@C
   SLESSetOptionsPrefix - Sets the prefix used for searching for all 
   SLES options in the database.

   Collective on SLES

   Input Parameter:
+  sles - the SLES context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   This prefix is particularly useful for nested use of SLES.  For
   example, the block Jacobi and block diagonal preconditioners use
   the prefix "sub_" for options relating to the individual blocks.  

.keywords: SLES, set, options, prefix, database

.seealso: SLESAppendOptionsPrefix(), SLESGetOptionsPrefix()
@*/
int SLESSetOptionsPrefix(SLES sles,char *prefix)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sles,SLES_COOKIE);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)sles, prefix); CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(sles->ksp,prefix);CHKERRQ(ierr);
  ierr = PCSetOptionsPrefix(sles->pc,prefix); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SLESAppendOptionsPrefix"
/*@C
   SLESAppendOptionsPrefix - Appends to the prefix used for searching for all 
   SLES options in the database.

   Collective on SLES

   Input Parameter:
+  sles - the SLES context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   This prefix is particularly useful for nested use of SLES.  For
   example, the block Jacobi and block diagonal preconditioners use
   the prefix "sub_" for options relating to the individual blocks.  

.keywords: SLES, append, options, prefix, database

.seealso: SLESSetOptionsPrefix(), SLESGetOptionsPrefix()
@*/
int SLESAppendOptionsPrefix(SLES sles,char *prefix)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sles,SLES_COOKIE);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)sles, prefix); CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(sles->ksp,prefix);CHKERRQ(ierr);
  ierr = PCAppendOptionsPrefix(sles->pc,prefix); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SLESGetOptionsPrefix"
/*@
   SLESGetOptionsPrefix - Gets the prefix used for searching for all 
   SLES options in the database.

   Not Collective

   Input Parameter:
.  sles - the SLES context

   Output Parameter:
.  prefix - pointer to the prefix string used

   Notes:
   This prefix is particularly useful for nested use of SLES.  For
   example, the block Jacobi and block diagonal preconditioners use
   the prefix "sub" for options relating to the individual blocks.  

.keywords: SLES, get, options, prefix, database

.seealso: SLESSetOptionsPrefix()
@*/
int SLESGetOptionsPrefix(SLES sles,char **prefix)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sles,SLES_COOKIE);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)sles, prefix); CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}

#undef __FUNC__  
#define __FUNC__ "SLESSetFromOptions"
/*@
   SLESSetFromOptions - Sets various SLES parameters from user options.
   Also takes all KSP and PC options.

   Collective on SLES

   Input Parameter:
.  sles - the SLES context

.keywords: SLES, set, options, database

.seealso: SLESPrintHelp()
@*/
int SLESSetFromOptions(SLES sles)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sles,SLES_COOKIE);
  ierr = KSPSetPC(sles->ksp,sles->pc);  CHKERRQ(ierr);
  ierr = KSPSetFromOptions(sles->ksp); CHKERRQ(ierr);
  ierr = PCSetFromOptions(sles->pc); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SLESCreate"
/*@C
   SLESCreate - Creates a linear equation solver context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  sles - the newly created SLES context

.keywords: SLES, create, context

.seealso: SLESSolve(), SLESDestroy()
@*/
int SLESCreate(MPI_Comm comm,SLES *outsles)
{
  int  ierr;
  SLES sles;

  PetscFunctionBegin;
  *outsles = 0;
  PetscHeaderCreate(sles,_p_SLES,int,SLES_COOKIE,0,"SLES",comm,SLESDestroy,SLESView);
  PLogObjectCreate(sles);
  sles->bops->publish = SLESPublish_Petsc;
  ierr = KSPCreate(comm,&sles->ksp); CHKERRQ(ierr);
  ierr = PCCreate(comm,&sles->pc); CHKERRQ(ierr);
  PLogObjectParent(sles,sles->ksp);
  PLogObjectParent(sles,sles->pc);
  sles->setupcalled = 0;
  *outsles = sles;
  PetscPublishAll(sles);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SLESDestroy"
/*@C
   SLESDestroy - Destroys the SLES context.

   Collective on SLES

   Input Parameters:
.  sles - the SLES context

.keywords: SLES, destroy, context

.seealso: SLESCreate(), SLESSolve()
@*/
int SLESDestroy(SLES sles)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sles,SLES_COOKIE);
  if (--sles->refct > 0) PetscFunctionReturn(0);

  ierr = KSPDestroy(sles->ksp); CHKERRQ(ierr);
  ierr = PCDestroy(sles->pc); CHKERRQ(ierr);
  PLogObjectDestroy(sles);
  PetscHeaderDestroy(sles);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SLESSetUp"
/*@
   SLESSetUp - Performs set up required for solving a linear system.

   Collective on SLES

   Input Parameters:
+  sles - the SLES context
.  b - the right hand side
-  x - location to hold solution

   Note:
   For basic use of the SLES solvers the user need not explicitly call
   SLESSetUp(), since these actions will automatically occur during
   the call to SLESSolve().  However, if one wishes to generate
   performance data for this computational phase (for example, for
   incomplete factorization using the ILU preconditioner) using the 
   PETSc log facilities, calling SLESSetUp() is required.

.keywords: SLES, solve, linear system

.seealso: SLESCreate(), SLESDestroy(), SLESDestroy()
@*/
int SLESSetUp(SLES sles,Vec b,Vec x)
{
  int ierr;
  KSP ksp;
  PC  pc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sles,SLES_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  ksp = sles->ksp; pc = sles->pc;
  KSPSetRhs(ksp,b);
  KSPSetSolution(ksp,x);
  KSPSetPC(ksp,pc);
  if (!sles->setupcalled) {
    PLogEventBegin(SLES_SetUp,sles,b,x,0);
    ierr = PCSetVector(pc,b); CHKERRQ(ierr);
    ierr = KSPSetUp(sles->ksp); CHKERRQ(ierr);
    ierr = PCSetUp(sles->pc); CHKERRQ(ierr);
    sles->setupcalled = 1;
    PLogEventEnd(SLES_SetUp,sles,b,x,0);
  }
  PetscFunctionReturn(0);
}

static int slesdoublecount = 0;

#undef __FUNC__  
#define __FUNC__ "SLESSolve"
/*@
   SLESSolve - Solves a linear system.

   Collective on SLES

   Input Parameters:
+  sles - the SLES context
-  b - the right hand side

   Output Parameters:
+  x - the approximate solution
-  its - the number of iterations until termination

   Notes:
     On return, the parameter "its" contains
+    <its> - the iteration number at which convergence
       was successfully reached, or
-    <-its> - the negative of the iteration at which
        divergence or breakdown was detected.

     If using a direct method (e.g., via the KSP solver
     KSPPREONLY and a preconditioner such as PCLU/PCILU),
     then its=1.  See KSPSetTolerances() and KSPDefaultConverged()
     for more details.  Also, see KSPSolve() for more details
     about iterative solver options.

   Setting a Nonzero Initial Guess:
     By default, SLES assumes an initial guess of zero by zeroing
     the initial value for the solution vector, x. To use a nonzero 
     initial guess, the user must call
.vb
        SLESGetKSP(sles,&ksp);
        KSPSetInitialGuessNonzero(ksp);
.ve

   Solving Successive Linear Systems:
     When solving multiple linear systems of the same size with the
     same method, several options are available.

     (1) To solve successive linear systems having the SAME
     preconditioner matrix (i.e., the same data structure with exactly
     the same matrix elements) but different right-hand-side vectors,
     the user should simply call SLESSolve() multiple times.  The
     preconditioner setup operations (e.g., factorization for ILU) will
     be done during the first call to SLESSolve() only; such operations
     will NOT be repeated for successive solves.

     (2) To solve successive linear systems that have DIFFERENT
     preconditioner matrices (i.e., the matrix elements and/or the
     matrix data structure change), the user MUST call SLESSetOperators()
     and SLESSolve() for each solve.  See SLESSetOperators() for
     options that can save work for such cases.


.keywords: SLES, solve, linear system

.seealso: SLESCreate(), SLESSetOperators(), SLESGetKSP(), KSPSetTolerances(),
          KSPDefaultConverged(), KSPSetInitialGuessNonzero(), KSPGetResidualNorm(),
          KSPSolve()
@*/
int SLESSolve(SLES sles,Vec b,Vec x,int *its)
{
  int        ierr, flg;
  KSP        ksp;
  PC         pc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sles,SLES_COOKIE);
  if (b == x) SETERRQ(PETSC_ERR_ARG_IDN,0,"b and x must be different vectors");
  ksp = sles->ksp; pc = sles->pc;
  ierr = KSPSetRhs(ksp,b);CHKERRQ(ierr);
  ierr = KSPSetSolution(ksp,x);CHKERRQ(ierr);
  ierr = KSPSetPC(ksp,pc);CHKERRQ(ierr);
  if (!sles->setupcalled) {
    ierr = SLESSetUp(sles,b,x); CHKERRQ(ierr);
  }
  ierr = PCSetUpOnBlocks(pc); CHKERRQ(ierr);
  if (!slesdoublecount) {PLogEventBegin(SLES_Solve,sles,b,x,0);} slesdoublecount++;
  ierr = PCPreSolve(pc,ksp); CHKERRQ(ierr);
  ierr = KSPSolve(ksp,its); CHKERRQ(ierr);
  ierr = PCPostSolve(pc,ksp); CHKERRQ(ierr);
  if (slesdoublecount == 1) {PLogEventEnd(SLES_Solve,sles,b,x,0);} slesdoublecount--;
  ierr = OptionsHasName(sles->prefix,"-sles_view", &flg); CHKERRQ(ierr); 
  if (flg) { ierr = SLESView(sles,VIEWER_STDOUT_WORLD); CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SLESSolveTrans"
/*@
   SLESSolveTrans - Solves the transpose of a linear system.

   Collective on SLES

   Input Parameters:
+  sles - the SLES context
-  b - the right hand side

   Output Parameters:
+  x - the approximate solution
-  its - the number of iterations until termination

   Notes:
     On return, the parameter "its" contains
+    <its> - the iteration number at which convergence
       was successfully reached, or
-    <-its> - the negative of the iteration at which
        divergence or breakdown was detected.

     Currently works only with the KSPPREONLY method.

   Setting a Nonzero Initial Guess:
     By default, SLES assumes an initial guess of zero by zeroing
     the initial value for the solution vector, x. To use a nonzero 
     initial guess, the user must call
.vb
        SLESGetKSP(sles,&ksp);
        KSPSetInitialGuessNonzero(ksp);
.ve

   Solving Successive Linear Systems:
     When solving multiple linear systems of the same size with the
     same method, several options are available.

     (1) To solve successive linear systems having the SAME
     preconditioner matrix (i.e., the same data structure with exactly
     the same matrix elements) but different right-hand-side vectors,
     the user should simply call SLESSolve() multiple times.  The
     preconditioner setup operations (e.g., factorization for ILU) will
     be done during the first call to SLESSolve() only; such operations
     will NOT be repeated for successive solves.

     (2) To solve successive linear systems that have DIFFERENT
     preconditioner matrices (i.e., the matrix elements and/or the
     matrix data structure change), the user MUST call SLESSetOperators()
     and SLESSolve() for each solve.  See SLESSetOperators() for
     options that can save work for such cases.


.keywords: SLES, solve, linear system

.seealso: SLESCreate(), SLESSetOperators(), SLESGetKSP(), KSPSetTolerances(),
          KSPDefaultConverged(), KSPSetInitialGuessNonzero(), KSPGetResidualNorm(),
          KSPSolve()
@*/
int SLESSolveTrans(SLES sles,Vec b,Vec x,int *its)
{
  int        ierr, flg;
  KSP        ksp;
  PC         pc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sles,SLES_COOKIE);
  if (b == x) SETERRQ(PETSC_ERR_ARG_IDN,0,"b and x must be different vectors");
  ksp = sles->ksp; pc = sles->pc;
  ierr = KSPSetRhs(ksp,b);CHKERRQ(ierr);
  ierr = KSPSetSolution(ksp,x);CHKERRQ(ierr);
  ierr = KSPSetPC(ksp,pc);CHKERRQ(ierr);
  if (!sles->setupcalled) {
    ierr = SLESSetUp(sles,b,x); CHKERRQ(ierr);
  }
  ierr = PCSetUpOnBlocks(pc); CHKERRQ(ierr);
  if (!slesdoublecount) {PLogEventBegin(SLES_Solve,sles,b,x,0);} slesdoublecount++;
  ierr = KSPSolveTrans(ksp,its); CHKERRQ(ierr);
  if (slesdoublecount == 1) {PLogEventEnd(SLES_Solve,sles,b,x,0);} slesdoublecount--;
  ierr = OptionsHasName(sles->prefix,"-sles_view", &flg); CHKERRQ(ierr); 
  if (flg) { ierr = SLESView(sles,VIEWER_STDOUT_WORLD); CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SLESGetKSP"
/*@C
   SLESGetKSP - Returns the KSP context for a SLES solver.

   Not Collective, but if KSP will be a parallel object if SLES is

   Input Parameter:
.  sles - the SLES context

   Output Parameter:
.  ksp - the Krylov space context

   Notes:  
   The user can then directly manipulate the KSP context to set various 
   options, etc.
   
.keywords: SLES, get, KSP, context

.seealso: SLESGetPC()
@*/
int SLESGetKSP(SLES sles,KSP *ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sles,SLES_COOKIE);
  *ksp = sles->ksp;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SLESGetPC"
/*@C
   SLESGetPC - Returns the preconditioner (PC) context for a SLES solver.

   Not Collective, but if PC will be a parallel object if SLES is

   Input Parameter:
.  sles - the SLES context

   Output Parameter:
.  pc - the preconditioner context

   Notes:  
   The user can then directly manipulate the PC context to set various 
   options, etc.

.keywords: SLES, get, PC, context

.seealso: SLESGetKSP()
@*/
int SLESGetPC(SLES sles,PC *pc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sles,SLES_COOKIE);
  *pc = sles->pc;
  PetscFunctionReturn(0);
}

#include "src/mat/matimpl.h"
#undef __FUNC__  
#define __FUNC__ "SLESSetOperators"
/*@
   SLESSetOperators - Sets the matrix associated with the linear system
   and a (possibly) different one associated with the preconditioner. 

   Collective on SLES and Mat

   Input Parameters:
+  sles - the SLES context
.  Amat - the matrix associated with the linear system
.  Pmat - the matrix to be used in constructing the preconditioner, usually the
          same as Amat. 
-  flag - flag indicating information about the preconditioner matrix structure
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
$      different Amat and Pmat matrices and want to reuse the
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


.keywords: SLES, set, operators, matrix, preconditioner, linear system

.seealso: SLESSolve()
@*/
int SLESSetOperators(SLES sles,Mat Amat,Mat Pmat,MatStructure flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sles,SLES_COOKIE);
  PetscValidHeaderSpecific(Amat,MAT_COOKIE);
  PetscValidHeaderSpecific(Pmat,MAT_COOKIE);
  PCSetOperators(sles->pc,Amat,Pmat,flag);
  sles->setupcalled = 0;  /* so that next solve call will call setup */
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SLESSetUpOnBlocks"
/*@
   SLESSetUpOnBlocks - Sets up the preconditioner for each block in
   the block Jacobi, block Gauss-Seidel, and overlapping Schwarz 
   methods.

   Collective on SLES

   Input Parameter:
.  sles - the SLES context

   Notes:
   SLESSetUpOnBlocks() is a routine that the user can optinally call for
   more precise profiling (via -log_summary) of the setup phase for these
   block preconditioners.  If the user does not call SLESSetUpOnBlocks(),
   it will automatically be called from within SLESSolve().
   
   Calling SLESSetUpOnBlocks() is the same as calling PCSetUpOnBlocks()
   on the PC context within the SLES context.

.keywords: SLES, setup, blocks

.seealso: PCSetUpOnBlocks(), SLESSetUp(), PCSetUp()
@*/
int SLESSetUpOnBlocks(SLES sles)
{
  int ierr;
  PC  pc;

  PetscFunctionBegin;
  ierr = SLESGetPC(sles,&pc); CHKERRQ(ierr);
  ierr = PCSetUpOnBlocks(pc); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



