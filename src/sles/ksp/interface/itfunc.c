#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: itfunc.c,v 1.102 1998/06/03 20:13:36 bsmith Exp bsmith $";
#endif
/*
      Interface KSP routines that the user calls.
*/

#include "src/ksp/kspimpl.h"   /*I "ksp.h" I*/

#undef __FUNC__  
#define __FUNC__ "KSPComputeExtremeSingularValues"
/*@
   KSPComputeExtremeSingularValues - Computes the extreme singular values
   for the preconditioned operator. Called after or during KSPSolve()
   (SLESSolve()).

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameters:
.  emin, emax - extreme singular values

   Notes:
   One must call KSPSetComputeSingularValues() before calling KSPSetUp() 
   (or use the option -ksp_compute_eigenvalues) in order for this routine to work correctly.

   Many users may just want to use the monitoring routine
   KSPSingularValueMonitor() (which can be set with option -ksp_singmonitor)
   to print the extreme singular values at each iteration of the linear solve.

.keywords: KSP, compute, extreme, singular, values

.seealso: KSPSetComputeSingularValues(), KSPSingularValueMonitor(), KSPComputeEigenvalues()
@*/
int KSPComputeExtremeSingularValues(KSP ksp,double *emax,double *emin)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscValidScalarPointer(emax);
  PetscValidScalarPointer(emin);
  if (!ksp->calc_sings) {
    SETERRQ(4,0,"Singular values not requested before KSPSetUp");
  }

  if (ksp->computeextremesingularvalues) {
    ierr = (*ksp->computeextremesingularvalues)(ksp,emax,emin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPComputeEigenvalues"
/*@
   KSPComputeEigenvalues - Computes the extreme eigenvalues for the
   preconditioned operator. Called after or during KSPSolve() (SLESSolve()).

   Not Collective

   Input Parameter:
+  ksp - iterative context obtained from KSPCreate()
-  n - size of arrays r and c. The number of eigenvalues computed (neig) will, in 
       general, be less than this.

   Output Parameters:
+  r - real part of computed eigenvalues
.  c - complex part of computed eigenvalues
-  neig - number of eigenvalues computed (will be less than or equal to n)
   
   Options Database Keys:
+  -ksp_compute_eigenvalues - Prints eigenvalues to stdout
-  -ksp_plot_eigenvalues - Plots eigenvalues in an x-window display

   Notes:
   The number of eigenvalues estimated depends on the size of the Krylov space
   generated during the KSPSolve() (that is the SLESSolve); for example, with 
   CG it corresponds to the number of CG iterations, for GMRES it is the number 
   of GMRES iterations SINCE the last restart. Any extra space in r[] and c[]
   will be ignored.

   KSPComputeEigenvalues() does not usually provide accurate estimates; it is
   intended only for assistance in understanding the convergence of iterative 
   methods, not for eigenanalysis. 

   One must call KSPSetComputeEigenvalues() before calling KSPSetUp() 
   in order for this routine to work correctly.

   Many users may just want to use the monitoring routine
   KSPSingularValueMonitor() (which can be set with option -ksp_singmonitor)
   to print the singular values at each iteration of the linear solve.

.keywords: KSP, compute, extreme, singular, values

.seealso: KSPSetComputeSingularValues(), KSPSingularValueMonitor(), KSPComputeExtremeSingularValues()
@*/
int KSPComputeEigenvalues(KSP ksp,int n,double *r,double *c,int *neig)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscValidScalarPointer(r);
  PetscValidScalarPointer(c);
  if (!ksp->calc_sings) {
    SETERRQ(4,0,"Eigenvalues not requested before KSPSetUp");
  }

  if (ksp->computeeigenvalues) {
    ierr = (*ksp->computeeigenvalues)(ksp,n,r,c,neig);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSetUp"
/*@
   KSPSetUp - Sets up the internal data structures for the
   later use of an iterative solver.

   Collective on KSP

   Input Parameter:
.  ksp   - iterative context obtained from KSPCreate()

.keywords: KSP, setup

.seealso: KSPCreate(), KSPSolve(), KSPDestroy()
@*/
int KSPSetUp(KSP ksp)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);

  if (!ksp->type_name){
    ierr = KSPSetType(ksp,KSPGMRES);CHKERRQ(ierr);
  }

  if (ksp->setupcalled) PetscFunctionReturn(0);
  ksp->setupcalled = 1;
  ierr = (*ksp->setup)(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "KSPSolve"
/*@
   KSPSolve - Solves linear system; usually not called directly, rather 
    it is called by a call to SLESSolve().

   Collective on KSP

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  its - number of iterations required

   Options Database:
+  -ksp_compute_eigenvalues - compute preconditioned operators eigenvalues
.  -ksp_plot_eigenvalues - plot the computed eigenvalues in an X-window
.  -ksp_compute_eigenvalues_explicitly - compute the eigenvalues by forming the 
      dense operator and useing LAPACK
-  -ksp_plot_eigenvalues_explicitly - plot the explicitly computing eigenvalues

   Notes:
   On return, the parameter "its" contains either the iteration
   number at which convergence was successfully reached, or the
   negative of the iteration at which divergence or breakdown was detected.

   If using a direct method (e.g., via the KSP solver
   KSPPREONLY and a preconditioner such as PCLU/PCILU),
   then its=1.  See KSPSetTolerances() and KSPDefaultConverged()
   for more details.

   Understanding Convergence:
   The routines KSPSetMonitor(), KSPComputeEigenvalues(), and
   KSPComputeEigenvaluesExplicitly() provide information on additional
   options to monitor convergence and print eigenvalue information.

.keywords: KSP, solve, linear system

.seealso: KSPCreate(), KSPSetUp(), KSPDestroy(), KSPSetTolerances(), KSPDefaultConverged(),
          SLESSolve(), KSPSolveTrans(), SLESGetKSP()
@*/
int KSPSolve(KSP ksp, int *its) 
{
  int        ierr,flag1,flag2,rank;
  Scalar     zero = 0.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscValidIntPointer(its);

  if (!ksp->setupcalled){ ierr = KSPSetUp(ksp); CHKERRQ(ierr);}
  if (ksp->guess_zero) { VecSet(&zero,ksp->vec_sol);}
  ierr = (*ksp->solve)(ksp,its); CHKERRQ(ierr);

  MPI_Comm_rank(ksp->comm,&rank);

  ierr = OptionsHasName(ksp->prefix,"-ksp_compute_eigenvalues",&flag1);CHKERRQ(ierr);
  ierr = OptionsHasName(ksp->prefix,"-ksp_plot_eigenvalues",&flag2);CHKERRQ(ierr);
  if (flag1 || flag2) {
    int    n = *its, i, neig;
    double *r,*c;
    r = (double *) PetscMalloc( 2*n*sizeof(double) ); CHKPTRQ(r);
    c = r + n;
    ierr = KSPComputeEigenvalues(ksp,n,r,c,&neig); CHKERRQ(ierr);
    if (flag1) {
      PetscPrintf(ksp->comm,"Iteratively computed eigenvalues\n");
      for ( i=0; i<neig; i++ ) {
        if (c[i] >= 0.0) PetscPrintf(ksp->comm,"%g + %gi\n",r[i],c[i]);
        else             PetscPrintf(ksp->comm,"%g - %gi\n",r[i],-c[i]);
      }
    }
    if (flag2 && !rank) {
      Viewer    viewer;
      Draw      draw;
      DrawSP    drawsp;

      ierr = ViewerDrawOpenX(PETSC_COMM_SELF,0,"Iteratively Computed Eigenvalues",
                             PETSC_DECIDE,PETSC_DECIDE,300,300,&viewer);CHKERRQ(ierr);
      ierr = ViewerDrawGetDraw(viewer,&draw); CHKERRQ(ierr);
      ierr = DrawSPCreate(draw,1,&drawsp); CHKERRQ(ierr);
      for ( i=0; i<neig; i++ ) {
        ierr = DrawSPAddPoint(drawsp,r+i,c+i); CHKERRQ(ierr);
      }
      ierr = DrawSPDraw(drawsp); CHKERRQ(ierr);
      ierr = DrawSPDestroy(drawsp); CHKERRQ(ierr);
      ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
    }
    PetscFree(r);
  }

  ierr = OptionsHasName(ksp->prefix,"-ksp_compute_eigenvalues_explicitly",&flag1); CHKERRQ(ierr);
  ierr = OptionsHasName(ksp->prefix,"-ksp_plot_eigenvalues_explicitly",&flag2); CHKERRQ(ierr);
  if (flag1 || flag2) {
    int    n, i;
    double *r,*c;
    ierr = VecGetSize(ksp->vec_sol,&n); CHKERRQ(ierr);
    r = (double *) PetscMalloc( 2*n*sizeof(double) ); CHKPTRQ(r);
    c = r + n;
    ierr = KSPComputeEigenvaluesExplicitly(ksp,n,r,c); CHKERRQ(ierr); 
    if (flag1) {
      PetscPrintf(ksp->comm,"Explicitly computed eigenvalues\n");
      for ( i=0; i<n; i++ ) {
        if (c[i] >= 0.0) PetscPrintf(ksp->comm,"%g + %gi\n",r[i],c[i]);
        else             PetscPrintf(ksp->comm,"%g - %gi\n",r[i],-c[i]);
      }
    }
    if (flag2 && !rank) {
      Viewer    viewer;
      Draw      draw;
      DrawSP    drawsp;

      ierr = ViewerDrawOpenX(PETSC_COMM_SELF,0,"Explicitly Computed Eigenvalues",0,320,300,300,&viewer); 
             CHKERRQ(ierr);
      ierr = ViewerDrawGetDraw(viewer,&draw); CHKERRQ(ierr);
      ierr = DrawSPCreate(draw,1,&drawsp); CHKERRQ(ierr);
      for ( i=0; i<n; i++ ) {
        ierr = DrawSPAddPoint(drawsp,r+i,c+i); CHKERRQ(ierr);
      }
      ierr = DrawSPDraw(drawsp); CHKERRQ(ierr);
      ierr = DrawSPDestroy(drawsp); CHKERRQ(ierr);
      ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
    }
    PetscFree(r);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSolveTrans"
/*@
   KSPSolveTrans - Solves the transpose of a linear system. Usually
     accessed through SLESSolveTrans().

   Collective on KSP

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  its - number of iterations required

   Notes:
   On return, the parameter "its" contains either the iteration
   number at which convergence was successfully reached, or the
   negative of the iteration at which divergence or breakdown was detected.

   Currently only supported by KSPType of KSPPREONLY. This routine is usally 
   only used internally by the BiCG solver on the subblocks in BJacobi and ASM.

.keywords: KSP, solve, linear system

.seealso: KSPCreate(), KSPSetUp(), KSPDestroy(), KSPSetTolerances(), KSPDefaultConverged(),
          SLESSolve(), SLESGetKSP()
@*/
int KSPSolveTrans(KSP ksp, int *its) 
{
  int        ierr,flag1,flag2,rank;
  Scalar     zero = 0.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscValidIntPointer(its);

  if (!ksp->setupcalled){ ierr = KSPSetUp(ksp); CHKERRQ(ierr);}
  if (ksp->guess_zero) { ierr = VecSet(&zero,ksp->vec_sol); CHKERRQ(ierr);}
  if (!ksp->solvetrans) SETERRQ(1,1,"No transpose solver for this Krylov method");
  ierr = (*ksp->solvetrans)(ksp,its); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPDestroy"
/*@C
   KSPDestroy - Destroys KSP context.

   Collective on KSP

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

.keywords: KSP, destroy

.seealso: KSPCreate(), KSPSetUp(), KSPSolve()
@*/
int KSPDestroy(KSP ksp)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (--ksp->refct > 0) PetscFunctionReturn(0);
  if (ksp->destroy) {
    ierr = (*ksp->destroy)(ksp); CHKERRQ(ierr);
  }
  if (ksp->xmonitor) KSPLGMonitorDestroy(ksp->xmonitor);
  PLogObjectDestroy(ksp);
  PetscHeaderDestroy(ksp);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSetPreconditionerSide"
/*@
    KSPSetPreconditionerSide - Sets the preconditioning side.

    Collective on KSP

    Input Parameter:
.   ksp - iterative context obtained from KSPCreate()

    Output Parameter:
.   side - the preconditioning side, where side is one of
.vb
      PC_LEFT - left preconditioning (default)
      PC_RIGHT - right preconditioning
      PC_SYMMETRIC - symmetric preconditioning
.ve

   Options Database Keys:
+  -ksp_left_pc - Sets left preconditioning
.  -ksp_right_pc - Sets right preconditioning
-  -ksp_symmetric_pc - Sets symmetric preconditioning

    Notes:
    Left preconditioning is used by default.  Symmetric preconditioning is
    currently available only for the KSPQCG method. Note, however, that
    symmetric preconditioning can be emulated by using either right or left
    preconditioning and a pre or post processing step.

.keywords: KSP, set, right, left, symmetric, side, preconditioner, flag

.seealso: KSPGetPreconditionerSide()
@*/
int KSPSetPreconditionerSide(KSP ksp,PCSide side)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->pc_side = side;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPGetPreconditionerSide"
/*@C
    KSPGetPreconditionerSide - Gets the preconditioning side.

    Not Collective

    Input Parameter:
.   ksp - iterative context obtained from KSPCreate()

    Output Parameter:
.   side - the preconditioning side, where side is one of
.vb
      PC_LEFT - left preconditioning (default)
      PC_RIGHT - right preconditioning
      PC_SYMMETRIC - symmetric preconditioning
.ve

.keywords: KSP, get, right, left, symmetric, side, preconditioner, flag

.seealso: KSPSetPreconditionerSide()
@*/
int KSPGetPreconditionerSide(KSP ksp, PCSide *side) 
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  *side = ksp->pc_side;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPGetTolerances"
/*@
   KSPGetTolerances - Gets the relative, absolute, divergence, and maximum
   iteration tolerances used by the default KSP convergence tests. 

   Not Collective

   Input Parameter:
.  ksp - the Krylov subspace context
  
   Output Parameters:
+  rtol - the relative convergence tolerance
.  atol - the absolute convergence tolerance
.  dtol - the divergence tolerance
-  maxits - maximum number of iterations

   Notes:
   The user can specify PETSC_NULL for any parameter that is not needed.

.keywords: KSP, get, tolerance, absolute, relative, divergence, convergence,
           maximum, iterations

.seealso: KSPSetTolerances()
@*/
int KSPGetTolerances(KSP ksp,double *rtol,double *atol,double *dtol,int *maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (atol)   *atol   = ksp->atol;
  if (rtol)   *rtol   = ksp->rtol;
  if (dtol)   *dtol   = ksp->divtol;
  if (maxits) *maxits = ksp->max_it;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSetTolerances"
/*@
   KSPSetTolerances - Sets the relative, absolute, divergence, and maximum
   iteration tolerances used by the default KSP convergence testers. 

   Collective on KSP

   Input Parameters:
+  ksp - the Krylov subspace context
.  rtol - the relative convergence tolerance
   (relative decrease in the residual norm)
.  atol - the absolute convergence tolerance 
   (absolute size of the residual norm)
.  dtol - the divergence tolerance
   (amount residual can increase before KSPDefaultConverged
   concludes that the method is diverging)
-  maxits - maximum number of iterations to use

   Options Database Keys:
+  -ksp_atol <atol> - Sets atol
.  -ksp_rtol <rtol> - Sets rtol
.  -ksp_divtol <dtol> - Sets dtol
-  -ksp_max_it <maxits> - Sets maxits

   Notes:
   Use PETSC_DEFAULT to retain the default value of any of the tolerances.

   See KSPDefaultConverged() for details on the use of these parameters
   in the default convergence test.  See also KSPSetConvergenceTest() 
   for setting user-defined stopping criteria.

.keywords: KSP, set, tolerance, absolute, relative, divergence, 
           convergence, maximum, iterations

.seealso: KSPGetTolerances(), KSPDefaultConverged(), KSPSetConvergenceTest()
@*/
int KSPSetTolerances(KSP ksp,double rtol,double atol,double dtol,int maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (atol != PETSC_DEFAULT)   ksp->atol   = atol;
  if (rtol != PETSC_DEFAULT)   ksp->rtol   = rtol;
  if (dtol != PETSC_DEFAULT)   ksp->divtol = dtol;
  if (maxits != PETSC_DEFAULT) ksp->max_it = maxits;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSetComputeResidual"
/*@
   KSPSetComputeResidual - Sets a flag to indicate whether the two norm 
   of the residual is calculated at each iteration.

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  flag - PETSC_TRUE or PETSC_FALSE

   Notes:
   Most Krylov methods do not yet take advantage of flag = PETSC_FALSE.

.keywords: KSP, set, residual, norm, calculate, flag
@*/
int KSPSetComputeResidual(KSP ksp,PetscTruth flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->calc_res   = flag;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSetUsePreconditionedResidual"
/*@
   KSPSetUsePreconditionedResidual - Sets a flag so that the two norm of the 
   preconditioned residual is used rather than the true residual, in the 
   default convergence tests.

   Collective on KSP

   Input Parameter:
.  ksp  - iterative context obtained from KSPCreate()

   Notes:
   Currently only CG, CHEBYCHEV, and RICHARDSON use this with left
   preconditioning.  All other methods always used the preconditioned
   residual.  With right preconditioning this flag is ignored, since 
   the preconditioned residual and true residual are the same.

   Options Database Key:
.  -ksp_preres - Activates KSPSetUsePreconditionedResidual()

.keywords: KSP, set, residual, precondition, flag
@*/
int KSPSetUsePreconditionedResidual(KSP ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->use_pres   = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSetInitialGuessNonzero"
/*@
   KSPSetInitialGuessNonzero - Tells the iterative solver that the 
   initial guess is nonzero; otherwise KSP assumes the initial guess
   is to be zero (and thus zeros it out before solving).

   Collective on KSP

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()

.keywords: KSP, set, initial guess, nonzero
@*/
int KSPSetInitialGuessNonzero(KSP ksp)
{
  PetscFunctionBegin;
  ksp->guess_zero   = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSetComputeSingularValues"
/*@
   KSPSetComputeSingularValues - Sets a flag so that the extreme singular 
   values will be calculated via a Lanczos or Arnoldi process as the linear 
   system is solved.

   Collective on KSP

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()

   Options Database Key:
.  -ksp_singmonitor - Activates KSPSetComputeSingularValues()

   Notes:
   Currently this option is not valid for all iterative methods.

   Many users may just want to use the monitoring routine
   KSPSingularValueMonitor() (which can be set with option -ksp_singmonitor)
   to print the singular values at each iteration of the linear solve.

.keywords: KSP, set, compute, singular values

.seealso: KSPComputeExtremeSingularValues(), KSPSingularValueMonitor()
@*/
int KSPSetComputeSingularValues(KSP ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->calc_sings  = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSetComputeEigenvalues"
/*@
   KSPSetComputeEigenvalues - Sets a flag so that the extreme eigenvalues
   values will be calculated via a Lanczos or Arnoldi process as the linear 
   system is solved.

   Collective on KSP

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()

   Notes:
   Currently this option is not valid for all iterative methods.

.keywords: KSP, set, compute, eigenvalues

.seealso: KSPComputeEigenvalues(), KSPComputeEigenvaluesExplicitly()
@*/
int KSPSetComputeEigenvalues(KSP ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->calc_sings  = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSetRhs"
/*@
   KSPSetRhs - Sets the right-hand-side vector for the linear system to
   be solved.

   Collective on KSP and Vec

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  b   - right-hand-side vector

.keywords: KSP, set, right-hand-side, rhs

.seealso: KSPGetRhs(), KSPSetSolution()
@*/
int KSPSetRhs(KSP ksp,Vec b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE);
  ksp->vec_rhs    = (b);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPGetRhs"
/*@C
   KSPGetRhs - Gets the right-hand-side vector for the linear system to
   be solved.

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  r - right-hand-side vector

.keywords: KSP, get, right-hand-side, rhs

.seealso: KSPSetRhs(), KSPGetSolution()
@*/
int KSPGetRhs(KSP ksp,Vec *r)
{   
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  *r = ksp->vec_rhs; 
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ "KSPSetSolution"
/*@
   KSPSetSolution - Sets the location of the solution for the 
   linear system to be solved.

   Collective on KSP and Vec

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  x   - solution vector

.keywords: KSP, set, solution

.seealso: KSPSetRhs(), KSPGetSolution()
@*/
int KSPSetSolution(KSP ksp, Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  ksp->vec_sol    = (x);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPGetSolution" 
/*@C
   KSPGetSolution - Gets the location of the solution for the 
   linear system to be solved.  Note that this may not be where the solution
   is stored during the iterative process; see KSPBuildSolution().

   Not Collective

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameters:
.  v - solution vector

.keywords: KSP, get, solution

.seealso: KSPGetRhs(), KSPSetSolution()
@*/
int KSPGetSolution(KSP ksp, Vec *v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);  *v = ksp->vec_sol; 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSetPC"
/*@
   KSPSetPC - Sets the preconditioner to be used to calculate the 
   application of the preconditioner on a vector. 

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  B   - the preconditioner object

   Notes:
   Use KSPGetPC() to retrieve the preconditioner context (for example,
   to free it at the end of the computations).

.keywords: KSP, set, precondition, Binv

.seealso: KSPGetPC()
@*/
int KSPSetPC(KSP ksp,PC B)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscValidHeaderSpecific(B,PC_COOKIE);
  ksp->B = B;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPGetPC"
/*@C
   KSPGetPC - Returns a pointer to the preconditioner context
   set with KSPSetPC().

   Not Collective

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  B - preconditioner context

.keywords: KSP, get, preconditioner, Binv

.seealso: KSPSetPC()
@*/
int KSPGetPC(KSP ksp, PC *B)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  *B = ksp->B; 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSetMonitor"
/*@C
   KSPSetMonitor - Sets an ADDITIONAL function to be called at every iteration to monitor 
   the residual/error etc.
      
   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
.  monitor - pointer to function (if this is PETSC_NULL, it turns off monitoring
-  mctx    - [optional] context for private data for the
             monitor routine (use PETSC_NULL if no context is desired)

   Calling Sequence of monitor:
$     monitor (KSP ksp, int it, double rnorm, void *mctx)

+  ksp - iterative context obtained from KSPCreate()
.  it - iteration number
.  rnorm - (estimated) 2-norm of (preconditioned) residual
-  mctx  - optional monitoring context, as set by KSPSetMonitor()

   Options Database Keys:
+    -ksp_monitor        - sets KSPDefaultMonitor()
.    -ksp_truemonitor    - sets KSPTrueMonitor()
.    -ksp_xmonitor       - sets line graph monitor,
                           uses KSPLGMonitorCreate()
.    -ksp_xtruemonitor   - sets line graph monitor,
                           uses KSPLGMonitorCreate()
.    -ksp_singmonitor    - sets KSPSingularValueMonitor()
.    -ksp_bsmonitor      - sets BlockSolve95 monitor that
                          prints both scaled and unscaled
                          residual norms
-    -ksp_cancelmonitors - cancels all monitors that have
                          been hardwired into a code by 
                          calls to KSPSetMonitor(), but
                          does not cancel those set via
                          the options database.

   Notes:  
   The default is to do nothing.  To print the residual, or preconditioned 
   residual if KSPSetUsePreconditionedResidual() was called, use 
   KSPDefaultMonitor() as the monitoring routine, with a null monitoring 
   context. 

   Several different monitoring routines may be set by calling
   KSPSetMonitor() multiple times; all will be called in the 
   order in which they were set.

.keywords: KSP, set, monitor

.seealso: KSPDefaultMonitor(), KSPLGMonitorCreate(), KSPClearMonitor()
@*/
int KSPSetMonitor(KSP ksp, int (*monitor)(KSP,int,double,void*), void *mctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (ksp->numbermonitors >= MAXKSPMONITORS) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Too KSP many monitors set");
  }

  ksp->monitor[ksp->numbermonitors]           = monitor;
  ksp->monitorcontext[ksp->numbermonitors++]  = (void*)mctx;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPClearMonitor"
/*@C
   KSPClearMonitor - Clears all monitors for a KSP object.

   Collective on KSP

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()

   Options Database Key:
.  -ksp_cancelmonitors - Cancels all monitors that have
    been hardwired into a code by calls to KSPSetMonitor(), 
    but does not cancel those set via the options database.

.keywords: KSP, set, monitor

.seealso: KSPDefaultMonitor(), KSPLGMonitorCreate(), KSPSetMonitor()
@*/
int KSPClearMonitor(KSP ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->numbermonitors = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPGetMonitorContext"
/*@C
   KSPGetMonitorContext - Gets the monitoring context, as set by 
   KSPSetMonitor() for the FIRST monitor only.

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  ctx - monitoring context

.keywords: KSP, get, monitor, context

.seealso: KSPDefaultMonitor(), KSPLGMonitorCreate()
@*/
int KSPGetMonitorContext(KSP ksp, void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  *ctx =      (ksp->monitorcontext[0]);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSetResidualHistory"
/*@
   KSPSetResidualHistory - Sets the array used to hold the residual history.
   If set, this array will contain the residual norms computed at each
   iteration of the solver.

   Not Collective

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
.  a   - array to hold history
-  na  - size of a

.keywords: KSP, set, residual, history, norm
@*/
int KSPSetResidualHistory(KSP ksp, double *a, int na)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (na) PetscValidScalarPointer(a);
  ksp->residual_history = a;
  ksp->res_hist_size    = na;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSetConvergenceTest"
/*@C
   KSPSetConvergenceTest - Sets the function to be used to determine
   convergence.  

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
.  converge - pointer to int function
-  cctx    - context for private data for the convergence routine (may be 
              null)

   Calling sequence of converge:
$     converge (KSP ksp, int it, double rnorm, void *mctx)

+  ksp - iterative context obtained from KSPCreate()
.  it - iteration number
.  rnorm - (estimated) 2-norm of (preconditioned) residual
-  cctx  - optional convergence context, as set by KSPSetConvergenceTest()

   Return value of converge:
   The convergence test should return 0 for not converged, 1 for 
   converged, and -1 for abort or failure to converge.  

   Notes:
   The default convergence test, KSPDefaultConverged(), aborts if the 
   residual grows to more than 10000 times the initial residual.

   The default is a combination of relative and absolute tolerances.  
   The residual value that is tested may be an approximation; routines 
   that need exact values should compute them.

.keywords: KSP, set, convergence, test, context

.seealso: KSPDefaultConverged(), KSPGetConvergenceContext()
@*/
int KSPSetConvergenceTest(KSP ksp,int (*converge)(KSP,int,double,void*),void *cctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ksp->converged = converge;	ksp->cnvP = (void*)cctx;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPGetConvergenceContext"
/*@C
   KSPGetConvergenceContext - Gets the convergence context set with 
   KSPSetConvergenceTest().  

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  ctx - monitoring context

.keywords: KSP, get, convergence, test, context

.seealso: KSPDefaultConverged(), KSPSetConvergenceTest()
@*/
int KSPGetConvergenceContext(KSP ksp, void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  *ctx = ksp->cnvP;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPBuildSolution"
/*@C
   KSPBuildSolution - Builds the approximate solution in a vector provided.
   This routine is NOT commonly needed (see SLESSolve()).

   Collective on KSP

   Input Parameter:
.  ctx - iterative context obtained from KSPCreate()

   Output Parameter: 
   Provide exactly one of
+  v - location to stash solution.   
-  V - the solution is returned in this location. This vector is created 
       internally. This vector should NOT be destroyed by the user with
       VecDestroy().

   Notes:
   This routine must be called after SLESSolve().
   This routine can be used in one of two ways
.vb
      KSPBuildSolution(ctx,PETSC_NULL,&V);
   or
      KSPBuildSolution(ctx,v,PETSC_NULL); 
.ve
   In the first case an internal vector is allocated to store the solution
   (the user cannot destroy this vector). In the second case the solution
   is generated in the vector that the user provides. Note that for certain 
   methods, such as KSPCG, the second case requires a copy of the solution,
   while in the first case the call is essentially free since it simply 
   returns the vector where the solution already is stored.

.keywords: KSP, build, solution

.seealso: KSPGetSolution(), KSPBuildResidual()
@*/
int KSPBuildSolution(KSP ctx, Vec v, Vec *V)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ctx,KSP_COOKIE);
  if (!V && !v) SETERRQ(PETSC_ERR_ARG_WRONG,0,"Must provide either v or V");
  if (!V) V = &v;
  ierr = (*ctx->buildsolution)(ctx,v,V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPBuildResidual"
/*@C
   KSPBuildResidual - Builds the residual in a vector provided.

   Collective on KSP

   Input Parameter:
.  ctx - iterative context obtained from KSPCreate()

   Output Parameters:
+  v - optional location to stash residual.  If v is not provided,
       then a location is generated.
.  t - work vector.  If not provided then one is generated.
-  V - the residual

   Notes:
   Regardless of whether or not v is provided, the residual is 
   returned in V.

.keywords: KSP, build, residual

.seealso: KSPBuildSolution()
@*/
int KSPBuildResidual(KSP ctx, Vec t, Vec v, Vec *V)
{
  int flag = 0, ierr;
  Vec w = v, tt = t;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ctx,KSP_COOKIE);
  if (!w) {
    ierr = VecDuplicate(ctx->vec_rhs,&w); CHKERRQ(ierr);
    PLogObjectParent((PetscObject)ctx,w);
  }
  if (!tt) {
    ierr = VecDuplicate(ctx->vec_rhs,&tt); CHKERRQ(ierr); flag = 1;
    PLogObjectParent((PetscObject)ctx,tt);
  }
  ierr = (*ctx->buildresidual)(ctx,tt,w,V); CHKERRQ(ierr);
  if (flag) {ierr = VecDestroy(tt); CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

