#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: umtr.c,v 1.58 1997/02/22 02:28:54 bsmith Exp balay $";
#endif

#include <math.h>
#include "src/snes/impls/umtr/umtr.h"                /*I "snes.h" I*/
#include "src/ksp/kspimpl.h"
#include "src/ksp/impls/qcg/qcg.h"
#include "pinclude/pviewer.h"

/*
    SNESSolve_UM_TR - Implements Newton's Method with a trust region approach 
    for solving unconstrained minimization problems.  

    The basic algorithm is taken from MINPACK-2 (dstrn).

    SNESSolve_UM_TR computes a local minimizer of a twice differentiable function
    f  by applying a trust region variant of Newton's method.  At each stage 
    of the algorithm, we us the prconditioned conjugate gradient method to
    determine an approximate minimizer of the quadratic equation

	     q(s) = g^T * s + .5 * s^T * H * s

    subject to the Euclidean norm trust region constraint

	     || D * s || <= delta,

    where delta is the trust region radius and D is a scaling matrix.
    Here g is the gradient and H is the Hessian matrix.

    Note:  SNESSolve_UM_TR MUST use the iterative solver KSPQCG; thus, we
           set KSPQCG in this routine regardless of what the user may have
           previously specified.
*/
#undef __FUNC__  
#define __FUNC__ "SNESSolve_UM_TR"
static int SNESSolve_UM_TR(SNES snes,int *outits)
{
  SNES_UMTR    *neP = (SNES_UMTR *) snes->data;
  int          maxits, i, history_len, nlconv, ierr, qits, newton;
  double       *gnorm, xnorm, max_val, *history, ftrial, delta;
  double       zero = 0.0, *f, two = 2.0, four = 4.0;
  Scalar       one = 1.0;
  Vec          X, G, Y, S, Xtrial;
  MatStructure flg = DIFFERENT_NONZERO_PATTERN;
  SLES         sles;
  KSP          ksp;
  KSP_QCG      *qcgP;

  nlconv        = 0;
  history	= snes->conv_hist;       /* convergence history */
  history_len	= snes->conv_hist_size;  /* convergence history length */
  maxits	= snes->max_its;         /* maximum number of iterations */
  X		= snes->vec_sol; 	 /* solution vector */
  G		= snes->vec_func;	 /* gradient vector */
  S		= snes->work[0]; 	 /* work vectors */
  Xtrial	= snes->work[1]; 
  Y		= snes->work[2]; 
  delta	        = neP->delta;           /* trust region radius */
  f		= &(snes->fc);		/* function to minimize */
  gnorm		= &(snes->norm);	/* gradient norm */

  ierr = VecNorm(X,NORM_2,&xnorm); CHKERRQ(ierr);              /* xnorm = || X || */
  snes->iter = 0;
  ierr = SNESComputeMinimizationFunction(snes,X,f); CHKERRQ(ierr); /* f(X) */
  ierr = SNESComputeGradient(snes,X,G); CHKERRQ(ierr);  /* G(X) <- gradient */
  ierr = VecNorm(G,NORM_2,gnorm); CHKERRQ(ierr);               /* gnorm = || G || */
  if (history) history[0] = *gnorm;
  SNESMonitor(snes,0,*gnorm);

  ierr = SNESGetSLES(snes,&sles); CHKERRQ(ierr);
  ierr = SLESGetKSP(sles,&ksp); CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPQCG); CHKERRQ(ierr);
  PLogInfo(snes,"SNESSolve_UM_TR: setting KSPType = KSPQCG\n");
  qcgP = (KSP_QCG *) ksp->data;

  for ( i=0; i<maxits && !nlconv; i++ ) {
    snes->iter = i+1;
    newton = 0;
    neP->success = 0;
    snes->nfailures = 0;
    ierr = SNESComputeHessian(snes,X,&snes->jacobian,&snes->jacobian_pre,&flg);
           CHKERRQ(ierr);
    ierr = SLESSetOperators(snes->sles,snes->jacobian,snes->jacobian_pre,flg);
           CHKERRQ(ierr);

    if (i == 0) {			/* Initialize delta */
      if (delta <= 0) {
        if (xnorm > zero) delta = neP->factor1*xnorm;
        else delta = neP->delta0;
        ierr = MatNorm(snes->jacobian,NORM_1,&max_val);
        if (ierr == PETSC_ERR_SUP) {
          PLogInfo(snes,"SNESSolve_UM_TR: Initial delta computed without matrix norm info");
        } else {
          CHKERRQ(ierr);
          if (PetscAbsDouble(max_val)<1.e-14)SETERRQ(1,0,"Hessian norm is too small");
          delta = PetscMax(delta,*gnorm/max_val);
        }
      } else { 
        delta = neP->delta0;
      }
    }
    do {
      /* Minimize the quadratic to compute the step s */
      qcgP->delta = delta;
      ierr = SLESSolve(snes->sles,G,S,&qits); CHKERRQ(ierr);
      snes->linear_its += PetscAbsInt(qits);
      if (qits < 0) SETERRQ(1,0,"Failure in SLESSolve");
      if (qcgP->info == 3) newton = 1;	            /* truncated Newton step */
      PLogInfo(snes,"SNESSolve_UM_TR: %d: ltsnrm=%g, delta=%g, q=%g, qits=%d\n", 
               i, qcgP->ltsnrm, delta, qcgP->quadratic, qits );

      ierr = VecWAXPY(&one,X,S,Xtrial); CHKERRQ(ierr); /* Xtrial <- X + S */
      ierr = VecNorm(Xtrial,NORM_2,&xnorm); CHKERRQ(ierr);
                           		               /* ftrial = f(Xtrial) */
      ierr = SNESComputeMinimizationFunction(snes,Xtrial,&ftrial); CHKERRQ(ierr);

      /* Compute the function reduction and the step size */
      neP->actred = *f - ftrial;
      neP->prered = -qcgP->quadratic;

      /* Adjust delta for the first Newton step */
      if ((i == 0) && (newton)) delta = PetscMin(delta,qcgP->ltsnrm);

      if (neP->actred < neP->eta1 * neP->prered) {  /* Unsuccessful step */

         PLogInfo(snes,"SNESSolve_UM_TR: Rejecting step\n");
         snes->nfailures += 1;

         /* If iterate is Newton step, reduce delta to current step length */
         if (newton) {
           delta = qcgP->ltsnrm;
           newton = 0;
         }
         delta /= four; 

      } else {          /* Successful iteration; adjust trust radius */

        neP->success = 1;
        PLogInfo(snes,"SNESSolve_UM_TR: Accepting step\n");
        if (newton) {
           delta = sqrt(qcgP->ltsnrm*delta);
           if (neP->actred < neP->eta2 * neP->prered) delta /= two;
        } else if (neP->actred < neP->eta2 * neP->prered)
           delta /= delta;
        else if ((neP->actred >= neP->eta3 * neP->prered) && 
           (neP->actred < neP->eta4 * neP->prered))
           delta *= two;
        else if (neP->actred >= neP->eta4 * neP->prered)
           delta *= four;
        else neP->sflag = 1;
      }

      neP->delta = delta;
      if ((*snes->converged)(snes,xnorm,*gnorm,ftrial,snes->cnvP)) nlconv = 1;
    } while (!neP->success && !nlconv);

    /* Question:  If (!neP->success && break), then last step was rejected, 
       but convergence was detected.  Should this update really happen? */
    *f = ftrial;
    ierr = VecCopy(Xtrial,X); CHKERRQ(ierr);
    snes->vec_sol_always = X;
    /* Note:  At last iteration, the gradient evaluation is unnecessary */
    ierr = SNESComputeGradient(snes,X,G); CHKERRQ(ierr);
    ierr = VecNorm(G,NORM_2,gnorm); CHKERRQ(ierr);
    if (history && history_len > i+1) history[i+1] = *gnorm;
    snes->vec_func_always = G;

    SNESMonitor(snes,i+1,*gnorm);
  }
  /* Verify solution is in corect location */
  if (X != snes->vec_sol) {
    ierr = VecCopy(X,snes->vec_sol); CHKERRQ(ierr);
    snes->vec_sol_always = snes->vec_sol;
    snes->vec_func_always = snes->vec_func; 
  }
  if (i == maxits) {
    PLogInfo(snes,"SNESSolve_UM_TR: Maximum number of iterations reached: %d\n",maxits);
    i--;
  }
  *outits = i;  /* not i+1, since update for i happens in loop above */
  if (history) snes->conv_act_size = (history_len < i+1) ? history_len : i+1;
  return 0;
}
/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "SNESSetUp_UM_TR"
static int SNESSetUp_UM_TR(SNES snes)
{
  int ierr;

  snes->nwork = 4;
  ierr = VecDuplicateVecs(snes->vec_sol,snes->nwork,&snes->work);CHKERRQ(ierr);
  PLogObjectParents(snes,snes->nwork,snes->work);
  snes->vec_sol_update_always = snes->work[3];
  return 0;
}
/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "SNESDestroy_UM_TR" /* ADIC Ignore */
static int SNESDestroy_UM_TR(PetscObject obj )
{
  SNES snes = (SNES) obj;
  int  ierr;

  if (snes->nwork) {
    ierr = VecDestroyVecs(snes->work,snes->nwork); CHKERRQ(ierr);
  }
  PetscFree(snes->data);
  return 0;
}
/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "SNESConverged_UM_TR"
/*@ 
   SNESConverged_UM_TR - Monitors the convergence of the SNESSolve_UM_TR()
   routine (default). 

   Input Parameters:
.  snes - the SNES context
.  xnorm - 2-norm of current iterate
.  gnorm - 2-norm of current gradient
.  f - objective function value
.  dummy - unused dummy context

   Returns:
$  1  if  ( f < fmin ),
$  2  if  ( abs(ared) <= rtol*abs(f) && 
$           pred <= rtol*abs(f) ),
$  3  if  ( delta <= deltatol*xnorm ),
$ -1  if  ( nfuncs > maxfunc ),
$ -2  if  ( abs(ared) <= epsmch && pred <= epsmch ),
$  0  otherwise,

   where
$    ared     - actual reduction
$    delta    - trust region paramenter
$    deltatol - trust region size tolerance,
$               set with SNESSetTrustRegionTolerance()
$    epsmch   - machine epsilon
$    fmin     - lower bound on function value,
$               set with SNESSetMinimizationFunctionTolerance()
$    nfunc    - number of function evaluations
$    maxfunc  - maximum number of function evaluations, 
$               set with SNESSetTolerances()
$    pred     - predicted reduction
$    rtol     - relative function tolerance, 
$               set with SNESSetTolerances()
@*/
int SNESConverged_UM_TR(SNES snes,double xnorm,double gnorm,double f,void *dummy)
{
  SNES_UMTR *neP = (SNES_UMTR *) snes->data;
  double    rtol = snes->rtol, delta = neP->delta,ared = neP->actred, pred = neP->prered;
  double    epsmch = 1.0e-14;   /* This must be fixed */

  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION)
    SETERRQ(1,0,"For SNES_UNCONSTRAINED_MINIMIZATION only");

  if (f != f) {
    PLogInfo(snes,"SNES:Failed to converged, function is NaN\n");
    return -3;
  }
  /* Test for successful convergence */
  if ((!neP->success || neP->sflag) && (delta <= snes->deltatol * xnorm)) {
    neP->sflag = 0;
    PLogInfo(snes,"SNESConverged_UM_TR: Trust region param satisfies tolerance: %g<=%g*%g\n",
             delta,snes->deltatol,xnorm);  
    return 3;
  }
  if ((PetscAbsDouble(ared) <= PetscAbsDouble(f) * rtol) && (pred) <= rtol*PetscAbsDouble(f)) {
    PLogInfo(snes,"SNESConverged_UM_TR:Actual (%g) and predicted (%g) reductions<%g*%g\n",
             PetscAbsDouble(ared),pred,rtol,PetscAbsDouble(f));
    return 2;
  }
  if (f < snes->fmin) {
    PLogInfo(snes,"SNESConverged_UM_TR:Function value (%g)<f_{minimum} (%g)\n",f,snes->fmin);
    return 1;
  }
  /* Test for termination and stringent tolerances. (failure and stop) */
  if ( (PetscAbsDouble(ared) <= epsmch) && pred <= epsmch ) {
    PLogInfo(snes,"SNESConverged_UM_TR:Actual (%g) and predicted (%g) reductions<epsmch (%g)\n",
             PetscAbsDouble(ared),pred,epsmch);
    return -2;
  }
  if (snes->nfuncs > snes->max_funcs) {
    PLogInfo(snes,"SNESConverged_UM_TR:Exceeded maximum number of function evaluations:%d>%d\n",
             snes->nfuncs, snes->max_funcs );
    return -1;
  }
  return 0;
}
/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "SNESSetFromOptions_UM_TR"
static int SNESSetFromOptions_UM_TR(SNES snes)
{
  SNES_UMTR *ctx = (SNES_UMTR *)snes->data;
  double    tmp;
  int       ierr, flg;

  ierr = OptionsGetDouble(snes->prefix,"-snes_um_eta1",&tmp,&flg); CHKERRQ(ierr);
  if (flg) {ctx->eta1 = tmp;}
  ierr = OptionsGetDouble(snes->prefix,"-snes_um_eta2",&tmp,&flg); CHKERRQ(ierr);
  if (flg) {ctx->eta2 = tmp;}
  ierr = OptionsGetDouble(snes->prefix,"-snes_um_eta3",&tmp,&flg); CHKERRQ(ierr);
  if (flg) {ctx->eta3 = tmp;}
  ierr = OptionsGetDouble(snes->prefix,"-snes_um_eta4",&tmp,&flg); CHKERRQ(ierr);
  if (flg) {ctx->eta4 = tmp;}
  ierr = OptionsGetDouble(snes->prefix,"-snes_um_delta0",&tmp,&flg); CHKERRQ(ierr);
  if (flg) {ctx->delta0 = tmp;}
  ierr = OptionsGetDouble(snes->prefix,"-snes_um_factor1",&tmp,&flg); CHKERRQ(ierr);
  if (flg) {ctx->factor1 = tmp;}

  return 0;
}
/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "SNESPrintHelp_UM_TR" /* ADIC Ignore */
static int SNESPrintHelp_UM_TR(SNES snes,char *p)
{
  SNES_UMTR *ctx = (SNES_UMTR *)snes->data;

  PetscPrintf(snes->comm," method SNES_UM_TR (umtr) for unconstrained minimization:\n");
  PetscPrintf(snes->comm,"   %ssnes_um_tr_eta1 <eta1> (default %g)\n",p,ctx->eta1);
  PetscPrintf(snes->comm,"   %ssnes_um_tr_eta2 <eta2> (default %g)\n",p,ctx->eta2);
  PetscPrintf(snes->comm,"   %ssnes_um_tr_eta3 <eta3> (default %g)\n",p,ctx->eta3);
  PetscPrintf(snes->comm,"   %ssnes_um_tr_eta4 <eta4> (default %g)\n",p,ctx->eta4);
  PetscPrintf(snes->comm,"   %ssnes_um_tr_delta0 <delta0> (default %g)\n",p,ctx->delta0);
  PetscPrintf(snes->comm,"   %ssnes_um_tr_factor1 <factor1> (default %g)\n",p,ctx->factor1);
  PetscPrintf(snes->comm,
    "   delta0, factor1: used to initialize trust region parameter\n");
  PetscPrintf(snes->comm,
    "   eta2, eta3, eta4: used to compute trust region parameter\n");
  PetscPrintf(snes->comm,
    "   eta1: step is unsuccessful if actred < eta1 * prered, where\n"); 
  PetscPrintf(snes->comm,
    "         pred = predicted reduction, actred = actual reduction\n");
  return 0;
}
/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "SNESView_UM_TR" /* ADIC Ignore */
static int SNESView_UM_TR(PetscObject obj,Viewer viewer)
{
  SNES       snes = (SNES)obj;
  SNES_UMTR  *tr = (SNES_UMTR *)snes->data;
  FILE       *fd;
  int        ierr;
  ViewerType vtype;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) { 
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    PetscFPrintf(snes->comm,fd,"    eta1=%g, eta1=%g, eta3=%g, eta4=%g\n",
                 tr->eta1,tr->eta2,tr->eta3,tr->eta4);
    PetscFPrintf(snes->comm,fd,"    delta0=%g, factor1=%g\n",tr->delta0,tr->factor1);
  }
  return 0;
}
/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "SNESCreate_UM_TR"
int SNESCreate_UM_TR(SNES snes)
{
  SNES_UMTR *neP;
  SLES      sles;
  PC        pc;
  int       ierr;

  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) 
    SETERRQ(1,0,"For SNES_UNCONSTRAINED_MINIMIZATION only");
  snes->type 		= SNES_UM_TR;
  snes->setup		= SNESSetUp_UM_TR;
  snes->solve		= SNESSolve_UM_TR;
  snes->destroy		= SNESDestroy_UM_TR;
  snes->converged	= SNESConverged_UM_TR;
  snes->printhelp       = SNESPrintHelp_UM_TR;
  snes->setfromoptions  = SNESSetFromOptions_UM_TR;
  snes->view            = SNESView_UM_TR;

  snes->nwork           = 0;

  neP			= PetscNew(SNES_UMTR); CHKPTRQ(neP);
  PLogObjectMemory(snes,sizeof(SNES_UM_TR));
  snes->data	        = (void *) neP;
  neP->delta0		= 1.0e-6;
  neP->delta 		= 0.0;
  neP->eta1		= 1.0e-4;
  neP->eta2		= 0.25;
  neP->eta3		= 0.50;
  neP->eta4		= 0.90;
  neP->factor1		= 1.0e-8;
  neP->actred		= 0.0;
  neP->prered		= 0.0;
  neP->success		= 0;
  neP->sflag		= 0;
 
  /* Set default preconditioner to be Jacobi, to override SLES default. */
  /* This implementation currently requires a symmetric preconditioner. */
  ierr = SNESGetSLES(snes,&sles); CHKERRQ(ierr);
  ierr = SLESGetPC(sles,&pc); CHKERRQ(ierr);
  ierr = PCSetType(pc,PCJACOBI); CHKERRQ(ierr);

  return 0;
}
