#include <petsc-private/linesearchimpl.h>
#include <petsc-private/snesimpl.h>

typedef struct {
  PetscReal *memory;

  PetscReal alpha;                      /* Initial reference factor >= 1 */
  PetscReal beta;                       /* Steplength determination < 1 */
  PetscReal beta_inf;           /* Steplength determination < 1 */
  PetscReal sigma;                      /* Acceptance criteria < 1) */
  PetscReal minimumStep;                /* Minimum step size */
  PetscReal lastReference;              /* Reference value of last iteration */

  PetscInt memorySize;          /* Number of functions kept in memory */
  PetscInt current;                     /* Current element for FIFO */
  PetscInt referencePolicy;             /* Integer for reference calculation rule */
  PetscInt replacementPolicy;   /* Policy for replacing values in memory */

  PetscBool nondescending;
  PetscBool memorySetup;


  Vec x;        /* Maintain reference to variable vector to check for changes */
  Vec work;
  Vec workstep;
} SNESLineSearch_ARMIJO;

#define REPLACE_FIFO 1
#define REPLACE_MRU  2

#define REFERENCE_MAX  1
#define REFERENCE_AVE  2
#define REFERENCE_MEAN 3

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchDestroy_Armijo"
static PetscErrorCode SNESLineSearchDestroy_Armijo(SNESLineSearch ls)
{
  SNESLineSearch_ARMIJO *armP = (SNESLineSearch_ARMIJO *)ls->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscFree(armP->memory);CHKERRQ(ierr);
  ierr = VecDestroy(&armP->x);CHKERRQ(ierr);
  ierr = VecDestroy(&armP->work);CHKERRQ(ierr);
  ierr = VecDestroy(&armP->workstep);CHKERRQ(ierr);
  ierr = PetscFree(ls->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchReset_Armijo"
static PetscErrorCode SNESLineSearchReset_Armijo(SNESLineSearch ls)
{
  SNESLineSearch_ARMIJO *armP = (SNESLineSearch_ARMIJO *)ls->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  if (armP->memory != NULL) {
    ierr = PetscFree(armP->memory);CHKERRQ(ierr);
  }
  armP->memorySetup = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetFromOptions_Armijo"
static PetscErrorCode SNESLineSearchSetFromOptions_Armijo(SNESLineSearch ls)
{
  SNESLineSearch_ARMIJO *armP = (SNESLineSearch_ARMIJO *)ls->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Armijo linesearch options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_linesearch_armijo_alpha", "initial reference constant", "", armP->alpha, &armP->alpha, 0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_linesearch_armijo_beta_inf", "decrease constant one", "", armP->beta_inf, &armP->beta_inf, 0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_linesearch_armijo_beta", "decrease constant", "", armP->beta, &armP->beta, 0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_linesearch_armijo_sigma", "acceptance constant", "", armP->sigma, &armP->sigma, 0);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_linesearch_armijo_memory_size", "number of historical elements", "", armP->memorySize, &armP->memorySize, 0);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_linesearch_armijo_reference_policy", "policy for updating reference value", "", armP->referencePolicy, &armP->referencePolicy, 0);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_linesearch_armijo_replacement_policy", "policy for updating memory", "", armP->replacementPolicy, &armP->replacementPolicy, 0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_linesearch_armijo_nondescending","Use nondescending armijo algorithm","",armP->nondescending,&armP->nondescending, 0);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchView_Armijo"
static PetscErrorCode SNESLineSearchView_Armijo(SNESLineSearch ls, PetscViewer pv)
{
  SNESLineSearch_ARMIJO *armP = (SNESLineSearch_ARMIJO *)ls->data;
  PetscBool            isascii;
  PetscErrorCode       ierr;


  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    //    ierr = PetscViewerASCIIPrintf(pv,"  maxf=%D, ftol=%g, gtol=%g\n",ls->max_funcs, (double)ls->rtol, (double)ls->ftol);CHKERRQ(ierr);
    ierr=PetscViewerASCIIPrintf(pv,"  Armijo linesearch",armP->alpha);CHKERRQ(ierr);
    if (armP->nondescending) {
      ierr = PetscViewerASCIIPrintf(pv, " (nondescending)");CHKERRQ(ierr);
    }

    ierr=PetscViewerASCIIPrintf(pv,": alpha=%g beta=%g ",(double)armP->alpha,(double)armP->beta);CHKERRQ(ierr);
    ierr=PetscViewerASCIIPrintf(pv,"sigma=%g ",(double)armP->sigma);CHKERRQ(ierr);
    ierr=PetscViewerASCIIPrintf(pv,"memsize=%D\n",armP->memorySize);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetup_Armijo"
PetscErrorCode SNESLineSearchSetup_Armijo (SNESLineSearch linesearch)
{
  /*  PetscErrorCode ierr;
  PetscFunctionBegin;
  // If this snes doesn't have an objective, then use the default 
  PetscErrorCode (*usermerit)(Vec,PetscReal*,
  if (!linesearch->ops->merit) {
    ierr = SNESGetObjective(snes,&usermerit);CHKERRQ(ierr);
    if (usermerit) {
      ierr = SNESLineSearchSetMerit(linesearch,SNESComputeObjective);CHKERRQ(ierr);
    } else {
      ierr = SNESLineSearchSetMerit(linesearch,SNESLineSearchDefaultMerit);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
   */
  return(0);
}
#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchApply_Armijo"
static PetscErrorCode  SNESLineSearchApply_Armijo(SNESLineSearch linesearch)
{
  PetscBool      changed_y, changed_w;
  SNESLineSearch_ARMIJO *armP;
  //Vec            X, F, Y, W;
  Vec            x,work,s,g,origs;
  SNES           snes;
  PetscReal      gnorm, xnorm, ynorm, lambda, minlambda, maxstep,f;
  PetscBool      domainerror;
  PetscReal      fact,ref,gdx,stol;
  PetscViewer    monitor;
  PetscInt       max_it;
  PetscInt       idx,i;
  PetscErrorCode (*merit)(SNES,Vec,PetscReal*);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  armP = (SNESLineSearch_ARMIJO*)linesearch->data;
  ierr = SNESLineSearchGetVecs(linesearch, &x, &g, &origs, &work, NULL);CHKERRQ(ierr);
  ierr = SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm);CHKERRQ(ierr);
  ierr = SNESLineSearchGetLambda(linesearch, &lambda);CHKERRQ(ierr);
  ierr = SNESLineSearchGetSNES(linesearch, &snes);CHKERRQ(ierr);
  ierr = SNESLineSearchGetMonitor(linesearch, &monitor);CHKERRQ(ierr);
  ierr = SNESLineSearchGetTolerances(linesearch,&minlambda,&maxstep,NULL,NULL,NULL,&max_it);CHKERRQ(ierr);
  ierr = SNESGetTolerances(snes,NULL,NULL,&stol,NULL,NULL);CHKERRQ(ierr);
  merit = linesearch->ops->merit;
  if (!merit) {
    merit = SNESLineSearchDefaultMerit;
    f = gnorm;
  } else {
     ierr = (*merit)(snes,work,&f);CHKERRQ(ierr);
  }
  if (!armP->work) {
    ierr = VecDuplicate(x,&armP->work);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&armP->workstep);CHKERRQ(ierr);
    armP->x = x;
    ierr = PetscObjectReference((PetscObject)armP->x);CHKERRQ(ierr);
  } else if (x != armP->x) {
    /* If x has changed, then recreate work */
    ierr = VecDestroy(&armP->work);CHKERRQ(ierr);
    ierr = VecDestroy(&armP->workstep);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&armP->work);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&armP->workstep);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)armP->x);CHKERRQ(ierr);
    armP->x = x;
    ierr = PetscObjectReference((PetscObject)armP->x);CHKERRQ(ierr);
  }

  ierr = SNESLineSearchSetSuccess(linesearch, PETSC_TRUE);CHKERRQ(ierr);

  /* precheck */
  ierr = SNESLineSearchPreCheck(linesearch,x,origs,&changed_y);CHKERRQ(ierr);

  /* update */

  /* ABOVE HERE FROM BASIC */
  /* Check to see of the memory has been allocated.  If not, allocate
     the historical array and populate it with the initial function
     values. */
  if (!armP->memory) {
    ierr = PetscMalloc1(armP->memorySize, &armP->memory );CHKERRQ(ierr);
  }

  if (!armP->memorySetup) {
    for (i = 0; i < armP->memorySize; i++) {
      armP->memory[i] = armP->alpha*(f);
    }

    armP->current = 0;
    armP->lastReference = armP->memory[0];
    armP->memorySetup=PETSC_TRUE;
  }

  /* Calculate reference value (MAX) */
  ref = armP->memory[0];
  idx = 0;

  for (i = 1; i < armP->memorySize; i++) {
    if (armP->memory[i] > ref) {
      ref = armP->memory[i];
      idx = i;
    }
  }

  if (armP->referencePolicy == REFERENCE_AVE) {
    ref = 0;
    for (i = 0; i < armP->memorySize; i++) {
      ref += armP->memory[i];
    }
    ref = ref / armP->memorySize;
    ref = PetscMax(ref, armP->memory[armP->current]);
  } else if (armP->referencePolicy == REFERENCE_MEAN) {
    ref = PetscMin(ref, 0.5*(armP->lastReference + armP->memory[armP->current]));
  }
  ierr = VecDotRealPart(g,origs,&gdx);CHKERRQ(ierr);

  if (PetscIsInfOrNanReal(gdx)) {
    ierr = PetscInfo1(linesearch,"Initial Line Search step * g is Inf or Nan (%g)\n",(double)gdx);CHKERRQ(ierr);
    ierr = SNESLineSearchSetSuccess(linesearch,PETSC_FALSE);CHKERRQ(ierr);
    snes->reason=SNES_DIVERGED_LINE_SEARCH;
    PetscFunctionReturn(0);
  }
  if (gdx >= 0.0) {
    ierr = PetscInfo1(linesearch,"Initial Line Search step is not descent direction (g's=%g), using -step\n",(double)gdx);CHKERRQ(ierr);
    gdx *= -1;
    ierr = VecCopy(origs,armP->workstep);CHKERRQ(ierr);
    ierr = VecScale(armP->workstep,-1.0);CHKERRQ(ierr);
    s = armP->workstep;
    /*
    ierr = SNESLineSearchSetSuccess(linesearch,PETSC_FALSE);CHKERRQ(ierr);
    snes->reason=SNES_DIVERGED_LINE_SEARCH;
     */
  } else {
    s = origs;
  }

  if (armP->nondescending) {
    fact = armP->sigma;
  } else {
    fact = armP->sigma * gdx;
  }
  while (lambda >= minlambda && (snes->nfuncs < snes->max_funcs)) {
    /* Calculate iterate */
    ierr = VecWAXPY(work,lambda,s,x);CHKERRQ(ierr);

    /* TODO: add projection here */

    /* Calculate function at new iterate */
    ierr = (*merit)(snes,work,&f);CHKERRQ(ierr);

    if (monitor) {
      ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(monitor,"    Line search:  step: %8.6g fmerit %14.12e\n", lambda, (double)f);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
    }
    if (PetscIsInfOrNanReal(f)) {
      lambda *= armP->beta_inf;
    } else {
      /* Check descent condition */
      if (armP->nondescending && f <= ref - lambda*fact*ref)
        break;
      if (!armP->nondescending && f <= ref + lambda*fact) {
        break;
      }

      lambda *= armP->beta;
    }
  }

  /* Check termination */
  if (PetscIsInfOrNanReal(f)) {
    ierr = PetscInfo(linesearch, "Function is inf or nan.\n");CHKERRQ(ierr);
    ierr = SNESLineSearchSetSuccess(linesearch,PETSC_FALSE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else if (lambda < minlambda) {
    ierr = PetscInfo(linesearch, "Step length is below tolerance.\n");CHKERRQ(ierr);
    ierr = SNESLineSearchSetSuccess(linesearch,PETSC_FALSE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else if (snes->nfuncs >= snes->max_funcs) {
    ierr = PetscInfo2(linesearch, "Number of line search function evals (%D) > maximum allowed (%D)\n",snes->nfuncs, snes->max_funcs);CHKERRQ(ierr);
    ierr = SNESLineSearchSetSuccess(linesearch,PETSC_FALSE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* Successful termination, update memory */
  armP->lastReference = ref;
  if (armP->replacementPolicy == REPLACE_FIFO) {
    armP->memory[armP->current++] = f;
    if (armP->current >= armP->memorySize) {
      armP->current = 0;
    }
  } else {
    armP->current = idx;
    armP->memory[idx] = f;
  }






  /* BELOW HERE FROM BASIC */


  if (linesearch->ops->viproject) {
    ierr = (*linesearch->ops->viproject)(snes, work);CHKERRQ(ierr);
  }

  /* postcheck */
  ierr = SNESLineSearchPostCheck(linesearch,x,s,work,&changed_y,&changed_w);CHKERRQ(ierr);
  if (changed_y) {
    ierr = VecWAXPY(work,-lambda,s,x);CHKERRQ(ierr);
    if (linesearch->ops->viproject) {
      ierr = (*linesearch->ops->viproject)(snes, work);CHKERRQ(ierr);
    }
  }
  if (linesearch->norms || snes->iter < max_it-1) {
    ierr = (*linesearch->ops->snesfunc)(snes,work,g);CHKERRQ(ierr);
    ierr = SNESGetFunctionDomainError(snes, &domainerror);CHKERRQ(ierr);
    if (domainerror) {
      ierr = SNESLineSearchSetSuccess(linesearch, PETSC_FALSE);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }

  if (linesearch->norms) {
    if (!linesearch->ops->vinorm) VecNormBegin(g, NORM_2, &linesearch->fnorm);
    ierr = VecNormBegin(s, NORM_2, &linesearch->ynorm);CHKERRQ(ierr);
    ierr = VecNormBegin(work, NORM_2, &linesearch->xnorm);CHKERRQ(ierr);
    if (!linesearch->ops->vinorm) VecNormEnd(g, NORM_2, &linesearch->fnorm);
    ierr = VecNormEnd(s, NORM_2, &linesearch->ynorm);CHKERRQ(ierr);
    ierr = VecNormEnd(work, NORM_2, &linesearch->xnorm);CHKERRQ(ierr);

    if (linesearch->ops->vinorm) {
      linesearch->fnorm = gnorm;

      ierr = (*linesearch->ops->vinorm)(snes, g, work, &linesearch->fnorm);CHKERRQ(ierr);
    } else {
      ierr = VecNorm(g,NORM_2,&linesearch->fnorm);CHKERRQ(ierr);
    }
  }

  /* copy the solution over */
  ierr = VecCopy(work, x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchCreate_Armijo"
/*MC
   SNESLINESEARCHARMIJO - This line search implementation is not a line
   search at all; it simply uses the full step.  Thus, this routine is intended
   for methods with well-scaled updates; i.e. Newton's method (SNESNEWTONLS), on
   well-behaved problems.

   Options Database Keys:
+   -snes_linesearch_damping (1.0) damping parameter.
-   -snes_linesearch_norms (true) whether to compute norms or not.

   Notes:
   For methods with ill-scaled updates (SNESNRICHARDSON, SNESNCG), a small
   damping parameter may yield satisfactory but slow convergence despite
   the simplicity of the line search.

   Level: advanced

.keywords: SNES, SNESLineSearch, damping

.seealso: SNESLineSearchCreate(), SNESLineSearchSetType()
M*/
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_Armijo(SNESLineSearch linesearch)
{
  SNESLineSearch_ARMIJO *armP;
  PetscErrorCode        ierr;
  PetscFunctionBegin;


  linesearch->ops->apply          = SNESLineSearchApply_Armijo;
  linesearch->ops->destroy        = SNESLineSearchDestroy_Armijo;
  linesearch->ops->setfromoptions = SNESLineSearchSetFromOptions_Armijo;
  linesearch->ops->reset          = SNESLineSearchReset_Armijo;
  linesearch->ops->view           = SNESLineSearchView_Armijo;
  linesearch->ops->setup          = SNESLineSearchSetup_Armijo;

  ierr = PetscNewLog(linesearch,&armP);CHKERRQ(ierr);
  linesearch->data = (void*)armP;
  armP->memory = NULL;
  armP->alpha = 1.0;
  armP->beta = 0.5;
  armP->beta_inf = 0.5;
  armP->sigma = 1e-4;
  armP->memorySize = 1;
  armP->referencePolicy = REFERENCE_MAX;
  armP->replacementPolicy = REPLACE_MRU;
  armP->nondescending=PETSC_FALSE;

  PetscFunctionReturn(0);
}
