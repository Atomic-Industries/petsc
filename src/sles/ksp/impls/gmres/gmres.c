
#ifndef lint
static char vcid[] = "$Id: gmres.c,v 1.74 1997/01/01 03:36:08 bsmith Exp balay $";
#endif

/*
    This file implements GMRES (a Generalized Minimal Residual) method.  
    Reference:  Saad and Schultz, 1986.

    The solver may be called recursively as long as all of the user-supplied
    routines can. This routine is meant to be compatible with execution on a
    parallel processor.  As such, it expects to be given routines for 
    all operations as well as a user-defined pointer to a distributed
    data structure.  THIS IS A DATA-STRUCTURE NEUTRAL IMPLEMENTATION.
  
    A context variable is used to hold internal data (the Hessenberg
    matrix and various parameters).

    Here are the routines that must be provided.  The generic parameters
    are:
	 ksp   - Iterative context.  See the generic iterative method
	         information

    Special routines needed only for gmres:

    void   orthog(  ksp, it )
        perform the orthogonalization of the vectors VV to VV+it.  A 
        basic version of this, defined in terms of vdot and maxpy, is
        available (in borthog.c) called void GMRESBasicOrthog;
	The user may use this routine to try alternate approaches.

    The calling sequence is the same as for all of the iterative methods.
    The special values (specific to GMRES) are:

    Some comments on left vs. right preconditioning, and restarts.
    Left and right preconditioning.
    If right preconditioning is chosen, then the problem being solved
    by gmres is actually
       My =  AB^-1 y = f
    so the initial residual is 
          r = f - Mx
    Note that B^-1 y = x or y = B x, and if x is non-zero, the initial
    residual is
          r = f - A x
    The final solution is then
          x = B^-1 y 

    If left preconditioning is chosen, then the problem being solved is
       My = B^-1 A x = B^-1 f,
    and the initial residual is
       r  = B^-1(f - Ax)

    Restarts:  Restarts are basically solves with x0 not equal to zero.
    Note that we can elliminate an extra application of B^-1 between
    restarts as long as we don't require that the solution at the end
    of a unsuccessful gmres iteration always be the solution x.
 */

#include <math.h>
#include <stdio.h>
#include "src/ksp/impls/gmres/gmresp.h"       /*I  "ksp.h"  I*/
#include "pinclude/pviewer.h"
#define GMRES_DELTA_DIRECTIONS 10
#define GMRES_DEFAULT_MAXK     30
static int    GMRESGetNewVectors( KSP ,int );
static int    GMRESUpdateHessenberg( KSP , int,double * );
static int    BuildGmresSoln(Scalar* ,Vec,Vec ,KSP, int);

#undef __FUNC__  
#define __FUNC__ "KSPSetUp_GMRES"
static int    KSPSetUp_GMRES(KSP ksp )
{
  unsigned  int size, hh, hes, rs, cc;
  int       ierr,  max_k, k;
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;

  if (ksp->pc_side == PC_SYMMETRIC)
    {SETERRQ(2,0,"no symmetric preconditioning for KSPGMRES");}
  max_k         = gmres->max_k;
  hh            = (max_k + 2) * (max_k + 1);
  hes           = (max_k + 1) * (max_k + 1);
  rs            = (max_k + 2);
  cc            = (max_k + 1);
  size          = (hh + hes + rs + 2*cc) * sizeof(Scalar);

  gmres->hh_origin  = (Scalar *) PetscMalloc(size);CHKPTRQ(gmres->hh_origin);
  PLogObjectMemory(ksp,size);
  gmres->hes_origin = gmres->hh_origin + hh;
  gmres->rs_origin  = gmres->hes_origin + hes;
  gmres->cc_origin  = gmres->rs_origin + rs;
  gmres->ss_origin  = gmres->cc_origin + cc;

  if (ksp->calc_sings) {
    /* Allocate workspace to hold Hessenberg matrix needed by Eispack */
    size = (max_k + 3)*(max_k + 9)*sizeof(Scalar);
    gmres->Rsvd = (Scalar *) PetscMalloc(size);CHKPTRQ(gmres->Rsvd);
    gmres->Dsvd = (double *) PetscMalloc(5*(max_k+2)*sizeof(double));CHKPTRQ(gmres->Dsvd);
  }

  /* Allocate array to hold pointers to user vectors.  Note that we need
   4 + max_k + 1 (since we need it+1 vectors, and it <= max_k) */
  gmres->vecs = (Vec *) PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(void *));
  CHKPTRQ(gmres->vecs);
  gmres->vecs_allocated = VEC_OFFSET + 2 + max_k;
  gmres->user_work = (Vec **)PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(void *));
  CHKPTRQ(gmres->user_work);
  gmres->mwork_alloc = (int *) PetscMalloc( (VEC_OFFSET+2+max_k)*sizeof(int) );
  CHKPTRQ(gmres->mwork_alloc);
  PLogObjectMemory(ksp,(VEC_OFFSET+2+max_k)*(2*sizeof(void *)+sizeof(int)));

  if (gmres->q_preallocate) {
    gmres->vv_allocated   = VEC_OFFSET + 2 + max_k;
    ierr = VecDuplicateVecs(ksp->vec_rhs,gmres->vv_allocated,&gmres->user_work[0]);
    CHKERRQ(ierr);
    PLogObjectParents(ksp,gmres->vv_allocated,gmres->user_work[0]);
    gmres->mwork_alloc[0] = gmres->vv_allocated;
    gmres->nwork_alloc    = 1;
    for (k=0; k<gmres->vv_allocated; k++)
	gmres->vecs[k] = gmres->user_work[0][k];
  }
  else {
    gmres->vv_allocated    = 5;
    ierr = VecDuplicateVecs(ksp->vec_rhs, 5,    &gmres->user_work[0]); CHKERRQ(ierr);
    PLogObjectParents(ksp,5,gmres->user_work[0]);
    gmres->mwork_alloc[0]  = 5;
    gmres->nwork_alloc     = 1;
    for (k=0; k<gmres->vv_allocated; k++)
	gmres->vecs[k] = gmres->user_work[0][k];
  }
  return 0;
}

/* 
    This routine computes the initial residual without making any assumptions
    about the solution.
 */
#undef __FUNC__  
#define __FUNC__ "GMRESResidual"
static int GMRESResidual(  KSP ksp,int restart )
{
  KSP_GMRES    *gmres = (KSP_GMRES *)(ksp->data);
  Scalar       mone = -1.0;
  Mat          Amat, Pmat;
  MatStructure pflag;
  int          ierr;

  ierr = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag); CHKERRQ(ierr);
  /* compute initial residual: f - M*x */
  /* (inv(b)*a)*x or (a*inv(b)*b)*x into dest */
  if (ksp->pc_side == PC_RIGHT) {
    /* we want a * binv * b * x, or just a * x for the first step */
    /* a*x into temp */
    ierr = MatMult(Amat,VEC_SOLN,VEC_TEMP ); CHKERRQ(ierr);
  }
  else {
    /* else we do binv * a * x */
    ierr = PCApplyBAorAB(ksp->B,ksp->pc_side,VEC_SOLN,VEC_TEMP,
                         VEC_TEMP_MATOP ); CHKERRQ(ierr);
  }
  /* This is an extra copy for the right-inverse case */
  ierr = VecCopy( VEC_BINVF, VEC_VV(0) ); CHKERRQ(ierr);
  ierr = VecAXPY( &mone, VEC_TEMP, VEC_VV(0) ); CHKERRQ(ierr);
      /* inv(b)(f - a*x) into dest */
  return 0;
}

/*
    Run gmres, possibly with restart.  Return residual history if requested.
    input parameters:
.        restart - 1 if restarting gmres, 0 otherwise
.	gmres  - structure containing parameters and work areas
.	itsSoFar- total number of iterations so far (from previous cycles)

    output parameters:
.        nres    - residuals (from preconditioned system) at each step.
                  If restarting, consider passing nres+it.  If null, 
                  ignored
.        itcount - number of iterations used.  nres[0] to nres[itcount]
                  are defined.  If null, ignored.
    Returns:
    0 on success, 1 on failure (did not converge)

    Notes:
    On entry, the value in vector VEC_VV(0) should be the initial residual
    (this allows shortcuts where the initial preconditioned residual is 0).
 */
#undef __FUNC__  
#define __FUNC__ "GMREScycle"
int GMREScycle(int *  itcount, int itsSoFar,int restart,KSP ksp,int *converged )
{
  double    res_norm, res, rtol;
  Scalar    tmp;
  int       hist_len= ksp->res_hist_size, cerr, ierr;
  double    hapbnd,*nres = ksp->residual_history,tt;
  /* Note that hapend is ignored in the code */
  int       it, hapend;
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);
  int       max_k = gmres->max_k, max_it = ksp->max_it;

  /* Question: on restart, compute the residual?  No; provide a restart 
     driver */

  it         = 0;
  *converged = 0;

  /* dest . dest */
  ierr   = VecNorm(VEC_VV(0),NORM_2,&res_norm); CHKERRQ(ierr);
  res    = res_norm;
  *RS(0) = res_norm;

  /* Do-nothing case: */
  if (res_norm == 0.0) {
    if (itcount) *itcount = 0;
    *converged = 1;
    return 0;
  }
  /* scale VEC_VV (the initial residual) */
  tmp = 1.0/res_norm; ierr = VecScale(&tmp , VEC_VV(0) ); CHKERRQ(ierr);

  if (!restart) {
    rtol      = ksp->rtol * res_norm;
    ksp->ttol = (ksp->atol > rtol) ? ksp->atol : rtol;
  }
  rtol= ksp->ttol;
  gmres->it = (it-1);  /* For converged */
  while (!(*converged = cerr = (*ksp->converged)(ksp,it+itsSoFar,res,ksp->cnvP))
           && it < max_k && it + itsSoFar < max_it) {
    if (nres && hist_len > it + itsSoFar) nres[it+itsSoFar]   = res;
    gmres->it = (it - 1);
    KSPMonitor(ksp,it + itsSoFar,res); 
    if (gmres->vv_allocated <= it + VEC_OFFSET + 1) {
      ierr = GMRESGetNewVectors(  ksp, it+1 );CHKERRQ(ierr);
    }
    ierr = PCApplyBAorAB(ksp->B,ksp->pc_side,VEC_VV(it),VEC_VV(1+it),
                         VEC_TEMP_MATOP); CHKERRQ(ierr);

    /* update hessenberg matrix and do Gram-Schmidt */
    (*gmres->orthog)(  ksp, it );

    /* vv(i+1) . vv(i+1) */
    ierr = VecNorm(VEC_VV(it+1),NORM_2,&tt); CHKERRQ(ierr);
    /* save the magnitude */
    *HH(it+1,it)    = tt;
    *HES(it+1,it)   = tt;

    /* check for the happy breakdown */
    hapbnd  = gmres->epsabs * PetscAbsScalar( *HH(it,it) / *RS(it) );
    if (hapbnd > gmres->haptol) hapbnd = gmres->haptol;
    if (tt > hapbnd) {
        tmp = 1.0/tt; ierr = VecScale( &tmp, VEC_VV(it+1) ); CHKERRQ(ierr);
    }
    else {
        /* We SHOULD probably abort the gmres step
           here.  This happens when the solution is exactly reached. */
      hapend = 1;
    }
    ierr = GMRESUpdateHessenberg( ksp, it, &res ); CHKERRQ(ierr);
    it++;
    gmres->it = (it-1);  /* For converged */
  }
  if (nres && hist_len > it + itsSoFar) nres[it + itsSoFar]   = res; 
  if (nres) 
    ksp->res_act_size = (hist_len < it + itsSoFar) ? hist_len : it+itsSoFar+1;
  gmres->it = it - 1;
  KSPMonitor( ksp,  it + itsSoFar, res );
  if (itcount) *itcount    = it;

  /*
    Down here we have to solve for the "best" coefficients of the Krylov
    columns, add the solution values together, and possibly unwind the
    preconditioning from the solution
   */
  if (it == 0) {
    *converged = 1;
    return 0;
  }

  /* Form the solution (or the solution so far) */
  ierr = BuildGmresSoln(RS(0),VEC_SOLN,VEC_SOLN,ksp,it-1); CHKERRQ(ierr);

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPSolve_GMRES"
static int KSPSolve_GMRES(KSP ksp,int *outits )
{
  int       ierr, restart, its, itcount, converged;
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;

  restart = 0;
  itcount = 0;
  /* Save binv*f */
  if (ksp->pc_side == PC_LEFT) {
    /* inv(b)*f */
    ierr = PCApply(ksp->B, VEC_RHS, VEC_BINVF ); CHKERRQ(ierr);
  }
  else if (ksp->pc_side == PC_RIGHT) {
    ierr = VecCopy( VEC_RHS, VEC_BINVF ); CHKERRQ(ierr);
  }
  /* Compute the initial (preconditioned) residual */
  if (!ksp->guess_zero) {
    ierr = GMRESResidual(  ksp, restart ); CHKERRQ(ierr);
  }
  else {
    ierr = VecCopy( VEC_BINVF, VEC_VV(0) ); CHKERRQ(ierr);
  }
    
  ierr = GMREScycle(&its, itcount, restart, ksp, &converged);CHKERRQ(ierr);
  while (!converged) {
    restart  = 1;
    itcount += its;
    ierr = GMRESResidual(  ksp, restart); CHKERRQ(ierr);
    if (itcount > ksp->max_it) break;
    /* need another check to make sure that gmres breaks out 
       at precisely the number of iterations chosen */
    ierr = GMREScycle(&its, itcount, restart, ksp, &converged);CHKERRQ(ierr);
  }
  itcount += its;      /* add in last call to GMREScycle */
  *outits = itcount;  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPAdjustWork_GMRES"
static int KSPAdjustWork_GMRES(KSP ksp )
{
  KSP_GMRES *gmres;
  int       i,ierr;

  if ( ksp->adjust_work_vectors ) {
    gmres = (KSP_GMRES *) ksp->data;
    for (i=0; i<gmres->vv_allocated; i++) {
      ierr = (*ksp->adjust_work_vectors)(ksp,gmres->user_work[i],gmres->mwork_alloc[i]); 
      CHKERRQ(ierr);
    }  
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPDestroy_GMRES"
static int KSPDestroy_GMRES(PetscObject obj)
{
  KSP       ksp = (KSP) obj;
  KSP_GMRES *gmres = (KSP_GMRES *) ksp->data;
  int       i;

  /* Free the Hessenberg matrix */
  if (gmres->hh_origin) PetscFree( gmres->hh_origin );

  /* Free the pointer to user variables */
  if (gmres->vecs) PetscFree( gmres->vecs );

  /* free work vectors */
  for (i=0; i<gmres->nwork_alloc; i++) {
    VecDestroyVecs(gmres->user_work[i], gmres->mwork_alloc[i] );
  }
  if (gmres->user_work)  PetscFree( gmres->user_work );
  if (gmres->mwork_alloc) PetscFree( gmres->mwork_alloc );
  if (gmres->nrs) PetscFree( gmres->nrs );
  if (gmres->sol_temp) VecDestroy(gmres->sol_temp);
  if (gmres->Rsvd) PetscFree(gmres->Rsvd);
  if (gmres->Dsvd) PetscFree(gmres->Dsvd);
  PetscFree( gmres ); 
  return 0;
}
/*
    BuildGmresSoln - create the solution from the starting vector and the
    current iterates.

    Input parameters:
        nrs - work area of size it + 1.
	vs  - index of initial guess
	vdest - index of result.  Note that vs may == vdest (replace
	        guess with the solution).

     This is an internal routine that knows about the GMRES internals.
 */
#undef __FUNC__  
#define __FUNC__ "BuildGmresSoln"
static int BuildGmresSoln(Scalar* nrs,Vec vs,Vec vdest,KSP ksp, int it )
{
  Scalar    tt, zero = 0.0, one = 1.0;
  int       ierr, ii, k, j;
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);

  /* Solve for solution vector that minimizes the residual */

  /* If it is < 0, no gmres steps have been performed */
  if (it < 0) {
    if (vdest != vs) {
      ierr = VecCopy( vs, vdest ); CHKERRQ(ierr);
    }
    return 0;
  }
  nrs[it] = *RS(it) / *HH(it,it);
  for (ii=1; ii<=it; ii++) {
    k   = it - ii;
    tt  = *RS(k);
    for (j=k+1; j<=it; j++) tt  = tt - *HH(k,j) * nrs[j];
    nrs[k]   = tt / *HH(k,k);
  }

  /* Accumulate the correction to the solution of the preconditioned problem in TEMP */
  ierr = VecSet( &zero, VEC_TEMP ); CHKERRQ(ierr);
  ierr = VecMAXPY(it+1, nrs, VEC_TEMP, &VEC_VV(0) ); CHKERRQ(ierr);

  /* If we preconditioned on the right, we need to solve for the correction to
     the unpreconditioned problem */
  if (ksp->pc_side == PC_RIGHT) {
    if (vdest != vs) {
      ierr = PCApply(ksp->B, VEC_TEMP, vdest ); CHKERRQ(ierr);
      ierr = VecAXPY( &one, vs, vdest ); CHKERRQ(ierr);
    }
    else {
      ierr = PCApply(ksp->B,VEC_TEMP,VEC_TEMP_MATOP); CHKERRQ(ierr);
      ierr = VecAXPY(&one,VEC_TEMP_MATOP,vdest); CHKERRQ(ierr);
    }
  }
  else if (ksp->pc_side == PC_LEFT) {
    if (vdest != vs) {
      ierr = VecCopy( VEC_TEMP, vdest ); CHKERRQ(ierr);
      ierr = VecAXPY( &one, vs, vdest ); CHKERRQ(ierr);
    }
    else {
      ierr = VecAXPY( &one, VEC_TEMP, vdest ); CHKERRQ(ierr);
    }
  }
  return 0;
}
/*
   Do the scalar work for the orthogonalization.  Return new residual.
 */
#undef __FUNC__  
#define __FUNC__ "GMRESUpdateHessenberg"
static int GMRESUpdateHessenberg( KSP ksp, int it, double *res )
{
  Scalar    *hh, *cc, *ss, tt;
  int       j;
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);

  hh  = HH(0,it);
  cc  = CC(0);
  ss  = SS(0);

  /* Apply all the previously computed plane rotations to the new column
     of the Hessenberg matrix */
  for (j=1; j<=it; j++) {
    tt  = *hh;
#if defined(PETSC_COMPLEX)
    *hh = conj(*cc) * tt + *ss * *(hh+1);
#else
    *hh = *cc * tt + *ss * *(hh+1);
#endif
    hh++;
    *hh = *cc++ * *hh - ( *ss++ * tt );
  }

  /*
    compute the new plane rotation, and apply it to:
     1) the right-hand-side of the Hessenberg system
     2) the new column of the Hessenberg matrix
    thus obtaining the updated value of the residual
  */
#if defined(PETSC_COMPLEX)
  tt        = sqrt( conj(*hh) * *hh + conj(*(hh+1)) * *(hh+1) );
#else
  tt        = sqrt( *hh * *hh + *(hh+1) * *(hh+1) );
#endif
  if (tt == 0.0) {SETERRQ(1,0,"Your matrix or preconditioner is the null operator");}
  *cc       = *hh / tt;
  *ss       = *(hh+1) / tt;
  *RS(it+1) = - ( *ss * *RS(it) );
#if defined(PETSC_COMPLEX)
  *RS(it)   = conj(*cc) * *RS(it);
  *hh       = conj(*cc) * *hh + *ss * *(hh+1);
#else
  *RS(it)   = *cc * *RS(it);
  *hh       = *cc * *hh + *ss * *(hh+1);
#endif
  *res      = PetscAbsScalar( *RS(it+1) );
  return 0;
}
/*
   This routine allocates more work vectors, starting from VEC_VV(it).
 */
#undef __FUNC__  
#define __FUNC__ "GMRESGetNewVectors"
static int GMRESGetNewVectors( KSP ksp,int it )
{
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;
  int       nwork = gmres->nwork_alloc,k, nalloc;

  nalloc = gmres->delta_allocate;
  /* Adjust the number to allocate to make sure that we don't exceed the
    number of available slots */
  if (it + VEC_OFFSET + nalloc >= gmres->vecs_allocated)
      nalloc = gmres->vecs_allocated - it - VEC_OFFSET;
  /* CHKPTRQ(nalloc); */
  if (nalloc == 0) return 0;

  gmres->vv_allocated += nalloc;
  VecDuplicateVecs(ksp->vec_rhs, nalloc,&gmres->user_work[nwork] );
  PLogObjectParents(ksp,nalloc,gmres->user_work[nwork]);CHKPTRQ(gmres->user_work[nwork]);
  gmres->mwork_alloc[nwork] = nalloc;
  for (k=0; k<nalloc; k++) {
    gmres->vecs[it+VEC_OFFSET+k] = gmres->user_work[nwork][k];
  }
  gmres->nwork_alloc++;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPGMRESSetRestart"
/*@
    KSPGMRESSetRestart - Sets the number of search directions 
    for GMRES before restart.

    Input Parameters:
.   ksp - the iterative context
.   max_k - the number of directions

    Options Database Key:
$   -ksp_gmres_restart <max_k>

    Note:
    The default value of max_k = 30.

.keywords: GMRES, set, restart

.seealso: KSPGMRESSetOrthogonalization(), KSPGMRESSetPreallocateVectors()
@*/
int KSPGMRESSetRestart(KSP ksp,int max_k )
{
  KSP_GMRES *gmres;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  gmres = (KSP_GMRES *)ksp->data;
  if (ksp->type != KSPGMRES) return 0;
  gmres->max_k = max_k;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPDefaultConverged_GMRES"
int KSPDefaultConverged_GMRES(KSP ksp,int n,double rnorm,void *dummy)
{
  if ( rnorm <= ksp->ttol ) return(1);
  else return(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPBuildSolution_GMRES"
static int KSPBuildSolution_GMRES(KSP ksp,Vec  ptr,Vec *result )
{
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data; 
  int       ierr;

  if (ptr == 0) {
    if (!gmres->sol_temp) {
      ierr = VecDuplicate(ksp->vec_sol,&gmres->sol_temp); CHKERRQ(ierr);
      PLogObjectParent(ksp,gmres->sol_temp);
    }
    ptr = gmres->sol_temp;
  }
  if (!gmres->nrs) {
    /* allocate the work area */
    gmres->nrs = (Scalar *)PetscMalloc((unsigned)(gmres->max_k*sizeof(Scalar)));
    PLogObjectMemory(ksp,gmres->max_k*sizeof(Scalar));
  }

  ierr = BuildGmresSoln(gmres->nrs,VEC_SOLN,ptr,ksp,gmres->it); CHKERRQ(ierr);
  *result = ptr; return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPGMRESSetOrthogonalization"
/*@C
  KSPGMRESSetOrthogonalization - Sets the orthogonalization routine used by GMRES.

  Input Parameters:
.   ksp   - iterative context obtained from KSPCreate
.   fcn   - Orthogonalization function

  Notes:
  Several orthogonalization routines are predefined.
$    KSPGMRESModifiedGramSchmidtOrthogonalization() - default.

$    KSPGMRESUnmodifiedGramSchmidtOrthogonalization() - 
       NOT recommended; however, for some problems, particularly
       when using parallel distributed vectors, this may be
       significantly faster.

$    KSPGMRESIROrthogonalization() - interative refinement
       version of KSPGMRESUnmodifiedGramSchmidtOrthogonalization(),
       which may be more numerically stable.

  Options Database Keys:
$  -ksp_gmres_unmodifiedgramschmidt
$  -ksp_gmres_irorthog

.keywords: GMRES, set, orthogonalization, Gram-Schmidt, iterative refinement

.seealso: KSPGMRESSetRestart(), KSPGMRESSetPreallocateVectors()
@*/
int KSPGMRESSetOrthogonalization( KSP ksp,int (*fcn)(KSP,int) )
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (ksp->type == KSPGMRES) {
    ((KSP_GMRES *)ksp->data)->orthog = fcn;
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPView_GMRES"
static int KSPView_GMRES(PetscObject obj,Viewer viewer)
{
  KSP         ksp = (KSP)obj;
  KSP_GMRES   *gmres = (KSP_GMRES *)ksp->data; 
  FILE        *fd;
  char        *cstr;
  int         ierr;
  ViewerType  vtype;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);

    if (gmres->orthog == KSPGMRESUnmodifiedGramSchmidtOrthogonalization) 
      cstr = "Unmodified Gram-Schmidt Orthogonalization";
    else if (gmres->orthog == KSPGMRESModifiedGramSchmidtOrthogonalization) 
      cstr = "Modified Gram-Schmidt Orthogonalization";
    else if (gmres->orthog == KSPGMRESIROrthogonalization) 
      cstr = "Unmodified Gram-Schmidt + Iterative Refinement Orthogonalization";
    else 
      cstr = "unknown orthogonalization";
    PetscFPrintf(ksp->comm,fd,"    GMRES: restart=%d, using %s\n",
               gmres->max_k,cstr);
  }
  return 0;
}

extern int KSPComputeExtremeSingularValues_GMRES(KSP,double *,double *);
extern int KSPComputeEigenvalues_GMRES(KSP,int,double *,double *);

#undef __FUNC__  
#define __FUNC__ "KSPCreate_GMRES"
int KSPCreate_GMRES(KSP ksp)
{
  KSP_GMRES *gmres;

  gmres = (KSP_GMRES*) PetscMalloc(sizeof(KSP_GMRES)); CHKPTRQ(gmres);
  PetscMemzero(gmres,sizeof(KSP_GMRES));
  PLogObjectMemory(ksp,sizeof(KSP_GMRES));
  ksp->data              = (void *) gmres;
  ksp->type              = KSPGMRES;
  ksp->converged         = KSPDefaultConverged_GMRES;
  ksp->buildsolution     = KSPBuildSolution_GMRES;

  ksp->setup             = KSPSetUp_GMRES;
  ksp->solver            = KSPSolve_GMRES;
  ksp->adjustwork        = KSPAdjustWork_GMRES;
  ksp->destroy           = KSPDestroy_GMRES;
  ksp->view              = KSPView_GMRES;
  ksp->computeextremesingularvalues = KSPComputeExtremeSingularValues_GMRES;
  ksp->computeeigenvalues           = KSPComputeEigenvalues_GMRES;

  gmres->haptol         = 1.0e-8;
  gmres->epsabs         = 1.0e-8;
  gmres->q_preallocate  = 0;
  gmres->delta_allocate = GMRES_DELTA_DIRECTIONS;
  gmres->orthog         = KSPGMRESModifiedGramSchmidtOrthogonalization;
  gmres->nrs            = 0;
  gmres->sol_temp       = 0;
  gmres->max_k          = GMRES_DEFAULT_MAXK;
  gmres->Rsvd           = 0;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPGMRESSetPreAllocateVectors"
/*@
    KSPGMRESSetPreAllocateVectors - Causes GMRES to preallocate all its
    needed work vectors at initial setup rather than the default, which 
    is to allocate them in chunks when needed.

    Input Paramter:
.   ksp   - iterative context obtained from KSPCreate

    Options Database Key:
$   -ksp_gmres_preallocate

.keywords: GMRES, preallocate, vectors

.seealso: KSPGMRESSetRestart(), KSPGMRESSetOrthogonalization()
@*/
int KSPGMRESSetPreAllocateVectors(KSP ksp)
{
  KSP_GMRES *gmres;

  if (ksp->type != KSPGMRES) return 0;
  gmres = (KSP_GMRES *)ksp->data;
  gmres->q_preallocate = 1;
  return 0;
}

