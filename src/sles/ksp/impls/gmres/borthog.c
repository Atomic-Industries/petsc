#ifndef lint
static char vcid[] = "$Id: borthog.c,v 1.34 1997/01/06 20:22:36 balay Exp balay $";
#endif
/*
    Routines used for the orthogonalization of the Hessenberg matrix.

    Note that for the complex numbers version, the VecDot() and
    VecMDot() arguments within the code MUST remain in the order
    given for correct computation of inner products.
*/
#include "src/ksp/impls/gmres/gmresp.h"
#include <math.h>

/*
    This is the basic orthogonalization routine using modified Gram-Schmidt.
 */
#undef __FUNC__  
#define __FUNC__ "KSPGMRESModifiedGramSchmidtOrthogonalization"
int KSPGMRESModifiedGramSchmidtOrthogonalization( KSP ksp,int it )
{
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);
  int       j;
  Scalar    *hh, *hes, tmp;

  PLogEventBegin(KSP_GMRESOrthogonalization,ksp,0,0,0);
  /* update Hessenberg matrix and do Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);
  for (j=0; j<=it; j++) {
    /* ( vv(it+1), vv(j) ) */
    VecDot( VEC_VV(it+1), VEC_VV(j), hh );
    *hes++   = *hh;
    /* vv(j) <- vv(j) - hh[j][it] vv(it) */
    tmp = - (*hh++);  VecAXPY(&tmp , VEC_VV(j), VEC_VV(it+1) );
  }
  PLogEventEnd(KSP_GMRESOrthogonalization,ksp,0,0,0);
  return 0;
}

/*
  This version uses UNMODIFIED Gram-Schmidt.  It is NOT recommended, 
  but it can give better performance when running in a parallel 
  environment

  Multiple applications of this can be used to provide a better 
  orthogonalization (but be careful of the HH and HES values).
 */
#undef __FUNC__  
#define __FUNC__ "KSPGMRESUnmodifiedGramSchmidtOrthogonalization"
int KSPGMRESUnmodifiedGramSchmidtOrthogonalization(KSP  ksp,int it )
{
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);
  int       j;
  Scalar    *hh, *hes;

  PLogEventBegin(KSP_GMRESOrthogonalization,ksp,0,0,0);
  /* update hessenberg matrix and do unmodified Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);

  /* 
   This is really a matrix-vector product, with the matrix stored
   as pointer to rows 
  */
  VecMDot( it+1, VEC_VV(it+1), &(VEC_VV(0)), hes );

  /*
    This is really a matrix-vector product: 
        [h[0],h[1],...]*[ v[0]; v[1]; ...] subtracted from v[it].
  */
  for (j=0; j<=it; j++) hh[j] = -hes[j];
  VecMAXPY(it+1, hh, VEC_VV(it+1),&VEC_VV(0) );
  for (j=0; j<=it; j++) hh[j] = -hh[j];
  PLogEventEnd(KSP_GMRESOrthogonalization,ksp,0,0,0);
  return 0;
}

/*
  This version uses iterative refinement of UNMODIFIED Gram-Schmidt.  
  It can give better performance when running in a parallel 
  environment and in some cases even in a sequential environment (because
  MAXPY has more data reuse).

  Care is taken to accumulate the updated HH/HES values.
 */
#undef __FUNC__  
#define __FUNC__ "KSPGMRESIROrthogonalization"
int KSPGMRESIROrthogonalization(KSP  ksp,int it )
{
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);
  int       j,ncnt;
  Scalar    *hh, *hes,shh[20], *lhh;
  double    dnorm;

  PLogEventBegin(KSP_GMRESOrthogonalization,ksp,0,0,0);
  /* Don't allocate small arrays */
  if (it < 20) lhh = shh;
  else {
    lhh = (Scalar *)PetscMalloc((it+1) * sizeof(Scalar)); CHKPTRQ(lhh);
  }
  
  /* update Hessenberg matrix and do unmodified Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);

  /* Clear hh and hes since we will accumulate values into them */
  for (j=0; j<=it; j++) {
    hh[j]  = 0.0;
    hes[j] = 0.0;
  }

  ncnt = 0;
  do {
    /* 
	 This is really a matrix-vector product, with the matrix stored
	 as pointer to rows 
    */
    VecMDot( it+1, VEC_VV(it+1), &(VEC_VV(0)), lhh ); /* <v,vnew> */

    /*
	 This is really a matrix vector product: 
	 [h[0],h[1],...]*[ v[0]; v[1]; ...] subtracted from v[it].
    */
    for (j=0; j<=it; j++) lhh[j] = - lhh[j];
    VecMAXPY(it+1, lhh, VEC_VV(it+1),&VEC_VV(0) );
    for (j=0; j<=it; j++) {
      hh[j]  -= lhh[j];     /* hh += <v,vnew> */
      hes[j] += lhh[j];     /* hes += - <v,vnew> */
    }

    /* Note that dnorm = (norm(d))**2 */
    dnorm = 0.0;
#if defined(PETSC_COMPLEX)
    for (j=0; j<=it; j++) dnorm += real(lhh[j] * conj(lhh[j]));
#else
    for (j=0; j<=it; j++) dnorm += lhh[j] * lhh[j];
#endif

    /* Continue until either we have only small corrections or we've done
	 as much work as a full orthogonalization (in terms of Mdots) */
  } while (dnorm > 1.0e-16 && ncnt++ < it);

  /* It would be nice to put ncnt somewhere.... */

  if (it >= 20) PetscFree( lhh );
  PLogEventEnd(KSP_GMRESOrthogonalization,ksp,0,0,0);
  return 0;
}

#include "pinclude/plapack.h"

/*  ---------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "KSPComputeExtremeSingularValues_GMRES"
int KSPComputeExtremeSingularValues_GMRES(KSP ksp,double *emax,double *emin)
{
  KSP_GMRES *gmres = (KSP_GMRES *) ksp->data;
  int       n = gmres->it + 1, N = gmres->max_k + 2, ierr, lwork = 5*N;
  int       idummy = N, i;
  Scalar    *R = gmres->Rsvd;
  double    *realpart = gmres->Dsvd;
  Scalar    *work = R + N*N, sdummy;

  if (n == 0) {
    *emax = *emin = 1.0;
    return 0;
  }
  /* copy R matrix to work space */
  PetscMemcpy(R,gmres->hh_origin,N*N*sizeof(Scalar));

  /* zero below diagonal garbage */
  for ( i=0; i<n; i++ ) {
    R[i*N+i+1] = 0.0;
  }
  
  /* compute Singular Values */
#if defined(PARCH_t3d)
  SETERRQ(1,0,"DGESVD not found on Cray T3D\n\
             Therefore not able to provide singular value estimates.");
#else
#if !defined(PETSC_COMPLEX)
  LAgesvd_("N","N",&n,&n,R,&N,realpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,&ierr);
#else
  LAgesvd_("N","N",&n,&n,R,&N,realpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,realpart+N,&ierr);
#endif
  if (ierr) SETERRQ(1,0,"Error in SVD");

  *emin = realpart[n-1];
  *emax = realpart[0];

  return 0;
#endif
}

#if !defined(PETSC_COMPLEX)
#undef __FUNC__  
#define __FUNC__ "KSPComputeEigenvalues_GMRES"
int KSPComputeEigenvalues_GMRES(KSP ksp,int nmax,double *r,double *c)
{
  KSP_GMRES *gmres = (KSP_GMRES *) ksp->data;
  int       n = gmres->it + 1, N = gmres->max_k + 1, ierr, lwork = 5*N;
  int       idummy = N, i,*perm;
  Scalar    *R = gmres->Rsvd;
  Scalar    *work = R + N*N;
  Scalar    *realpart = gmres->Dsvd, *imagpart = realpart + N ;
  Scalar    sdummy;

  if (nmax < n) SETERRQ(1,0,"Not enough room in r and c for eigenvalues");

  if (n == 0) {
    return 0;
  }
  /* copy R matrix to work space */
  PetscMemcpy(R,gmres->hes_origin,N*N*sizeof(Scalar));

  /* compute eigenvalues */
  LAgeev_("N","N",&n,R,&N,realpart,imagpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,&ierr);
  if (ierr) SETERRQ(1,0,"Error in Lapack routine");
  perm = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(perm);
  for ( i=0; i<n; i++ ) { perm[i] = i;}
  ierr = PetscSortDoubleWithPermutation(n,realpart,perm); CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    r[i] = realpart[perm[i]];
    c[i] = imagpart[perm[i]];
  }
  PetscFree(perm);
  
  return 0;
}
#else
#undef __FUNC__  
#define __FUNC__ "KSPComputeEigenvalues_GMRES"
int KSPComputeEigenvalues_GMRES(KSP ksp,int nmax,double *r,double *c)
{
  KSP_GMRES *gmres = (KSP_GMRES *) ksp->data;
  int       n = gmres->it + 1, N = gmres->max_k + 1, ierr, lwork = 5*N;
  int       idummy = N, i,*perm;
  Scalar    *R = gmres->Rsvd;
  Scalar    *work = R + N*N;
  Scalar    *eigs = work + 5*N;
  Scalar    sdummy;

  if (nmax < n) SETERRQ(1,0,"Not enough room in r and c for eigenvalues");

  if (n == 0) {
    return 0;
  }
  /* copy R matrix to work space */
  PetscMemcpy(R,gmres->hes_origin,N*N*sizeof(Scalar));

  /* compute eigenvalues */
  LAgeev_("N","N",&n,R,&N,eigs,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,gmres->Dsvd,&ierr);
  if (ierr) SETERRQ(1,0,"Error in Lapack routine");
  perm = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(perm);
  for ( i=0; i<n; i++ ) { perm[i] = i;}
  for ( i=0; i<n; i++ ) { r[i]    = real(eigs[i]);}
  ierr = PetscSortDoubleWithPermutation(n,r,perm); CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    r[i] = PetscReal(eigs[perm[i]]);
    c[i] = imag(eigs[perm[i]]);
  }
  PetscFree(perm);
  
  return 0;
}
#endif



