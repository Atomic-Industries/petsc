/*
  Common tools for constructing discretizations
*/
#if !defined(__PETSCDT_H)
#define __PETSCDT_H

#include <petscsys.h>

/*S
  PetscQuadrature - Quadrature rule for integration.

  Level: developer

.seealso:  PetscQuadratureCreate(), PetscQuadratureDestroy()
S*/
typedef struct _p_PetscQuadrature *PetscQuadrature;

PETSC_EXTERN PetscErrorCode PetscQuadratureCreate(MPI_Comm, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscQuadratureDuplicate(PetscQuadrature, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscQuadratureGetOrder(PetscQuadrature, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscQuadratureSetOrder(PetscQuadrature, PetscInt);
PETSC_EXTERN PetscErrorCode PetscQuadratureGetNumComponents(PetscQuadrature, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscQuadratureSetNumComponents(PetscQuadrature, PetscInt);
PETSC_EXTERN PetscErrorCode PetscQuadratureGetData(PetscQuadrature, PetscInt*, PetscInt*, PetscInt*, const PetscReal *[], const PetscReal *[]);
PETSC_EXTERN PetscErrorCode PetscQuadratureSetData(PetscQuadrature, PetscInt, PetscInt, PetscInt, const PetscReal [], const PetscReal []);
PETSC_EXTERN PetscErrorCode PetscQuadratureView(PetscQuadrature, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscQuadratureDestroy(PetscQuadrature *);

PETSC_EXTERN PetscErrorCode PetscQuadratureExpandComposite(PetscQuadrature, PetscInt, const PetscReal[], const PetscReal[], PetscQuadrature *);

PETSC_EXTERN PetscErrorCode PetscDTLegendreEval(PetscInt,const PetscReal*,PetscInt,const PetscInt*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTGaussQuadrature(PetscInt,PetscReal,PetscReal,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTReconstructPoly(PetscInt,PetscInt,const PetscReal*,PetscInt,const PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTGaussTensorQuadrature(PetscInt,PetscInt,PetscInt,PetscReal,PetscReal,PetscQuadrature*);
PETSC_EXTERN PetscErrorCode PetscDTGaussJacobiQuadrature(PetscInt,PetscInt,PetscInt,PetscReal,PetscReal,PetscQuadrature*);

PETSC_EXTERN PetscErrorCode PetscDTTanhSinhTensorQuadrature(PetscInt, PetscInt, PetscReal, PetscReal, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscDTTanhSinhIntegrate(void (*)(PetscReal, PetscReal *), PetscReal, PetscReal, PetscInt, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTTanhSinhIntegrateMPFR(void (*)(PetscReal, PetscReal *), PetscReal, PetscReal, PetscInt, PetscReal *);

PETSC_EXTERN PetscErrorCode PetscDTAltVApply(PetscInt, PetscInt, const PetscReal *, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVWedge(PetscInt, PetscInt, PetscInt, const PetscReal *, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVWedgeMatrix(PetscInt, PetscInt, PetscInt, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVPullback(PetscInt, PetscInt, const PetscReal *, PetscInt, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVPullbackMatrix(PetscInt, PetscInt, const PetscReal *, PetscInt, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVInterior(PetscInt, PetscInt, const PetscReal *, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVInteriorMatrix(PetscInt, PetscInt, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTAltVStar(PetscInt, PetscInt, PetscInt, const PetscReal *, PetscReal *);

PETSC_STATIC_INLINE PetscErrorCode PetscDTBinomial(PetscInt n, PetscInt k, PetscInt *binomial)
{
  PetscReal binom = 1;
  PetscInt  i;

  PetscFunctionBeginHot;
  k = PetscMin(k, n - k);
  if (n < 0 || k < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Binomal arguments (%D %D) must be non-negative\n", n, k);
  for (i = 0; i < k; i++) binom = (binom * (n - i)) / (i + 1);
  *binomial = binom;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDTEnumPerm(PetscInt n, PetscInt k, PetscInt *work, PetscInt *perm, PetscBool *isOdd)
{
  PetscBool odd = PETSC_FALSE;
  PetscInt  i;
  PetscInt *w = &work[n - 2];

  PetscFunctionBeginHot;
  for (i = 2; i <= n; i++) {
    *(w--) = k % i;
    k /= i;
  }
  for (i = 0; i < n; i++) perm[i] = i;
  for (i = 0; i < n - 1; i++) {
    PetscInt s = work[i];
    PetscInt swap = perm[i];

    perm[i] = perm[i + s];
    perm[i + s] = swap;
    odd ^= (!!s);
  }
  if (isOdd) *isOdd = odd;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDTEnumSubset(PetscInt n, PetscInt k, PetscInt j, PetscInt *subset)
{
  PetscInt       Nk, i, l;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = PetscDTBinomial(n, k, &Nk);CHKERRQ(ierr);
  for (i = 0, l = 0; i < n && l < k; i++) {
    PetscInt Nminuskminus = (Nk * (k - l)) / (n - i);
    PetscInt Nminusk = Nk - Nminuskminus;

    if (j < Nminuskminus) {
      subset[l++] = i;
      Nk = Nminuskminus;
    } else {
      j -= Nminuskminus;
      Nk = Nminusk;
    }
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDTSubsetIndex(PetscInt n, PetscInt k, const PetscInt *subset, PetscInt *index)
{
  PetscInt       i, j = 0, l, Nk;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = PetscDTBinomial(n, k, &Nk);CHKERRQ(ierr);
  for (i = 0, l = 0; i < n && l < k; i++) {
    PetscInt Nminuskminus = (Nk * (k - l)) / (n - i);
    PetscInt Nminusk = Nk - Nminuskminus;

    if (subset[l] == i) {
      l++;
      Nk = Nminuskminus;
    } else {
      j += Nminuskminus;
      Nk = Nminusk;
    }
  }
  *index = j;
  PetscFunctionReturn(0);
}


PETSC_STATIC_INLINE PetscErrorCode PetscDTEnumSplit(PetscInt n, PetscInt k, PetscInt j, PetscInt *subset, PetscBool *isOdd)
{
  PetscInt       i, l, m, *subcomp, Nk;
  PetscBool      odd;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = PetscDTBinomial(n, k, &Nk);CHKERRQ(ierr);
  odd = PETSC_FALSE;
  subcomp = &subset[k];
  for (i = 0, l = 0, m = 0; i < n && l < k; i++) {
    PetscInt Nminuskminus = (Nk * (k - l)) / (n - i);
    PetscInt Nminusk = Nk - Nminuskminus;

    if (j < Nminuskminus) {
      subset[l++] = i;
      Nk = Nminuskminus;
    } else {
      subcomp[m++] = i;
      j -= Nminuskminus;
      odd ^= ((k - l) & 1);
      Nk = Nminusk;
    }
  }
  for (; i < n; i++) {
    subcomp[m++] = i;
  }
  if (isOdd) *isOdd = odd;
  PetscFunctionReturn(0);
}

#endif
