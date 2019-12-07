#include <petsc/private/petscimpl.h>
#include <petsc/private/dtimpl.h>

/*@
   PetscDTAltVApply - Apply a k-form to a set of k N-dimensional vectors

   Input Arguments:
+  N - the dimension of the space
.  k - the index of the alternating form
.  w - the alternating form
-  v - the set of k vectors of size N, size [k x N], row-major

   Output Arguments:
.  wv - w[v[0],...,v[k-1]]

   Level: intermediate

.seealso: PetscDTAltVPullback(), PetscDTAltVPullbackMatrix()
@*/
PetscErrorCode PetscDTAltVApply(PetscInt N, PetscInt k, const PetscReal *w, const PetscReal *v, PetscReal *wv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (N < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid dimension");
  if (k < 0 || k > N) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  if (N <= 3) {
    if (!k) {
      *wv = w[0];
    } else {
      if (N == 1)        {*wv = w[0] * v[0];}
      else if (N == 2) {
        if (k == 1)      {*wv = w[0] * v[0] + w[1] * v[1];}
        else             {*wv = w[0] * (v[0] * v[3] - v[1] * v[2]);}
      } else {
        if (k == 1)      {*wv = w[0] * v[0] + w[1] * v[1] + w[2] * v[2];}
        else if (k == 2) {
          *wv = w[0] * (v[0] * v[4] - v[1] * v[3]) +
                w[1] * (v[0] * v[5] - v[2] * v[3]) +
                w[2] * (v[1] * v[5] - v[2] * v[4]);
        } else {
          *wv = w[0] * (v[0] * (v[4] * v[8] - v[5] * v[7]) +
                        v[1] * (v[5] * v[6] - v[3] * v[8]) +
                        v[2] * (v[3] * v[7] - v[4] * v[6]));
        }
      }
    }
  } else {
    PetscInt Nk, Nf;
    PetscInt *subset, *perm;
    PetscInt i, j, l;
    PetscReal sum = 0.;

    ierr = PetscDTFactorialInt(k, &Nf);CHKERRQ(ierr);
    ierr = PetscDTBinomialInt(N, k, &Nk);CHKERRQ(ierr);
    ierr = PetscMalloc2(k, &subset, k, &perm);CHKERRQ(ierr);
    for (i = 0; i < Nk; i++) {
      PetscReal subsum = 0.;

      ierr = PetscDTEnumSubset(N, k, i, subset);CHKERRQ(ierr);
      for (j = 0; j < Nf; j++) {
        PetscBool permOdd;
        PetscReal prod;

        ierr = PetscDTEnumPerm(k, j, perm, &permOdd);CHKERRQ(ierr);
        prod = permOdd ? -1. : 1.;
        for (l = 0; l < k; l++) {
          prod *= v[perm[l] * N + subset[l]];
        }
        subsum += prod;
      }
      sum += w[i] * subsum;
    }
    ierr = PetscFree2(subset, perm);CHKERRQ(ierr);
    *wv = sum;
  }
  PetscFunctionReturn(0);
}

/*@
   PetscDTAltVWedge - Compute the wedge product of two alternating forms

   Input Arguments:
+  N - the dimension of the space
.  j - the index of the form a
.  k - the index of the form b
.  a - the j-form
-  b - the k-form

   Output Arguments:
.  awedgeb - a wedge b, a (j+k)-form

   Level: intermediate

.seealso: PetscDTAltVWedgeMatrix(), PetscDTAltVPullback(), PetscDTAltVPullbackMatrix()
@*/
PetscErrorCode PetscDTAltVWedge(PetscInt N, PetscInt j, PetscInt k, const PetscReal *a, const PetscReal *b, PetscReal *awedgeb)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (N < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid dimension");
  if (j < 0 || k < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "negative form degree");
  if (j + k > N) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Wedge greater than dimension");
  if (N <= 3) {
    PetscInt Njk;

    ierr = PetscDTBinomialInt(N, j+k, &Njk);CHKERRQ(ierr);
    if (!j)      {for (i = 0; i < Njk; i++) {awedgeb[i] = a[0] * b[i];}}
    else if (!k) {for (i = 0; i < Njk; i++) {awedgeb[i] = a[i] * b[0];}}
    else {
      if (N == 2) {awedgeb[0] = a[0] * b[1] - a[1] * b[0];}
      else {
        if (j+k == 2) {
          awedgeb[0] = a[0] * b[1] - a[1] * b[0];
          awedgeb[1] = a[0] * b[2] - a[2] * b[0];
          awedgeb[2] = a[1] * b[2] - a[2] * b[1];
        } else {
          awedgeb[0] = a[0] * b[2] - a[1] * b[1] + a[2] * b[0];
        }
      }
    }
  } else {
    PetscInt  Njk;
    PetscInt  JKj;
    PetscInt *subset, *subsetjk, *subsetj, *subsetk;
    PetscInt  i;

    ierr = PetscDTBinomialInt(N, j+k, &Njk);CHKERRQ(ierr);
    ierr = PetscDTBinomialInt(j+k, j, &JKj);CHKERRQ(ierr);
    ierr = PetscMalloc4(j+k, &subset, j+k, &subsetjk, j, &subsetj, k, &subsetk);CHKERRQ(ierr);
    for (i = 0; i < Njk; i++) {
      PetscReal sum = 0.;
      PetscInt  l;

      ierr = PetscDTEnumSubset(N, j+k, i, subset);CHKERRQ(ierr);
      for (l = 0; l < JKj; l++) {
        PetscBool jkOdd;
        PetscInt  m, jInd, kInd;

        ierr = PetscDTEnumSplit(j+k, j, l, subsetjk, &jkOdd);CHKERRQ(ierr);
        for (m = 0; m < j; m++) {
          subsetj[m] = subset[subsetjk[m]];
        }
        for (m = 0; m < k; m++) {
          subsetk[m] = subset[subsetjk[j+m]];
        }
        ierr = PetscDTSubsetIndex(N, j, subsetj, &jInd);CHKERRQ(ierr);
        ierr = PetscDTSubsetIndex(N, k, subsetk, &kInd);CHKERRQ(ierr);
        sum += jkOdd ? -(a[jInd] * b[kInd]) : (a[jInd] * b[kInd]);
      }
      awedgeb[i] = sum;
    }
    ierr = PetscFree4(subset, subsetjk, subsetj, subsetk);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   PetscDTAltVWedgeMatrix - Compute the matrix defined by the wedge product with an alternating form

   Input Arguments:
+  N - the dimension of the space
.  j - the index of the form a
.  k - the index of the form that (a wedge) will be applied to
-  a - the j-form

   Output Arguments:
.  awedge - a wedge, an [(N choose j+k) x (N choose k)] matrix in row-major order, such that (a wedge) * b = a wedge b

   Level: intermediate

.seealso: PetscDTAltVPullback(), PetscDTAltVPullbackMatrix()
@*/
PetscErrorCode PetscDTAltVWedgeMatrix(PetscInt N, PetscInt j, PetscInt k, const PetscReal *a, PetscReal *awedgeMat)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (N < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid dimension");
  if (j < 0 || k < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "negative form degree");
  if (j + k > N) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Wedge greater than dimension");
  if (N <= 3) {
    PetscInt Njk;

    ierr = PetscDTBinomialInt(N, j+k, &Njk);CHKERRQ(ierr);
    if (!j) {
      for (i = 0; i < Njk * Njk; i++) {awedgeMat[i] = 0.;}
      for (i = 0; i < Njk; i++) {awedgeMat[i * (Njk + 1)] = a[0];}
    } else if (!k) {
      for (i = 0; i < Njk; i++) {awedgeMat[i] = a[i];}
    } else {
      if (N == 2) {
        awedgeMat[0] = -a[1]; awedgeMat[1] =  a[0];
      } else {
        if (j+k == 2) {
          awedgeMat[0] = -a[1]; awedgeMat[1] =  a[0]; awedgeMat[2] =    0.;
          awedgeMat[3] = -a[2]; awedgeMat[4] =    0.; awedgeMat[5] =  a[0];
          awedgeMat[6] =    0.; awedgeMat[7] = -a[2]; awedgeMat[8] =  a[1];
        } else {
          awedgeMat[0] =  a[2]; awedgeMat[1] = -a[1]; awedgeMat[2] =  a[0];
        }
      }
    }
  } else {
    PetscInt  Njk;
    PetscInt  Nk;
    PetscInt  JKj, i;
    PetscInt *subset, *subsetjk, *subsetj, *subsetk;

    ierr = PetscDTBinomialInt(N,   k,   &Nk);CHKERRQ(ierr);
    ierr = PetscDTBinomialInt(N,   j+k, &Njk);CHKERRQ(ierr);
    ierr = PetscDTBinomialInt(j+k, j,   &JKj);CHKERRQ(ierr);
    ierr = PetscMalloc4(j+k, &subset, j+k, &subsetjk, j, &subsetj, k, &subsetk);CHKERRQ(ierr);
    for (i = 0; i < Njk * Nk; i++) awedgeMat[i] = 0.;
    for (i = 0; i < Njk; i++) {
      PetscInt  l;

      ierr = PetscDTEnumSubset(N, j+k, i, subset);CHKERRQ(ierr);
      for (l = 0; l < JKj; l++) {
        PetscBool jkOdd;
        PetscInt  m, jInd, kInd;

        ierr = PetscDTEnumSplit(j+k, j, l, subsetjk, &jkOdd);CHKERRQ(ierr);
        for (m = 0; m < j; m++) {
          subsetj[m] = subset[subsetjk[m]];
        }
        for (m = 0; m < k; m++) {
          subsetk[m] = subset[subsetjk[j+m]];
        }
        ierr = PetscDTSubsetIndex(N, j, subsetj, &jInd);CHKERRQ(ierr);
        ierr = PetscDTSubsetIndex(N, k, subsetk, &kInd);CHKERRQ(ierr);
        awedgeMat[i * Nk + kInd] += jkOdd ? - a[jInd] : a[jInd];
      }
    }
    ierr = PetscFree4(subset, subsetjk, subsetj, subsetk);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   PetscDTAltVPullback - Compute the pullback of an alternating form under a linear transformation

   Input Arguments:
+  N - the dimension of the origin space
.  M - the dimension of the image space
.  L - the linear transformation, an [M x N] matrix in row-major format
.  k - the index of the form.  A negative form degree indicates that Pullback should be conjugated by the Hodge star operator (see note)
-  w - the k-form in the image space

   Output Arguments:
.  Lstarw - the pullback of w to a k-form in the origin space

   Level: intermediate

   Note: negative form degrees accomodate, e.g., H-div conforming vector fields.  An H-div conforming vector field stores its degrees of freedom as (dx, dy, dz), like a 1-form,
   but its normal trace is integrated on faces, like a 2-form.  The correct pullback then is to apply the Hodge star transformation from 1-form to 2-form, pullback as a 2-form,
   then the inverse Hodge star transformation.

.seealso: PetscDTAltVPullbackMatrix(), PetscDTAltVStar()
@*/
PetscErrorCode PetscDTAltVPullback(PetscInt N, PetscInt M, const PetscReal *L, PetscInt k, const PetscReal *w, PetscReal *Lstarw)
{
  PetscInt         i, j, Nk, Mk;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (N < 0 || M < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid dimensions");
  if (PetscAbsInt(k) > N || PetscAbsInt(k) > M) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  if (N <= 3 && M <= 3) {

    ierr = PetscDTBinomialInt(M, PetscAbsInt(k), &Mk);CHKERRQ(ierr);
    ierr = PetscDTBinomialInt(N, PetscAbsInt(k), &Nk);CHKERRQ(ierr);
    if (!k) {
      Lstarw[0] = w[0];
    } else if (k == 1) {
      for (i = 0; i < Nk; i++) {
        PetscReal sum = 0.;

        for (j = 0; j < Mk; j++) {sum += L[j * Nk + i] * w[j];}
        Lstarw[i] = sum;
      }
    } else if (k == -1) {
      PetscReal mult[3] = {1., -1., 1.};

      for (i = 0; i < Nk; i++) {
        PetscReal sum = 0.;

        for (j = 0; j < Mk; j++) {
          sum += L[(Mk - 1 - j) * Nk + (Nk - 1 - i)] * w[j] * mult[j];
        }
        Lstarw[i] = mult[i] * sum;
      }
    } else if (k == 2) {
      PetscInt pairs[3][2] = {{0,1},{0,2},{1,2}};

      for (i = 0; i < Nk; i++) {
        PetscReal sum = 0.;
        for (j = 0; j < Mk; j++) {
          sum += (L[pairs[j][0] * N + pairs[i][0]] * L[pairs[j][1] * N + pairs[i][1]] -
                  L[pairs[j][1] * N + pairs[i][0]] * L[pairs[j][0] * N + pairs[i][1]]) * w[j];
        }
        Lstarw[i] = sum;
      }
    } else if (k == -2) {
      PetscInt  pairs[3][2] = {{1,2},{2,0},{0,1}};
      PetscInt  offi = (N == 2) ? 2 : 0;
      PetscInt  offj = (M == 2) ? 2 : 0;

      for (i = 0; i < Nk; i++) {
        PetscReal sum   = 0.;

        for (j = 0; j < Mk; j++) {
          sum += (L[pairs[offj + j][0] * N + pairs[offi + i][0]] *
                  L[pairs[offj + j][1] * N + pairs[offi + i][1]] -
                  L[pairs[offj + j][1] * N + pairs[offi + i][0]] *
                  L[pairs[offj + j][0] * N + pairs[offi + i][1]]) * w[j];

        }
        Lstarw[i] = sum;
      }
    } else {
      PetscReal detL = L[0] * (L[4] * L[8] - L[5] * L[7]) +
                       L[1] * (L[5] * L[6] - L[3] * L[8]) +
                       L[2] * (L[3] * L[7] - L[4] * L[6]);

      for (i = 0; i < Nk; i++) {Lstarw[i] = detL * w[i];}
    }
  } else {
    PetscInt         Nf, l, p;
    PetscReal       *Lw, *Lwv;
    PetscInt        *subsetw, *subsetv;
    PetscInt        *perm;
    PetscReal       *walloc = NULL;
    const PetscReal *ww = NULL;
    PetscBool        negative = PETSC_FALSE;

    ierr = PetscDTBinomialInt(M, PetscAbsInt(k), &Mk);CHKERRQ(ierr);
    ierr = PetscDTBinomialInt(N, PetscAbsInt(k), &Nk);CHKERRQ(ierr);
    ierr = PetscDTFactorialInt(PetscAbsInt(k), &Nf);CHKERRQ(ierr);
    if (k < 0) {
      negative = PETSC_TRUE;
      k = -k;
      ierr = PetscMalloc1(Mk, &walloc);CHKERRQ(ierr);
      ierr = PetscDTAltVStar(M, M - k, 1, w, walloc);CHKERRQ(ierr);
      ww = walloc;
    } else {
      ww = w;
    }
    ierr = PetscMalloc5(k, &subsetw, k, &subsetv, k, &perm, N * k, &Lw, k * k, &Lwv);CHKERRQ(ierr);
    for (i = 0; i < Nk; i++) Lstarw[i] = 0.;
    for (i = 0; i < Mk; i++) {
      ierr = PetscDTEnumSubset(M, k, i, subsetw);CHKERRQ(ierr);
      for (j = 0; j < Nk; j++) {
        ierr = PetscDTEnumSubset(N, k, j, subsetv);CHKERRQ(ierr);
        for (p = 0; p < Nf; p++) {
          PetscReal prod;
          PetscBool isOdd;

          ierr = PetscDTEnumPerm(k, p, perm, &isOdd);CHKERRQ(ierr);
          prod = isOdd ? -ww[i] : ww[i];
          for (l = 0; l < k; l++) {
            prod *= L[subsetw[perm[l]] * N + subsetv[l]];
          }
          Lstarw[j] += prod;
        }
      }
    }
    if (negative) {
      PetscReal *sLsw;

      ierr = PetscMalloc1(Nk, &sLsw);CHKERRQ(ierr);
      ierr = PetscDTAltVStar(N, N - k, -1,  Lstarw, sLsw);CHKERRQ(ierr);
      for (i = 0; i < Nk; i++) Lstarw[i] = sLsw[i];
      ierr = PetscFree(sLsw);CHKERRQ(ierr);
    }
    ierr = PetscFree5(subsetw, subsetv, perm, Lw, Lwv);CHKERRQ(ierr);
    ierr = PetscFree(walloc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   PetscDTAltVPullbackMatrix - Compute the pullback matrix for k-forms under a linear transformation

   Input Arguments:
+  N - the dimension of the origin space
.  M - the dimension of the image space
.  L - the linear transformation, an [M x N] matrix in row-major format
-  k - the index of the alternating forms.  A negative form degree indicates that the pullback should be conjugated by the Hodge star operator (see note in PetscDTAltvPullback())

   Output Arguments:
.  Lstar - the pullback matrix, an [(N choose k) x (M choose k)] matrix in row-major format such that Lstar * w = L^* w

   Level: intermediate

.seealso: PetscDTAltVPullback(), PetscDTAltVStar()
@*/
PetscErrorCode PetscDTAltVPullbackMatrix(PetscInt N, PetscInt M, const PetscReal *L, PetscInt k, PetscReal *Lstar)
{
  PetscInt        Nk, Mk, Nf, i, j, l, p;
  PetscReal      *Lw, *Lwv;
  PetscInt       *subsetw, *subsetv;
  PetscInt       *perm;
  PetscBool       negative = PETSC_FALSE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (N < 0 || M < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid dimensions");
  if (PetscAbsInt(k) > N || PetscAbsInt(k) > M) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  if (N <= 3 && M <= 3) {
    PetscReal mult[3] = {1., -1., 1.};

    ierr = PetscDTBinomialInt(M, PetscAbsInt(k), &Mk);CHKERRQ(ierr);
    ierr = PetscDTBinomialInt(N, PetscAbsInt(k), &Nk);CHKERRQ(ierr);
    if (!k) {
      Lstar[0] = 1.;
    } else if (k == 1) {
      for (i = 0; i < Nk; i++) {for (j = 0; j < Mk; j++) {Lstar[i * Mk + j] = L[j * Nk + i];}}
    } else if (k == -1) {
      for (i = 0; i < Nk; i++) {
        for (j = 0; j < Mk; j++) {
          Lstar[i * Mk + j] = L[(Mk - 1 - j) * Nk + (Nk - 1 - i)] * mult[i] * mult[j];
        }
      }
    } else if (k == 2) {
      PetscInt pairs[3][2] = {{0,1},{0,2},{1,2}};

      for (i = 0; i < Nk; i++) {
        for (j = 0; j < Mk; j++) {
          Lstar[i * Mk + j] = L[pairs[j][0] * N + pairs[i][0]] *
                              L[pairs[j][1] * N + pairs[i][1]] -
                              L[pairs[j][1] * N + pairs[i][0]] *
                              L[pairs[j][0] * N + pairs[i][1]];
        }
      }
    } else if (k == -2) {
      PetscInt  pairs[3][2] = {{1,2},{2,0},{0,1}};
      PetscInt  offi = (N == 2) ? 2 : 0;
      PetscInt  offj = (M == 2) ? 2 : 0;

      for (i = 0; i < Nk; i++) {
        for (j = 0; j < Mk; j++) {
          Lstar[i * Mk + j] = L[pairs[offj + j][0] * N + pairs[offi + i][0]] *
                              L[pairs[offj + j][1] * N + pairs[offi + i][1]] -
                              L[pairs[offj + j][1] * N + pairs[offi + i][0]] *
                              L[pairs[offj + j][0] * N + pairs[offi + i][1]];
        }
      }
    } else {
      PetscReal detL = L[0] * (L[4] * L[8] - L[5] * L[7]) +
                       L[1] * (L[5] * L[6] - L[3] * L[8]) +
                       L[2] * (L[3] * L[7] - L[4] * L[6]);

      for (i = 0; i < Nk; i++) {Lstar[i] = detL;}
    }
  } else {
    if (k < 0) {
      negative = PETSC_TRUE;
      k = -k;
    }
    ierr = PetscDTBinomialInt(M, PetscAbsInt(k), &Mk);CHKERRQ(ierr);
    ierr = PetscDTBinomialInt(N, PetscAbsInt(k), &Nk);CHKERRQ(ierr);
    ierr = PetscDTFactorialInt(PetscAbsInt(k), &Nf);CHKERRQ(ierr);
    ierr = PetscMalloc5(M, &subsetw, N, &subsetv, k, &perm, N * k, &Lw, k * k, &Lwv);CHKERRQ(ierr);
    for (i = 0; i < Nk * Mk; i++) Lstar[i] = 0.;
    for (i = 0; i < Mk; i++) {
      PetscBool iOdd;
      PetscInt  iidx, jidx;

      ierr = PetscDTEnumSplit(M, k, i, subsetw, &iOdd);CHKERRQ(ierr);
      iidx = negative ? Mk - 1 - i : i;
      iOdd = negative ? iOdd ^ ((k * (M-k)) & 1) : PETSC_FALSE;
      for (j = 0; j < Nk; j++) {
        PetscBool jOdd;

        ierr = PetscDTEnumSplit(N, k, j, subsetv, &jOdd);CHKERRQ(ierr);
        jidx = negative ? Nk - 1 - j : j;
        jOdd = negative ? iOdd ^ jOdd ^ ((k * (N-k)) & 1) : PETSC_FALSE;
        for (p = 0; p < Nf; p++) {
          PetscReal prod;
          PetscBool isOdd;

          ierr = PetscDTEnumPerm(k, p, perm, &isOdd);CHKERRQ(ierr);
          isOdd ^= jOdd;
          prod = isOdd ? -1. : 1.;
          for (l = 0; l < k; l++) {
            prod *= L[subsetw[perm[l]] * N + subsetv[l]];
          }
          Lstar[jidx * Mk + iidx] += prod;
        }
      }
    }
    ierr = PetscFree5(subsetw, subsetv, perm, Lw, Lwv);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   PetscDTAltVInterior - Compute the interior product of an alternating form with a vector

   Input Arguments:
+  N - the dimension of the origin space
.  k - the index of the alternating forms
.  w - the k-form
-  v - the N dimensional vector

   Output Arguments:
.  wIntv - the (k-1) form (w int v).

   Level: intermediate

.seealso: PetscDTAltVInteriorMatrix(), PetscDTAltVInteriorPattern(), PetscDTAltVPullback(), PetscDTAltVPullbackMatrix()
@*/
PetscErrorCode PetscDTAltVInterior(PetscInt N, PetscInt k, const PetscReal *w, const PetscReal *v, PetscReal *wIntv)
{
  PetscInt        i, Nk, Nkm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (k <= 0 || k > N) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  ierr = PetscDTBinomialInt(N, k,   &Nk);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(N, k-1, &Nkm);CHKERRQ(ierr);
  if (N <= 3) {
    if (k == 1) {
      PetscReal sum = 0.;

      for (i = 0; i < N; i++) {
        sum += w[i] * v[i];
      }
      wIntv[0] = sum;
    } else if (k == N) {
      PetscReal mult[3] = {1., -1., 1.};

      for (i = 0; i < N; i++) {
        wIntv[N - 1 - i] = w[0] * v[i] * mult[i];
      }
    } else {
      wIntv[0] = - w[0]*v[1] - w[1]*v[2];
      wIntv[1] =   w[0]*v[0] - w[2]*v[2];
      wIntv[2] =   w[1]*v[0] + w[2]*v[1];
    }
  } else {
    PetscInt       *subset, *work;

    ierr = PetscMalloc2(k, &subset, k, &work);CHKERRQ(ierr);
    for (i = 0; i < Nkm; i++) wIntv[i] = 0.;
    for (i = 0; i < Nk; i++) {
      PetscInt  j, l, m;

      ierr = PetscDTEnumSubset(N, k, i, subset);CHKERRQ(ierr);
      for (j = 0; j < k; j++) {
        PetscInt  idx;
        PetscBool flip = (j & 1);

        for (l = 0, m = 0; l < k; l++) {
          if (l != j) work[m++] = subset[l];
        }
        ierr = PetscDTSubsetIndex(N, k - 1, work, &idx);CHKERRQ(ierr);
        wIntv[idx] += flip ? -(w[i] * v[subset[j]]) :  (w[i] * v[subset[j]]);
      }
    }
    ierr = PetscFree2(subset, work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   PetscDTAltVInteriorMatrix - Compute the matrix of the linear transformation induced on an alternating form by the interior product with a vector

   Input Arguments:
+  N - the dimension of the origin space
.  k - the index of the alternating forms
-  v - the N dimensional vector

   Output Arguments:
.  intvMat - an [(N choose (k-1)) x (N choose k)] matrix, row-major: (intvMat) * w = (w int v)

   Level: intermediate

.seealso: PetscDTAltVInterior(), PetscDTAltVInteriorPattern(), PetscDTAltVPullback(), PetscDTAltVPullbackMatrix()
@*/
PetscErrorCode PetscDTAltVInteriorMatrix(PetscInt N, PetscInt k, const PetscReal *v, PetscReal *intvMat)
{
  PetscInt        i, Nk, Nkm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (k <= 0 || k > N) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  ierr = PetscDTBinomialInt(N, k,   &Nk);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(N, k-1, &Nkm);CHKERRQ(ierr);
  if (N <= 3) {
    if (k == 1) {
      for (i = 0; i < N; i++) intvMat[i] = v[i];
    } else if (k == N) {
      PetscReal mult[3] = {1., -1., 1.};

      for (i = 0; i < N; i++) intvMat[N - 1 - i] = v[i] * mult[i];
    } else {
      intvMat[0] = -v[1]; intvMat[1] = -v[2]; intvMat[2] =    0.;
      intvMat[3] =  v[0]; intvMat[4] =    0.; intvMat[5] = -v[2];
      intvMat[6] =    0.; intvMat[7] =  v[0]; intvMat[8] =  v[1];
    }
  } else {
    PetscInt       *subset, *work;

    ierr = PetscMalloc2(k, &subset, k, &work);CHKERRQ(ierr);
    for (i = 0; i < Nk * Nkm; i++) intvMat[i] = 0.;
    for (i = 0; i < Nk; i++) {
      PetscInt  j, l, m;

      ierr = PetscDTEnumSubset(N, k, i, subset);CHKERRQ(ierr);
      for (j = 0; j < k; j++) {
        PetscInt  idx;
        PetscBool flip = (j & 1);

        for (l = 0, m = 0; l < k; l++) {
          if (l != j) work[m++] = subset[l];
        }
        ierr = PetscDTSubsetIndex(N, k - 1, work, &idx);CHKERRQ(ierr);
        intvMat[idx * Nk + i] += flip ? -v[subset[j]] :  v[subset[j]];
      }
    }
    ierr = PetscFree2(subset, work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   PetscDTAltVInteriorPattern - compute the sparsity and sign pattern of the interior product matrix computed in PetscDTAltVInteriorMatrix()

   Input Arguments:
+  N - the dimension of the origin space
-  k - the index of the alternating forms

   Output Arguments:
.  indices - The interior product matrix has (N choose k) * k non-zeros.  indices[i][0] and indices[i][1] are the row and column of a non-zero,
   and its value is equal to the vector coordinate v[j] if indices[i][2] = j, or -v[j] if indices[i][2] = -(j+1)

   Level: intermediate

   Note: this form is useful when the interior product needs to be computed at multiple locations, as when computing the Koszul differential

.seealso: PetscDTAltVInterior(), PetscDTAltVInteriorMatrix(), PetscDTAltVPullback(), PetscDTAltVPullbackMatrix()
@*/
PetscErrorCode PetscDTAltVInteriorPattern(PetscInt N, PetscInt k, PetscInt (*indices)[3])
{
  PetscInt        i, Nk, Nkm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (k <= 0 || k > N) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  ierr = PetscDTBinomialInt(N, k,   &Nk);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(N, k-1, &Nkm);CHKERRQ(ierr);
  if (N <= 3) {
    if (k == 1) {
      for (i = 0; i < N; i++) {
        indices[i][0] = 0;
        indices[i][1] = i;
        indices[i][2] = i;
      }
    } else if (k == N) {
      PetscInt val[3] = {0, -2, 2};

      for (i = 0; i < N; i++) {
        indices[i][0] = N - 1 - i;
        indices[i][1] = 0;
        indices[i][2] = val[i];
      }
    } else {
      indices[0][0] = 0; indices[0][1] = 0; indices[0][2] = -(1 + 1);
      indices[1][0] = 0; indices[1][1] = 1; indices[1][2] = -(2 + 1);
      indices[2][0] = 1; indices[2][1] = 0; indices[2][2] = 0;
      indices[3][0] = 1; indices[3][1] = 2; indices[3][2] = -(2 + 1);
      indices[4][0] = 2; indices[4][1] = 1; indices[4][2] = 0;
      indices[5][0] = 2; indices[5][1] = 2; indices[5][2] = 1;
    }
  } else {
    PetscInt       *subset, *work;

    ierr = PetscMalloc2(k, &subset, k, &work);CHKERRQ(ierr);
    for (i = 0; i < Nk; i++) {
      PetscInt  j, l, m;

      ierr = PetscDTEnumSubset(N, k, i, subset);CHKERRQ(ierr);
      for (j = 0; j < k; j++) {
        PetscInt  idx;
        PetscBool flip = (j & 1);

        for (l = 0, m = 0; l < k; l++) {
          if (l != j) work[m++] = subset[l];
        }
        ierr = PetscDTSubsetIndex(N, k - 1, work, &idx);CHKERRQ(ierr);
        indices[i * k + j][0] = idx;
        indices[i * k + j][1] = i;
        indices[i * k + j][2] = flip ? -(subset[j] + 1) : subset[j];
      }
    }
    ierr = PetscFree2(subset, work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   PetscDTAltVStar - Apply a power of the Hodge star transformation to an alternating form

   Input Arguments:
+  N - the dimension of the space
.  k - the index of the alternating form
.  pow - the number of times to apply the Hodge star operator
-  w - the alternating form

   Output Arguments:
.  starw = (star)^pow w

   Level: intermediate

.seealso: PetscDTAltVPullback(), PetscDTAltVPullbackMatrix()
@*/
PetscErrorCode PetscDTAltVStar(PetscInt N, PetscInt k, PetscInt pow, const PetscReal *w, PetscReal *starw)
{
  PetscInt        Nk, i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (k < 0 || k > N) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid form degree");
  ierr = PetscDTBinomialInt(N, k, &Nk);CHKERRQ(ierr);
  pow = pow % 4;
  pow = (pow + 4) % 4; /* make non-negative */
  /* pow is now 0, 1, 2, 3 */
  if (N <= 3) {
    if (pow & 1) {
      PetscReal mult[3] = {1., -1., 1.};

      for (i = 0; i < Nk; i++) starw[Nk - 1 - i] = w[i] * mult[i];
    } else {
      for (i = 0; i < Nk; i++) starw[i] = w[i];
    }
    if (pow > 1 && ((k * (N - k)) & 1)) {
      for (i = 0; i < Nk; i++) starw[i] = -starw[i];
    }
  } else {
    PetscInt       *subset;

    ierr = PetscMalloc1(N, &subset);CHKERRQ(ierr);
    if (pow % 2) {
      PetscInt l = (pow == 1) ? k : N - k;
      for (i = 0; i < Nk; i++) {
        PetscBool sOdd;
        PetscInt  j, idx;

        ierr = PetscDTEnumSplit(N, l, i, subset, &sOdd);CHKERRQ(ierr);
        ierr = PetscDTSubsetIndex(N, l, subset, &idx);CHKERRQ(ierr);
        ierr = PetscDTSubsetIndex(N, N-l, &subset[l], &j);CHKERRQ(ierr);
        starw[j] = sOdd ? -w[idx] : w[idx];
      }
    } else {
      for (i = 0; i < Nk; i++) starw[i] = w[i];
    }
    /* star^2 = -1^(k * (N - k)) */
    if (pow > 1 && (k * (N - k)) % 2) {
      for (i = 0; i < Nk; i++) starw[i] = -starw[i];
    }
    ierr = PetscFree(subset);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
