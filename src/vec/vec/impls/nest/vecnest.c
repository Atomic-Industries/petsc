
#include <../src/vec/vec/impls/nest/vecnestimpl.h> /*I  "petscvec.h"   I*/
#include <petscdevice.h>

/* check all blocks are filled */
static PetscErrorCode VecAssemblyBegin_Nest(Vec v) {
  Vec_Nest      *vs  = (Vec_Nest *)v->data;
  Vec           *vsv = vs->v;
  const PetscInt nr  = vs->nb;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < nr; ++i) {
    PetscCheck(vsv[i], PetscObjectComm((PetscObject)v), PETSC_ERR_SUP, "Nest vector cannot contain NULL blocks");
    PetscCall(VecAssemblyBegin(vsv[i]));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecAssemblyEnd_Nest(Vec v) {
  Vec_Nest      *vs  = (Vec_Nest *)v->data;
  Vec           *vsv = vs->v;
  const PetscInt nr  = vs->nb;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < nr; ++i) PetscCall(VecAssemblyEnd(vsv[i]));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecDestroy_Nest(Vec v, PetscDeviceContext dctx) {
  Vec_Nest      *vs  = (Vec_Nest *)v->data;
  Vec           *vsv = vs->v;
  const PetscInt nr  = vs->nb;

  PetscFunctionBegin;
  if (vsv) {
    for (PetscInt i = 0; i < nr; ++i) PetscCall(VecDestroyAsync(dctx, vsv + i));
    PetscCall(PetscFree(vs->v));
  }
  for (PetscInt i = 0; i < nr; ++i) PetscCall(ISDestroy(vs->is + i));
  PetscCall(PetscFree(vs->is));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "VecNestGetSubVec_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "VecNestGetSubVecs_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "VecNestSetSubVec_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "VecNestSetSubVecs_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "VecNestGetSize_C", NULL));
  PetscCall(PetscFree(vs));
  PetscFunctionReturn(0);
}

/* supports nested blocks */
static PetscErrorCode VecCopy_Nest(Vec x, Vec y, PetscDeviceContext dctx) {
  Vec_Nest      *bx  = (Vec_Nest *)x->data;
  Vec           *bxv = bx->v, *byv = ((Vec_Nest *)y->data)->v;
  const PetscInt nr = bx->nb;

  PetscFunctionBegin;
  PetscCheckTypeName(y, VECNEST);
  VecNestCheckCompatible2(x, 1, y, 2);
  for (PetscInt i = 0; i < nr; ++i) PetscCall(VecCopyAsync(bxv[i], byv[i], dctx));
  PetscFunctionReturn(0);
}

/* supports nested blocks */
static PetscErrorCode VecDuplicate_Nest(Vec x, Vec *y, PetscDeviceContext dctx) {
  Vec_Nest      *bx = (Vec_Nest *)x->data;
  Vec           *sub, *bxv = bx->v;
  const PetscInt nr = bx->nb;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(nr, &sub));
  for (PetscInt i = 0; i < nr; ++i) PetscCall(VecDuplicateAsync(bxv[i], sub + i, dctx));
  PetscCall(VecCreateNest(PetscObjectComm((PetscObject)x), nr, bx->is, sub, y));
  for (PetscInt i = 0; i < nr; ++i) PetscCall(VecDestroyAsync(dctx, sub + i));
  PetscCall(PetscFree(sub));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecXDot_Nest(Vec x, Vec y, PetscManagedScalar scal, PetscDeviceContext dctx, PetscErrorCode (*const VecXDot_Fn)(Vec, Vec, PetscScalar *)) {
  Vec_Nest      *bx  = (Vec_Nest *)x->data;
  Vec           *bxv = bx->v, *byv = ((Vec_Nest *)y->data)->v;
  const PetscInt nr  = bx->nb;
  PetscScalar    tmp = 0.0;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < nr; ++i) {
    PetscScalar x_dot_y;

    PetscCall(VecXDot_Fn(bxv[i], byv[i], &x_dot_y));
    tmp += x_dot_y;
  }
  PetscCall(PetscManagedScalarSetValues(dctx, scal, PETSC_MEMTYPE_HOST, &tmp, 1));
  PetscFunctionReturn(0);
}

/* supports nested blocks */
static PetscErrorCode VecDot_Nest(Vec x, Vec y, PetscManagedScalar val, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(VecXDot_Nest(x, y, val, dctx, VecDot));
  PetscFunctionReturn(0);
}

/* supports nested blocks */
static PetscErrorCode VecTDot_Nest(Vec x, Vec y, PetscManagedScalar val, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(VecXDot_Nest(x, y, val, dctx, VecTDot));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecDotNorm2_Nest(Vec x, Vec y, PetscManagedScalar dp, PetscManagedScalar nm, PetscDeviceContext dctx) {
  Vec_Nest      *bx  = (Vec_Nest *)x->data;
  Vec           *bxv = bx->v, *byv = ((Vec_Nest *)y->data)->v;
  const PetscInt nr  = bx->nb;
  PetscScalar    _dp = 0.0, _nm = 0.0;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < nr; ++i) {
    PetscScalar x_dot_y;
    PetscReal   norm2_y;

    PetscCall(VecDotNorm2(bxv[i], byv[i], &x_dot_y, &norm2_y));
    _dp += x_dot_y;
    _nm += norm2_y;
  }
  PetscCall(PetscManagedScalarSetValues(dctx, dp, PETSC_MEMTYPE_HOST, &_dp, 1));
  PetscCall(PetscManagedScalarSetValues(dctx, nm, PETSC_MEMTYPE_HOST, &_nm, 1));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecAZPZ_Nest_Private(Vec x, PetscManagedScalar alpha, Vec y, PetscDeviceContext dctx, PetscErrorCode (*VecAZPZAsync_Fn)(Vec, PetscManagedScalar, Vec, PetscDeviceContext)) {
  Vec_Nest      *bx  = (Vec_Nest *)x->data;
  Vec           *bxv = bx->v, *byv = ((Vec_Nest *)y->data)->v;
  const PetscInt nr = bx->nb;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < nr; ++i) PetscCall(VecAZPZAsync_Fn(bxv[i], alpha, byv[i], dctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecAXPY_Nest(Vec x, PetscManagedScalar alpha, Vec y, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(VecAZPZ_Nest_Private(x, alpha, y, dctx, VecAXPYAsync));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecAYPX_Nest(Vec x, PetscManagedScalar alpha, Vec y, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(VecAZPZ_Nest_Private(x, alpha, y, dctx, VecAYPXAsync));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecAXPBY_Nest(Vec x, PetscManagedScalar alpha, PetscManagedScalar beta, Vec y, PetscDeviceContext dctx) {
  Vec_Nest      *bx  = (Vec_Nest *)x->data;
  Vec           *bxv = bx->v, *byv = ((Vec_Nest *)y->data)->v;
  const PetscInt nr = bx->nb;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < nr; ++i) PetscCall(VecAXPBYAsync(byv[i], alpha, beta, bxv[i], dctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecAXPBYPCZ_Nest(Vec z, PetscManagedScalar alpha, PetscManagedScalar beta, PetscManagedScalar gamma, Vec x, Vec y, PetscDeviceContext dctx) {
  Vec_Nest      *bx  = (Vec_Nest *)x->data;
  Vec           *bxv = bx->v, *byv = ((Vec_Nest *)y->data)->v, *bzv = ((Vec_Nest *)z->data)->v;
  const PetscInt nr = bx->nb;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < nr; ++i) PetscCall(VecAXPBYPCZAsync(bzv[i], alpha, beta, gamma, bxv[i], byv[i], dctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScale_Nest(Vec x, PetscManagedScalar alpha, PetscDeviceContext dctx) {
  Vec_Nest      *bx  = (Vec_Nest *)x->data;
  Vec           *bxv = bx->v;
  const PetscInt nr  = bx->nb;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < nr; ++i) PetscCall(VecScaleAsync(bxv[i], alpha, dctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecPointwiseApply_Nest(Vec w, Vec x, Vec y, PetscDeviceContext dctx, PetscErrorCode (*const VecPointwiseApply_Fn)(Vec, Vec, Vec, PetscDeviceContext)) {
  Vec_Nest      *bw  = (Vec_Nest *)w->data;
  Vec           *bwv = bw->v, *bxv = ((Vec_Nest *)x->data)->v, *byv = ((Vec_Nest *)y->data)->v;
  const PetscInt nr = bw->nb;

  PetscFunctionBegin;
  VecNestCheckCompatible3(w, 1, x, 2, y, 3);
  for (PetscInt i = 0; i < nr; ++i) PetscCall(VecPointwiseApply_Fn(bwv[i], bxv[i], byv[i], dctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecPointwiseMult_Nest(Vec w, Vec x, Vec y, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(VecPointwiseApply_Nest(w, x, y, dctx, VecPointwiseMultAsync));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecPointwiseDivide_Nest(Vec w, Vec x, Vec y, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(VecPointwiseApply_Nest(w, x, y, dctx, VecPointwiseDivideAsync));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecReciprocal_Nest(Vec x, PetscDeviceContext dctx) {
  Vec_Nest      *bx  = (Vec_Nest *)x->data;
  Vec           *bxv = bx->v;
  const PetscInt nr  = bx->nb;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < nr; ++i) PetscCall(VecReciprocalAsync(bxv[i], dctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecNorm_Nest(Vec xin, NormType type, PetscManagedReal z, PetscDeviceContext dctx) {
  Vec_Nest      *bx  = (Vec_Nest *)xin->data;
  Vec           *bxv = bx->v;
  const PetscInt nr  = bx->nb;
  PetscReal      _z;

  PetscFunctionBegin;
  switch (type) {
  case NORM_2: {
    PetscScalar dot;

    PetscCall(VecDot(xin, xin, &dot));
    _z = PetscAbsScalar(PetscSqrtScalar(dot));
  } break;
  case NORM_1:
  case NORM_INFINITY:
    _z = 0.0;
    for (PetscInt i = 0; i < nr; ++i) {
      PetscReal z_i;

      PetscCall(VecNorm(bxv[i], type, &z_i));
      if (type == NORM_1) {
        _z += z_i;
      } else if (z_i > _z) {
        _z = z_i;
      }
    }
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)xin), PETSC_ERR_SUP, "No support for NormType %s", NormTypes[type]);
  }
  PetscCall(PetscManagedRealSetValues(dctx, z, PETSC_MEMTYPE_HOST, &_z, 1));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMAXPY_Nest(Vec y, PetscManagedInt nv, PetscManagedScalar alpha, Vec *x, PetscDeviceContext dctx) {
  PetscInt *nvptr;

  PetscFunctionBegin;
  PetscCall(PetscManagedIntGetValues(dctx, nv, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &nvptr));
  /* Do axpy on each vector, v */
  for (PetscInt v = 0, nvval = *nvptr; v < nvval; ++v) {
    PetscManagedScalar tmp;

    PetscCall(PetscManagedScalarGetSubRange(dctx, alpha, v, 1, &tmp));
    PetscCall(VecAXPYAsync(y, tmp, x[v], dctx));
    PetscCall(PetscManagedScalarRestoreSubRange(dctx, alpha, &tmp));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMXDot_Nest_Private(Vec x, PetscManagedInt nv, const Vec y[], PetscManagedScalar val, PetscDeviceContext dctx, PetscErrorCode (*const VecDotFunc)(Vec, Vec, PetscManagedScalar, PetscDeviceContext)) {
  PetscInt *nvptr;

  PetscFunctionBegin;
  PetscCall(PetscManagedIntGetValues(dctx, nv, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &nvptr));
  for (PetscInt j = 0, nvval = *nvptr; j < nvval; ++j) {
    PetscManagedScalar tmp;

    PetscCall(PetscManagedScalarGetSubRange(dctx, val, j, 1, &tmp));
    PetscCall(VecDotFunc(x, y[j], tmp, dctx));
    PetscCall(PetscManagedScalarRestoreSubRange(dctx, val, &tmp));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMDot_Nest(Vec x, PetscManagedInt nv, const Vec y[], PetscManagedScalar val, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(VecMXDot_Nest_Private(x, nv, y, val, dctx, VecDotAsync));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMTDot_Nest(Vec x, PetscManagedInt nv, const Vec y[], PetscManagedScalar val, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(VecMXDot_Nest_Private(x, nv, y, val, dctx, VecTDotAsync));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSet_Nest(Vec x, PetscManagedScalar alpha, PetscDeviceContext dctx) {
  Vec_Nest      *bx  = (Vec_Nest *)x->data;
  Vec           *bxv = bx->v;
  const PetscInt nr  = bx->nb;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < nr; ++i) PetscCall(VecSetAsync(bxv[i], alpha, dctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecConjugate_Nest(Vec x, PetscDeviceContext dctx) {
  Vec_Nest      *bx  = (Vec_Nest *)x->data;
  Vec           *bxv = bx->v;
  const PetscInt nr  = bx->nb;

  PetscFunctionBegin;
  for (PetscInt j = 0; j < nr; ++j) PetscCall(VecConjugateAsync(bxv[j], dctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSwap_Nest(Vec x, Vec y, PetscDeviceContext dctx) {
  Vec_Nest      *bx  = (Vec_Nest *)x->data;
  Vec           *bxv = bx->v, *byv = ((Vec_Nest *)y->data)->v;
  const PetscInt nr = bx->nb;

  PetscFunctionBegin;
  VecNestCheckCompatible2(x, 1, y, 2);
  for (PetscInt i = 0; i < nr; ++i) PetscCall(VecSwapAsync(bxv[i], byv[i], dctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecWAXPY_Nest(Vec w, PetscManagedScalar alpha, Vec x, Vec y, PetscDeviceContext dctx) {
  Vec_Nest      *bx  = (Vec_Nest *)x->data;
  Vec           *bxv = bx->v, *byv = ((Vec_Nest *)y->data)->v, *bwv = ((Vec_Nest *)w->data)->v;
  const PetscInt nr = bx->nb;

  PetscFunctionBegin;
  VecNestCheckCompatible3(w, 1, x, 3, y, 4);
  for (PetscInt i = 0; i < nr; ++i) PetscCall(VecWAXPYAsync(bwv[i], alpha, bxv[i], byv[i], dctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMinMax_Nest_Recursive(Vec x, PetscInt *cnt, PetscInt *p, PetscReal *val, PetscErrorCode (*const VecFun)(Vec, PetscInt *, PetscReal *), int (*const cmp)(PetscReal, PetscReal)) {
  PetscBool isnest;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)x, VECNEST, &isnest));
  if (isnest) {
    const Vec_Nest *bx  = (Vec_Nest *)x->data;
    const Vec      *bxv = bx->v;
    const PetscInt  nr  = bx->nb;

    /* now descend recursively */
    for (PetscInt i = 0; i < nr; ++i) PetscCall(VecMinMax_Nest_Recursive(bxv[i], cnt, p, val, VecFun, cmp));
  } else {
    PetscReal entry_val;
    PetscInt  entry_loc, len;

    /* not nest */
    PetscCall(VecFun(x, &entry_loc, &entry_val));
    if (cmp(entry_val, *val)) {
      *val = entry_val;
      *p   = entry_loc + (*cnt);
    }
    PetscCall(VecGetSize(x, &len));
    *cnt += len;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMinMax_Nest_Private(Vec x, PetscManagedInt p, PetscManagedReal m, PetscDeviceContext dctx, PetscReal minit, PetscErrorCode (*const VecFun)(Vec, PetscInt *, PetscReal *), int (*const cmp)(PetscReal, PetscReal)) {
  PetscInt ptmp = 0, cnt = 0;

  PetscFunctionBegin;
  PetscCall(VecMinMax_Nest_Recursive(x, &cnt, &ptmp, &minit, VecFun, cmp));
  if (p) PetscCall(PetscManagedIntSetValues(dctx, p, PETSC_MEMTYPE_HOST, &ptmp, 1));
  PetscCall(PetscManagedRealSetValues(dctx, m, PETSC_MEMTYPE_HOST, &minit, 1));
  PetscFunctionReturn(0);
}

static int PetscRealGT(PetscReal l, PetscReal r) {
  return l > r;
}

/* supports nested blocks */
static PetscErrorCode VecMax_Nest(Vec x, PetscManagedInt p, PetscManagedReal max, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(VecMinMax_Nest_Private(x, p, max, dctx, PETSC_MIN_REAL, VecMax, PetscRealGT));
  PetscFunctionReturn(0);
}

static int PetscRealLT(PetscReal l, PetscReal r) {
  return l < r;
}

static PetscErrorCode VecMin_Nest(Vec x, PetscManagedInt p, PetscManagedReal min, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(VecMinMax_Nest_Private(x, p, min, dctx, PETSC_MAX_REAL, VecMin, PetscRealLT));
  PetscFunctionReturn(0);
}

/* supports nested blocks */
static PetscErrorCode VecView_Nest(Vec x, PetscViewer viewer) {
  Vec_Nest *bx = (Vec_Nest *)x->data;
  PetscBool isascii;
  PetscInt  i;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "VecNest, rows=%" PetscInt_FMT ",  structure: \n", bx->nb));
    for (i = 0; i < bx->nb; i++) {
      VecType  type;
      char     name[256] = "", prefix[256] = "";
      PetscInt NR;

      PetscCall(VecGetSize(bx->v[i], &NR));
      PetscCall(VecGetType(bx->v[i], &type));
      if (((PetscObject)bx->v[i])->name) PetscCall(PetscSNPrintf(name, sizeof(name), "name=\"%s\", ", ((PetscObject)bx->v[i])->name));
      if (((PetscObject)bx->v[i])->prefix) PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "prefix=\"%s\", ", ((PetscObject)bx->v[i])->prefix));

      PetscCall(PetscViewerASCIIPrintf(viewer, "(%" PetscInt_FMT ") : %s%stype=%s, rows=%" PetscInt_FMT " \n", i, name, prefix, type, NR));

      PetscCall(PetscViewerASCIIPushTab(viewer)); /* push1 */
      PetscCall(VecView(bx->v[i], viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer)); /* pop1 */
    }
    PetscCall(PetscViewerASCIIPopTab(viewer)); /* pop0 */
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSize_Nest_Recursive(Vec x, PetscBool globalsize, PetscInt *L) {
  Vec_Nest *bx;
  PetscInt  size, i, nr;
  PetscBool isnest;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)x, VECNEST, &isnest));
  if (!isnest) {
    /* Not nest */
    if (globalsize) PetscCall(VecGetSize(x, &size));
    else PetscCall(VecGetLocalSize(x, &size));
    *L = *L + size;
    PetscFunctionReturn(0);
  }

  /* Otherwise we have a nest */
  bx = (Vec_Nest *)x->data;
  nr = bx->nb;

  /* now descend recursively */
  for (i = 0; i < nr; i++) PetscCall(VecSize_Nest_Recursive(bx->v[i], globalsize, L));
  PetscFunctionReturn(0);
}

/* Returns the sum of the global size of all the consituent vectors in the nest */
static PetscErrorCode VecGetSize_Nest(Vec x, PetscInt *N) {
  PetscFunctionBegin;
  *N = x->map->N;
  PetscFunctionReturn(0);
}

/* Returns the sum of the local size of all the consituent vectors in the nest */
static PetscErrorCode VecGetLocalSize_Nest(Vec x, PetscInt *n) {
  PetscFunctionBegin;
  *n = x->map->n;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMaxPointwiseDivide_Nest(Vec x, Vec y, PetscManagedReal max, PetscDeviceContext dctx) {
  const Vec_Nest *bx  = (Vec_Nest *)x->data;
  const Vec      *bxv = bx->v, *byv = ((Vec_Nest *)y->data)->v;
  const PetscInt  nr = bx->nb;
  PetscReal       m  = 0.0;

  PetscFunctionBegin;
  VecNestCheckCompatible2(x, 1, y, 2);
  for (PetscInt i = 0; i < nr; ++i) {
    PetscReal local_max;

    PetscCall(VecMaxPointwiseDivide(bxv[i], byv[i], &local_max));
    if (local_max > m) m = local_max;
  }
  PetscCall(PetscManagedRealSetValues(dctx, max, PETSC_MEMTYPE_HOST, &m, 1));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecGetSubVector_Nest(Vec X, IS is, Vec *x) {
  Vec_Nest *bx = (Vec_Nest *)X->data;
  PetscInt  i;

  PetscFunctionBegin;
  *x = NULL;
  for (i = 0; i < bx->nb; i++) {
    PetscBool issame = PETSC_FALSE;
    PetscCall(ISEqual(is, bx->is[i], &issame));
    if (issame) {
      *x = bx->v[i];
      PetscCall(PetscObjectReference((PetscObject)(*x)));
      break;
    }
  }
  PetscCheck(*x, PetscObjectComm((PetscObject)is), PETSC_ERR_ARG_OUTOFRANGE, "Index set not found in nested Vec");
  PetscFunctionReturn(0);
}

static PetscErrorCode VecRestoreSubVector_Nest(Vec X, IS is, Vec *x) {
  PetscFunctionBegin;
  PetscCall(VecDestroy(x));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecGetArray_Nest(Vec X, PetscScalar **x, PetscDeviceContext PETSC_UNUSED dctx) {
  Vec_Nest *bx = (Vec_Nest *)X->data;
  PetscInt  i, m, rstart, rend;

  PetscFunctionBegin;
  PetscCall(VecGetOwnershipRange(X, &rstart, &rend));
  PetscCall(VecGetLocalSize(X, &m));
  PetscCall(PetscMalloc1(m, x));
  for (i = 0; i < bx->nb; i++) {
    Vec                subvec = bx->v[i];
    IS                 isy    = bx->is[i];
    PetscInt           j, sm;
    const PetscInt    *ixy;
    const PetscScalar *y;
    PetscCall(VecGetLocalSize(subvec, &sm));
    PetscCall(VecGetArrayRead(subvec, &y));
    PetscCall(ISGetIndices(isy, &ixy));
    for (j = 0; j < sm; j++) {
      PetscInt ix = ixy[j];
      PetscCheck(ix >= rstart && rend > ix, PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for getting array from nonlocal subvec");
      (*x)[ix - rstart] = y[j];
    }
    PetscCall(ISRestoreIndices(isy, &ixy));
    PetscCall(VecRestoreArrayRead(subvec, &y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecRestoreArray_Nest(Vec X, PetscScalar **x, PetscDeviceContext PETSC_UNUSED dctx) {
  Vec_Nest *bx = (Vec_Nest *)X->data;
  PetscInt  i, m, rstart, rend;

  PetscFunctionBegin;
  PetscCall(VecGetOwnershipRange(X, &rstart, &rend));
  PetscCall(VecGetLocalSize(X, &m));
  for (i = 0; i < bx->nb; i++) {
    Vec             subvec = bx->v[i];
    IS              isy    = bx->is[i];
    PetscInt        j, sm;
    const PetscInt *ixy;
    PetscScalar    *y;
    PetscCall(VecGetLocalSize(subvec, &sm));
    PetscCall(VecGetArray(subvec, &y));
    PetscCall(ISGetIndices(isy, &ixy));
    for (j = 0; j < sm; j++) {
      PetscInt ix = ixy[j];
      PetscCheck(ix >= rstart && rend > ix, PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for getting array from nonlocal subvec");
      y[j] = (*x)[ix - rstart];
    }
    PetscCall(ISRestoreIndices(isy, &ixy));
    PetscCall(VecRestoreArray(subvec, &y));
  }
  PetscCall(PetscFree(*x));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecRestoreArrayRead_Nest(Vec PETSC_UNUSED X, const PetscScalar **x, PetscDeviceContext PETSC_UNUSED dctx) {
  PetscFunctionBegin;
  PetscCall(PetscFree(*x));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecConcatenate_Nest(PetscInt nx, const Vec X[], Vec *Y, IS *x_is[]) {
  PetscFunctionBegin;
  PetscCheck(nx <= 0, PetscObjectComm((PetscObject)(*X)), PETSC_ERR_SUP, "VecConcatenate() is not supported for VecNest");
  PetscFunctionReturn(0);
}

static PetscErrorCode VecNestSetOps_Private(struct _VecOps *ops) {
  PetscFunctionBegin;
  ops->duplicate               = VecDuplicate_Nest;
  ops->duplicatevecs           = VecDuplicateVecs_Default;
  ops->destroyvecs             = VecDestroyVecs_Default;
  ops->dot                     = VecDot_Nest;
  ops->mdot                    = VecMDot_Nest;
  ops->norm                    = VecNorm_Nest;
  ops->tdot                    = VecTDot_Nest;
  ops->mtdot                   = VecMTDot_Nest;
  ops->scale                   = VecScale_Nest;
  ops->copy                    = VecCopy_Nest;
  ops->set                     = VecSet_Nest;
  ops->swap                    = VecSwap_Nest;
  ops->axpy                    = VecAXPY_Nest;
  ops->axpby                   = VecAXPBY_Nest;
  ops->maxpy                   = VecMAXPY_Nest;
  ops->aypx                    = VecAYPX_Nest;
  ops->waxpy                   = VecWAXPY_Nest;
  ops->axpbypcz                = NULL;
  ops->pointwisemult           = VecPointwiseMult_Nest;
  ops->pointwisedivide         = VecPointwiseDivide_Nest;
  ops->setvalues               = NULL;
  ops->assemblybegin           = VecAssemblyBegin_Nest;
  ops->assemblyend             = VecAssemblyEnd_Nest;
  ops->getarray                = VecGetArray_Nest;
  ops->getsize                 = VecGetSize_Nest;
  ops->getlocalsize            = VecGetLocalSize_Nest;
  ops->restorearray            = VecRestoreArray_Nest;
  ops->restorearrayread        = VecRestoreArrayRead_Nest;
  ops->max                     = VecMax_Nest;
  ops->min                     = VecMin_Nest;
  ops->setrandom               = NULL;
  ops->setoption               = NULL;
  ops->setvaluesblocked        = NULL;
  ops->destroy                 = VecDestroy_Nest;
  ops->view                    = VecView_Nest;
  ops->placearray              = NULL;
  ops->replacearray            = NULL;
  ops->dot_local               = VecDot_Nest;
  ops->tdot_local              = VecTDot_Nest;
  ops->norm_local              = VecNorm_Nest;
  ops->mdot_local              = VecMDot_Nest;
  ops->mtdot_local             = VecMTDot_Nest;
  ops->load                    = NULL;
  ops->reciprocal              = VecReciprocal_Nest;
  ops->conjugate               = VecConjugate_Nest;
  ops->setlocaltoglobalmapping = NULL;
  ops->setvalueslocal          = NULL;
  ops->resetarray              = NULL;
  ops->setfromoptions          = NULL;
  ops->maxpointwisedivide      = VecMaxPointwiseDivide_Nest;
  ops->load                    = NULL;
  ops->pointwisemax            = NULL;
  ops->pointwisemaxabs         = NULL;
  ops->pointwisemin            = NULL;
  ops->getvalues               = NULL;
  ops->sqrt                    = NULL;
  ops->abs                     = NULL;
  ops->exp                     = NULL;
  ops->shift                   = NULL;
  ops->create                  = NULL;
  ops->stridegather            = NULL;
  ops->stridescatter           = NULL;
  ops->dotnorm2                = VecDotNorm2_Nest;
  ops->getsubvector            = VecGetSubVector_Nest;
  ops->restoresubvector        = VecRestoreSubVector_Nest;
  ops->axpbypcz                = VecAXPBYPCZ_Nest;
  ops->concatenate             = VecConcatenate_Nest;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecNestGetSubVecs_Private(Vec x, PetscInt m, const PetscInt idxm[], Vec vec[]) {
  Vec_Nest *b = (Vec_Nest *)x->data;
  PetscInt  i;
  PetscInt  row;

  PetscFunctionBegin;
  if (!m) PetscFunctionReturn(0);
  for (i = 0; i < m; i++) {
    row = idxm[i];
    PetscCheck(row < b->nb, PetscObjectComm((PetscObject)x), PETSC_ERR_ARG_OUTOFRANGE, "Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT, row, b->nb - 1);
    vec[i] = b->v[row];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecNestGetSubVec_Nest(Vec X, PetscInt idxm, Vec *sx) {
  PetscFunctionBegin;
  PetscCall(PetscObjectStateIncrease((PetscObject)X));
  PetscCall(VecNestGetSubVecs_Private(X, 1, &idxm, sx));
  PetscFunctionReturn(0);
}

/*@
 VecNestGetSubVec - Returns a single, sub-vector from a nest vector.

 Not collective

 Input Parameters:
+  X  - nest vector
-  idxm - index of the vector within the nest

 Output Parameter:
.  sx - vector at index idxm within the nest

 Notes:

 Level: developer

.seealso: `VecNestGetSize()`, `VecNestGetSubVecs()`
@*/
PetscErrorCode VecNestGetSubVec(Vec X, PetscInt idxm, Vec *sx) {
  PetscFunctionBegin;
  PetscUseMethod(X, "VecNestGetSubVec_C", (Vec, PetscInt, Vec *), (X, idxm, sx));
  PetscFunctionReturn(0);
}

PetscErrorCode VecNestGetSubVecs_Nest(Vec X, PetscInt *N, Vec **sx) {
  Vec_Nest *b = (Vec_Nest *)X->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectStateIncrease((PetscObject)X));
  if (N) *N = b->nb;
  if (sx) *sx = b->v;
  PetscFunctionReturn(0);
}

/*@C
 VecNestGetSubVecs - Returns the entire array of vectors defining a nest vector.

 Not collective

 Input Parameter:
.  X  - nest vector

 Output Parameters:
+  N - number of nested vecs
-  sx - array of vectors

 Notes:
 The user should not free the array sx.

 Fortran Notes:
 The caller must allocate the array to hold the subvectors.

 Level: developer

.seealso: `VecNestGetSize()`, `VecNestGetSubVec()`
@*/
PetscErrorCode VecNestGetSubVecs(Vec X, PetscInt *N, Vec **sx) {
  PetscFunctionBegin;
  PetscUseMethod(X, "VecNestGetSubVecs_C", (Vec, PetscInt *, Vec **), (X, N, sx));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecNestSetSubVec_Private(Vec X, PetscInt idxm, Vec x) {
  Vec_Nest *bx = (Vec_Nest *)X->data;
  PetscInt  i, offset = 0, n = 0, bs;
  IS        is;
  PetscBool issame = PETSC_FALSE;
  PetscInt  N      = 0;

  /* check if idxm < bx->nb */
  PetscCheck(idxm < bx->nb, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Out of range index value %" PetscInt_FMT " maximum %" PetscInt_FMT, idxm, bx->nb);

  PetscFunctionBegin;
  PetscCall(VecDestroy(&bx->v[idxm]));      /* destroy the existing vector */
  PetscCall(VecDuplicate(x, &bx->v[idxm])); /* duplicate the layout of given vector */
  PetscCall(VecCopy(x, bx->v[idxm]));       /* copy the contents of the given vector */

  /* check if we need to update the IS for the block */
  offset = X->map->rstart;
  for (i = 0; i < idxm; i++) {
    n = 0;
    PetscCall(VecGetLocalSize(bx->v[i], &n));
    offset += n;
  }

  /* get the local size and block size */
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetBlockSize(x, &bs));

  /* create the new IS */
  PetscCall(ISCreateStride(PetscObjectComm((PetscObject)x), n, offset, 1, &is));
  PetscCall(ISSetBlockSize(is, bs));

  /* check if they are equal */
  PetscCall(ISEqual(is, bx->is[idxm], &issame));

  if (!issame) {
    /* The IS of given vector has a different layout compared to the existing block vector.
     Destroy the existing reference and update the IS. */
    PetscCall(ISDestroy(&bx->is[idxm]));
    PetscCall(ISDuplicate(is, &bx->is[idxm]));
    PetscCall(ISCopy(is, bx->is[idxm]));

    offset += n;
    /* Since the current IS[idxm] changed, we need to update all the subsequent IS */
    for (i = idxm + 1; i < bx->nb; i++) {
      /* get the local size and block size */
      PetscCall(VecGetLocalSize(bx->v[i], &n));
      PetscCall(VecGetBlockSize(bx->v[i], &bs));

      /* destroy the old and create the new IS */
      PetscCall(ISDestroy(&bx->is[i]));
      PetscCall(ISCreateStride(((PetscObject)bx->v[i])->comm, n, offset, 1, &bx->is[i]));
      PetscCall(ISSetBlockSize(bx->is[i], bs));

      offset += n;
    }

    n = 0;
    PetscCall(VecSize_Nest_Recursive(X, PETSC_TRUE, &N));
    PetscCall(VecSize_Nest_Recursive(X, PETSC_FALSE, &n));
    PetscCall(PetscLayoutSetSize(X->map, N));
    PetscCall(PetscLayoutSetLocalSize(X->map, n));
  }

  PetscCall(ISDestroy(&is));
  PetscFunctionReturn(0);
}

PetscErrorCode VecNestSetSubVec_Nest(Vec X, PetscInt idxm, Vec sx) {
  PetscFunctionBegin;
  PetscCall(PetscObjectStateIncrease((PetscObject)X));
  PetscCall(VecNestSetSubVec_Private(X, idxm, sx));
  PetscFunctionReturn(0);
}

/*@
   VecNestSetSubVec - Set a single component vector in a nest vector at specified index.

   Not collective

   Input Parameters:
+  X  - nest vector
.  idxm - index of the vector within the nest vector
-  sx - vector at index idxm within the nest vector

   Notes:
   The new vector sx does not have to be of same size as X[idxm]. Arbitrary vector layouts are allowed.

   Level: developer

.seealso: `VecNestSetSubVecs()`, `VecNestGetSubVec()`
@*/
PetscErrorCode VecNestSetSubVec(Vec X, PetscInt idxm, Vec sx) {
  PetscFunctionBegin;
  PetscUseMethod(X, "VecNestSetSubVec_C", (Vec, PetscInt, Vec), (X, idxm, sx));
  PetscFunctionReturn(0);
}

PetscErrorCode VecNestSetSubVecs_Nest(Vec X, PetscInt N, PetscInt *idxm, Vec *sx) {
  PetscInt i;

  PetscFunctionBegin;
  PetscCall(PetscObjectStateIncrease((PetscObject)X));
  for (i = 0; i < N; i++) PetscCall(VecNestSetSubVec_Private(X, idxm[i], sx[i]));
  PetscFunctionReturn(0);
}

/*@C
   VecNestSetSubVecs - Sets the component vectors at the specified indices in a nest vector.

   Not collective

   Input Parameters:
+  X  - nest vector
.  N - number of component vecs in sx
.  idxm - indices of component vecs that are to be replaced
-  sx - array of vectors

   Notes:
   The components in the vector array sx do not have to be of the same size as corresponding
   components in X. The user can also free the array "sx" after the call.

   Level: developer

.seealso: `VecNestGetSize()`, `VecNestGetSubVec()`
@*/
PetscErrorCode VecNestSetSubVecs(Vec X, PetscInt N, PetscInt *idxm, Vec *sx) {
  PetscFunctionBegin;
  PetscUseMethod(X, "VecNestSetSubVecs_C", (Vec, PetscInt, PetscInt *, Vec *), (X, N, idxm, sx));
  PetscFunctionReturn(0);
}

PetscErrorCode VecNestGetSize_Nest(Vec X, PetscInt *N) {
  Vec_Nest *b = (Vec_Nest *)X->data;

  PetscFunctionBegin;
  *N = b->nb;
  PetscFunctionReturn(0);
}

/*@
 VecNestGetSize - Returns the size of the nest vector.

 Not collective

 Input Parameter:
.  X  - nest vector

 Output Parameter:
.  N - number of nested vecs

 Notes:

 Level: developer

.seealso: `VecNestGetSubVec()`, `VecNestGetSubVecs()`
@*/
PetscErrorCode VecNestGetSize(Vec X, PetscInt *N) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(X, VEC_CLASSID, 1);
  PetscValidIntPointer(N, 2);
  PetscUseMethod(X, "VecNestGetSize_C", (Vec, PetscInt *), (X, N));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSetUp_Nest_Private(Vec V, PetscInt nb, Vec x[]) {
  Vec_Nest *ctx = (Vec_Nest *)V->data;
  PetscInt  i;

  PetscFunctionBegin;
  if (ctx->setup_called) PetscFunctionReturn(0);

  ctx->nb = nb;
  PetscCheck(ctx->nb >= 0, PetscObjectComm((PetscObject)V), PETSC_ERR_ARG_WRONG, "Cannot create VECNEST with < 0 blocks.");

  /* Create space */
  PetscCall(PetscMalloc1(ctx->nb, &ctx->v));
  for (i = 0; i < ctx->nb; i++) {
    ctx->v[i] = x[i];
    PetscCall(PetscObjectReference((PetscObject)x[i]));
    /* Do not allocate memory for internal sub blocks */
  }

  PetscCall(PetscMalloc1(ctx->nb, &ctx->is));

  ctx->setup_called = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSetUp_NestIS_Private(Vec V, PetscInt nb, IS is[]) {
  Vec_Nest *ctx = (Vec_Nest *)V->data;
  PetscInt  i, offset, m, n, M, N;

  PetscFunctionBegin;
  if (is) { /* Do some consistency checks and reference the is */
    offset = V->map->rstart;
    for (i = 0; i < ctx->nb; i++) {
      PetscCall(ISGetSize(is[i], &M));
      PetscCall(VecGetSize(ctx->v[i], &N));
      PetscCheck(M == N, PetscObjectComm((PetscObject)V), PETSC_ERR_ARG_INCOMP, "In slot %" PetscInt_FMT ", IS of size %" PetscInt_FMT " is not compatible with Vec of size %" PetscInt_FMT, i, M, N);
      PetscCall(ISGetLocalSize(is[i], &m));
      PetscCall(VecGetLocalSize(ctx->v[i], &n));
      PetscCheck(m == n, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "In slot %" PetscInt_FMT ", IS of local size %" PetscInt_FMT " is not compatible with Vec of local size %" PetscInt_FMT, i, m, n);
      if (PetscDefined(USE_DEBUG)) { /* This test can be expensive */
        PetscInt  start;
        PetscBool contiguous;
        PetscCall(ISContiguousLocal(is[i], offset, offset + n, &start, &contiguous));
        PetscCheck(contiguous, PetscObjectComm((PetscObject)V), PETSC_ERR_SUP, "Index set %" PetscInt_FMT " is not contiguous with layout of matching vector", i);
        PetscCheck(start == 0, PetscObjectComm((PetscObject)V), PETSC_ERR_SUP, "Index set %" PetscInt_FMT " introduces overlap or a hole", i);
      }
      PetscCall(PetscObjectReference((PetscObject)is[i]));
      ctx->is[i] = is[i];
      offset += n;
    }
  } else { /* Create a contiguous ISStride for each entry */
    offset = V->map->rstart;
    for (i = 0; i < ctx->nb; i++) {
      PetscInt bs;
      PetscCall(VecGetLocalSize(ctx->v[i], &n));
      PetscCall(VecGetBlockSize(ctx->v[i], &bs));
      PetscCall(ISCreateStride(((PetscObject)ctx->v[i])->comm, n, offset, 1, &ctx->is[i]));
      PetscCall(ISSetBlockSize(ctx->is[i], bs));
      offset += n;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   VecCreateNest - Creates a new vector containing several nested subvectors, each stored separately

   Collective on Vec

   Input Parameters:
+  comm - Communicator for the new Vec
.  nb - number of nested blocks
.  is - array of nb index sets describing each nested block, or NULL to pack subvectors contiguously
-  x - array of nb sub-vectors

   Output Parameter:
.  Y - new vector

   Level: advanced

.seealso: `VecCreate()`, `MatCreateNest()`, `DMSetVecType()`, `VECNEST`
@*/
PetscErrorCode VecCreateNest(MPI_Comm comm, PetscInt nb, IS is[], Vec x[], Vec *Y) {
  Vec       V;
  Vec_Nest *s;
  PetscInt  n, N;

  PetscFunctionBegin;
  PetscCall(VecCreate(comm, &V));
  PetscCall(PetscObjectChangeTypeName((PetscObject)V, VECNEST));

  /* allocate and set pointer for implememtation data */
  PetscCall(PetscNew(&s));
  V->data         = (void *)s;
  s->setup_called = PETSC_FALSE;
  s->nb           = -1;
  s->v            = NULL;

  PetscCall(VecSetUp_Nest_Private(V, nb, x));

  n = N = 0;
  PetscCall(VecSize_Nest_Recursive(V, PETSC_TRUE, &N));
  PetscCall(VecSize_Nest_Recursive(V, PETSC_FALSE, &n));
  PetscCall(PetscLayoutSetSize(V->map, N));
  PetscCall(PetscLayoutSetLocalSize(V->map, n));
  PetscCall(PetscLayoutSetBlockSize(V->map, 1));
  PetscCall(PetscLayoutSetUp(V->map));

  PetscCall(VecSetUp_NestIS_Private(V, nb, is));

  PetscCall(VecNestSetOps_Private(V->ops));
  V->petscnative = PETSC_FALSE;

  /* expose block api's */
  PetscCall(PetscObjectComposeFunction((PetscObject)V, "VecNestGetSubVec_C", VecNestGetSubVec_Nest));
  PetscCall(PetscObjectComposeFunction((PetscObject)V, "VecNestGetSubVecs_C", VecNestGetSubVecs_Nest));
  PetscCall(PetscObjectComposeFunction((PetscObject)V, "VecNestSetSubVec_C", VecNestSetSubVec_Nest));
  PetscCall(PetscObjectComposeFunction((PetscObject)V, "VecNestSetSubVecs_C", VecNestSetSubVecs_Nest));
  PetscCall(PetscObjectComposeFunction((PetscObject)V, "VecNestGetSize_C", VecNestGetSize_Nest));

  *Y = V;
  PetscFunctionReturn(0);
}

/*MC
  VECNEST - VECNEST = "nest" - Vector type consisting of nested subvectors, each stored separately.

  Level: intermediate

  Notes:
  This vector type reduces the number of copies for certain solvers applied to multi-physics problems.
  It is usually used with MATNEST and DMComposite via DMSetVecType().

.seealso: `VecCreate()`, `VecType`, `VecCreateNest()`, `MatCreateNest()`
M*/
