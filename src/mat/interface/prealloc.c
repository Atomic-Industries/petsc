#include <petsc/private/matimpl.h> /*I "petscmat.h" I*/
#include <petsc/private/hashmapijv.h>

typedef struct {
  PetscHMapIJV ht;
  PetscInt    *dnz, *onz;
  PetscInt    *dnzu, *onzu;
  PetscBool    nooffproc;
  PetscBool    used;

  struct _MatOps ops;
} Mat_Hash;

/*
   Code currently only works for AIJ matrix, for BAIJ if there are calls to MatSetValues() (not the block version) the indexing
   needs to be converted to block indexing below to get the correct preallocation
*/
PetscErrorCode MatSetValues_Hash(Mat A, PetscInt m, const PetscInt *rows, PetscInt n, const PetscInt *cols, const PetscScalar *values, InsertMode addv)
{
  Mat_Hash *p = (Mat_Hash *)A->hash_ctx;
  PetscInt  rStart, rEnd, r, cStart, cEnd, c;

  PetscFunctionBegin;
  PetscCall(MatGetOwnershipRange(A, &rStart, &rEnd));
  PetscCall(MatGetOwnershipRangeColumn(A, &cStart, &cEnd));
  for (r = 0; r < m; ++r) {
    PetscHashIJKey key;
    PetscBool      missing;

    key.i = rows[r];
    if (key.i < 0) continue;
    if ((key.i < rStart) || (key.i >= rEnd)) {
      PetscCall(MatStashValuesRow_Private(&A->stash, key.i, n, cols, values, PETSC_FALSE));
    } else { /* Hash table is for blocked rows/cols */
      PetscScalar val;
      key.i = rows[r];
      for (c = 0; c < n; ++c) {
        key.j = cols[c];
        if (key.j < 0) continue;
        // needs to be fixed
        val = values[r + c * n];
        switch (addv) {
        case INSERT_VALUES:
          PetscCall(PetscHMapIJVQuerySet(p->ht, key, val, &missing));
          break;
        case ADD_VALUES:
          PetscCall(PetscHMapIJVQueryAdd(p->ht, key, val, &missing));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "InsertMode not supported");
        }
        if (missing) {
          if ((key.j >= cStart) && (key.j < cEnd)) {
            ++p->dnz[key.i - rStart];
            if (key.j >= key.i) ++p->dnzu[key.i - rStart];
          } else {
            ++p->onz[key.i - rStart];
            if (key.j >= key.i) ++p->onzu[key.i - rStart];
          }
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyBegin_Hash(Mat A, MatAssemblyType type)
{
  PetscInt nstash, reallocs;

  PetscFunctionBegin;
  PetscCall(MatStashScatterBegin_Private(A, &A->stash, A->rmap->range));
  PetscCall(MatStashGetInfo_Private(&A->stash, &nstash, &reallocs));
  PetscCall(PetscInfo(A, "Stash has %" PetscInt_FMT " entries, uses %" PetscInt_FMT " mallocs.\n", nstash, reallocs));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_Hash(Mat A, MatAssemblyType type)
{
  PetscHashIter  hi;
  PetscHashIJKey key;
  PetscScalar   *values;
  PetscMPIInt    nm;
  PetscInt      *cols, rStart, rEnd, *rowstarts;
  PetscScalar   *val, value;
  PetscInt      *row, *col;
  PetscInt       n, i, j, rstart, ncols, flg;
  Mat_Hash      *p = (Mat_Hash *)A->hash_ctx;

  PetscFunctionBegin;
  p->nooffproc = PETSC_TRUE;
  while (1) {
    PetscCall(MatStashScatterGetMesg_Private(&A->stash, &nm, &row, &col, &val, &flg));
    if (flg) p->nooffproc = PETSC_FALSE;
    if (!flg) break;

    for (i = 0; i < nm;) {
      /* Now identify the consecutive vals belonging to the same row */
      for (j = i, rstart = row[j]; j < nm; j++) {
        if (row[j] != rstart) break;
      }
      if (j < nm) ncols = j - i;
      else ncols = nm - i;
      /* Now assemble all these values with a single function call */
      PetscCall(MatSetValues_Hash(A, 1, row + i, ncols, col + i, val + i, A->insertmode));
      i = j;
    }
  }
  PetscCall(MatStashScatterEnd_Private(&A->stash));
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &p->nooffproc, 1, MPIU_BOOL, MPI_LAND, PetscObjectComm((PetscObject)A)));
  if (type != MAT_FINAL_ASSEMBLY) PetscFunctionReturn(0);

  A->insertmode   = NOT_SET_VALUES; /* this was set by the previous calls to MatSetValues() */
  A->preallocated = PETSC_FALSE;    /* this was set for the MatSetValues_Hash() to work */

  PetscCall(PetscMemcpy(&A->ops, &p->ops, sizeof(struct _MatOps)));

  /* move values from hash format to matrix type format */
  PetscCall(MatXAIJSetPreallocation(A, 1, p->dnz, p->onz, p->dnzu, p->onzu));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(MatSetOption(A, MAT_NO_OFF_PROC_ENTRIES, p->nooffproc));
  PetscCall(MatGetOwnershipRange(A, &rStart, &rEnd));
  PetscCall(PetscHMapIJVGetSize(p->ht, &n));
  PetscCall(PetscMalloc2(n, &cols, rEnd - rStart + 1, &rowstarts));
  PetscCall(PetscMalloc1(n, &values));
  rowstarts[0] = 0;
  for (PetscInt i = 0; i < rEnd - rStart; i++) { rowstarts[i + 1] = rowstarts[i] + p->dnz[i] + p->onz[i]; }
  PetscCheck(rowstarts[rEnd - rStart] == n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Hash claims %" PetscInt_FMT " entries, but dnz+onz counts %" PetscInt_FMT, n, rowstarts[rEnd - rStart]);

  PetscHashIterBegin(p->ht, hi);
  while (!PetscHashIterAtEnd(p->ht, hi)) {
    PetscHashIterGetKey(p->ht, hi, key);
    PetscInt lrow         = key.i - rStart;
    cols[rowstarts[lrow]] = key.j;
    PetscHashIterGetVal(p->ht, hi, value);
    values[rowstarts[lrow]] = value;
    rowstarts[lrow]++;
    PetscHashIterNext(p->ht, hi);
  }
  PetscCall(PetscHMapIJVDestroy(&p->ht));

  for (PetscInt i = 0; i < rEnd - rStart; i++) {
    PetscInt grow = rStart + i;
    PetscInt end = rowstarts[i], start = end - p->dnz[i] - p->onz[i];
    PetscCall(MatSetValues(A, 1, &grow, end - start, &cols[start], &values[start], INSERT_VALUES));
  }
  PetscCall(PetscFree(values));
  PetscCall(PetscFree2(cols, rowstarts));
  PetscCall(MatSetOption(A, MAT_NO_OFF_PROC_ENTRIES, PETSC_FALSE));

  PetscCall(MatStashDestroy_Private(&A->stash));
  PetscCall(PetscHMapIJVDestroy(&p->ht));
  PetscCall(PetscFree4(p->dnz, p->onz, p->dnzu, p->onzu));
  PetscCall(PetscFree(p));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatSetUp_Default(Mat A)
{
  Mat_Hash *p;
  PetscInt  m;

  PetscFunctionBegin;
  PetscCall(PetscInfo(A, "Using hash-based MatSetValues() because no preallocation provided\n"));
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));

  PetscCall(PetscNew(&p));
  A->hash_ctx = (void *)p;
  PetscCall(MatGetLocalSize(A, &m, NULL));
  PetscCall(PetscHMapIJVCreate(&p->ht));
  /* there is no way to check if a MatStash has been initialized */
  PetscCall(MatStashDestroy_Private(&A->stash));
  PetscCall(MatStashCreate_Private(PetscObjectComm((PetscObject)A), 1, &A->stash));
  PetscCall(PetscCalloc4(m, &p->dnz, m, &p->onz, m, &p->dnzu, m, &p->onzu));

  PetscCall(PetscMemcpy(&p->ops, &A->ops, sizeof(struct _MatOps)));
  PetscCall(PetscMemzero(&A->ops, sizeof(struct _MatOps)));

  A->ops->assemblybegin = MatAssemblyBegin_Hash;
  A->ops->assemblyend   = MatAssemblyEnd_Hash;
  A->ops->setvalues     = MatSetValues_Hash;
  A->preallocated       = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   MatXAIJSetNoPreallocation - use an efficient hash based code to handle `MatSetValues()` before the first
    `MatAssemblyBegin()` and `MatAssemblyEnd()`. At that time the object switches to its usual proceeedure.

 Input Parameter:
.  A - the matrix

 Level: beginnger

.seealso: `Mat`, `MatCreate()`, `MatXAIJSetNoPreallocation()`
@*/
PetscErrorCode MatXAIJSetNoPreallocation(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscInfo(A, "Requesting hash-based MatSetValues()\n"));
  A->ops->setup = MatSetUp_Default;
  PetscFunctionReturn(0);
}
