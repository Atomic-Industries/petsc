#if !defined(_matelemdenseimpl_h)
#define _matelemdenseimpl_h

#include <../src/mat/impls/elemental/matelemimpl.h> /*I "petscmat.h" I*/

typedef struct {
  PetscInt commsize;
  PetscInt m[2];       /* Number of entries in a local block of the row (column) space */
  PetscInt mr[2];      /* First incomplete/ragged rank of (row) column space.
                          We expose a blocked ordering to the user because that is what all other PETSc infrastructure uses.
                          With the blocked ordering when the number of processes do not evenly divide the vector size,
                          we still need to be able to convert from PETSc/blocked ordering to VC/VR ordering. */
  El::Grid                                     *grid;
  El::DistMatrix<PetscElemScalar>              *emat;
  El::Matrix<PetscElemScalar>                  *esubmat; /* Used for adding off-proc matrix entries */
  El::AxpyInterface<PetscElemScalar>           *interface;
  El::DistMatrix<PetscInt,El::VC,El::STAR> *pivot; /* pivot vector representing the pivot matrix P in PA = LU */
} Mat_ElemDense;

typedef struct {
  El::Grid *grid;
  PetscInt   grid_refct;
} Mat_ElemDense_Grid;

PETSC_STATIC_INLINE void P2RO(Mat A,PetscInt rc,PetscInt p,PetscInt *rank,PetscInt *offset)
{
  Mat_ElemDense *a       = (Mat_ElemDense*)A->data;
  PetscInt      critical = a->m[rc]*a->mr[rc];
  if (p < critical) {
    *rank   = p / a->m[rc];
    *offset = p % a->m[rc];
  } else {
    *rank   = a->mr[rc] + (p - critical) / (a->m[rc] - 1);
    *offset = (p - critical) % (a->m[rc] - 1);
  }
}
PETSC_STATIC_INLINE void RO2P(Mat A,PetscInt rc,PetscInt rank,PetscInt offset,PetscInt *p)
{
  Mat_ElemDense *a = (Mat_ElemDense*)A->data;
  if (rank < a->mr[rc]) {
    *p = rank*a->m[rc] + offset;
  } else {
    *p = a->mr[rc]*a->m[rc] + (rank - a->mr[rc])*(a->m[rc]-1) + offset;
  }
}

PETSC_STATIC_INLINE void E2RO(Mat A,PetscInt rc,PetscInt p,PetscInt *rank,PetscInt *offset)
{
  Mat_ElemDense *a = (Mat_ElemDense*)A->data;
  *rank   = p % a->commsize;
  *offset = p / a->commsize;
}
PETSC_STATIC_INLINE void RO2E(Mat A,PetscInt rc,PetscInt rank,PetscInt offset,PetscInt *e)
{
  Mat_ElemDense *a = (Mat_ElemDense*)A->data;
  *e = offset * a->commsize + rank;
}

#endif
