#ifndef __TSHISTORYIMPL_H
#define __TSHISTORYIMPL_H

#include <petsc/private/tsimpl.h>

struct _n_TSHistory {
  PETSCHEADER(PetscOps);

  PetscReal *hist;    /* time history */
  PetscInt  *hist_id; /* stores the stepid in time history */
  PetscInt  n;        /* current number of steps stored */
  PetscBool sorted;   /* if the history is sorted in ascending order */
  PetscReal c;        /* current capacity of hist */
  PetscReal s;        /* reallocation size */
};

PETSC_INTERN PetscErrorCode TSHistoryCreate(MPI_Comm,TSHistory*);
PETSC_INTERN PetscErrorCode TSHistoryDestroy(TSHistory*);
PETSC_INTERN PetscErrorCode TSHistorySetHistory(TSHistory,PetscInt,PetscReal[]);
PETSC_INTERN PetscErrorCode TSHistoryGetLocFromTime(TSHistory,PetscReal,PetscInt*);
PETSC_INTERN PetscErrorCode TSHistoryUpdate(TSHistory,PetscInt,PetscReal);
PETSC_INTERN PetscErrorCode TSHistoryGetTimeStep(TSHistory,PetscBool,PetscInt,PetscReal*);
#endif
