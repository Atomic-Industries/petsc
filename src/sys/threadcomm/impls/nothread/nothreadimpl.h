
#if !defined(__NOTHREADIMPLH)
#define __NOTHREADIMPLH

#include <petsc-private/threadcommimpl.h>

PETSC_EXTERN PetscErrorCode PetscThreadCommInit_NoThread(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_NoThread(PetscThreadComm);

#endif
