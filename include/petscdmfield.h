#if !defined(__PETSCDMFIELD_H)
#define      __PETSCDMFIELD_H
#include <petscdm.h>
#include <petscdt.h>

/*S
    DMField - PETSc object for defining a field on a mesh topology

    Level: intermediate
S*/
typedef struct _p_DMField* DMField;

PETSC_EXTERN PetscErrorCode DMFieldInitializePackage(void);
PETSC_EXTERN PetscErrorCode DMFieldFinalizePackage(void);

PETSC_EXTERN PetscClassId DMFIELD_CLASSID;

/*J
    DMFieldType - String with the name of a DMField method

    Level: intermediate
J*/
typedef const char *DMFieldType;
#define DMFIELDDA    "da"
#define DMFIELDDS    "ds"
#define DMFIELDSHELL "shell"

PETSC_EXTERN PetscFunctionList DMFieldList;
PETSC_EXTERN PetscErrorCode    DMFieldSetType(DMField, DMFieldType);
PETSC_EXTERN PetscErrorCode    DMFieldGetType(DMField, DMFieldType*);
PETSC_EXTERN PetscErrorCode    DMFieldRegister(const char[],PetscErrorCode (*)(DMField));

typedef enum {DMFIELD_VERTEX,DMFIELD_EDGE,DMFIELD_FACET,DMFIELD_CELL} DMFieldContinuity;
PETSC_EXTERN const char *const DMFieldContinuities[];

PETSC_EXTERN PetscErrorCode    DMFieldDestroy(DMField*);
PETSC_EXTERN PetscErrorCode    DMFieldView(DMField,PetscViewer);

PETSC_EXTERN PetscErrorCode    DMFieldGetDM(DMField,DM*);
PETSC_EXTERN PetscErrorCode    DMFieldGetNumComponents(DMField,PetscInt*);
PETSC_EXTERN PetscErrorCode    DMFieldGetContinuity(DMField,DMFieldContinuity*);

PETSC_EXTERN PetscErrorCode    DMFieldEvaluate(DMField,Vec,PetscDataType,void*,void*,void*);
PETSC_EXTERN PetscErrorCode    DMFieldEvaluateFE(DMField,PetscInt,const PetscInt*,PetscQuadrature,PetscDataType,void*,void*,void*);
PETSC_EXTERN PetscErrorCode    DMFieldEvaluateFV(DMField,PetscInt,const PetscInt*,PetscDataType,void*,void*,void*);

PETSC_EXTERN PetscErrorCode    DMFieldCreateDA(DM,PetscInt,const PetscScalar *,DMField *);
PETSC_EXTERN PetscErrorCode    DMFieldCreateDS(DM,PetscInt,Vec,DMField *);

PETSC_EXTERN PetscErrorCode    DMFieldCreateShell(DM,PetscInt,DMFieldContinuity,void *,DMField *);
PETSC_EXTERN PetscErrorCode    DMFieldShellGetContext(DMField,void *);
PETSC_EXTERN PetscErrorCode    DMFieldShellSetEvaluate(DMField,PetscErrorCode(*)(DMField,Vec,PetscDataType,void*,void*,void*));
PETSC_EXTERN PetscErrorCode    DMFieldShellSetEvaluateFE(DMField,PetscErrorCode(*)(DMField,PetscInt,const PetscInt*,PetscQuadrature,PetscDataType,void*,void*,void*));
PETSC_EXTERN PetscErrorCode    DMFieldShellSetEvaluateFV(DMField,PetscErrorCode(*)(DMField,PetscInt,const PetscInt*,PetscDataType,void*,void*,void*));

#endif
