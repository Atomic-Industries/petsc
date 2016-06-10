#if !defined(__PETSCDMSWARM_H)
#define __PETSCDMSWARM_H

#include <petscdm.h>

typedef enum {
  DMSWARM_BASIC=0,
  DMSWARM_PIC
} DMSwarmType;

typedef enum {
  DMSWARM_MIGRATE_BASIC=0,
  DMSWARM_MIGRATE_DMCELLNSCATTER,
  DMSWARM_MIGRATE_DMCELLEXACT,
  DMSWARM_MIGRATE_USER
} DMSwarmMigrateType;

typedef enum {
  DMSWARM_COLLECT_BASIC=0,
  DMSWARM_COLLECT_DMDABOUNDINGBOX,
  DMSWARM_COLLECT_GENERAL,
  DMSWARM_COLLECT_USER
} DMSwarmCollectType;

PETSC_EXTERN const char* DMSwarmTypeNames[];
PETSC_EXTERN const char* DMSwarmMigrateTypeNames[];
PETSC_EXTERN const char* DMSwarmCollectTypeNames[];

PETSC_EXTERN const char DMSwarmField_pid[];
PETSC_EXTERN const char DMSwarmField_rank[];
PETSC_EXTERN const char DMSwarmPICField_coor[];


PETSC_EXTERN PetscErrorCode DMSwarmCreateGlobalVectorFromField(DM dm,const char fieldname[],Vec *vec);
PETSC_EXTERN PetscErrorCode DMSwarmDestroyGlobalVectorFromField(DM dm,const char fieldname[],Vec *vec);
PETSC_EXTERN PetscErrorCode DMSwarmCreateGlobalVector(DM, const char[], Vec *);

PETSC_EXTERN PetscErrorCode DMSwarmInitializeFieldRegister(DM dm);
PETSC_EXTERN PetscErrorCode DMSwarmFinalizeFieldRegister(DM dm);
PETSC_EXTERN PetscErrorCode DMSwarmSetLocalSizes(DM dm,PetscInt nlocal,PetscInt buffer);
PETSC_EXTERN PetscErrorCode DMSwarmRegisterPetscDatatypeField(DM dm,const char fieldname[],PetscInt blocksize,PetscDataType type);
PETSC_EXTERN PetscErrorCode DMSwarmRegisterUserStructField(DM dm,const char fieldname[],size_t size);
PETSC_EXTERN PetscErrorCode DMSwarmRegisterUserDatatypeField(DM dm,const char fieldname[],size_t size,PetscInt blocksize);
PETSC_EXTERN PetscErrorCode DMSwarmGetField(DM dm,const char fieldname[],PetscInt *blocksize,PetscDataType *type,void **data);
PETSC_EXTERN PetscErrorCode DMSwarmRestoreField(DM dm,const char fieldname[],PetscInt *blocksize,PetscDataType *type,void **data);

PETSC_EXTERN PetscErrorCode DMSwarmVectorDefineField(DM dm,const char fieldname[]);

PETSC_EXTERN PetscErrorCode DMSwarmAddPoint(DM dm);
PETSC_EXTERN PetscErrorCode DMSwarmAddNPoints(DM dm,PetscInt npoints);
PETSC_EXTERN PetscErrorCode DMSwarmRemovePoint(DM dm);
PETSC_EXTERN PetscErrorCode DMSwarmRemovePointAtIndex(DM dm,PetscInt idx);

PETSC_EXTERN PetscErrorCode DMSwarmGetLocalSize(DM dm,PetscInt *nlocal);
PETSC_EXTERN PetscErrorCode DMSwarmGetSize(DM dm,PetscInt *n);
PETSC_EXTERN PetscErrorCode DMSwarmMigrate(DM dm,PetscBool remove_sent_points);

PETSC_EXTERN PetscErrorCode DMSwarmCollectViewCreate(DM dm);
PETSC_EXTERN PetscErrorCode DMSwarmCollectViewDestroy(DM dm);
PETSC_EXTERN PetscErrorCode DMSwarmSetCellDM(DM dm,DM dmcell);
PETSC_EXTERN PetscErrorCode DMSwarmGetCellDM(DM dm,DM *dmcell);

PETSC_EXTERN PetscErrorCode DMSwarmSetType(DM dm,DMSwarmType stype);

#endif

