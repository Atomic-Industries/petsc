/* $Id: is.h,v 1.39 1997/10/01 22:47:58 bsmith Exp bsmith $ */

/*
   An index set is a generalization of a subset of integers.  Index sets
   are used for defining scatters and gathers.
*/
#if !defined(__IS_PACKAGE)
#define __IS_PACKAGE
#include "petsc.h"

#define IS_COOKIE PETSC_COOKIE+2

typedef struct _p_IS* IS;

/*
    Default index set data structures that PETSc provides.
*/
typedef enum {IS_GENERAL=0, IS_STRIDE=1, IS_BLOCK = 2} ISType;
extern int   ISCreateGeneral(MPI_Comm,int,int *,IS *);
extern int   ISCreateBlock(MPI_Comm,int,int,int *,IS *);
extern int   ISCreateStride(MPI_Comm,int,int,int,IS *);

extern int   ISDestroy(IS);

extern int   ISSetPermutation(IS);
extern int   ISPermutation(IS,PetscTruth*); 
extern int   ISSetIdentity(IS);
extern int   ISIdentity(IS,PetscTruth*);

extern int   ISGetIndices(IS,int **);
extern int   ISRestoreIndices(IS,int **);
extern int   ISGetSize(IS,int *);
extern int   ISInvertPermutation(IS,IS*);
extern int   ISView(IS,Viewer);
extern int   ISEqual(IS, IS, PetscTruth *);
extern int   ISSort(IS);
extern int   ISSorted(IS, PetscTruth *);
extern int   ISDifference(IS,IS,IS*);
extern int   ISSum(IS,IS,IS*);

extern int   ISBlock(IS,PetscTruth*);
extern int   ISBlockGetIndices(IS,int **);
extern int   ISBlockRestoreIndices(IS,int **);
extern int   ISBlockGetSize(IS,int *);
extern int   ISBlockGetBlockSize(IS,int *);

extern int   ISStride(IS,PetscTruth*);
extern int   ISStrideGetInfo(IS,int *,int*);

extern int   ISDuplicate(IS, IS *);
/* --------------------------------------------------------------------------*/

/*
   ISLocalToGlobalMappings are mappings from an arbitrary
  local ordering from 0 to n-1 to a global PETSc ordering 
  used by a vector or matrix.

   Note: mapping from Local to Global is scalable; but Global
  to local may not be if the range of global values represented locally
  is very large.
*/
#define IS_LTOGM_COOKIE PETSC_COOKIE+12
typedef struct _p_ISLocalToGlobalMapping* ISLocalToGlobalMapping;

extern int ISLocalToGlobalMappingCreate(MPI_Comm,int, int*, ISLocalToGlobalMapping*);
extern int ISLocalToGlobalMappingCreateIS(IS,ISLocalToGlobalMapping *);
extern int ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping);
extern int ISLocalToGlobalMappingApply(ISLocalToGlobalMapping,int,int*,int *);
extern int ISLocalToGlobalMappingApplyIS(ISLocalToGlobalMapping,IS,IS*);
typedef enum {IS_GTOLM_MASK,IS_GTOLM_DROP} ISGlobalToLocalMappingType;
extern int ISGlobalToLocalMappingApply(ISLocalToGlobalMapping,ISGlobalToLocalMappingType,
                                       int,int *,int*,int *);

/* --------------------------------------------------------------------------*/

/*
     ISColorings are sets of IS's that define a coloring
   of the underlying indices
*/
struct _p_ISColoring {
  int      n;
  IS       *is;
  MPI_Comm comm;
};
typedef struct _p_ISColoring* ISColoring;

extern int ISColoringDestroy(ISColoring);
extern int ISColoringView(ISColoring,Viewer);
extern int ISColoringCreate(MPI_Comm,int,int*,ISColoring*);

/* --------------------------------------------------------------------------*/

/*
     ISPartitioning are sets of IS's that define a partioning
   of the underlying indices. This is the same as a ISColoring.
*/
#define ISPartitioning        ISColoring
#define ISPartitioningView    ISColoringView
#define ISPartitioningCreate  ISColoringCreate
#define ISPartitioningDestroy ISColoringDestroy
extern int ISPartitioningToLocalIS(ISPartitioning,IS*);

#endif




