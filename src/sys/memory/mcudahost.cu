#if defined(PETSC_HAVE_CUDA)

#include <petscsys.h>             /*I   "petscsys.h"   I*/
#include <petsccublas.h>          /* Needed to provide CHKERRCUDA() */

static PetscErrorCode PetscCudaHostMalloc(size_t a,PetscBool clear,int lineno,const char function[],const char filename[],void **result)
{
  cudaError_t ierr;
  ierr = cudaMallocHost(result,a);CHKERRCUDA(ierr);
  return 0;
}

static PetscErrorCode PetscCudaHostFree(void *aa,int lineno,const char function[],const char filename[])
{
  cudaError_t ierr;
  ierr = cudaFreeHost(aa);CHKERRCUDA(ierr);
  return 0;
}

static PetscErrorCode PetscCudaHostRealloc(size_t a,int lineno,const char function[],const char filename[],void **result)
{
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"CUDA has no Realloc()");
  return 0;
}

static PetscErrorCode (*PetscMallocOld)(size_t,PetscBool,int,const char[],const char[],void**);
static PetscErrorCode (*PetscFreeOld)(void*,int,const char[],const char[]);

/*@C
   PetscMallocSetCUDAHost - Set PetscMalloc to use CUDAHostMalloc
     Switch the current malloc and free routines to the CUDA malloc and free routines

   Not Collective

   Level: developer

   Notes:
     This provides a way to use the CUDA malloc and free routines temporarily. One
     can switch back to the previous choice by calling PetscMallocResetCUDAHost().

.seealso: PetscMallocResetCUDAHost()
@*/
PetscErrorCode PetscMallocSetCUDAHost(void)
{
  PetscFunctionBegin;
  /* Save the previous choice */
  PetscMallocOld = PetscTrMalloc;
  PetscFreeOld   = PetscTrFree;
  PetscTrMalloc  = PetscCudaHostMalloc;
  PetscTrFree    = PetscCudaHostFree;
  PetscFunctionReturn(0);
}

/*@C
   PetscMallocResetCUDAHost - Reset the changes made by PetscMallocSetCUDAHost

   Not Collective

   Level: developer

.seealso: PetscMallocSetCUDAHost()
@*/
PetscErrorCode PetscMallocResetCUDAHost(void)
{
  PetscFunctionBegin;
  PetscTrMalloc = PetscMallocOld;
  PetscTrFree   = PetscFreeOld;
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_CUDA */
