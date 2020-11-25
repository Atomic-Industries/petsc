/*
   Implements the sequential cuda vectors.
*/

#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_CXX_COMPLEX_FIX

#include <petscconf.h>
#include <petsc/private/vecimpl.h>
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc/private/cudavecimpl.h>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

/*
    Allocates space for the vector array on the GPU if it does not exist.
    Does NOT change the PetscCUDAFlag for the vector
    Does NOT zero the CUDA array

 */
PetscErrorCode VecCUDAAllocateCheck(Vec v)
{
  PetscErrorCode ierr;
  cudaError_t    err;
  Vec_CUDA       *veccuda;
  PetscBool      option_set;

  PetscFunctionBegin;
  if (!v->spptr) {
    PetscReal pinned_memory_min;
    ierr = PetscMalloc(sizeof(Vec_CUDA),&v->spptr);CHKERRQ(ierr);
    veccuda = (Vec_CUDA*)v->spptr;
    err = cudaMalloc((void**)&veccuda->GPUarray_allocated,sizeof(PetscScalar)*((PetscBLASInt)v->map->n));CHKERRCUDA(err);
    veccuda->GPUarray = veccuda->GPUarray_allocated;
    veccuda->stream = 0;  /* using default stream */
    if (v->offloadmask == PETSC_OFFLOAD_UNALLOCATED) {
      if (v->data && ((Vec_Seq*)v->data)->array) {
        v->offloadmask = PETSC_OFFLOAD_CPU;
      } else {
        v->offloadmask = PETSC_OFFLOAD_GPU;
      }
    }
    pinned_memory_min = 0;

    /* Need to parse command line for minimum size to use for pinned memory allocations on host here.
       Note: This same code duplicated in VecCreate_SeqCUDA_Private() and VecCreate_MPICUDA_Private(). Is there a good way to avoid this? */
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)v),((PetscObject)v)->prefix,"VECCUDA Options","Vec");CHKERRQ(ierr);
    ierr = PetscOptionsReal("-vec_pinned_memory_min","Minimum size (in bytes) for an allocation to use pinned memory on host","VecSetPinnedMemoryMin",pinned_memory_min,&pinned_memory_min,&option_set);CHKERRQ(ierr);
    if (option_set) v->minimum_bytes_pinned_memory = pinned_memory_min;
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Copies a vector from the CPU to the GPU unless we already have an up-to-date copy on the GPU */
PetscErrorCode VecCUDACopyToGPU(Vec v)
{
  PetscErrorCode ierr;
  cudaError_t    err;
  Vec_CUDA       *veccuda;
  PetscScalar    *varray;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  ierr = VecCUDAAllocateCheck(v);CHKERRQ(ierr);
  if (v->offloadmask == PETSC_OFFLOAD_CPU) {
    ierr               = PetscLogEventBegin(VEC_CUDACopyToGPU,v,0,0,0);CHKERRQ(ierr);
    veccuda            = (Vec_CUDA*)v->spptr;
    varray             = veccuda->GPUarray;
    err                = cudaMemcpy(varray,((Vec_Seq*)v->data)->array,v->map->n*sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(err);
    ierr               = PetscLogCpuToGpu((v->map->n)*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr               = PetscLogEventEnd(VEC_CUDACopyToGPU,v,0,0,0);CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

/*
     VecCUDACopyFromGPU - Copies a vector from the GPU to the CPU unless we already have an up-to-date copy on the CPU
*/
PetscErrorCode VecCUDACopyFromGPU(Vec v)
{
  PetscErrorCode ierr;
  cudaError_t    err;
  Vec_CUDA       *veccuda;
  PetscScalar    *varray;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  ierr = VecCUDAAllocateCheckHost(v);CHKERRQ(ierr);
  if (v->offloadmask == PETSC_OFFLOAD_GPU) {
    ierr               = PetscLogEventBegin(VEC_CUDACopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    veccuda            = (Vec_CUDA*)v->spptr;
    varray             = veccuda->GPUarray;
    err                = cudaMemcpy(((Vec_Seq*)v->data)->array,varray,v->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUDA(err);
    ierr               = PetscLogGpuToCpu((v->map->n)*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr               = PetscLogEventEnd(VEC_CUDACopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    v->offloadmask     = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

/*MC
   VECSEQCUDA - VECSEQCUDA = "seqcuda" - The basic sequential vector, modified to use CUDA

   Options Database Keys:
. -vec_type seqcuda - sets the vector type to VECSEQCUDA during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateSeqWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateSeq(), VecSetPinnedMemoryMin()
M*/

PetscErrorCode VecAYPX_SeqCUDA(Vec yin,PetscScalar alpha,Vec xin)
{
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscErrorCode    ierr;
  PetscBLASInt      one = 1,bn = 0;
  PetscScalar       sone = 1.0;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cberr;
  cudaError_t       err;

  PetscFunctionBegin;
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArray(yin,&yarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (alpha == (PetscScalar)0.0) {
    err = cudaMemcpy(yarray,xarray,bn*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRCUDA(err);
  } else if (alpha == (PetscScalar)1.0) {
    cberr = cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuFlops(1.0*yin->map->n);CHKERRQ(ierr);
  } else {
    cberr = cublasXscal(cublasv2handle,bn,&alpha,yarray,one);CHKERRCUBLAS(cberr);
    cberr = cublasXaxpy(cublasv2handle,bn,&sone,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuFlops(2.0*yin->map->n);CHKERRQ(ierr);
  }
  err  = WaitForCUDA();CHKERRCUDA(err);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArray(yin,&yarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPY_SeqCUDA(Vec yin,PetscScalar alpha,Vec xin)
{
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscErrorCode    ierr;
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cberr;
  PetscBool         yiscuda,xiscuda;
  cudaError_t       err;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) PetscFunctionReturn(0);
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)yin,&yiscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)xin,&xiscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  if (xiscuda && yiscuda) {
    ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArray(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    err  = WaitForCUDA();CHKERRCUDA(err);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArray(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2.0*yin->map->n);CHKERRQ(ierr);
  } else {
    ierr = VecAXPY_Seq(yin,alpha,xin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseDivide_SeqCUDA(Vec win, Vec xin, Vec yin)
{
  PetscInt                              n = xin->map->n;
  const PetscScalar                     *xarray=NULL,*yarray=NULL;
  PetscScalar                           *warray=NULL;
  thrust::device_ptr<const PetscScalar> xptr,yptr;
  thrust::device_ptr<PetscScalar>       wptr;
  PetscErrorCode                        ierr;
  cudaError_t                           err;

  PetscFunctionBegin;
  ierr = VecCUDAGetArrayWrite(win,&warray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    wptr = thrust::device_pointer_cast(warray);
    xptr = thrust::device_pointer_cast(xarray);
    yptr = thrust::device_pointer_cast(yarray);
    thrust::transform(xptr,xptr+n,yptr,wptr,thrust::divides<PetscScalar>());
    err  = WaitForCUDA();CHKERRCUDA(err);
  } catch (char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(win,&warray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecWAXPY_SeqCUDA(Vec win,PetscScalar alpha,Vec xin, Vec yin)
{
  const PetscScalar *xarray=NULL,*yarray=NULL;
  PetscScalar       *warray=NULL;
  PetscErrorCode    ierr;
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cberr;
  cudaError_t       err;

  PetscFunctionBegin;
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(win->map->n,&bn);CHKERRQ(ierr);
  if (alpha == (PetscScalar)0.0) {
    ierr = VecCopy_SeqCUDA(yin,win);CHKERRQ(ierr);
  } else {
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayWrite(win,&warray);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    err = cudaMemcpy(warray,yarray,win->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRCUDA(err);
    cberr = cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,warray,one);CHKERRCUBLAS(cberr);
    err  = WaitForCUDA();CHKERRCUDA(err);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2*win->map->n);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayWrite(win,&warray);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecMAXPY_SeqCUDA(Vec xin, PetscInt nv,const PetscScalar *alpha,Vec *y)
{
  PetscErrorCode ierr;
  cudaError_t    err;
  PetscInt       n = xin->map->n,j,j_rem;
  PetscScalar    alpha0,alpha1,alpha2,alpha3;

  PetscFunctionBegin;
  ierr = PetscLogGpuFlops(nv*2.0*n);CHKERRQ(ierr);
  switch (j_rem=nv&0x3) {
    case 3:
      alpha0 = alpha[0];
      alpha1 = alpha[1];
      alpha2 = alpha[2];
      alpha += 3;
      ierr   = VecAXPY_SeqCUDA(xin,alpha0,y[0]);CHKERRQ(ierr);
      ierr   = VecAXPY_SeqCUDA(xin,alpha1,y[1]);CHKERRQ(ierr);
      ierr   = VecAXPY_SeqCUDA(xin,alpha2,y[2]);CHKERRQ(ierr);
      y   += 3;
      break;
    case 2:
      alpha0 = alpha[0];
      alpha1 = alpha[1];
      alpha +=2;
      ierr   = VecAXPY_SeqCUDA(xin,alpha0,y[0]);CHKERRQ(ierr);
      ierr   = VecAXPY_SeqCUDA(xin,alpha1,y[1]);CHKERRQ(ierr);
      y +=2;
      break;
    case 1:
      alpha0 = *alpha++;
      ierr   = VecAXPY_SeqCUDA(xin,alpha0,y[0]);CHKERRQ(ierr);
      y     +=1;
      break;
  }
  for (j=j_rem; j<nv; j+=4) {
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha2 = alpha[2];
    alpha3 = alpha[3];
    alpha += 4;
    ierr   = VecAXPY_SeqCUDA(xin,alpha0,y[0]);CHKERRQ(ierr);
    ierr   = VecAXPY_SeqCUDA(xin,alpha1,y[1]);CHKERRQ(ierr);
    ierr   = VecAXPY_SeqCUDA(xin,alpha2,y[2]);CHKERRQ(ierr);
    ierr   = VecAXPY_SeqCUDA(xin,alpha3,y[3]);CHKERRQ(ierr);
    y   += 4;
  }
  err  = WaitForCUDA();CHKERRCUDA(err);
  PetscFunctionReturn(0);
}

PetscErrorCode VecDot_SeqCUDA(Vec xin,Vec yin,PetscScalar *z)
{
  const PetscScalar *xarray,*yarray;
  PetscErrorCode    ierr;
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cerr;

  PetscFunctionBegin;
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  /* arguments y, x are reversed because BLAS complex conjugates the first argument, PETSc the second */
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cerr = cublasXdot(cublasv2handle,bn,yarray,one,xarray,one,z);CHKERRCUBLAS(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  if (xin->map->n >0) {
    ierr = PetscLogGpuFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

//
// CUDA kernels for MDot to follow
//

// set work group size to be a power of 2 (128 is usually a good compromise between portability and speed)
#define MDOT_WORKGROUP_SIZE 128
#define MDOT_WORKGROUP_NUM  128

#if !defined(PETSC_USE_COMPLEX)
#define WARP_SIZE             32
#define FULL_WARP_MASK        0xffffffff
#define MAX_THREADS_PER_BLOCK 1024

static __device__ __forceinline__ float PetscCUDADot(float2 a, float2 b)
{
  return a.x*b.x+a.y*b.y;
}

static __device__ __forceinline__ float PetscCUDADot(float4 a, float4 b)
{
    return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;
}


#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 600))
#else
/* Compute capability 6 or lower doesn't have atomic add for doubles */
static __device__ __forceinline__ double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull,assumed,__double_as_longlong(val+__longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

/* For some reason compiler can't deduce derived types, so do it by force here */
static __device__ __forceinline__ PetscScalar PetscCUDAAtomicAdd(PetscScalar *address, PetscScalar val)
{
#if defined(PETSC_USE_REAL_SINGLE)
  return (PetscScalar)atomicAdd((float *)address, (float)val);
#elif defined(PETSC_USE_REAL_DOUBLE)
  return (PetscScalar)atomicAdd((double *)address, (double)val);
#elif defined(PETSC_USE_REAL___FLOAT128)
  return (PetscScalar)atomicAdd((__fp128 *)address, (__fp128)val);
#elif defined(PETSC_USE_REAL___FP16)
  return (PetscScalar)atomicAdd((half *)address, (half)val);
#endif
}

/*
   Perform a reduction over a warp, where tx%WARP_SIZE == 0 holds the final reduced
   result, you must also pass in a mask to signify which threads in the warp are
   participating in the reduction
*/
template <typename T>
__device__ __forceinline__ void warpReduce(unsigned int warpMask, T& redVal)
{
  /* __shfl_down deprecated after CUDA 9, no clue when __shfl_down was introduced however */
#if (CUDART_VERSION >= 9000)
#pragma unroll
  for (int i = WARP_SIZE/2; i > 0; i >>= 1) redVal += __shfl_down_sync(warpMask, redVal, i);
#else
#pragma unroll
  for (int i = WARP_SIZE/2; i > 0; i >>= 1) redVal += __shfl_down(redVal, i);
#endif
}

template <typename vec_t>
__global__ void VecMDot_SeqCUDA_kernel(const vec_t *__restrict__ x, const vec_t *__restrict__ y0[], PetscInt vecLen, PetscInt ny, vec_t *result)
{
  constexpr size_t copy_size = sizeof(float4)/sizeof(vec_t);
  /* Hold X slice in shared memory */
  __shared__ vec_t shm[MAX_THREADS_PER_BLOCK*copy_size];
  __shared__ vec_t warpSums[2][WARP_SIZE];
  const PetscInt tx = threadIdx.x, bx = blockIdx.x, bdx = blockDim.x;
  const PetscInt laneId = tx%WARP_SIZE, warpId = tx/WARP_SIZE;
  /* Every thread performs copy_size number of additions, float = 4, double = 2, __fp128 = 1 */
  const PetscInt idx = bx*bdx*copy_size+tx*copy_size;
  /* Buffer selection */
  PetscInt flipflop = 0;
  vec_t sum;
  vec_t yslice[copy_size];

  /* Load slice of X into shared memory with vectorized load */
  /* TODO HANDLE NON ALIGNED VECLEN */
  *((float4 *)(shm+copy_size*tx)) = tx < vecLen ? *((float4 *)(x+idx)) : make_float4(0,0,0,0);
#pragma unroll
  for (PetscInt i = 0; i < ny; ++i) {
    /* Load our slice of the y vector into registers */
    *((float4 *)yslice) = *((float4 *)(y0+i*vecLen+idx));

    /* Essentially a constexpr */
    switch (copy_size) {
    case 4: /* float */
      sum = PetscCUDADot(((float4 *)(shm))[4*tx], ((float4 *)(yslice))[0]);
      break;
    case 2: /* double */
      sum = PetscCUDADot(((float2 *)(shm))[2*tx], ((float2 *)(yslice))[0])+dot(((float2 *)(shm))[2*tx+1], ((float2 *)(yslice))[1]);
      break;
    case 1: /* __fp128 */
      sum = shm[tx]*yslice[0];
      break;
    default:
      /* Not expected, crash the kernel */
      __trap();
      break;
    }
    /* Block-wide warp reduction such that only laneId == 0 for each warp has reduced sum */
    warpReduce(FULL_WARP_MASK, sum);
    /* Only warp leader stores intermediate sums, threads must diverge here */
    if (!laneId) warpSums[flipflop][warpId] = sum;
    /* First warp performs final reduction, no thread divergence */
    if (!warpId) {
      warpReduce(FULL_WARP_MASK, warpSums[flipflop][laneId]);
      /* Again, only warp leader writes */
      if (!laneId) PetscCUDAAtomicAdd(&(result[ny]), warpSums[flipflop][laneId]);
    }
    flipflop = 1-flipflop;
    /* Sync every other */
    if (flipflop) __syncthreads();
  }
  return;
}

#undef WARP_SIZE
#undef FULL_WARP_MASK
// M = 2:
__global__ void VecMDot_SeqCUDA_kernel2(const PetscScalar *x,const PetscScalar *y0,const PetscScalar *y1,
                                        PetscInt size, PetscScalar *group_results)
{
  __shared__ PetscScalar tmp_buffer[2*MDOT_WORKGROUP_SIZE];
  PetscInt entries_per_group = (size - 1) / gridDim.x + 1;
  entries_per_group = (entries_per_group == 0) ? 1 : entries_per_group;  // for very small vectors, a group should still do some work
  PetscInt vec_start_index = blockIdx.x * entries_per_group;
  PetscInt vec_stop_index  = PetscMin((blockIdx.x + 1) * entries_per_group, size); // don't go beyond vec size

  PetscScalar entry_x    = 0;
  PetscScalar group_sum0 = 0;
  PetscScalar group_sum1 = 0;
  for (PetscInt i = vec_start_index + threadIdx.x; i < vec_stop_index; i += blockDim.x) {
    entry_x     = x[i];   // load only once from global memory!
    group_sum0 += entry_x * y0[i];
    group_sum1 += entry_x * y1[i];
  }
  tmp_buffer[threadIdx.x]                       = group_sum0;
  tmp_buffer[threadIdx.x + MDOT_WORKGROUP_SIZE] = group_sum1;

  // parallel reduction
  for (PetscInt stride = blockDim.x/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      tmp_buffer[threadIdx.x                      ] += tmp_buffer[threadIdx.x+stride                      ];
      tmp_buffer[threadIdx.x + MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + MDOT_WORKGROUP_SIZE];
    }
  }

  // write result of group to group_results
  if (threadIdx.x == 0) {
    group_results[blockIdx.x]             = tmp_buffer[0];
    group_results[blockIdx.x + gridDim.x] = tmp_buffer[MDOT_WORKGROUP_SIZE];
  }
}

// M = 3:
__global__ void VecMDot_SeqCUDA_kernel3(const PetscScalar *x,const PetscScalar *y0,const PetscScalar *y1,const PetscScalar *y2,
                                        PetscInt size, PetscScalar *group_results)
{
  __shared__ PetscScalar tmp_buffer[3*MDOT_WORKGROUP_SIZE];
  PetscInt entries_per_group = (size - 1) / gridDim.x + 1;
  entries_per_group = (entries_per_group == 0) ? 1 : entries_per_group;  // for very small vectors, a group should still do some work
  PetscInt vec_start_index = blockIdx.x * entries_per_group;
  PetscInt vec_stop_index  = PetscMin((blockIdx.x + 1) * entries_per_group, size); // don't go beyond vec size

  PetscScalar entry_x    = 0;
  PetscScalar group_sum0 = 0;
  PetscScalar group_sum1 = 0;
  PetscScalar group_sum2 = 0;
  for (PetscInt i = vec_start_index + threadIdx.x; i < vec_stop_index; i += blockDim.x) {
    entry_x     = x[i];   // load only once from global memory!
    group_sum0 += entry_x * y0[i];
    group_sum1 += entry_x * y1[i];
    group_sum2 += entry_x * y2[i];
  }
  tmp_buffer[threadIdx.x]                           = group_sum0;
  tmp_buffer[threadIdx.x +     MDOT_WORKGROUP_SIZE] = group_sum1;
  tmp_buffer[threadIdx.x + 2 * MDOT_WORKGROUP_SIZE] = group_sum2;

  // parallel reduction
  for (PetscInt stride = blockDim.x/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      tmp_buffer[threadIdx.x                          ] += tmp_buffer[threadIdx.x+stride                          ];
      tmp_buffer[threadIdx.x +     MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride +     MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 2 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 2 * MDOT_WORKGROUP_SIZE];
    }
  }

  // write result of group to group_results
  if (threadIdx.x == 0) {
    group_results[blockIdx.x                ] = tmp_buffer[0];
    group_results[blockIdx.x +     gridDim.x] = tmp_buffer[    MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 2 * gridDim.x] = tmp_buffer[2 * MDOT_WORKGROUP_SIZE];
  }
}

// M = 4:
__global__ void VecMDot_SeqCUDA_kernel4(const PetscScalar *x,const PetscScalar *y0,const PetscScalar *y1,const PetscScalar *y2,const PetscScalar *y3,
                                        PetscInt size, PetscScalar *group_results)
{
  __shared__ PetscScalar tmp_buffer[4*MDOT_WORKGROUP_SIZE];
  PetscInt entries_per_group = (size - 1) / gridDim.x + 1;
  entries_per_group = (entries_per_group == 0) ? 1 : entries_per_group;  // for very small vectors, a group should still do some work
  PetscInt vec_start_index = blockIdx.x * entries_per_group;
  PetscInt vec_stop_index  = PetscMin((blockIdx.x + 1) * entries_per_group, size); // don't go beyond vec size

  PetscScalar entry_x    = 0;
  PetscScalar group_sum0 = 0;
  PetscScalar group_sum1 = 0;
  PetscScalar group_sum2 = 0;
  PetscScalar group_sum3 = 0;
  for (PetscInt i = vec_start_index + threadIdx.x; i < vec_stop_index; i += blockDim.x) {
    entry_x     = x[i];   // load only once from global memory!
    group_sum0 += entry_x * y0[i];
    group_sum1 += entry_x * y1[i];
    group_sum2 += entry_x * y2[i];
    group_sum3 += entry_x * y3[i];
  }
  tmp_buffer[threadIdx.x]                           = group_sum0;
  tmp_buffer[threadIdx.x +     MDOT_WORKGROUP_SIZE] = group_sum1;
  tmp_buffer[threadIdx.x + 2 * MDOT_WORKGROUP_SIZE] = group_sum2;
  tmp_buffer[threadIdx.x + 3 * MDOT_WORKGROUP_SIZE] = group_sum3;

  // parallel reduction
  for (PetscInt stride = blockDim.x/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      tmp_buffer[threadIdx.x                          ] += tmp_buffer[threadIdx.x+stride                          ];
      tmp_buffer[threadIdx.x +     MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride +     MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 2 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 2 * MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 3 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 3 * MDOT_WORKGROUP_SIZE];
    }
  }

  // write result of group to group_results
  if (threadIdx.x == 0) {
    group_results[blockIdx.x                ] = tmp_buffer[0];
    group_results[blockIdx.x +     gridDim.x] = tmp_buffer[    MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 2 * gridDim.x] = tmp_buffer[2 * MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 3 * gridDim.x] = tmp_buffer[3 * MDOT_WORKGROUP_SIZE];
  }
}

// M = 8:
__global__ void VecMDot_SeqCUDA_kernel8(const PetscScalar *x,const PetscScalar *y0,const PetscScalar *y1,const PetscScalar *y2,const PetscScalar *y3,
                                          const PetscScalar *y4,const PetscScalar *y5,const PetscScalar *y6,const PetscScalar *y7,
                                          PetscInt size, PetscScalar *group_results)
{
  __shared__ PetscScalar tmp_buffer[8*MDOT_WORKGROUP_SIZE];
  PetscInt entries_per_group = (size - 1) / gridDim.x + 1;
  entries_per_group = (entries_per_group == 0) ? 1 : entries_per_group;  // for very small vectors, a group should still do some work
  PetscInt vec_start_index = blockIdx.x * entries_per_group;
  PetscInt vec_stop_index  = PetscMin((blockIdx.x + 1) * entries_per_group, size); // don't go beyond vec size

  PetscScalar entry_x    = 0;
  PetscScalar group_sum0 = 0;
  PetscScalar group_sum1 = 0;
  PetscScalar group_sum2 = 0;
  PetscScalar group_sum3 = 0;
  PetscScalar group_sum4 = 0;
  PetscScalar group_sum5 = 0;
  PetscScalar group_sum6 = 0;
  PetscScalar group_sum7 = 0;
  for (PetscInt i = vec_start_index + threadIdx.x; i < vec_stop_index; i += blockDim.x) {
    entry_x     = x[i];   // load only once from global memory!
    group_sum0 += entry_x * y0[i];
    group_sum1 += entry_x * y1[i];
    group_sum2 += entry_x * y2[i];
    group_sum3 += entry_x * y3[i];
    group_sum4 += entry_x * y4[i];
    group_sum5 += entry_x * y5[i];
    group_sum6 += entry_x * y6[i];
    group_sum7 += entry_x * y7[i];
  }
  tmp_buffer[threadIdx.x]                           = group_sum0;
  tmp_buffer[threadIdx.x +     MDOT_WORKGROUP_SIZE] = group_sum1;
  tmp_buffer[threadIdx.x + 2 * MDOT_WORKGROUP_SIZE] = group_sum2;
  tmp_buffer[threadIdx.x + 3 * MDOT_WORKGROUP_SIZE] = group_sum3;
  tmp_buffer[threadIdx.x + 4 * MDOT_WORKGROUP_SIZE] = group_sum4;
  tmp_buffer[threadIdx.x + 5 * MDOT_WORKGROUP_SIZE] = group_sum5;
  tmp_buffer[threadIdx.x + 6 * MDOT_WORKGROUP_SIZE] = group_sum6;
  tmp_buffer[threadIdx.x + 7 * MDOT_WORKGROUP_SIZE] = group_sum7;

  // parallel reduction
  for (PetscInt stride = blockDim.x/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      tmp_buffer[threadIdx.x                          ] += tmp_buffer[threadIdx.x+stride                          ];
      tmp_buffer[threadIdx.x +     MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride +     MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 2 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 2 * MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 3 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 3 * MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 4 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 4 * MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 5 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 5 * MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 6 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 6 * MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 7 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 7 * MDOT_WORKGROUP_SIZE];
    }
  }

  // write result of group to group_results
  if (threadIdx.x == 0) {
    group_results[blockIdx.x                ] = tmp_buffer[0];
    group_results[blockIdx.x +     gridDim.x] = tmp_buffer[    MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 2 * gridDim.x] = tmp_buffer[2 * MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 3 * gridDim.x] = tmp_buffer[3 * MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 4 * gridDim.x] = tmp_buffer[4 * MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 5 * gridDim.x] = tmp_buffer[5 * MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 6 * gridDim.x] = tmp_buffer[6 * MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 7 * gridDim.x] = tmp_buffer[7 * MDOT_WORKGROUP_SIZE];
  }
}
#endif /* !defined(PETSC_USE_COMPLEX) */

PetscErrorCode VecMDot_SeqCUDA_NEW(Vec xin, PetscInt nv, const Vec yin[], PetscScalar *z)
{
  const PetscScalar **d_y;
  const PetscScalar *d_x;
  const PetscInt    xBlocks = (PetscInt)PetscCeilReal((PetscReal)(xin->map->n)/(PetscReal)(MAX_THREADS_PER_BLOCK));
  const dim3        dimGrid(xBlocks), dimBlock(MAX_THREADS_PER_BLOCK);
  /* How many elems we can copy to shm per vectorized load */
  PetscScalar       *d_z;
  PetscInt          i;
  cudaError_t       cerr;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (nv <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Number of vectors provided to VecMDot_SeqCUDA not positive.");
  if (!xin->map->n) {
    for (i = 0; i < nv; ++i) z[i] = 0;
    PetscFunctionReturn(0);
  }
  cerr = cudaMalloc(&d_y, nv*sizeof(PetscScalar *));CHKERRCUDA(cerr);
  cerr = cudaMalloc(&d_z, nv*sizeof(PetscScalar));CHKERRCUDA(cerr);
  for (i = 0; i < nv; ++i) {ierr = VecCUDAGetArrayRead(yin[i], &(d_y[i]));CHKERRQ(ierr);}
  ierr = VecCUDAGetArrayRead(xin, &d_x);CHKERRQ(ierr);
  VecMDot_SeqCUDA_kernel<<<dimGrid, dimBlock>>>(d_x, d_y, xin->map->n, nv, d_z);
  ierr = VecCUDARestoreArrayRead(xin, &d_x);CHKERRQ(ierr);
  for (i = 0; i < nv; ++i) {ierr = VecCUDARestoreArrayRead(yin[i], &(d_y[i]));CHKERRQ(ierr);}
  cerr = cudaMemcpy(z, d_z, nv*sizeof(PetscScalar), cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
  cerr = cudaFree(d_z);CHKERRCUDA(cerr);
  cerr = cudaFree(d_y);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}
#undef MAX_THREADS_PER_BLOCK

PetscErrorCode VecMDot_SeqCUDA(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode    ierr;
  PetscInt          i,n = xin->map->n,current_y_index = 0;
  const PetscScalar *xptr,*y0ptr,*y1ptr,*y2ptr,*y3ptr,*y4ptr,*y5ptr,*y6ptr,*y7ptr;
  PetscScalar       *group_results_gpu;
#if !defined(PETSC_USE_COMPLEX)
  PetscInt          j;
  PetscScalar       group_results_cpu[MDOT_WORKGROUP_NUM * 8]; // we process at most eight vectors in one kernel
#endif
  cudaError_t       cuda_ierr;
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cberr;

  PetscFunctionBegin;
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
  if (nv <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Number of vectors provided to VecMDot_SeqCUDA not positive.");
  /* Handle the case of local size zero first */
  if (!xin->map->n) {
    for (i=0; i<nv; ++i) z[i] = 0;
    PetscFunctionReturn(0);
  }

  // allocate scratchpad memory for the results of individual work groups:
  cuda_ierr = cudaMalloc((void**)&group_results_gpu, sizeof(PetscScalar) * MDOT_WORKGROUP_NUM * 8);CHKERRCUDA(cuda_ierr);

  ierr = VecCUDAGetArrayRead(xin,&xptr);CHKERRQ(ierr);

  while (current_y_index < nv)
  {
    switch (nv - current_y_index) {

      case 7:
      case 6:
      case 5:
      case 4:
        ierr = VecCUDAGetArrayRead(yin[current_y_index  ],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+2],&y2ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+3],&y3ptr);CHKERRQ(ierr);
        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        cberr = cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,&z[current_y_index+1]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y2ptr,one,xptr,one,&z[current_y_index+2]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y3ptr,one,xptr,one,&z[current_y_index+3]);CHKERRCUBLAS(cberr);
        ierr  = PetscLogGpuTimeEnd();CHKERRQ(ierr);
#else
        // run kernel:
        VecMDot_SeqCUDA_kernel4<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,y2ptr,y3ptr,n,group_results_gpu);
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

        // copy results back to
        cuda_ierr = cudaMemcpy(group_results_cpu,group_results_gpu,sizeof(PetscScalar) * MDOT_WORKGROUP_NUM * 4,cudaMemcpyDeviceToHost);CHKERRCUDA(cuda_ierr);

        // sum group results into z:
        for (j=0; j<4; ++j) {
          z[current_y_index + j] = 0;
          for (i=j*MDOT_WORKGROUP_NUM; i<(j+1)*MDOT_WORKGROUP_NUM; ++i) z[current_y_index + j] += group_results_cpu[i];
        }
#endif
        ierr = VecCUDARestoreArrayRead(yin[current_y_index  ],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+2],&y2ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+3],&y3ptr);CHKERRQ(ierr);
        current_y_index += 4;
        break;

      case 3:
        ierr = VecCUDAGetArrayRead(yin[current_y_index  ],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+2],&y2ptr);CHKERRQ(ierr);

        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        cberr = cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,&z[current_y_index+1]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y2ptr,one,xptr,one,&z[current_y_index+2]);CHKERRCUBLAS(cberr);
        ierr  = PetscLogGpuTimeEnd();CHKERRQ(ierr);
#else
        // run kernel:
        VecMDot_SeqCUDA_kernel3<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,y2ptr,n,group_results_gpu);
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

        // copy results back to
        cuda_ierr = cudaMemcpy(group_results_cpu,group_results_gpu,sizeof(PetscScalar) * MDOT_WORKGROUP_NUM * 3,cudaMemcpyDeviceToHost);CHKERRCUDA(cuda_ierr);

        // sum group results into z:
        for (j=0; j<3; ++j) {
          z[current_y_index + j] = 0;
          for (i=j*MDOT_WORKGROUP_NUM; i<(j+1)*MDOT_WORKGROUP_NUM; ++i) z[current_y_index + j] += group_results_cpu[i];
        }
#endif
        ierr = VecCUDARestoreArrayRead(yin[current_y_index  ],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+2],&y2ptr);CHKERRQ(ierr);
        current_y_index += 3;
        break;

      case 2:
        ierr = VecCUDAGetArrayRead(yin[current_y_index],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        cberr = cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,&z[current_y_index+1]);CHKERRCUBLAS(cberr);
        ierr  = PetscLogGpuTimeEnd();CHKERRQ(ierr);
#else
        // run kernel:
        VecMDot_SeqCUDA_kernel2<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,n,group_results_gpu);
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

        // copy results back to
        cuda_ierr = cudaMemcpy(group_results_cpu,group_results_gpu,sizeof(PetscScalar) * MDOT_WORKGROUP_NUM * 2,cudaMemcpyDeviceToHost);CHKERRCUDA(cuda_ierr);

        // sum group results into z:
        for (j=0; j<2; ++j) {
          z[current_y_index + j] = 0;
          for (i=j*MDOT_WORKGROUP_NUM; i<(j+1)*MDOT_WORKGROUP_NUM; ++i) z[current_y_index + j] += group_results_cpu[i];
        }
#endif
        ierr = VecCUDARestoreArrayRead(yin[current_y_index],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        current_y_index += 2;
        break;

      case 1:
        ierr = VecCUDAGetArrayRead(yin[current_y_index],&y0ptr);CHKERRQ(ierr);
        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
        cberr = cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]);CHKERRCUBLAS(cberr);
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index],&y0ptr);CHKERRQ(ierr);
        current_y_index += 1;
        break;

      default: // 8 or more vectors left
        ierr = VecCUDAGetArrayRead(yin[current_y_index  ],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+2],&y2ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+3],&y3ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+4],&y4ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+5],&y5ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+6],&y6ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+7],&y7ptr);CHKERRQ(ierr);
        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        cberr = cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,&z[current_y_index+1]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y2ptr,one,xptr,one,&z[current_y_index+2]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y3ptr,one,xptr,one,&z[current_y_index+3]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y4ptr,one,xptr,one,&z[current_y_index+4]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y5ptr,one,xptr,one,&z[current_y_index+5]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y6ptr,one,xptr,one,&z[current_y_index+6]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y7ptr,one,xptr,one,&z[current_y_index+7]);CHKERRCUBLAS(cberr);
        ierr  = PetscLogGpuTimeEnd();CHKERRQ(ierr);
#else
        // run kernel:
        VecMDot_SeqCUDA_kernel8<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,y2ptr,y3ptr,y4ptr,y5ptr,y6ptr,y7ptr,n,group_results_gpu);
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

        // copy results back to
        cuda_ierr = cudaMemcpy(group_results_cpu,group_results_gpu,sizeof(PetscScalar) * MDOT_WORKGROUP_NUM * 8,cudaMemcpyDeviceToHost);CHKERRCUDA(cuda_ierr);

        // sum group results into z:
        for (j=0; j<8; ++j) {
          z[current_y_index + j] = 0;
          for (i=j*MDOT_WORKGROUP_NUM; i<(j+1)*MDOT_WORKGROUP_NUM; ++i) z[current_y_index + j] += group_results_cpu[i];
        }
#endif
        ierr = VecCUDARestoreArrayRead(yin[current_y_index  ],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+2],&y2ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+3],&y3ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+4],&y4ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+5],&y5ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+6],&y6ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+7],&y7ptr);CHKERRQ(ierr);
        current_y_index += 8;
        break;
    }
  }
  ierr = VecCUDARestoreArrayRead(xin,&xptr);CHKERRQ(ierr);

  cuda_ierr = cudaFree(group_results_gpu);CHKERRCUDA(cuda_ierr);
  ierr = PetscLogGpuFlops(PetscMax(nv*(2.0*n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef MDOT_WORKGROUP_SIZE
#undef MDOT_WORKGROUP_NUM

PetscErrorCode VecSet_SeqCUDA(Vec xin,PetscScalar alpha)
{
  PetscInt                        n = xin->map->n;
  PetscScalar                     *xarray = NULL;
  thrust::device_ptr<PetscScalar> xptr;
  PetscErrorCode                  ierr;
  cudaError_t                     err;

  PetscFunctionBegin;
  ierr = VecCUDAGetArrayWrite(xin,&xarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (alpha == (PetscScalar)0.0) {
    err = cudaMemset(xarray,0,n*sizeof(PetscScalar));CHKERRCUDA(err);
  } else {
    try {
      xptr = thrust::device_pointer_cast(xarray);
      thrust::fill(xptr,xptr+n,alpha);
    } catch (char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
    }
  }
  err  = WaitForCUDA();CHKERRCUDA(err);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(xin,&xarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecScale_SeqCUDA(Vec xin,PetscScalar alpha)
{
  PetscScalar    *xarray;
  PetscErrorCode ierr;
  PetscBLASInt   one = 1,bn = 0;
  cublasHandle_t cublasv2handle;
  cublasStatus_t cberr;
  cudaError_t    err;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    ierr = VecSet_SeqCUDA(xin,alpha);CHKERRQ(ierr);
    err  = WaitForCUDA();CHKERRCUDA(err);
  } else if (alpha != (PetscScalar)1.0) {
    ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
    ierr = VecCUDAGetArray(xin,&xarray);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasXscal(cublasv2handle,bn,&alpha,xarray,one);CHKERRCUBLAS(cberr);
    ierr = VecCUDARestoreArray(xin,&xarray);CHKERRQ(ierr);
    err  = WaitForCUDA();CHKERRCUDA(err);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  }
  ierr = PetscLogGpuFlops(xin->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_SeqCUDA(Vec xin,Vec yin,PetscScalar *z)
{
  const PetscScalar *xarray,*yarray;
  PetscErrorCode    ierr;
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cerr;

  PetscFunctionBegin;
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cerr = cublasXdotu(cublasv2handle,bn,xarray,one,yarray,one,z);CHKERRCUBLAS(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  if (xin->map->n > 0) {
    ierr = PetscLogGpuFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecCopy_SeqCUDA(Vec xin,Vec yin)
{
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscErrorCode    ierr;
  cudaError_t       err;

  PetscFunctionBegin;
  if (xin != yin) {
    if (xin->offloadmask == PETSC_OFFLOAD_GPU) {
      PetscBool yiscuda;

      ierr = PetscObjectTypeCompareAny((PetscObject)yin,&yiscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
      ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
      if (yiscuda) {
        ierr = VecCUDAGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
      } else {
        ierr = VecGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
      }
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      if (yiscuda) {
        err = cudaMemcpy(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRCUDA(err);
      } else {
        err = cudaMemcpy(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUDA(err);
      }
      err  = WaitForCUDA();CHKERRCUDA(err);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
      if (yiscuda) {
        ierr = VecCUDARestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
      } else {
        ierr = VecRestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
      }
    } else if (xin->offloadmask == PETSC_OFFLOAD_CPU) {
      /* copy in CPU if we are on the CPU */
      ierr = VecCopy_SeqCUDA_Private(xin,yin);CHKERRQ(ierr);
    } else if (xin->offloadmask == PETSC_OFFLOAD_BOTH) {
      /* if xin is valid in both places, see where yin is and copy there (because it's probably where we'll want to next use it) */
      if (yin->offloadmask == PETSC_OFFLOAD_CPU) {
        /* copy in CPU */
        ierr = VecCopy_SeqCUDA_Private(xin,yin);CHKERRQ(ierr);
      } else if (yin->offloadmask == PETSC_OFFLOAD_GPU) {
        /* copy in GPU */
        ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
        err  = cudaMemcpy(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRCUDA(err);
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
      } else if (yin->offloadmask == PETSC_OFFLOAD_BOTH) {
        /* xin and yin are both valid in both places (or yin was unallocated before the earlier call to allocatecheck
           default to copy in GPU (this is an arbitrary choice) */
        ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
        err  = cudaMemcpy(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRCUDA(err);
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
      } else {
        ierr = VecCopy_SeqCUDA_Private(xin,yin);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecSwap_SeqCUDA(Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscBLASInt   one = 1,bn = 0;
  PetscScalar    *xarray,*yarray;
  cublasHandle_t cublasv2handle;
  cublasStatus_t cberr;
  cudaError_t    err;

  PetscFunctionBegin;
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
  if (xin != yin) {
    ierr = VecCUDAGetArray(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArray(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasXswap(cublasv2handle,bn,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    err  = WaitForCUDA();CHKERRCUDA(err);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecCUDARestoreArray(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArray(yin,&yarray);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBY_SeqCUDA(Vec yin,PetscScalar alpha,PetscScalar beta,Vec xin)
{
  PetscErrorCode    ierr;
  PetscScalar       a = alpha,b = beta;
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscBLASInt      one = 1, bn = 0;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cberr;
  cudaError_t       err;

  PetscFunctionBegin;
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
  if (a == (PetscScalar)0.0) {
    ierr = VecScale_SeqCUDA(yin,beta);CHKERRQ(ierr);
  } else if (b == (PetscScalar)1.0) {
    ierr = VecAXPY_SeqCUDA(yin,alpha,xin);CHKERRQ(ierr);
  } else if (a == (PetscScalar)1.0) {
    ierr = VecAYPX_SeqCUDA(yin,beta,xin);CHKERRQ(ierr);
  } else if (b == (PetscScalar)0.0) {
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArray(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    err = cudaMemcpy(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRCUDA(err);
    cberr = cublasXscal(cublasv2handle,bn,&alpha,yarray,one);CHKERRCUBLAS(cberr);
    err  = WaitForCUDA();CHKERRCUDA(err);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(xin->map->n);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArray(yin,&yarray);CHKERRQ(ierr);
  } else {
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArray(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasXscal(cublasv2handle,bn,&beta,yarray,one);CHKERRCUBLAS(cberr);
    cberr = cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArray(yin,&yarray);CHKERRQ(ierr);
    err  = WaitForCUDA();CHKERRCUDA(err);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(3.0*xin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBYPCZ_SeqCUDA(Vec zin,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  cudaError_t    err;
  PetscInt       n = zin->map->n;

  PetscFunctionBegin;
  if (gamma == (PetscScalar)1.0) {
    /* z = ax + b*y + z */
    ierr = VecAXPY_SeqCUDA(zin,alpha,xin);CHKERRQ(ierr);
    ierr = VecAXPY_SeqCUDA(zin,beta,yin);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(4.0*n);CHKERRQ(ierr);
  } else {
    /* z = a*x + b*y + c*z */
    ierr = VecScale_SeqCUDA(zin,gamma);CHKERRQ(ierr);
    ierr = VecAXPY_SeqCUDA(zin,alpha,xin);CHKERRQ(ierr);
    ierr = VecAXPY_SeqCUDA(zin,beta,yin);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(5.0*n);CHKERRQ(ierr);
  }
  err  = WaitForCUDA();CHKERRCUDA(err);
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseMult_SeqCUDA(Vec win,Vec xin,Vec yin)
{
  PetscInt                              n = win->map->n;
  const PetscScalar                     *xarray,*yarray;
  PetscScalar                           *warray;
  thrust::device_ptr<const PetscScalar> xptr,yptr;
  thrust::device_ptr<PetscScalar>       wptr;
  PetscErrorCode                        ierr;
  cudaError_t                           err;

  PetscFunctionBegin;
  ierr = VecCUDAGetArray(win,&warray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    wptr = thrust::device_pointer_cast(warray);
    xptr = thrust::device_pointer_cast(xarray);
    yptr = thrust::device_pointer_cast(yarray);
    thrust::transform(xptr,xptr+n,yptr,wptr,thrust::multiplies<PetscScalar>());
    err  = WaitForCUDA();CHKERRCUDA(err);
  } catch (char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArray(win,&warray);CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* should do infinity norm in cuda */

PetscErrorCode VecNorm_SeqCUDA(Vec xin,NormType type,PetscReal *z)
{
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n;
  PetscBLASInt      one = 1, bn = 0;
  const PetscScalar *xarray;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cberr;
  cudaError_t       err;

  PetscFunctionBegin;
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasXnrm2(cublasv2handle,bn,xarray,one,z);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(PetscMax(2.0*n-1,0.0));CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    int  i;
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasIXamax(cublasv2handle,bn,xarray,one,&i);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    if (bn) {
      PetscScalar zs;
      err = cudaMemcpy(&zs,xarray+i-1,sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUDA(err);
      *z = PetscAbsScalar(zs);
    } else *z = 0.0;
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  } else if (type == NORM_1) {
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasXasum(cublasv2handle,bn,xarray,one,z);CHKERRCUBLAS(cberr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(PetscMax(n-1.0,0.0));CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    ierr = VecNorm_SeqCUDA(xin,NORM_1,z);CHKERRQ(ierr);
    ierr = VecNorm_SeqCUDA(xin,NORM_2,z+1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotNorm2_SeqCUDA(Vec s, Vec t, PetscScalar *dp, PetscScalar *nm)
{
  PetscErrorCode ierr;
  cudaError_t    err;
  PetscReal      n = s->map->n;

  PetscFunctionBegin;
  ierr = VecDot_SeqCUDA(s,t,dp);CHKERRQ(ierr);
  ierr = VecDot_SeqCUDA(t,t,nm);CHKERRQ(ierr);
  err  = WaitForCUDA();CHKERRCUDA(err);
  ierr = PetscLogGpuFlops(4.0*n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroy_SeqCUDA(Vec v)
{
  PetscErrorCode ierr;
  cudaError_t    err;

  PetscFunctionBegin;
  if (v->spptr) {
    if (((Vec_CUDA*)v->spptr)->GPUarray_allocated) {
      err = cudaFree(((Vec_CUDA*)v->spptr)->GPUarray_allocated);CHKERRCUDA(err);
      ((Vec_CUDA*)v->spptr)->GPUarray_allocated = NULL;
    }
    if (((Vec_CUDA*)v->spptr)->stream) {
      err = cudaStreamDestroy(((Vec_CUDA*)v->spptr)->stream);CHKERRCUDA(err);
    }
  }
  ierr = VecDestroy_SeqCUDA_Private(v);CHKERRQ(ierr);
  ierr = PetscFree(v->spptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_COMPLEX)
struct conjugate
{
  __host__ __device__
    PetscScalar operator()(PetscScalar x)
    {
      return PetscConj(x);
    }
};
#endif

PetscErrorCode VecConjugate_SeqCUDA(Vec xin)
{
#if defined(PETSC_USE_COMPLEX)
  PetscScalar                     *xarray;
  PetscErrorCode                  ierr;
  PetscInt                        n = xin->map->n;
  thrust::device_ptr<PetscScalar> xptr;
  cudaError_t                     err;

  PetscFunctionBegin;
  ierr = VecCUDAGetArray(xin,&xarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    xptr = thrust::device_pointer_cast(xarray);
    thrust::transform(xptr,xptr+n,xptr,conjugate());
    err  = WaitForCUDA();CHKERRCUDA(err);
  } catch (char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = VecCUDARestoreArray(xin,&xarray);CHKERRQ(ierr);
#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetLocalVector_SeqCUDA(Vec v,Vec w)
{
  PetscErrorCode ierr;
  cudaError_t    err;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(w,VEC_CLASSID,2);
  PetscCheckTypeName(w,VECSEQCUDA);
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  if (w->data) {
    if (((Vec_Seq*)w->data)->array_allocated) {
      if (w->pinned_memory) {
        ierr = PetscMallocSetCUDAHost();CHKERRQ(ierr);
      }
      ierr = PetscFree(((Vec_Seq*)w->data)->array_allocated);CHKERRQ(ierr);
      if (w->pinned_memory) {
        ierr = PetscMallocResetCUDAHost();CHKERRQ(ierr);
        w->pinned_memory = PETSC_FALSE;
      }
    }
    ((Vec_Seq*)w->data)->array = NULL;
    ((Vec_Seq*)w->data)->unplacedarray = NULL;
  }
  if (w->spptr) {
    PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
    if (((Vec_CUDA*)w->spptr)->GPUarray) {
      err = cudaFree(((Vec_CUDA*)w->spptr)->GPUarray);CHKERRCUDA(err);
      ((Vec_CUDA*)w->spptr)->GPUarray = NULL;
    }
    if (((Vec_CUDA*)w->spptr)->stream) {
      err = cudaStreamDestroy(((Vec_CUDA*)w->spptr)->stream);CHKERRCUDA(err);
    }
    ierr = PetscFree(w->spptr);CHKERRQ(ierr);
  }

  if (v->petscnative) {
    ierr = PetscFree(w->data);CHKERRQ(ierr);
    w->data = v->data;
    w->offloadmask = v->offloadmask;
    w->pinned_memory = v->pinned_memory;
    w->spptr = v->spptr;
    ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
  } else {
    ierr = VecGetArray(v,&((Vec_Seq*)w->data)->array);CHKERRQ(ierr);
    w->offloadmask = PETSC_OFFLOAD_CPU;
    ierr = VecCUDAAllocateCheck(w);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreLocalVector_SeqCUDA(Vec v,Vec w)
{
  PetscErrorCode ierr;
  cudaError_t    err;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(w,VEC_CLASSID,2);
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  PetscCheckTypeName(w,VECSEQCUDA);

  if (v->petscnative) {
    v->data = w->data;
    v->offloadmask = w->offloadmask;
    v->pinned_memory = w->pinned_memory;
    v->spptr = w->spptr;
    w->data = 0;
    w->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
    w->spptr = 0;
  } else {
    ierr = VecRestoreArray(v,&((Vec_Seq*)w->data)->array);CHKERRQ(ierr);
    if ((Vec_CUDA*)w->spptr) {
      err = cudaFree(((Vec_CUDA*)w->spptr)->GPUarray);CHKERRCUDA(err);
      ((Vec_CUDA*)w->spptr)->GPUarray = NULL;
      if (((Vec_CUDA*)v->spptr)->stream) {
        err = cudaStreamDestroy(((Vec_CUDA*)w->spptr)->stream);CHKERRCUDA(err);
      }
      ierr = PetscFree(w->spptr);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
