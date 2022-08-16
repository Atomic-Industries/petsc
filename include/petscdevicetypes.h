#ifndef PETSCDEVICETYPES_H
#define PETSCDEVICETYPES_H

#include <petscsys.h> /*I <petscdevicetypes.h> I*/

/* SUBMANSEC = Sys */

/*E
  PetscMemType - Memory type of a pointer

  Developer Note:
  Encoding of the bitmask in binary: xxxxyyyz

$ z = 0                - Host memory
$ z = 1                - Device memory
$ yyy = 000            - CUDA-related memory
$ yyy = 001            - HIP-related memory
$ yyy = 010            - SYCL-related memory
$ xxxxyyy1 = 0000,0001 - CUDA memory
$ xxxxyyy1 = 0001,0001 - CUDA NVSHMEM memory
$ xxxxyyy1 = 0000,0011 - HIP memory
$ xxxxyyy1 = 0000,0101 - SYCL memory

  Other types of memory, e.g., CUDA managed memory, can be added when needed.

  Level: beginner

  Notes:
  `PETSC_MEMTYPE_KOKKOS` depends on the Kokkos backend configuration

  Developer Notes:
  This enum uses a function (`PetscMemTypes()`) to convert to string representation so cannot
  be used in `PetscOptionsEnum()`.

.seealso: `VecGetArrayAndMemType()`, `PetscSFBcastWithMemTypeBegin()`, `PetscSFReduceWithMemTypeBegin()`
E*/
typedef enum {
  PETSC_MEMTYPE_HOST    = 0,
  PETSC_MEMTYPE_DEVICE  = 0x01,
  PETSC_MEMTYPE_CUDA    = 0x01,
  PETSC_MEMTYPE_NVSHMEM = 0x11,
  PETSC_MEMTYPE_HIP     = 0x03,
  PETSC_MEMTYPE_SYCL    = 0x05,
} PetscMemType;
#if PetscDefined(HAVE_CUDA)
#define PETSC_MEMTYPE_KOKKOS PETSC_MEMTYPE_CUDA
#elif PetscDefined(HAVE_HIP)
#define PETSC_MEMTYPE_KOKKOS PETSC_MEMTYPE_HIP
#elif PetscDefined(HAVE_SYCL)
#define PETSC_MEMTYPE_KOKKOS PETSC_MEMTYPE_SYCL
#else
#define PETSC_MEMTYPE_KOKKOS PETSC_MEMTYPE_HOST
#endif

#define PetscMemTypeHost(m)    (((m)&0x1) == PETSC_MEMTYPE_HOST)
#define PetscMemTypeDevice(m)  (((m)&0x1) == PETSC_MEMTYPE_DEVICE)
#define PetscMemTypeCUDA(m)    (((m)&0xF) == PETSC_MEMTYPE_CUDA)
#define PetscMemTypeHIP(m)     (((m)&0xF) == PETSC_MEMTYPE_HIP)
#define PetscMemTypeSYCL(m)    (((m)&0xF) == PETSC_MEMTYPE_SYCL)
#define PetscMemTypeNVSHMEM(m) ((m) == PETSC_MEMTYPE_NVSHMEM)

PETSC_NODISCARD static inline PETSC_CONSTEXPR_14 const char *PetscMemTypes(PetscMemType mtype) {
#ifdef __cplusplus
  static_assert(PETSC_MEMTYPE_CUDA == PETSC_MEMTYPE_DEVICE, "");
#endif
  switch (mtype) {
#define PETSC_CASE_NAME(v) \
  case v: return PetscStringize(v)
    PETSC_CASE_NAME(PETSC_MEMTYPE_HOST);
    /* PETSC_CASE_NAME(PETSC_MEMTYPE_DEVICE); same as PETSC_MEMTYPE_CUDA */
    PETSC_CASE_NAME(PETSC_MEMTYPE_CUDA);
    PETSC_CASE_NAME(PETSC_MEMTYPE_NVSHMEM);
    PETSC_CASE_NAME(PETSC_MEMTYPE_HIP);
    PETSC_CASE_NAME(PETSC_MEMTYPE_SYCL);
#undef PETSC_CASE_NAME
  }
  PetscUnreachable();
  return "invalid";
}

#define PETSC_OFFLOAD_VECKOKKOS_DEPRECATED PETSC_OFFLOAD_VECKOKKOS PETSC_DEPRECATED_ENUM("Use PETSC_OFFLOAD_KOKKOS (since version 3.17.0)")

/*E
  PetscOffloadMask - indicates which memory (CPU, GPU, or none) contains valid data

$ PETSC_OFFLOAD_UNALLOCATED - no memory contains valid matrix entries; NEVER used for vectors
$ PETSC_OFFLOAD_GPU         - GPU has valid vector/matrix entries
$ PETSC_OFFLOAD_CPU         - CPU has valid vector/matrix entries
$ PETSC_OFFLOAD_BOTH        - Both GPU and CPU have valid vector/matrix entries and they match
$ PETSC_OFFLOAD_KOKKOS      - Reserved for Kokkos matrix and vector. It means the offload is managed by Kokkos, thus this flag itself cannot tell you where the valid data is.

  Developer Notes:
  This enum uses a function (`PetscOffloadMasks()`) to convert to string representation so
  cannot be used in `PetscOptionsEnum()`.

  Level: developer
E*/
typedef enum {
  PETSC_OFFLOAD_UNALLOCATED          = 0x0,
  PETSC_OFFLOAD_CPU                  = 0x1,
  PETSC_OFFLOAD_GPU                  = 0x2,
  PETSC_OFFLOAD_BOTH                 = 0x3,
  PETSC_OFFLOAD_VECKOKKOS_DEPRECATED = 0x100,
  PETSC_OFFLOAD_KOKKOS               = 0x100
} PetscOffloadMask;

#define PetscOffloadUnallocated(m) ((m) == PETSC_OFFLOAD_UNALLOCATED)
#define PetscOffloadHost(m)        (((m)&PETSC_OFFLOAD_CPU) == PETSC_OFFLOAD_CPU)
#define PetscOffloadDevice(m)      (((m)&PETSC_OFFLOAD_GPU) == PETSC_OFFLOAD_GPU)

#if defined(__cplusplus)
// Some overzealous older gcc versions warn that the comparisons below are always true. Neat
// that it can detect this, but the tautology *is* the point of the static_assert()!
#if defined(__GNUC__) && __GNUC__ >= 6 && !PetscDefined(HAVE_WINDOWS_COMPILERS)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-compare"
#endif
static_assert(!PetscOffloadHost(PETSC_OFFLOAD_UNALLOCATED), "");
static_assert(PetscOffloadHost(PETSC_OFFLOAD_BOTH), "");
static_assert(!PetscOffloadHost(PETSC_OFFLOAD_GPU), "");
static_assert(PetscOffloadHost(PETSC_OFFLOAD_BOTH), "");
static_assert(!PetscOffloadHost(PETSC_OFFLOAD_KOKKOS), "");

static_assert(!PetscOffloadDevice(PETSC_OFFLOAD_UNALLOCATED), "");
static_assert(!PetscOffloadDevice(PETSC_OFFLOAD_CPU), "");
static_assert(PetscOffloadDevice(PETSC_OFFLOAD_GPU), "");
static_assert(PetscOffloadDevice(PETSC_OFFLOAD_BOTH), "");
static_assert(!PetscOffloadDevice(PETSC_OFFLOAD_KOKKOS), "");
#if defined(__GNUC__) && __GNUC__ >= 6 && !PetscDefined(HAVE_WINDOWS_COMPILERS)
#pragma GCC diagnostic pop
#endif
#endif

PETSC_NODISCARD static inline PETSC_CONSTEXPR_14 const char *PetscOffloadMasks(PetscOffloadMask mask) {
  switch (mask) {
#define PETSC_CASE_RETURN(v) \
  case v: return PetscStringize(v)
    PETSC_CASE_RETURN(PETSC_OFFLOAD_UNALLOCATED);
    PETSC_CASE_RETURN(PETSC_OFFLOAD_CPU);
    PETSC_CASE_RETURN(PETSC_OFFLOAD_GPU);
    PETSC_CASE_RETURN(PETSC_OFFLOAD_BOTH);
    PETSC_CASE_RETURN(PETSC_OFFLOAD_KOKKOS);
#undef PETSC_CASE_RETURN
  }
  PetscUnreachable();
  return "invalid";
}

/*E
  PetscDeviceInitType - Initialization strategy for `PetscDevice`

$ PETSC_DEVICE_INIT_NONE  - PetscDevice is never initialized
$ PETSC_DEVICE_INIT_LAZY  - PetscDevice is initialized on demand
$ PETSC_DEVICE_INIT_EAGER - PetscDevice is initialized as soon as possible

  Notes:
  PETSC_DEVICE_INIT_NONE implies that any initialization of PetscDevice is disallowed and
  doing so results in an error. Useful to ensure that no accelerator is used in a program.

  Level: beginner

.seealso: `PetscDevice`, `PetscDeviceType`, `PetscDeviceInitialize()`,
`PetscDeviceInitialized()`, `PetscDeviceCreate()`
E*/
typedef enum {
  PETSC_DEVICE_INIT_NONE,
  PETSC_DEVICE_INIT_LAZY,
  PETSC_DEVICE_INIT_EAGER
} PetscDeviceInitType;
PETSC_EXTERN const char *const PetscDeviceInitTypes[];

/*E
  PetscDeviceType - Kind of accelerator device backend

$ PETSC_DEVICE_HOST - Host, no accelerator backend found
$ PETSC_DEVICE_CUDA - CUDA enabled GPU
$ PETSC_DEVICE_HIP  - ROCM/HIP enabled GPU
$ PETSC_DEVICE_SYCL - SYCL enabled device
$ PETSC_DEVICE_MAX  - Always 1 greater than the largest valid PetscDeviceType, invalid type, do not use

  Notes:
  One can also use the `PETSC_DEVICE_DEFAULT()` routine to get the current default `PetscDeviceType`.

  Level: beginner

.seealso: `PetscDevice`, `PetscDeviceInitType`, `PetscDeviceCreate()`, `PETSC_DEVICE_DEFAULT()`
E*/
typedef enum {
  PETSC_DEVICE_HOST,
  PETSC_DEVICE_CUDA,
  PETSC_DEVICE_HIP,
  PETSC_DEVICE_SYCL,
  PETSC_DEVICE_MAX
} PetscDeviceType;
PETSC_EXTERN const char *const PetscDeviceTypes[];

/*S
  PetscDevice - Object to manage an accelerator "device" (usually a GPU)

  Notes:
  This object is used to house configuration and state of a device, but does not offer any
  ability to interact with or drive device computation. This functionality is facilitated
  instead by the `PetscDeviceContext` object.

  Level: beginner

.seealso: `PetscDeviceType`, `PetscDeviceInitType`, `PetscDeviceCreate()`,
`PetscDeviceConfigure()`, `PetscDeviceDestroy()`, `PetscDeviceContext`,
`PetscDeviceContextSetDevice()`, `PetscDeviceContextGetDevice()`
S*/
typedef struct _n_PetscDevice *PetscDevice;

/*E
  PetscStreamType - Stream blocking mode, indicates how a stream implementation will interact
  with the default "NULL" stream, which is usually blocking.

$ PETSC_STREAM_GLOBAL_BLOCKING    - Alias for NULL stream. Any stream of this type will block the host for all other streams to finish work before starting its operations.
$ PETSC_STREAM_DEFAULT_BLOCKING   - Stream will act independent of other streams, but will still be blocked by actions on the NULL stream.
$ PETSC_STREAM_GLOBAL_NONBLOCKING - Stream is truly asynchronous, and is blocked by nothing, not even the NULL stream.
$ PETSC_STREAM_MAX                - Always 1 greater than the largest PetscStreamType, do not use

  Level: intermediate

.seealso: `PetscDeviceContextSetStreamType()`, `PetscDeviceContextGetStreamType()`
E*/
typedef enum {
  PETSC_STREAM_GLOBAL_BLOCKING,
  PETSC_STREAM_DEFAULT_BLOCKING,
  PETSC_STREAM_GLOBAL_NONBLOCKING,
  PETSC_STREAM_MAX
} PetscStreamType;
PETSC_EXTERN const char *const PetscStreamTypes[];

/*E
  PetscDeviceContextJoinMode - Describes the type of join operation to perform in
  `PetscDeviceContextJoin()`

$ PETSC_DEVICE_CONTEXT_JOIN_DESTROY - Destroy all incoming sub-contexts after join.
$ PETSC_DEVICE_CONTEXT_JOIN_SYNC    - Synchronize incoming sub-contexts after join.
$ PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC - Do not synchronize incoming sub-contexts after join.

  Level: beginner

.seealso: `PetscDeviceContext`, `PetscDeviceContextFork()`, `PetscDeviceContextJoin()`
E*/
typedef enum {
  PETSC_DEVICE_CONTEXT_JOIN_DESTROY,
  PETSC_DEVICE_CONTEXT_JOIN_SYNC,
  PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC
} PetscDeviceContextJoinMode;
PETSC_EXTERN const char *const PetscDeviceContextJoinModes[];

/*S
  PetscDeviceContext - Container to manage stream dependencies and the various solver handles
  for asynchronous device compute.

  Level: beginner

.seealso: `PetscDevice`, `PetscDeviceContextCreate()`, `PetscDeviceContextSetDevice()`,
`PetscDeviceContextDestroy()`,`PetscDeviceContextFork()`, `PetscDeviceContextJoin()`
S*/
typedef struct _p_PetscDeviceContext *PetscDeviceContext;

/*E
  PetscDeviceCopyMode - Describes the copy direction of a device-aware memcpy

$ PETSC_DEVICE_COPY_HTOH - Copy from host memory to host memory
$ PETSC_DEVICE_COPY_DTOH - Copy from device memory to host memory
$ PETSC_DEVICE_COPY_HTOD - Copy from host memory to device memory
$ PETSC_DEVICE_COPY_DTOD - Copy from device memory to device memory
$ PETSC_DEVICE_COPY_AUTO - Infer the copy direction from the pointers

  Level: beginner

.seealso: `PetscDeviceArrayCopy()`, `PetscDeviceMemcpy()`
E*/
typedef enum {
  PETSC_DEVICE_COPY_HTOH,
  PETSC_DEVICE_COPY_DTOH,
  PETSC_DEVICE_COPY_HTOD,
  PETSC_DEVICE_COPY_DTOD,
  PETSC_DEVICE_COPY_AUTO,
} PetscDeviceCopyMode;
PETSC_EXTERN const char *const PetscDeviceCopyModes[];

/*E
  PetscMemoryAccessMode - Describes the intended usage of a memory region

$ PETSC_MEMORY_ACCESS_READ       - Read only
$ PETSC_MEMORY_ACCESS_WRITE      - Write only
$ PETSC_MEMORY_ACCESS_READ_WRITE - Read and write

  Notes:
  This enum is a bitmask with the following encoding (assuming 2 bit):

.vb
  PETSC_MEMORY_ACCESS_READ       = 0b01
  PETSC_MEMORY_ACCESS_WRITE      = 0b10
  PETSC_MEMORY_ACCESS_READ_WRITE = 0b11

  // consequently
  PETSC_MEMORY_ACCESS_READ | PETSC_MEMORY_ACCESS_WRITE = PETSC_MEMORY_ACCESS_READ_WRITE
.ve

  Developer Notes:
  This enum uses a function (`PetscMemoryAccessModes()`) to convert values to string
  representation, so cannot be used in `PetscOptionsEnum()`.

  Level: beginner
E*/
typedef enum {
  PETSC_MEMORY_ACCESS_READ       = 0x1, // 01
  PETSC_MEMORY_ACCESS_WRITE      = 0x2, // 10
  PETSC_MEMORY_ACCESS_READ_WRITE = 0x3, // 11
} PetscMemoryAccessMode;

#define PetscMemoryAccessRead(m)  (((m)&PETSC_MEMORY_ACCESS_READ) == PETSC_MEMORY_ACCESS_READ)
#define PetscMemoryAccessWrite(m) (((m)&PETSC_MEMORY_ACCESS_WRITE) == PETSC_MEMORY_ACCESS_WRITE)

#if defined(__cplusplus)
// Some overzealous older gcc versions warn that the comparisons below are always true. Neat
// that it can detect this, but the tautology *is* the point of the static_assert()!
#if defined(__GNUC__) && __GNUC__ >= 6 && !PetscDefined(HAVE_WINDOWS_COMPILERS)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-compare"
#endif
static_assert(PetscMemoryAccessRead(PETSC_MEMORY_ACCESS_READ), "");
static_assert(PetscMemoryAccessRead(PETSC_MEMORY_ACCESS_READ_WRITE), "");
static_assert(!PetscMemoryAccessRead(PETSC_MEMORY_ACCESS_WRITE), "");
static_assert(PetscMemoryAccessWrite(PETSC_MEMORY_ACCESS_WRITE), "");
static_assert(PetscMemoryAccessWrite(PETSC_MEMORY_ACCESS_READ_WRITE), "");
static_assert(!PetscMemoryAccessWrite(PETSC_MEMORY_ACCESS_READ), "");
static_assert((PETSC_MEMORY_ACCESS_READ | PETSC_MEMORY_ACCESS_WRITE) == PETSC_MEMORY_ACCESS_READ_WRITE, "");
#if defined(__GNUC__) && __GNUC__ >= 6 && !PetscDefined(HAVE_WINDOWS_COMPILERS)
#pragma GCC diagnostic pop
#endif
#endif

PETSC_NODISCARD static inline PETSC_CONSTEXPR_14 const char *PetscMemoryAccessModes(PetscMemoryAccessMode mode) {
  switch (mode) {
#define PETSC_CASE_RETURN(v) \
  case v: return PetscStringize(v)
    PETSC_CASE_RETURN(PETSC_MEMORY_ACCESS_READ);
    PETSC_CASE_RETURN(PETSC_MEMORY_ACCESS_WRITE);
    PETSC_CASE_RETURN(PETSC_MEMORY_ACCESS_READ_WRITE);
#undef PETSC_CASE_RETURN
  }
  PetscUnreachable();
  return "invalid";
}

/*E
  PetscDeviceContextOption

$ PETSC_DEVICE_CONTEXT_ALLOW_ORPHANS - Allow a `PetscDeviceContext` to be destroyed with dangling dependencies

  Level: intermediate

.seealso: `PetscDeviceContextSetOption()`
E*/
typedef enum {
  PETSC_DEVICE_CONTEXT_ALLOW_ORPHANS,
} PetscDeviceContextOption;

typedef enum {
  PETSC_OPERATOR_PLUS,
  PETSC_OPERATOR_MINUS,
  PETSC_OPERATOR_MULTIPLY,
  PETSC_OPERATOR_DIVIDE,
  PETSC_OPERATOR_EQUAL
} PetscOperatorType;

#define PetscTypeSuffix Scalar
#include "petscmanagedtype.inl"
#define PetscTypeSuffix Real
#include "petscmanagedtype.inl"
#define PetscTypeSuffix Int
#include "petscmanagedtype.inl"
#endif /* PETSCDEVICETYPES_H */
