#ifndef PETSCVECCUPMIMPL_H
#define PETSCVECCUPMIMPL_H

#include <petsc/private/vecimpl.h>
#include <../src/vec/vec/impls/dvecimpl.h> // for Vec_Seq

#if PetscDefined(HAVE_CUDA)
PETSC_INTERN PetscErrorCode VecCreate_CUDA(Vec, PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecCreate_SeqCUDA(Vec, PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecCreate_MPICUDA(Vec, PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecCUDAGetArrays_Private(Vec, const PetscScalar **, const PetscScalar **, PetscOffloadMask *, PetscDeviceContext);
#endif

#if PetscDefined(HAVE_HIP)
PETSC_INTERN PetscErrorCode VecCreate_HIP(Vec, PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecCreate_SeqHIP(Vec, PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecCreate_MPIHIP(Vec, PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecHIPGetArrays_Private(Vec, const PetscScalar **, const PetscScalar **, PetscOffloadMask *, PetscDeviceContext);
#endif

#if PetscDefined(HAVE_NVSHMEM)
PETSC_INTERN PetscErrorCode PetscNvshmemInitializeCheck(void);
PETSC_INTERN PetscErrorCode PetscNvshmemMalloc(size_t, void **);
PETSC_INTERN PetscErrorCode PetscNvshmemCalloc(size_t, void **);
PETSC_INTERN PetscErrorCode PetscNvshmemFree_Private(void *);
#define PetscNvshmemFree(ptr) ((ptr) && (PetscNvshmemFree_Private(ptr) || ((ptr) = PETSC_NULLPTR, 0)))
PETSC_INTERN PetscErrorCode PetscNvshmemSum(PetscInt, PetscScalar *, const PetscScalar *);
PETSC_INTERN PetscErrorCode PetscNvshmemMax(PetscInt, PetscReal *, const PetscReal *);
PETSC_INTERN PetscErrorCode VecNormAsync_NVSHMEM(Vec, NormType, PetscReal *);
PETSC_INTERN PetscErrorCode VecAllocateNVSHMEM_SeqCUDA(Vec);
#else
#define PetscNvshmemFree(ptr) 0
#endif

#if defined(__cplusplus) && PetscDefined(HAVE_DEVICE)
#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupmblasinterface.hpp>

#include <limits>     // std::numeric_limits
#include <cstring>    // std::memset
#include <functional> // std::ref

namespace Petsc {

namespace vec {

namespace cupm {

namespace impl {

namespace {

// a simple RAII helper for PetscMallocSet[CUDA|HIP]Host(). it exists because integrating the
// regular versions would be an enormous pain to square with the templated types...
template <device::cupm::DeviceType T>
class UseCUPMHostAlloc_ : device::cupm::impl::Interface<T> {
  // would have loved to just do
  //
  // const auto oldmalloc = PetscTrMalloc;
  //
  // but in order to use auto the member needs to be static; in order to be static it must
  // also be constexpr -- which in turn requires an initializer (also implicitly required by
  // auto). But constexpr needs a constant expression initializer, so we can't initialize it
  // with global (mutable) variables...
#define DECLTYPE_AUTO(left, right) decltype(right) left = right
  const DECLTYPE_AUTO(oldmalloc_, PetscTrMalloc);
  const DECLTYPE_AUTO(oldfree_, PetscTrFree);
  const DECLTYPE_AUTO(oldrealloc_, PetscTrRealloc);
#undef DECLTYPE_AUTO
  const bool v_;

public:
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(interface_type, T);

  UseCUPMHostAlloc_(bool useit) noexcept : v_(useit) {
    if (useit) {
      // all unused arguments are un-named, this saves having to add PETSC_UNUSED to them all
      PetscTrMalloc = [](size_t sz, PetscBool clear, int, const char *, const char *, void **ptr) {
        PetscFunctionBegin;
        PetscCallCUPM(cupmMallocHost(ptr, sz));
        if (clear) std::memset(*ptr, 0, sz);
        PetscFunctionReturn(0);
      };
      PetscTrFree = [](void *ptr, int, const char *, const char *) {
        PetscFunctionBegin;
        PetscCallCUPM(cupmFreeHost(ptr));
        PetscFunctionReturn(0);
      };
      PetscTrRealloc = [](size_t, int, const char *, const char *, void **) {
        // REVIEW ME: can be implemented by malloc->copy->free?
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "%s has no realloc()", cupmName());
      };
    }
  }

  PETSC_NODISCARD auto value() const PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(v_);

  ~UseCUPMHostAlloc_() noexcept {
    if (v_) {
      PetscTrMalloc  = oldmalloc_;
      PetscTrFree    = oldfree_;
      PetscTrRealloc = oldrealloc_;
    }
  }
};

template <device::cupm::DeviceType T>
class WithCUPMBlasPointerMode : public device::cupm::impl::BlasInterface<T> {
  PETSC_CUPMBLAS_INHERIT_INTERFACE_TYPEDEFS_USING(interface_type, T);

  const cupmBlasHandle_t     &handle_;
  const cupmBlasPointerMode_t mode_;

#define PetscCallCUPMBLASAbort(...) \
  do { \
    const cupmBlasError_t cberr = __VA_ARGS__; \
    if (PetscUnlikely(cberr != CUPMBLAS_STATUS_SUCCESS)) { SETERRABORT(PETSC_COMM_SELF, PETSC_ERR_GPU, "%s error %d (%s)", cupmBlasName(), static_cast<PetscErrorCode>(cberr), cupmBlasGetErrorName(cberr)); } \
  } while (0)

  static cupmBlasPointerMode_t get_old_mode(const cupmBlasHandle_t &handle) noexcept {
    cupmBlasPointerMode_t mode;

    PetscFunctionBegin;
    PetscCallCUPMBLASAbort(cupmBlasGetPointerMode(handle, &mode));
    PetscFunctionReturn(mode);
  }

public:
  WithCUPMBlasPointerMode(const cupmBlasHandle_t &handle, cupmBlasPointerMode_t mode) noexcept : handle_(handle), mode_(get_old_mode(handle)) {
    PetscFunctionBegin;
    PetscCallCUPMBLASAbort(cupmBlasSetPointerMode(handle, mode));
    PetscFunctionReturnVoid();
  }

  WithCUPMBlasPointerMode(const cupmBlasHandle_t &handle, PetscMemType mtype) noexcept : WithCUPMBlasPointerMode(handle, PetscMemTypeDevice(mtype) ? CUPMBLAS_POINTER_MODE_DEVICE : CUPMBLAS_POINTER_MODE_HOST) { }

  ~WithCUPMBlasPointerMode() noexcept {
    PetscFunctionBegin;
    // REVIWE ME: swapping back and forth is kind of expensive...
    PetscCallCUPMBLASAbort(cupmBlasSetPointerMode(handle_, mode_));
    PetscFunctionReturnVoid();
  }
#undef PetscCallCUPMBLASAbort

  constexpr WithCUPMBlasPointerMode(WithCUPMBlasPointerMode &&) noexcept            = default;
  constexpr WithCUPMBlasPointerMode &operator=(WithCUPMBlasPointerMode &&) noexcept = default;
};

struct no_op {
  template <typename... T>
  constexpr PetscErrorCode operator()(T &&...) const noexcept {
    return 0;
  }
};

template <typename T>
struct CooPair {
  using value_type = T;
  using size_type  = PetscCount;

  value_type *&device;
  value_type *&host;
  size_type    size;
};

template <typename U>
static constexpr CooPair<U> make_coo_pair(U *&device, U *&host, PetscCount size) noexcept {
  return {device, host, size};
}

} // anonymous namespace

// forward declarations
template <device::cupm::DeviceType>
struct VecSeq_CUPM;
template <device::cupm::DeviceType>
struct VecMPI_CUPM;

// Base class for the VecSeq and VecMPI CUPM implementations. On top of the usual DeviceType
// template parameter it also uses CRTP to be able to use values/calls specific to either
// VecSeq or VecMPI. This is in effect "inside-out" polymorphism.
template <device::cupm::DeviceType T, typename Derived>
class Vec_CUPMBase : device::cupm::impl::BlasInterface<T> {
public:
  PETSC_CUPMBLAS_INHERIT_INTERFACE_TYPEDEFS_USING(cupmBlasInterface_t, T);

private:
  PETSC_CXX_COMPAT_DECL(PetscErrorCode ResetAllocatedDevicePtr_(PetscDeviceContext, Vec, PetscScalar * = nullptr));
  template <typename CastFunctionType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode VecAllocateCheck_(Vec, void *&, CastFunctionType &&));

  PETSC_CXX_COMPAT_DECL(PetscErrorCode CheckPointerMatchesMemType_(const void *ptr, PetscMemType mtype)) {
    PetscFunctionBegin;
    if (PetscDefined(USE_DEBUG) && ptr) {
      PetscMemType ptr_mtype;

      PetscCall(PetscCUPMGetMemType(ptr, &ptr_mtype));
      if (mtype == PETSC_MEMTYPE_HOST) {
        PetscCheck(PetscMemTypeHost(ptr_mtype), PETSC_COMM_SELF, PETSC_ERR_POINTER, "Pointer %p declared as %s does not match actual memtype %s", ptr, PetscMemTypes(mtype), PetscMemTypes(ptr_mtype));
      } else if (mtype == PETSC_MEMTYPE_DEVICE) {
        // generic "device" memory should only care if the actual memtype is also generically
        // "device"
        PetscCheck(PetscMemTypeDevice(ptr_mtype), PETSC_COMM_SELF, PETSC_ERR_POINTER, "Pointer %p declared as %s does not match actual memtype %s", ptr, PetscMemTypes(mtype), PetscMemTypes(ptr_mtype));
      } else {
        PetscCheck(mtype == ptr_mtype, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Pointer %p declared as %s does not match actual memtype %s", ptr, PetscMemTypes(mtype), PetscMemTypes(ptr_mtype));
      }
    }
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode GetHandleDispatch_(PetscDeviceContext dctx, cupmBlasHandle_t *handle, cupmStream_t *stream)) {
    PetscFunctionBegin;
    if (PetscDefined(USE_DEBUG)) {
      PetscDeviceType dtype;

      PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
      PetscCheckCompatibleDeviceTypes(PETSC_DEVICE_CUPM(), -1, dtype, 1);
    }
    if (handle) PetscCall(PetscDeviceContextGetBLASHandle_Internal(dctx, handle));
    if (stream) PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx, stream));
    PetscFunctionReturn(0);
  }

  // RAII versions of the get/restore array routines. Determines constness of the pointer type,
  // holds the pointer itself provides the implicit conversion operator
  template <PetscMemType MT, PetscMemoryAccessMode MA>
  struct vector_array {
    static const auto memory_type = MT;
    static const auto access_type = MA;

    using value_type              = PetscScalar;
    using pointer_type            = value_type *;
    using const_pointer_type      = const value_type *;
    using cupm_pointer_type       = cupmScalar_t *;
    using const_cupm_pointer_type = const cupmScalar_t *;

    pointer_type      data() const noexcept { return ptr_; }
    cupm_pointer_type cupmdata() const noexcept { return cupmScalarCast(ptr_); }

    operator pointer_type() const noexcept { return const_cast<pointer_type>(ptr_); }

    // in case pointer_type == cupmscalar_pointer_type we don't want this overload to exist, so
    // we make a dummy template parameter to allow SFINAE to nix it for us
    template <typename U = pointer_type, typename = util::enable_if_t<!std::is_same<U, cupm_pointer_type>::value>>
    operator cupm_pointer_type() const noexcept {
      return cupmScalarCast(const_cast<pointer_type>(ptr_));
    }

    vector_array(PetscDeviceContext dctx, Vec v) noexcept : ptr_(initialize_(dctx, v)), dctx_(dctx), v_(v) { }

    ~vector_array() noexcept {
      PetscFunctionBegin;
      PetscCallAbort(PETSC_COMM_SELF, restorearray_async<MT, MA>(v_, &ptr_, dctx_));
      PetscFunctionReturnVoid();
    }

    constexpr vector_array(vector_array &&) noexcept            = default;
    constexpr vector_array &operator=(vector_array &&) noexcept = default;

  private:
    pointer_type       ptr_;
    PetscDeviceContext dctx_;
    Vec                v_;

    PETSC_CXX_COMPAT_DECL(pointer_type initialize_(PetscDeviceContext dctx, Vec v)) {
      pointer_type array = nullptr;

      PetscFunctionBegin;
      PetscCallAbort(PETSC_COMM_SELF, getarray_async<MT, MA, true>(v, &array, dctx));
      PetscFunctionReturn(array);
    }
  };

protected:
  PETSC_CXX_COMPAT_DECL(PetscErrorCode VecView_Debug(Vec v, const char *message = "")) {
    const auto   pobj  = PetscObjectCast(v);
    const auto   vimpl = VecIMPLCast(v);
    const auto   vcu   = VecCUPMCast(v);
    PetscMemType mtype;
    PetscBool    device_mem;
    MPI_Comm     comm;

    PetscFunctionBegin;
    PetscValidPointer(vimpl, 1);
    PetscValidPointer(vcu, 1);
    PetscCall(PetscObjectGetComm(pobj, &comm));
    PetscCall(PetscPrintf(comm, "---------- %s ----------\n", message));
    PetscCall(PetscObjectPrintClassNamePrefixType(pobj, PETSC_VIEWER_STDOUT_(comm)));
    PetscCall(PetscPrintf(comm, "Address:             %p\n", v));
    PetscCall(PetscPrintf(comm, "Size:                %" PetscInt_FMT "\n", v->map->n));
    PetscCall(PetscPrintf(comm, "Offload mask:        %s\n", PetscOffloadMasks(v->offloadmask)));
    PetscCall(PetscPrintf(comm, "Host ptr:            %p\n", vimpl->array));
    PetscCall(PetscPrintf(comm, "Device ptr:          %p\n", vcu->array_d));
    PetscCall(PetscPrintf(comm, "Device alloced ptr:  %p\n", vcu->array_allocated_d));
    PetscCall(cupmGetMemType(vcu->array_d, &mtype));
    device_mem = static_cast<PetscBool>(PetscMemTypeDevice(mtype));
    PetscCall(PetscPrintf(comm, "dptr is device mem?  %s\n", PetscBools[device_mem]));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(auto GetHandles_(PetscDeviceContext dctx, cupmBlasHandle_t *handle, cupmStream_t *stream = nullptr))
  PETSC_DECLTYPE_AUTO_RETURNS(GetHandleDispatch_(dctx, handle, stream));

  PETSC_CXX_COMPAT_DECL(auto GetHandles_(PetscDeviceContext dctx, cupmStream_t *stream))
  PETSC_DECLTYPE_AUTO_RETURNS(GetHandleDispatch_(dctx, nullptr, stream));

  PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMAllocateCheck_(Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode VecIMPLAllocateCheck_(Vec));

public:
  struct Vec_CUPM {
    PetscScalar *array_d;           // gpu data
    PetscScalar *array_allocated_d; // does PETSc own the array ptr?
    PetscBool    nvshmem;           // is array allocated in nvshmem? It is used to allocate
                                    // Mvctx->lvec in nvshmem
    // COO stuff
    PetscCount  *jmap1_d; // [m+1]: i-th entry of the vector has jmap1[i+1]-jmap1[i] repeats
                          // in COO arrays
    PetscCount  *perm1_d; // [tot1]: permutation array for local entries
    PetscCount  *imap2_d; // [nnz2]: i-th unique entry in recvbuf is imap2[i]-th entry in
                          // the vector
    PetscCount  *jmap2_d; // [nnz2+1]
    PetscCount  *perm2_d; // [recvlen]
    PetscCount  *Cperm_d; // [sendlen]: permutation array to fill sendbuf[]. 'C' for
                          // communication
    // Buffers for remote values in VecSetValuesCOO()
    PetscScalar *sendbuf_d;
    PetscScalar *recvbuf_d;
  };

  PETSC_CXX_COMPAT_DECL(constexpr auto VecCUPMCast(Vec v))
  PETSC_DECLTYPE_AUTO_RETURNS(static_cast<Vec_CUPM *>(v->spptr));
  // This is a trick to get around the fact that in CRTP the derived class is not yet fully
  // defined because Base<Derived> must necessarily be instantiated before Derived is
  // complete. By using a dummy template parameter we make the type "dependent" and so will
  // only be determined when the derived class is instantiated (and therefore fully defined)
  template <typename U = Derived>
  PETSC_CXX_COMPAT_DECL(constexpr auto VecIMPLCast(Vec v))
  PETSC_DECLTYPE_AUTO_RETURNS(U::VecIMPLCast_(v));

  PETSC_CXX_COMPAT_DECL(constexpr PetscLogEvent VEC_CUPMCopyToGPU()) { return T == device::cupm::DeviceType::CUDA ? VEC_CUDACopyToGPU : VEC_HIPCopyToGPU; }

  PETSC_CXX_COMPAT_DECL(constexpr PetscLogEvent VEC_CUPMCopyFromGPU()) { return T == device::cupm::DeviceType::CUDA ? VEC_CUDACopyFromGPU : VEC_HIPCopyFromGPU; }

  PETSC_CXX_COMPAT_DECL(constexpr VecType VECSEQCUPM()) { return T == device::cupm::DeviceType::CUDA ? VECSEQCUDA : VECSEQHIP; }

  PETSC_CXX_COMPAT_DECL(constexpr VecType VECMPICUPM()) { return T == device::cupm::DeviceType::CUDA ? VECMPICUDA : VECMPIHIP; }

  template <typename U = Derived>
  PETSC_CXX_COMPAT_DECL(constexpr VecType VECTYPE()) {
    return U::VECTYPE_();
  }

  PETSC_CXX_COMPAT_DECL(constexpr PetscRandomType PETSCDEVICERAND()) {
    // REVIEW ME: HIP default rng?
    return T == device::cupm::DeviceType::CUDA ? PETSCCURAND : PETSCRANDER48;
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode CUPMBlasIntCast(PetscInt x, cupmBlasInt_t *y)) {
    using petsc_type = decltype(x);
    using blas_type  = util::remove_pointer_t<decltype(y)>;

    PetscFunctionBegin;
    if (!std::is_same<petsc_type, blas_type>::value) {
      PetscCheck(x <= std::numeric_limits<blas_type>::max(), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "%" PetscInt_FMT " is too big for %s, which may be restricted to 32 bit integers", x, cupmBlasName());
    }
    PetscCheck(x >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Passing negative integer to %s routine: %" PetscInt_FMT, cupmBlasName(), x);
    *y = static_cast<blas_type>(x);
    PetscFunctionReturn(0);
  }

  // data movement
  PETSC_CXX_COMPAT_DECL(PetscErrorCode HostAllocateCheck_(PetscDeviceContext, Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode DeviceAllocateCheck_(PetscDeviceContext, Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode CopyToDevice_(PetscDeviceContext, Vec, bool = false));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode CopyToHost_(PetscDeviceContext, Vec, bool = false));
  // need functions to create the vector arrays, otherwise using them as an unnamed temporary
  // leads to most vexing parse
  PETSC_CXX_COMPAT_DECL(auto DeviceArrayRead(PetscDeviceContext dctx, Vec v))
  PETSC_DECLTYPE_AUTO_RETURNS(vector_array<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ>{dctx, v});
  PETSC_CXX_COMPAT_DECL(auto DeviceArrayWrite(PetscDeviceContext dctx, Vec v))
  PETSC_DECLTYPE_AUTO_RETURNS(vector_array<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_WRITE>{dctx, v});
  PETSC_CXX_COMPAT_DECL(auto DeviceArrayReadWrite(PetscDeviceContext dctx, Vec v))
  PETSC_DECLTYPE_AUTO_RETURNS(vector_array<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ_WRITE>{dctx, v});
  PETSC_CXX_COMPAT_DECL(auto HostArrayRead(PetscDeviceContext dctx, Vec v))
  PETSC_DECLTYPE_AUTO_RETURNS(vector_array<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ>{dctx, v});
  PETSC_CXX_COMPAT_DECL(auto HostArrayWrite(PetscDeviceContext dctx, Vec v))
  PETSC_DECLTYPE_AUTO_RETURNS(vector_array<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_WRITE>{dctx, v});
  PETSC_CXX_COMPAT_DECL(auto HostArrayReadWrite(PetscDeviceContext dctx, Vec v))
  PETSC_DECLTYPE_AUTO_RETURNS(vector_array<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ_WRITE>{dctx, v});

  // accessors
  template <PetscMemType, PetscMemoryAccessMode, bool = false>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getarray_async(Vec, PetscScalar **, PetscDeviceContext));
  template <PetscMemType, PetscMemoryAccessMode>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode restorearray_async(Vec, PetscScalar **, PetscDeviceContext));
  template <PetscMemoryAccessMode>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getarrayandmemtype_async(Vec, PetscScalar **, PetscMemType *, PetscDeviceContext));
  template <PetscMemoryAccessMode>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode restorearrayandmemtype_async(Vec, PetscScalar **, PetscDeviceContext));
  template <PetscMemType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode replacearray_async(Vec, const PetscScalar *, PetscDeviceContext));

  // common ops shared between Seq and MPI
  PETSC_CXX_COMPAT_DECL(PetscErrorCode Create_CUPM(Vec, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode Create_CUPMBase(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscDeviceContext, Vec *, PetscBool, PetscLayout /*reference*/ = nullptr));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode Initialize_CUPMBase(Vec, PetscBool, PetscScalar *, PetscScalar *, PetscDeviceContext));
  template <typename F>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode Destroy_CUPMBase(Vec, PetscDeviceContext, F &&));
  template <typename SetupFunctionT = no_op>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode Duplicate_CUPMBase(Vec, Vec *, PetscDeviceContext, SetupFunctionT && = SetupFunctionT{}));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode BindToCPU_CUPMBase(Vec, PetscBool, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode GetArrays_CUPMBase(Vec, const PetscScalar **, const PetscScalar **, PetscOffloadMask *, PetscDeviceContext));
  template <PetscMemType, typename F>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode ResetArray_CUPMBase(Vec, F &&, PetscDeviceContext));
  template <PetscMemType, typename F>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode PlaceArray_CUPMBase(Vec, const PetscScalar *, F &&, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode ResetPreallocationCOO_CUPMBase(Vec, PetscDeviceContext));
  template <std::size_t NCount = 0, std::size_t NScal = 0>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode SetPreallocationCOO_CUPMBase(Vec, PetscCount, const PetscInt[], PetscDeviceContext, const std::array<CooPair<PetscCount>, NCount> & = {}, const std::array<CooPair<PetscScalar>, NScal> & = {}));

  // utility for using cupmHostAlloc()
  PETSC_CXX_COMPAT_DECL(auto UseCUPMHostAlloc(bool b))
  PETSC_DECLTYPE_AUTO_RETURNS(UseCUPMHostAlloc_<T>(b));
  PETSC_CXX_COMPAT_DECL(auto UseCUPMHostAlloc(PetscBool b))
  PETSC_DECLTYPE_AUTO_RETURNS(UseCUPMHostAlloc(static_cast<bool>(b)));
  PETSC_CXX_COMPAT_DECL(auto UseCUPMHostAlloc(Vec v))
  PETSC_DECLTYPE_AUTO_RETURNS(UseCUPMHostAlloc(v->pinned_memory == PETSC_TRUE));
};

template <device::cupm::DeviceType T, typename D>
template <PetscMemType MT, PetscMemoryAccessMode MA>
const PetscMemType Vec_CUPMBase<T, D>::vector_array<MT, MA>::memory_type;

template <device::cupm::DeviceType T, typename D>
template <PetscMemType MT, PetscMemoryAccessMode MA>
const PetscMemoryAccessMode Vec_CUPMBase<T, D>::vector_array<MT, MA>::access_type;

template <device::cupm::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::ResetAllocatedDevicePtr_(PetscDeviceContext dctx, Vec v, PetscScalar *new_value)) {
  const auto vcu          = VecCUPMCast(v);
  auto      &device_array = vcu->array_allocated_d;

  PetscFunctionBegin;
  if (device_array) {
    if (PetscDefined(HAVE_NVSHMEM) && vcu->nvshmem) {
      PetscCall(PetscNvshmemFree(device_array));
    } else {
      cupmStream_t stream;

      PetscCall(GetHandles_(dctx, &stream));
      PetscCallCUPM(cupmFreeAsync(device_array, stream));
    }
  }
  device_array = new_value;
  PetscFunctionReturn(0);
}

PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMCheckMinimumPinnedMemory_Internal(Vec v)) {
  auto      mem = static_cast<PetscInt>(v->minimum_bytes_pinned_memory);
  PetscBool flg;

  PetscFunctionBegin;
  PetscObjectOptionsBegin(PetscObjectCast(v));
  PetscCall(PetscOptionsRangeInt("-vec_pinned_memory_min", "Minimum size (in bytes) for an allocation to use pinned memory on host", "VecSetPinnedMemoryMin", mem, &mem, &flg, 0, std::numeric_limits<decltype(mem)>::max()));
  if (flg) v->minimum_bytes_pinned_memory = mem;
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T, typename D>
template <typename CastFunctionType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::VecAllocateCheck_(Vec v, void *&dest, CastFunctionType &&cast)) {
  PetscFunctionBegin;
  if (PetscUnlikely(!dest)) {
    auto          impl = cast(v);
    cupmBlasInt_t bn;

    PetscCall(PetscNewLog(PetscObjectCast(v), &impl));
    dest = impl;
    // do a cast to blasint check because if blasint cant hold the size, then any subsequent
    // cupmblas calls can't use it either. Doing this now this means we don't have to check
    // during every function
    PetscCall(CUPMBlasIntCast(v->map->n, &bn));
  }
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::VecIMPLAllocateCheck_(Vec v)) {
  PetscFunctionBegin;
  PetscCall(VecAllocateCheck_(v, v->data, VecIMPLCast<D>));
  PetscFunctionReturn(0);
}

// allocate the Vec_CUPM struct. this is normally done through DeviceAllocateCheck_(), but in
// certain circumstances (such as when the user places the device array) we do not want to do
// the full DeviceAllocateCheck_() as it also allocates the array
template <device::cupm::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::VecCUPMAllocateCheck_(Vec v)) {
  PetscFunctionBegin;
  PetscCall(VecAllocateCheck_(v, v->spptr, VecCUPMCast));
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::HostAllocateCheck_(PetscDeviceContext, Vec v)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCall(VecIMPLAllocateCheck_(v));
  if (auto &alloc = VecIMPLCast(v)->array_allocated) PetscFunctionReturn(0);
  else {
    const auto n      = v->map->n;
    const auto nbytes = n * sizeof(*alloc);

    PetscCall(VecCUPMCheckMinimumPinnedMemory_Internal(v));
    {
      const auto useit = UseCUPMHostAlloc(nbytes > v->minimum_bytes_pinned_memory);

      v->pinned_memory = static_cast<decltype(v->pinned_memory)>(useit.value());
      PetscCall(PetscMalloc1(n, &alloc));
    }
    PetscCall(PetscLogObjectMemory(PetscObjectCast(v), nbytes));
    if (!VecIMPLCast(v)->array) VecIMPLCast(v)->array = alloc;
    if (v->offloadmask == PETSC_OFFLOAD_UNALLOCATED) v->offloadmask = PETSC_OFFLOAD_CPU;
  }
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::DeviceAllocateCheck_(PetscDeviceContext dctx, Vec v)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCall(VecCUPMAllocateCheck_(v));
  if (auto &alloc = VecCUPMCast(v)->array_d) PetscFunctionReturn(0);
  else {
    auto        &array_allocated_d = VecCUPMCast(v)->array_allocated_d;
    cupmStream_t stream;

    PetscCall(GetHandles_(dctx, &stream));
    PetscCall(PetscCUPMMallocAsync(&array_allocated_d, v->map->n, stream));
    alloc = array_allocated_d;
    if (v->offloadmask == PETSC_OFFLOAD_UNALLOCATED) {
      const auto vimp = VecIMPLCast(v);
      v->offloadmask  = (vimp && vimp->array) ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
    }
  }
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::CopyToDevice_(PetscDeviceContext dctx, Vec v, bool forceasync)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCall(DeviceAllocateCheck_(dctx, v));
  if (v->offloadmask == PETSC_OFFLOAD_CPU) {
    cupmStream_t stream;

    PetscCall(GetHandles_(dctx, &stream));
    PetscCall(PetscLogEventBegin(VEC_CUPMCopyToGPU(), v, 0, 0, 0));
    PetscCall(PetscCUPMMemcpyAsync(VecCUPMCast(v)->array_d, VecIMPLCast(v)->array, v->map->n, cupmMemcpyHostToDevice, stream, forceasync));
    PetscCall(PetscLogEventEnd(VEC_CUPMCopyToGPU(), v, 0, 0, 0));
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::CopyToHost_(PetscDeviceContext dctx, Vec v, bool forceasync)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCall(HostAllocateCheck_(dctx, v));
  if (v->offloadmask == PETSC_OFFLOAD_GPU) {
    cupmStream_t stream;

    PetscCall(GetHandles_(dctx, &stream));
    PetscCall(PetscLogEventBegin(VEC_CUPMCopyFromGPU(), v, 0, 0, 0));
    PetscCall(PetscCUPMMemcpyAsync(VecIMPLCast(v)->array, VecCUPMCast(v)->array_d, v->map->n, cupmMemcpyDeviceToHost, stream, forceasync));
    PetscCall(PetscLogEventEnd(VEC_CUPMCopyFromGPU(), v, 0, 0, 0));
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

#define STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED(mtype) static_assert((mtype == PETSC_MEMTYPE_HOST) || (mtype == PETSC_MEMTYPE_DEVICE), "")

// v->ops->getarray[read|write] or VecCUPMGetArray[Read|Write]()
template <device::cupm::DeviceType T, typename D>
template <PetscMemType mtype, PetscMemoryAccessMode access, bool force_async>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::getarray_async(Vec v, PetscScalar **a, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED(mtype);
  constexpr auto hostmem    = PetscMemTypeHost(mtype);
  const auto     oldmask    = v->offloadmask;
  auto          &mask       = v->offloadmask;
  auto           maybe_sync = false;

  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQCUPM(), VECMPICUPM());
  if (PetscMemoryAccessRead(access)) {
    // READ or READ_WRITE
    if (((oldmask == PETSC_OFFLOAD_GPU) && hostmem) || ((oldmask == PETSC_OFFLOAD_CPU) && !hostmem)) {
      // if we move the data we should set the flag to synchronize later on
      maybe_sync = true;
    }
    PetscCall((hostmem ? CopyToHost_ : CopyToDevice_)(dctx, v, force_async));
  } else {
    // WRITE only
    PetscCall((hostmem ? HostAllocateCheck_ : DeviceAllocateCheck_)(dctx, v));
  }
  *a = hostmem ? VecIMPLCast(v)->array : VecCUPMCast(v)->array_d;
  // if unallocated previously we should zero things out if we intend to read
  if ((oldmask == PETSC_OFFLOAD_UNALLOCATED) && PetscMemoryAccessRead(access)) {
    const auto n = v->map->n;

    if (hostmem) {
      PetscCall(PetscArrayzero(*a, n));
    } else {
      cupmStream_t stream;

      PetscCall(GetHandles_(dctx, &stream));
      PetscCall(PetscCUPMMemsetAsync(*a, 0, n, stream, force_async));
      maybe_sync = true;
    }
  }
  // update the offloadmask if we intend to write, since we assume immediately modified
  if (PetscMemoryAccessWrite(access)) {
    PetscCall(VecSetErrorIfLocked(v, 1));
    // REVIEW ME: this should probably also call PetscObjectStateIncrease() since we assume it
    // is immediately modified
    mask = hostmem ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
  }
  // if we are a globally blocking stream and we have MOVED data then we should synchronize,
  // since even doing async calls on the NULL stream is not synchronous
  if (!force_async && maybe_sync) {
    PetscStreamType stype;

    PetscCall(PetscDeviceContextGetStreamType(dctx, &stype));
    if (stype == PETSC_STREAM_GLOBAL_BLOCKING) PetscCall(PetscDeviceContextSynchronize(dctx));
  }
  PetscFunctionReturn(0);
}

// v->ops->restorearray[read|write] or VecCUPMRestoreArray[Read|Write]()
template <device::cupm::DeviceType T, typename D>
template <PetscMemType mtype, PetscMemoryAccessMode access>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::restorearray_async(Vec v, PetscScalar **a, PetscDeviceContext)) {
  NVTX_RANGE;
  STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED(mtype);

  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQCUPM(), VECMPICUPM());
  if (PetscMemoryAccessWrite(access)) {
    // WRITE or READ_WRITE
    PetscCall(PetscObjectStateIncrease(PetscObjectCast(v)));
    v->offloadmask = PetscMemTypeHost(mtype) ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
  }
  if (a) {
    PetscCall(CheckPointerMatchesMemType_(*a, mtype));
    *a = nullptr;
  }
  PetscFunctionReturn(0);
}

// v->ops->getarrayandmemtype
template <device::cupm::DeviceType T, typename D>
template <PetscMemoryAccessMode access>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::getarrayandmemtype_async(Vec v, PetscScalar **a, PetscMemType *mtype, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCall(getarray_async<PETSC_MEMTYPE_DEVICE, access>(v, a, dctx));
  if (mtype) *mtype = (PetscDefined(HAVE_NVSHMEM) && VecCUPMCast(v)->nvshmem) ? PETSC_MEMTYPE_NVSHMEM : PETSC_MEMTYPE_CUPM();
  PetscFunctionReturn(0);
}

// v->ops->restorearrayandmemtype
template <device::cupm::DeviceType T, typename D>
template <PetscMemoryAccessMode access>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::restorearrayandmemtype_async(Vec v, PetscScalar **a, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCall(restorearray_async<PETSC_MEMTYPE_DEVICE, access>(v, a, dctx));
  PetscFunctionReturn(0);
}

// v->ops->replacearray or VecCUPMReplaceArray()
template <device::cupm::DeviceType T, typename D>
template <PetscMemType mtype>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::replacearray_async(Vec v, const PetscScalar *a, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED(mtype);
  PetscCheckTypeNames(v, VECSEQCUPM(), VECMPICUPM());
  PetscCall(CheckPointerMatchesMemType_(a, mtype));
  if (PetscMemTypeHost(mtype)) {
    PetscCall(VecIMPLAllocateCheck_(v));
    {
      const auto vimpl      = VecIMPLCast(v);
      auto      &host_array = vimpl->array_allocated;

      // make sure the users array has the latest values.
      // REVIEW ME: why? we're about to free it
      if (host_array != vimpl->array) PetscCall(CopyToHost_(dctx, v));
      if (host_array) {
        const auto useit = UseCUPMHostAlloc(v);
        PetscCall(PetscFree(host_array));
      }
      host_array       = const_cast<PetscScalar *>(a);
      vimpl->array     = host_array;
      v->pinned_memory = PETSC_FALSE; // REVIEW ME: we can determine this
      v->offloadmask   = PETSC_OFFLOAD_CPU;
    }
  } else {
    PetscCall(VecCUPMAllocateCheck_(v));
    PetscCall(ResetAllocatedDevicePtr_(dctx, v, const_cast<PetscScalar *>(a)));
    // don't update the offloadmask if placed pointer is NULL
    if ((VecCUPMCast(v)->array_d = VecCUPMCast(v)->array_allocated_d)) { v->offloadmask = PETSC_OFFLOAD_GPU; }
    PetscCall(PetscObjectStateIncrease(PetscObjectCast(v)));
  }
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                      Common core between Seq and MPI                               //

// VecCreate_CUPM()
template <device::cupm::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::Create_CUPM(Vec v, PetscDeviceContext)) {
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm(PetscObjectCast(v)), &size));
  PetscCall(VecSetType(v, size > 1 ? VECMPICUPM() : VECSEQCUPM()));
  PetscFunctionReturn(0);
}

// VecCreateCUPM()
template <device::cupm::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::Create_CUPMBase(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, PetscDeviceContext, Vec *v, PetscBool call_set_type, PetscLayout reference)) {
  PetscFunctionBegin;
  PetscCall(VecCreate(comm, v));
  if (reference) PetscCall(PetscLayoutReference(reference, &(*v)->map));
  PetscCall(VecSetSizes(*v, n, N));
  if (bs) PetscCall(VecSetBlockSize(*v, bs));
  if (call_set_type) PetscCall(VecSetType(*v, VECTYPE()));
  PetscFunctionReturn(0);
}

// VecCreateIMPL_CUPM()
template <device::cupm::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::Initialize_CUPMBase(Vec v, PetscBool allocate_missing, PetscScalar *host_array, PetscScalar *device_array, PetscDeviceContext dctx)) {
  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUPM()));
  PetscCall(PetscObjectChangeTypeName(PetscObjectCast(v), VECTYPE()));
  PetscCall(D::bindtocpu_async(v, PETSC_FALSE, dctx));
  if (device_array) {
    PetscCall(CheckPointerMatchesMemType_(device_array, PETSC_MEMTYPE_CUPM()));
    PetscCall(VecCUPMAllocateCheck_(v));
    VecCUPMCast(v)->array_d = device_array;
  }
  if (host_array) {
    PetscCall(CheckPointerMatchesMemType_(host_array, PETSC_MEMTYPE_HOST));
    PetscCall(HostAllocateCheck_(dctx, v));
    VecIMPLCast(v)->array = host_array;
  }
  if (allocate_missing) {
    PetscCall(DeviceAllocateCheck_(dctx, v));
    PetscCall(HostAllocateCheck_(dctx, v));
    // REVIEW ME: junchao, is this needed with new calloc() branch? VecSet() will call
    // set_async() for reference
    // calls device-version
    PetscCall(VecSet(v, 0));
    // zero the host while device is underway
    PetscCall(PetscArrayzero(VecIMPLCast(v)->array, v->map->n));
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  } else {
    if (host_array) {
      v->offloadmask = device_array ? PETSC_OFFLOAD_BOTH : PETSC_OFFLOAD_CPU;
    } else {
      v->offloadmask = device_array ? PETSC_OFFLOAD_GPU : PETSC_OFFLOAD_UNALLOCATED;
    }
  }
  PetscFunctionReturn(0);
}

// v->ops->destroy
template <device::cupm::DeviceType T, typename D>
template <typename F>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::Destroy_CUPMBase(Vec v, PetscDeviceContext dctx, F &&VecDestroy_IMPLS)) {
  PetscFunctionBegin;
  if (const auto vcu = VecCUPMCast(v)) {
    PetscCall(ResetAllocatedDevicePtr_(dctx, v));
    PetscCall(ResetPreallocationCOO_CUPMBase(v, dctx));
    PetscCall(PetscFree(v->spptr));
  }
  PetscCall(PetscObjectSAWsViewOff(PetscObjectCast(v)));
  if (const auto vimpl = VecIMPLCast(v)) {
    if (auto &array_allocated = vimpl->array_allocated) {
      const auto useit = UseCUPMHostAlloc(v);

      // do this ourselves since we may want to use the cupm functions
      PetscCall(PetscFree(array_allocated));
    }
  }
  v->pinned_memory = PETSC_FALSE;
  PetscCall(VecDestroy_IMPLS(v, dctx));
  PetscFunctionReturn(0);
}

// v->ops->duplicate
template <device::cupm::DeviceType T, typename D>
template <typename SetupFunctionT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::Duplicate_CUPMBase(Vec v, Vec *y, PetscDeviceContext dctx, SetupFunctionT &&DerivedCreateIMPLCUPM_Async)) {
  NVTX_RANGE;
  // if the derived setup is the default no_op then we should call VecSetType()
  constexpr auto call_set_type = static_cast<PetscBool>(std::is_same<SetupFunctionT, no_op>::value);
  const auto     vobj          = PetscObjectCast(v);
  const auto     map           = v->map;
  PetscInt       bs;

  PetscFunctionBegin;
  PetscCall(VecGetBlockSize(v, &bs));
  PetscCall(Create_CUPMBase(PetscObjectComm(vobj), bs, map->n, map->N, dctx, y, call_set_type, map));
  // Derived class can set up the remainder of the data structures here
  PetscCall(DerivedCreateIMPLCUPM_Async(*y, dctx));
  // If the other vector is bound to CPU then the memcpy of the ops struct will give the
  // duplicated vector the host "getarray" function which does not lazily allocate the array
  // (as it is assumed to always exist). So we force allocation here, before we overwrite the
  // ops
  if (v->boundtocpu) PetscCall(HostAllocateCheck_(dctx, *y));
  // in case the user has done some VecSetOps() tomfoolery
  PetscCall(PetscMemcpy((*y)->ops, v->ops, sizeof(*v->ops)));
  {
    const auto yobj = PetscObjectCast(*y);
    PetscCall(PetscObjectListDuplicate(vobj->olist, &yobj->olist));
    PetscCall(PetscFunctionListDuplicate(vobj->qlist, &yobj->qlist));
  }
  (*y)->stash.donotstash   = v->stash.donotstash;
  (*y)->stash.ignorenegidx = v->stash.ignorenegidx;
  (*y)->map->bs            = std::abs(v->map->bs);
  (*y)->bstash.bs          = v->bstash.bs;
  PetscFunctionReturn(0);
}

// v->ops->resetarray or VecCUPMResetArray()
template <device::cupm::DeviceType T, typename D>
template <PetscMemType mtype, typename F>
// yes (probably Jed :)), ideal world these should be arguments not template parameters. But I
// need to assign this function to a C compatible function pointer, so something like default
// arguments don't work no? Stubs seem like overkill too...
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::ResetArray_CUPMBase(Vec v, F &&VecResetArray_IMPL, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED(mtype);
  PetscCheckTypeNames(v, VECSEQCUPM(), VECMPICUPM());
  // REVIEW ME:
  // this is wildly inefficient but must be done if we assume that the placed array must have
  // correct values
  if (PetscMemTypeHost(mtype)) {
    PetscCall(CopyToHost_(dctx, v));
    PetscCall(VecResetArray_IMPL(v, dctx));
    v->offloadmask = PETSC_OFFLOAD_CPU;
  } else {
    PetscCall(VecIMPLAllocateCheck_(v));
    PetscCall(VecCUPMAllocateCheck_(v));
    {
      const auto vcu        = VecCUPMCast(v);
      const auto vimpl      = VecIMPLCast(v);
      auto      &host_array = vimpl->unplacedarray;

      PetscCall(CheckPointerMatchesMemType_(host_array, PETSC_MEMTYPE_DEVICE));
      PetscCall(CopyToDevice_(dctx, v));
      PetscCall(PetscObjectStateIncrease(PetscObjectCast(v)));
      // Need to reset the offloadmask. If we had a stashed pointer we are on the GPU,
      // otherwise check if the host has a valid pointer. If neither, then we are not allocated.
      if ((vcu->array_d = host_array)) {
        v->offloadmask = PETSC_OFFLOAD_GPU;
        host_array     = nullptr;
      } else if (vimpl->array) {
        v->offloadmask = PETSC_OFFLOAD_CPU;
      } else {
        v->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
      }
    }
  }
  PetscFunctionReturn(0);
}

// v->ops->placearray or VecCUPMPlaceArray()
template <device::cupm::DeviceType T, typename D>
template <PetscMemType mtype, typename F>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::PlaceArray_CUPMBase(Vec v, const PetscScalar *a, F &&VecPlaceArray_IMPL, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED(mtype);
  PetscCheckTypeNames(v, VECSEQCUPM(), VECMPICUPM());
  PetscCall(CheckPointerMatchesMemType_(a, mtype));
  if (PetscMemTypeHost(mtype)) {
    PetscCall(CopyToHost_(dctx, v));
    PetscCall(VecPlaceArray_IMPL(v, a, dctx));
    v->offloadmask = PETSC_OFFLOAD_CPU;
  } else {
    PetscCall(VecCUPMAllocateCheck_(v));
    PetscCall(VecIMPLAllocateCheck_(v));
    {
      auto &device_array = VecCUPMCast(v)->array_d;
      auto &backup_array = VecIMPLCast(v)->unplacedarray;

      PetscCheck(!backup_array, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "VecPlaceArray() was already called on this vector, without a call to VecResetArray()");
      PetscCall(PetscObjectStateIncrease(PetscObjectCast(v)));
      backup_array = device_array;
      // only update the offload mask if we actually assign a pointer
      if ((device_array = const_cast<PetscScalar *>(a))) v->offloadmask = PETSC_OFFLOAD_GPU;
    }
  }
  PetscFunctionReturn(0);
}

#undef STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED

namespace {

PETSC_CXX_COMPAT_DECL(PetscErrorCode ChangeDefaultRandType(PetscRandomType target, char **ptr)) {
  PetscFunctionBegin;
  PetscValidPointer(ptr, 2);
  PetscValidCharPointer(*ptr, 2);
  if (std::strcmp(target, *ptr)) {
    PetscCall(PetscFree(*ptr));
    PetscCall(PetscStrallocpy(target, ptr));
  }
  PetscFunctionReturn(0);
}

} // anonymous namespace

#define VecSetOp_CUPM(op_name, op_host, ...) \
  do { \
    if (usehost) { \
      v->ops->op_name = op_host; \
    } else { \
      v->ops->op_name = __VA_ARGS__; \
    } \
  } while (0)

// v->ops->bindtocpu
template <device::cupm::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::BindToCPU_CUPMBase(Vec v, PetscBool usehost, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  if ((v->boundtocpu = usehost)) PetscCall(CopyToHost_(dctx, v));
  PetscCall(ChangeDefaultRandType(usehost ? PETSCRANDER48 : PETSCDEVICERAND(), &v->defaultrandtype));

  // set the base functions that are guaranteed to be the same for both
  v->ops->duplicate    = D::duplicate_async;
  v->ops->create       = D::create_async;
  v->ops->destroy      = D::destroy_async;
  v->ops->bindtocpu    = D::bindtocpu_async;
  v->ops->replacearray = replacearray_async<PETSC_MEMTYPE_HOST>;

  // set device-only common functions
  VecSetOp_CUPM(dotnorm2, nullptr, D::dotnorm2_async);
  VecSetOp_CUPM(getarray, nullptr, getarray_async<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ_WRITE>);
  VecSetOp_CUPM(restorearray, nullptr, restorearray_async<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ_WRITE>);
  VecSetOp_CUPM(getarraywrite, nullptr, getarray_async<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_WRITE>);
  VecSetOp_CUPM(restorearraywrite, nullptr, restorearray_async<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_WRITE>);

  VecSetOp_CUPM(getarrayread, nullptr, [](Vec v, const PetscScalar **a, PetscDeviceContext dctx) { return getarray_async<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ>(v, const_cast<PetscScalar **>(a), dctx); });
  VecSetOp_CUPM(restorearrayread, nullptr, [](Vec v, const PetscScalar **a, PetscDeviceContext dctx) { return restorearray_async<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ>(v, const_cast<PetscScalar **>(a), dctx); });

  VecSetOp_CUPM(getarrayandmemtype, nullptr, getarrayandmemtype_async<PETSC_MEMORY_ACCESS_READ_WRITE>);
  VecSetOp_CUPM(restorearrayandmemtype, nullptr, restorearrayandmemtype_async<PETSC_MEMORY_ACCESS_READ_WRITE>);

  VecSetOp_CUPM(getarraywriteandmemtype, nullptr, getarrayandmemtype_async<PETSC_MEMORY_ACCESS_WRITE>);
  VecSetOp_CUPM(restorearraywriteandmemtype, nullptr, [](Vec v, PetscScalar **a, PetscMemType *, PetscDeviceContext dctx) { return restorearrayandmemtype_async<PETSC_MEMORY_ACCESS_WRITE>(v, a, dctx); });

  VecSetOp_CUPM(getarrayreadandmemtype, nullptr, [](Vec v, const PetscScalar **a, PetscMemType *m, PetscDeviceContext d) { return getarrayandmemtype_async<PETSC_MEMORY_ACCESS_READ>(v, const_cast<PetscScalar **>(a), m, d); });
  VecSetOp_CUPM(restorearrayreadandmemtype, nullptr, [](Vec v, const PetscScalar **a, PetscDeviceContext d) { return restorearrayandmemtype_async<PETSC_MEMORY_ACCESS_READ>(v, const_cast<PetscScalar **>(a), d); });

  // set the functions that are always sequential
  using VecSeq_T = VecSeq_CUPM<T>;
  VecSetOp_CUPM(scale, VecScale_Seq, VecSeq_T::scale_async);
  VecSetOp_CUPM(copy, VecCopy_Seq, VecSeq_T::copy_async);
  VecSetOp_CUPM(set, VecSet_Seq, VecSeq_T::set_async);
  VecSetOp_CUPM(swap, VecSwap_Seq, VecSeq_T::swap_async);
  VecSetOp_CUPM(axpy, VecAXPY_Seq, VecSeq_T::axpy_async);
  VecSetOp_CUPM(axpby, VecAXPBY_Seq, VecSeq_T::axpby_async);
  VecSetOp_CUPM(maxpy, VecMAXPY_Seq, VecSeq_T::maxpy_async);
  VecSetOp_CUPM(aypx, VecAYPX_Seq, VecSeq_T::aypx_async);
  VecSetOp_CUPM(waxpy, VecWAXPY_Seq, VecSeq_T::waxpy_async);
  VecSetOp_CUPM(axpbypcz, VecAXPBYPCZ_Seq, VecSeq_T::axpbypcz_async);
  VecSetOp_CUPM(pointwisemult, VecPointwiseMult_Seq, VecSeq_T::pointwisemult_async);
  VecSetOp_CUPM(pointwisedivide, VecPointwiseDivide_Seq, VecSeq_T::pointwisedivide_async);
  VecSetOp_CUPM(setrandom, VecSetRandom_Seq, VecSeq_T::setrandom_async);
  VecSetOp_CUPM(dot_local, VecDot_Seq, VecSeq_T::dot_async);
  VecSetOp_CUPM(tdot_local, VecTDot_Seq, VecSeq_T::tdot_async);
  VecSetOp_CUPM(norm_local, VecNorm_Seq, VecSeq_T::norm_async);
  VecSetOp_CUPM(mdot_local, VecMDot_Seq, VecSeq_T::mdot_async);
  VecSetOp_CUPM(reciprocal, VecReciprocal_Default, VecSeq_T::reciprocal_async);
  VecSetOp_CUPM(shift, nullptr, VecSeq_T::shift_async);
  VecSetOp_CUPM(getlocalvector, nullptr, VecSeq_T::template getlocalvector_async<PETSC_MEMORY_ACCESS_READ_WRITE>);
  VecSetOp_CUPM(restorelocalvector, nullptr, VecSeq_T::template restorelocalvector_async<PETSC_MEMORY_ACCESS_READ_WRITE>);
  VecSetOp_CUPM(getlocalvectorread, nullptr, VecSeq_T::template getlocalvector_async<PETSC_MEMORY_ACCESS_READ>);
  VecSetOp_CUPM(restorelocalvectorread, nullptr, VecSeq_T::template restorelocalvector_async<PETSC_MEMORY_ACCESS_READ>);
  VecSetOp_CUPM(sum, nullptr, VecSeq_T::sum_async);
  PetscFunctionReturn(0);
}

// Called from VecGetSubVector()
template <device::cupm::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::GetArrays_CUPMBase(Vec v, const PetscScalar **host_array, const PetscScalar **device_array, PetscOffloadMask *mask, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQCUPM(), VECMPICUPM());
  if (host_array) {
    PetscCall(HostAllocateCheck_(dctx, v));
    *host_array = VecIMPLCast(v)->array;
  }
  if (device_array) {
    PetscCall(DeviceAllocateCheck_(dctx, v));
    *device_array = VecCUPMCast(v)->array_d;
  }
  if (mask) *mask = v->offloadmask;
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::ResetPreallocationCOO_CUPMBase(Vec v, PetscDeviceContext dctx)) {
  PetscFunctionBegin;
  if (const auto vcu = VecCUPMCast(v)) {
    const auto   cntptrs = util::make_array(std::ref(vcu->jmap1_d), std::ref(vcu->perm1_d), std::ref(vcu->imap2_d), std::ref(vcu->jmap2_d), std::ref(vcu->perm2_d), std::ref(vcu->Cperm_d));
    cupmStream_t stream;

    PetscCall(GetHandles_(dctx, &stream));
    for (auto &&ptr : cntptrs) PetscCallCUPM(cupmFreeAsync(ptr.get(), stream));
    for (auto &&ptr : util::make_array(std::ref(vcu->sendbuf_d), std::ref(vcu->recvbuf_d))) { PetscCallCUPM(cupmFreeAsync(ptr.get(), stream)); }
  }
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T, typename D>
template <std::size_t NCount, std::size_t NScal>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T, D>::SetPreallocationCOO_CUPMBase(Vec v, PetscCount ncoo, const PetscInt coo_i[], PetscDeviceContext dctx, const std::array<CooPair<PetscCount>, NCount> &extra_cntptrs, const std::array<CooPair<PetscScalar>, NScal> &bufptrs)) {
  const auto vimpl = VecIMPLCast(v);

  PetscFunctionBegin;
  PetscCall(ResetPreallocationCOO_CUPMBase(v, dctx));
  // need to instantiate the private pointer if not already
  PetscCall(VecCUPMAllocateCheck_(v));
  {
    cupmStream_t stream;
    const auto   vcu     = VecCUPMCast(v);
    const auto   cntptrs = util::concat_array(util::make_array(make_coo_pair(vcu->jmap1_d, vimpl->jmap1, v->map->n + 1), make_coo_pair(vcu->perm1_d, vimpl->perm1, vimpl->tot1)), extra_cntptrs);

    PetscCall(GetHandles_(dctx, &stream));
    // allocate
    for (auto &elem : cntptrs) PetscCall(PetscCUPMMallocAsync(&elem.device, elem.size, stream));
    for (auto &elem : bufptrs) PetscCall(PetscCUPMMallocAsync(&elem.device, elem.size, stream));
    // copy
    for (const auto &elem : cntptrs) { PetscCall(PetscCUPMMemcpyAsync(elem.device, elem.host, elem.size, cupmMemcpyHostToDevice, stream)); }
    for (const auto &elem : bufptrs) { PetscCall(PetscCUPMMemcpyAsync(elem.device, elem.host, elem.size, cupmMemcpyHostToDevice, stream)); }
  }
  PetscFunctionReturn(0);
}

#define PETSC_VEC_CUPM_BASE_CLASS_HEADER(name, Tp, Derived) \
  using name = ::Petsc::vec::cupm::impl::Vec_CUPMBase<Tp, Derived>; \
  friend name; \
  /* introspection */ \
  using name::VecCUPMCast; \
  using name::VecIMPLCast; \
  using name::VECTYPE; \
  using name::VECSEQCUPM; \
  using name::VECMPICUPM; \
  using name::VecView_Debug; \
  /* utility */ \
  using typename name::Vec_CUPM; \
  using name::UseCUPMHostAlloc; \
  using name::GetHandles_; \
  using name::VecCUPMAllocateCheck_; \
  using name::VecIMPLAllocateCheck_; \
  using name::HostAllocateCheck_; \
  using name::DeviceAllocateCheck_; \
  using name::CopyToDevice_; \
  using name::CopyToHost_; \
  using name::getarray_async; \
  using name::restorearray_async; \
  using name::getarrayandmemtype_async; \
  using name::restorearrayandmemtype_async; \
  using name::replacearray_async; \
  /* base functions */ \
  using name::Create_CUPMBase; \
  using name::Initialize_CUPMBase; \
  using name::Destroy_CUPMBase; \
  using name::Duplicate_CUPMBase; \
  using name::BindToCPU_CUPMBase; \
  using name::Create_CUPM; \
  using name::DeviceArrayRead; \
  using name::DeviceArrayWrite; \
  using name::DeviceArrayReadWrite; \
  using name::HostArrayRead; \
  using name::HostArrayWrite; \
  using name::HostArrayReadWrite; \
  using name::ResetArray_CUPMBase; \
  using name::PlaceArray_CUPMBase; \
  using name::ResetPreallocationCOO_CUPMBase; \
  using name::SetPreallocationCOO_CUPMBase; \
  /* blas interface */ \
  PETSC_CUPMBLAS_INHERIT_INTERFACE_TYPEDEFS_USING(cupmBlasInterface_t, Tp)

} // namespace impl

} // namespace cupm

} // namespace vec

} // namespace Petsc

#endif // __cplusplus && PetscDefined(HAVE_DEVICE)

#endif // PETSCVECCUPMIMPL_H
