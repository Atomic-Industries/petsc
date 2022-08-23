#ifndef PETSCVECSEQCUPM_HPP
#define PETSCVECSEQCUPM_HPP

#define PETSC_SKIP_SPINLOCK // REVIEW ME: why

#include <petsc/private/veccupmimpl.h> /*I <petscvec.h> I*/
#include <petsc/private/randomimpl.h>  // for _p_PetscRandom

#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

// TODO
// - refactor the AXPY's for code reuse
// - figure out how to template which thrust namespace to use so we can do
//   thrust::<backend>::par.on(stream)
// - maybe reintroduce PetscDeviceMalloc()?
// - There is also an overloaded version of cudaMallocAsync that takes the same arguments as
//   cudaMallocFromPoolAsync
// - touch up the docs for both implementations
// - pick one of the VecGetArray<modifier>() to explain data movement semantics in the docs and
//   have everyone else refer to it
// - remove the cuda and hip separate versions
// - remove bindtocpu?
// - do rocblas instead of hipblas
// - remove this define and use the right header (i.e. clean up the headers first)

namespace Petsc {

namespace Vector {

namespace CUPM {

namespace Impl {

namespace {

template <bool>
struct UseComplexTag { };

} // anonymous namespace

template <Device::CUPM::DeviceType T>
struct VecSeq_CUPM : Vec_CUPMBase<T, VecSeq_CUPM<T>> {
  PETSC_VEC_CUPM_BASE_CLASS_HEADER(base_type, T, VecSeq_CUPM<T>);

private:
  PETSC_CXX_COMPAT_DECL(constexpr auto VecIMPLCast_(Vec v)) PETSC_DECLTYPE_AUTO_RETURNS(static_cast<Vec_Seq *>(v->data));
  PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 auto VECTYPE_()) PETSC_DECLTYPE_AUTO_RETURNS(VECSEQCUPM());

  // common core for min and max
  template <typename TupleFuncT, typename UnaryFuncT>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode minmax_async_(TupleFuncT &&, UnaryFuncT &&, PetscReal, Vec, PetscInt *, PetscReal *));
  // common core for pointwise binary and pointwise unary thrust functions
  template <typename BinaryFuncT>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwisebinary_async_(BinaryFuncT &&, Vec, Vec, Vec));
  template <typename UnaryFuncT>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwiseunary_async_(UnaryFuncT &&, Vec, Vec /*out*/ = nullptr));
  // mdot dispatchers
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_async_(UseComplexTag<true>, Vec, PetscInt, const Vec[], PetscScalar *));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_async_(UseComplexTag<false>, Vec, PetscInt, const Vec[], PetscScalar *));
  // dispatcher for the actual kernels for mdot when NOT configured for complex, called by
  // mdot_async_(use_complex_tag<false>,...)
  template <int>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_kernel_dispatch_(PetscDeviceContext, cupmStream_t, const PetscScalar *, const Vec[], PetscInt, PetscScalar **, PetscScalar *, PetscInt *));
  // common core for the various create routines
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createseqcupm_async_(Vec, PetscScalar * /*host_ptr*/ = nullptr, PetscScalar * /*device_ptr*/ = nullptr));

public:
  // callable directly via a bespoke function
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create_async(Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createseqcupm_async(MPI_Comm, PetscInt, PetscInt, Vec *, PetscBool));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createseqcupmwithbotharrays_async(MPI_Comm, PetscInt, PetscInt, const PetscScalar[], const PetscScalar[], Vec *));

  // callable indirectly via function pointers
  PETSC_CXX_COMPAT_DECL(PetscErrorCode duplicate_async(Vec, Vec *));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode aypx_async(Vec, PetscScalar, Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode axpy_async(Vec, PetscScalar, Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwisedivide_async(Vec, Vec, Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwisemult_async(Vec, Vec, Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode reciprocal_async(Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode waxpy_async(Vec, PetscScalar, Vec, Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode maxpy_async(Vec, PetscInt, const PetscScalar *, Vec *));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode dot_async(Vec, Vec, PetscScalar *));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_async(Vec, PetscInt, const Vec[], PetscScalar *));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode set_async(Vec, PetscScalar));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode scale_async(Vec, PetscScalar));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode tdot_async(Vec, Vec, PetscScalar *));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode copy_async(Vec, Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode swap_async(Vec, Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode axpby_async(Vec, PetscScalar, PetscScalar, Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode axpbypcz_async(Vec, PetscScalar, PetscScalar, PetscScalar, Vec, Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode norm_async(Vec, NormType, PetscReal *));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode dotnorm2_async(Vec, Vec, PetscScalar *, PetscScalar *));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy_async(Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode conjugate_async(Vec));
  template <MemoryAccess>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getlocalvector_async(Vec, Vec));
  template <MemoryAccess>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode restorelocalvector_async(Vec, Vec));
  template <PetscMemType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode resetarray_async(Vec));
  template <PetscMemType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode placearray_async(Vec, const PetscScalar *));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode max_async(Vec, PetscInt *, PetscReal *));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode min_async(Vec, PetscInt *, PetscReal *));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode sum_async(Vec, PetscScalar *));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode shift_async(Vec, PetscScalar));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode setrandom_async(Vec, PetscRandom));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode bindtocpu_async(Vec, PetscBool));
};

// ================================================================================== //
//                                                                                    //
//                                  utility methods                                   //
//                                                                                    //
// ================================================================================== //

// ================================================================================== //
//                                  array accessors                                   //

#define CHKERRTHRUST(...) \
  do { \
    try { \
      __VA_ARGS__; \
    } catch (const thrust::system_error &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Thrust error: %s", ex.what()); } \
  } while (0)

namespace {

struct PetscLogGpuTime_ {
  /* lets hope barry doesn't notice these */
  PetscLogGpuTime_() noexcept { PetscCallAbort(PETSC_COMM_SELF, PetscLogGpuTimeBegin()); }
  ~PetscLogGpuTime_() noexcept { PetscCallAbort(PETSC_COMM_SELF, PetscLogGpuTimeEnd()); }
};

} // anonymous namespace

#if PetscDefined(USING_NVCC)
#if !PetscDefined(USE_DEBUG) && (THRUST_VERSION >= 101600)
#define thrust_call_par_on(func, s, ...) func(thrust::cuda::par_nosync.on(s), __VA_ARGS__)
#else
#define thrust_call_par_on(func, s, ...) func(thrust::cuda::par.on(s), __VA_ARGS__)
#endif
#elif PetscDefined(USING_HCC) // rocThrust has no par_nosync
#define thrust_call_par_on(func, s, ...) func(thrust::hip::par.on(s), __VA_ARGS__)
#else
#define thrust_call_par_on(func, s, ...) func(__VA_ARGS__)
#endif

#define THRUST_CALL(...) \
  [&] { \
    const auto timer = PetscLogGpuTime_{}; \
    return thrust_call_par_on(__VA_ARGS__); \
  }()

template <Device::CUPM::DeviceType T>
template <typename BinaryFuncT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwisebinary_async_(BinaryFuncT &&binary, Vec win, Vec xin, Vec yin)) {
  const auto         n = xin->map->n;
  PetscDeviceContext dctx;
  cupmStream_t       stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx, &stream));
  CHKERRTHRUST(auto xptr = thrust::device_pointer_cast(DeviceArrayRead(dctx, xin).ptr); auto yptr = thrust::device_pointer_cast(DeviceArrayRead(dctx, yin).ptr); auto wptr = thrust::device_pointer_cast(DeviceArrayWrite(dctx, win).ptr);

               THRUST_CALL(thrust::transform, stream, xptr, xptr + n, yptr, wptr, std::forward<BinaryFuncT>(binary)););
  PetscCall(PetscLogGpuFlops(n));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
template <typename UnaryFuncT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwiseunary_async_(UnaryFuncT &&unary, Vec xin, Vec yin)) {
  const auto         n = xin->map->n;
  PetscDeviceContext dctx;
  cupmStream_t       stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx, &stream));
  if (xin == yin || !yin) { // in-place
    CHKERRTHRUST(auto xptr = thrust::device_pointer_cast(DeviceArrayReadWrite(dctx, xin).ptr);

                 THRUST_CALL(thrust::transform, stream, xptr, xptr + n, xptr, std::forward<UnaryFuncT>(unary)););
  } else {
    CHKERRTHRUST(auto xptr = thrust::device_pointer_cast(DeviceArrayRead(dctx, xin).ptr); auto yptr = thrust::device_pointer_cast(DeviceArrayWrite(dctx, yin).ptr);

                 THRUST_CALL(thrust::transform, stream, xptr, xptr + n, yptr, std::forward<UnaryFuncT>(unary)););
  }
  PetscCall(PetscLogGpuFlops(n));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::createseqcupm_async_(Vec v, PetscScalar *host_array, PetscScalar *device_array)) {
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm(PetscObjectCast(v)), &size));
  PetscCheck(size <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Must create VecSeq on communicator of size 1, have size %d", size);
  // REVIEW ME: remove me
  PetscCheck(!VecIMPLCast(v), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Creating VecSeq for the second time!");
  PetscCall(VecCreate_Seq_Private(v, host_array));
  PetscCall(Initialize_CUPMBase(v, PETSC_FALSE, host_array, device_array));
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                                                                    //
//                                  public methods                                    //
//                                                                                    //
// ================================================================================== //

// ================================================================================== //
//                             constructors/destructors                               //

// v->ops->create
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::create_async(Vec v)) {
  PetscFunctionBegin;
  PetscCall(createseqcupm_async_(v));
  PetscFunctionReturn(0);
}

// VecCreateSeqCUPM()
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::createseqcupm_async(MPI_Comm comm, PetscInt bs, PetscInt n, Vec *v, PetscBool call_set_type)) {
  PetscFunctionBegin;
  PetscCall(Create_CUPMBase(comm, bs, n, n, v, call_set_type));
  PetscFunctionReturn(0);
}

// VecCreateSeqCUPMWithArrays()
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::createseqcupmwithbotharrays_async(MPI_Comm comm, PetscInt bs, PetscInt n, const PetscScalar host_array[], const PetscScalar device_array[], Vec *v)) {
  PetscFunctionBegin;
  // do NOT call VecSetType(), otherwise ops->create() -> create_async() ->
  // createseqcupm_async_() is called!
  PetscCall(createseqcupm_async(comm, bs, n, v, PETSC_FALSE));
  PetscCall(createseqcupm_async_(*v, PetscRemoveConstCast(host_array), PetscRemoveConstCast(device_array)));
  PetscFunctionReturn(0);
}

// v->ops->duplicate
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::duplicate_async(Vec v, Vec *y)) {
  PetscFunctionBegin;
  PetscCall(Duplicate_CUPMBase(v, y));
  PetscFunctionReturn(0);
}

// v->ops->destroy
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::destroy_async(Vec v)) {
  PetscFunctionBegin;
  PetscCall(Destroy_CUPMBase(v));
  PetscCall(VecDestroy_Seq(v));
  PetscFunctionReturn(0);
}

// v->ops->bindtocpu
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::bindtocpu_async(Vec v, PetscBool usehost)) {
  PetscFunctionBegin;
  PetscCall(BindToCPU_CUPMBase(v, usehost));

  // REVIEW ME: this absolutely should be some sort of bulk mempcy rather than this mess
  VecSetOp_CUPM(dot, VecDot_Seq, dot_async);
  VecSetOp_CUPM(norm, VecNorm_Seq, norm_async);
  VecSetOp_CUPM(tdot, VecTDot_Seq, tdot_async);
  VecSetOp_CUPM(mdot, VecMDot_Seq, mdot_async);
  VecSetOp_CUPM(resetarray, VecResetArray_Seq, resetarray_async<PETSC_MEMTYPE_HOST>);
  VecSetOp_CUPM(placearray, VecPlaceArray_Seq, placearray_async<PETSC_MEMTYPE_HOST>);
  v->ops->mtdot = v->ops->mtdot_local = VecMTDot_Seq;
  VecSetOp_CUPM(conjugate, VecConjugate_Seq, conjugate_async);
  VecSetOp_CUPM(max, VecMax_Seq, max_async);
  VecSetOp_CUPM(min, VecMin_Seq, min_async);
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                    mutatators                                      //

// v->ops->getlocalvector or v->ops->getlocalvectorread
template <Device::CUPM::DeviceType T>
template <MemoryAccess access>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::getlocalvector_async(Vec v, Vec w)) {
  PetscBool wisseqcupm;

  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQCUPM(), VECMPICUPM());
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(w), VECSEQCUPM(), &wisseqcupm));
  if (wisseqcupm) {
    if (const auto wseq = VecIMPLCast(w)) {
      if (wseq->array_allocated) {
        const auto useit = UseCUPMHostAlloc(w);

        PetscCall(PetscFree(wseq->array_allocated));
        w->pinned_memory = PETSC_FALSE;
      }
      wseq->array = wseq->unplacedarray = nullptr;
    }
    if (const auto wcu = VecCUPMCast(w)) {
      if (auto device_array = wcu->device_array) {
        cupmStream_t stream;

        PetscCall(GetHandles_(&stream));
        PetscCallCUPM(cupmFreeAsync(device_array, stream));
      }
      PetscCall(PetscFree(w->spptr /* wcu */));
    }
  }
  if (v->petscnative && wisseqcupm) {
    PetscCall(PetscFree(w->data));
    w->data          = v->data;
    w->offloadmask   = v->offloadmask;
    w->pinned_memory = v->pinned_memory;
    w->spptr         = v->spptr;
    PetscCall(PetscObjectStateIncrease(PetscObjectCast(w)));
  } else {
    const auto arrayptr = &VecIMPLCast(w)->array;
    if (access == MemoryAccess::READ) {
      PetscCall(VecGetArrayRead(v, const_cast<const PetscScalar **>(arrayptr)));
    } else {
      PetscCall(VecGetArray(v, arrayptr));
    }
    w->offloadmask = PETSC_OFFLOAD_CPU;
    if (wisseqcupm) {
      PetscDeviceContext dctx;

      PetscCall(GetHandles_(&dctx));
      PetscCall(DeviceAllocateCheck_(dctx, w));
    }
  }
  PetscFunctionReturn(0);
}

// v->ops->restorelocalvector or v->ops->restorelocalvectorread
template <Device::CUPM::DeviceType T>
template <MemoryAccess access>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::restorelocalvector_async(Vec v, Vec w)) {
  PetscBool wisseqcupm;

  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQCUPM(), VECMPICUPM());
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(w), VECSEQCUPM(), &wisseqcupm));
  if (v->petscnative && wisseqcupm) {
    v->data          = w->data;
    v->offloadmask   = w->offloadmask;
    v->pinned_memory = w->pinned_memory;
    v->spptr         = w->spptr;
    w->data          = nullptr;
    w->spptr         = nullptr;
    w->offloadmask   = PETSC_OFFLOAD_UNALLOCATED;
  } else {
    auto array = &VecIMPLCast(w)->array;
    if (access == MemoryAccess::READ) {
      PetscCall(VecRestoreArrayRead(v, const_cast<const PetscScalar **>(array)));
    } else {
      PetscCall(VecRestoreArray(v, array));
    }
    if (w->spptr && wisseqcupm) {
      cupmStream_t stream;

      PetscCall(GetHandles_(&stream));
      PetscCallCUPM(cupmFreeAsync(VecCUPMCast(w)->device_array, stream));
      PetscCall(PetscFree(w->spptr));
    }
  }
  PetscFunctionReturn(0);
}

// v->ops->resetarray or VecCUPMResetArray()
template <Device::CUPM::DeviceType T>
template <PetscMemType mtype>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::resetarray_async(Vec v)) {
  PetscFunctionBegin;
  PetscCall(base_type::template ResetArray_CUPMBase<mtype>(v, VecResetArray_Seq));
  PetscFunctionReturn(0);
}

// v->ops->placearray or VecCUPMPlaceArray()
template <Device::CUPM::DeviceType T>
template <PetscMemType mtype>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::placearray_async(Vec v, const PetscScalar *a)) {
  PetscFunctionBegin;
  PetscCall(base_type::template PlaceArray_CUPMBase<mtype>(v, a, VecPlaceArray_Seq));
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                   compute methods                                  //

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::aypx_async(Vec yin, PetscScalar alpha, Vec xin)) {
  const auto         n = static_cast<cupmBlasInt_t>(yin->map->n);
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) {
    const auto   nbytes = n * sizeof(PetscScalar);
    cupmStream_t stream;

    PetscCall(GetHandles_(&dctx, &stream));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUPM(cupmMemcpyAsync(DeviceArrayWrite(dctx, yin).ptr, DeviceArrayRead(dctx, xin).ptr, nbytes, cupmMemcpyDeviceToDevice, stream));
    PetscCall(PetscLogGpuTimeEnd());
  } else {
    const auto       alphaIsOne = alpha == PetscScalar(1.0);
    cupmBlasHandle_t cupmBlasHandle;

    PetscCall(GetHandles_(&dctx, &cupmBlasHandle));
    {
      const auto calpha = makeCupmScalar(alpha);
      const auto yarray = DeviceArrayReadWrite(dctx, yin);
      const auto xarray = DeviceArrayRead(dctx, xin);

      PetscCallCUPMBLAS(cupmBlasSetPointerMode(cupmBlasHandle, CUPMBLAS_POINTER_MODE_HOST));
      PetscCall(PetscLogGpuTimeBegin());
      if (alphaIsOne) {
        PetscCallCUPMBLAS(cupmBlasXaxpy(cupmBlasHandle, n, &calpha, xarray, 1, yarray, 1));
      } else {
        const auto sone = makeCupmScalar(1.0);

        PetscCallCUPMBLAS(cupmBlasXscal(cupmBlasHandle, n, &calpha, yarray, 1));
        PetscCallCUPMBLAS(cupmBlasXaxpy(cupmBlasHandle, n, &sone, xarray, 1, yarray, 1));
      }
      PetscCall(PetscLogGpuTimeEnd());
    }
    PetscCall(PetscLogGpuFlops((alphaIsOne ? 1 : 2) * n));
    PetscCall(PetscLogCpuToGpuScalar(sizeof(alpha)));
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::axpy_async(Vec yin, PetscScalar alpha, Vec xin)) {
  PetscBool xiscupm;

  PetscFunctionBegin;
  if (PetscUnlikely(alpha == PetscScalar(0.0))) PetscFunctionReturn(0);
  PetscCall(PetscObjectTypeCompareAny(PetscObjectCast(xin), &xiscupm, VECSEQCUPM(), VECMPICUPM(), ""));
  if (xiscupm) {
    const auto         n      = static_cast<cupmBlasInt_t>(yin->map->n);
    const auto         calpha = makeCupmScalar(alpha);
    cupmBlasHandle_t   cupmBlasHandle;
    PetscDeviceContext dctx;

    PetscCall(GetHandles_(&dctx, &cupmBlasHandle));
    PetscCallCUPMBLAS(cupmBlasSetPointerMode(cupmBlasHandle, CUPMBLAS_POINTER_MODE_HOST));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUPMBLAS(cupmBlasXaxpy(cupmBlasHandle, n, &calpha, DeviceArrayRead(dctx, xin), 1, DeviceArrayReadWrite(dctx, yin), 1));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(2 * n));
    PetscCall(PetscLogCpuToGpuScalar(sizeof(alpha)));
  } else {
    PetscCall(VecAXPY_Seq(yin, alpha, xin));
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwisedivide_async(Vec win, Vec xin, Vec yin)) {
  PetscFunctionBegin;
  if (xin->boundtocpu || yin->boundtocpu) {
    PetscCall(VecPointwiseDivide_Seq(win, xin, yin));
  } else {
    PetscCall(pointwisebinary_async_(thrust::divides<PetscScalar>(), win, xin, yin));
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwisemult_async(Vec win, Vec xin, Vec yin)) {
  PetscFunctionBegin;
  if (xin->boundtocpu || yin->boundtocpu) {
    PetscCall(VecPointwiseMult_Seq(win, xin, yin));
  } else {
    PetscCall(pointwisebinary_async_(thrust::multiplies<PetscScalar>(), win, xin, yin));
  }
  PetscFunctionReturn(0);
}

namespace detail {

struct reciprocal {
  PETSC_HOSTDEVICE_DECL PetscScalar operator()(PetscScalar s) const {
    // yes all of this verbosity is needed because sometimes PetscScalar is a thrust::complex
    // and then it matter whether we do s ? true : false vs s == 0, as well as whether we wrap
    // everything in PetscScalar...
    return s == PetscScalar{0.0} ? s : PetscScalar{1.0} / s;
  }
};

} // namespace detail

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::reciprocal_async(Vec xin)) {
  PetscFunctionBegin;
  PetscCall(pointwiseunary_async_(detail::reciprocal(), xin));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::waxpy_async(Vec win, PetscScalar alpha, Vec xin, Vec yin)) {
  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) {
    PetscCall(copy_async(yin, win));
  } else {
    const auto         n      = win->map->n;
    const auto         calpha = makeCupmScalar(alpha);
    PetscDeviceContext dctx;
    cupmBlasHandle_t   cupmBlasHandle;
    cupmStream_t       stream;

    PetscCall(GetHandles_(&dctx, &cupmBlasHandle, &stream));
    PetscCallCUPMBLAS(cupmBlasSetPointerMode(cupmBlasHandle, CUPMBLAS_POINTER_MODE_HOST));
    PetscCall(PetscLogGpuTimeBegin());
    {
      const auto warray = DeviceArrayWrite(dctx, win);
      PetscCallCUPM(cupmMemcpyAsync(warray.ptr, DeviceArrayRead(dctx, yin).ptr, n * sizeof(typename decltype(warray)::value_type), cupmMemcpyDeviceToDevice, stream));
      PetscCallCUPMBLAS(cupmBlasXaxpy(cupmBlasHandle, n, &calpha, DeviceArrayRead(dctx, xin), 1, warray, 1));
    }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(2 * n));
    PetscCall(PetscLogCpuToGpuScalar(sizeof(calpha)));
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::maxpy_async(Vec xin, PetscInt nv, const PetscScalar *alpha, Vec *y)) {
  const auto         n = xin->map->n;
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx, &cupmBlasHandle));
  {
    const auto xarray = DeviceArrayReadWrite(dctx, xin);

    PetscCall(cupmBlasSetPointerModeFromPointer(cupmBlasHandle, alpha));
    PetscCall(PetscLogGpuTimeBegin());
    for (decltype(nv) j = 0; j < nv; ++j) { PetscCallCUPMBLAS(cupmBlasXaxpy(cupmBlasHandle, n, cupmScalarCast(alpha + j), DeviceArrayRead(dctx, y[j]), 1, xarray, 1)); }
    PetscCall(PetscLogGpuTimeEnd());
  }
  PetscCall(PetscLogGpuFlops(nv * 2 * n));
  PetscCall(PetscLogCpuToGpuScalar(nv * sizeof(*alpha)));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::dot_async(Vec xin, Vec yin, PetscScalar *z)) {
  const auto         n = xin->map->n;
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx, &cupmBlasHandle));
  // arguments y, x are reversed because BLAS complex conjugates the first argument, PETSc the
  // second
  PetscCall(cupmBlasSetPointerModeFromPointer(cupmBlasHandle, z));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUPMBLAS(cupmBlasXdot(cupmBlasHandle, n, DeviceArrayRead(dctx, yin), 1, DeviceArrayRead(dctx, xin), 1, cupmScalarCast(z)));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(PetscMax(2 * n - 1, 0)));
  PetscCall(PetscLogGpuToCpuScalar(sizeof(*z)));
  PetscFunctionReturn(0);
}

#define MDOT_WORKGROUP_NUM  128
#define MDOT_WORKGROUP_SIZE MDOT_WORKGROUP_NUM

namespace kernels {

PETSC_HOSTDEVICE_DECL static PetscInt EntriesPerGroup(PetscInt size) {
  const auto group_entries = (size - 1) / gridDim.x + 1;
  // for very small vectors, a group should still do some work
  return group_entries ? group_entries : 1;
}

template <int N>
PETSC_KERNEL_DECL static void mdot_kernel(const PetscScalar *PETSC_RESTRICT x, const PetscScalar *PETSC_RESTRICT y[N], PetscInt size, PetscScalar *PETSC_RESTRICT results) {
  static_assert(N > 1, "");
  PETSC_SHAREDMEM_DECL PetscScalar shmem[N * MDOT_WORKGROUP_SIZE];
  // HIP -- for whatever reason -- has threadIdx, blockIdx, blockDim, and gridDim as separate
  // types, so each of these go on separate lines...
  const auto                       tx       = threadIdx.x;
  const auto                       bx       = blockIdx.x;
  const auto                       bdx      = blockDim.x;
  const auto                       gdx      = gridDim.x;
  const auto                       worksize = EntriesPerGroup(size);
  const auto                       begin    = tx + bx * worksize;
  const auto                       end      = min((bx + 1) * worksize, size);
  const PetscScalar               *ylocal[N];
  PetscScalar                      sumlocal[N];

#pragma unroll
  for (auto i = 0; i < N; ++i) {
    sumlocal[i] = 0;
    ylocal[i]   = y[i]; // load pointer once
  }

#pragma unroll
  for (auto i = begin; i < end; i += bdx) {
    const auto xi = x[i]; // load only once from global memory!

#pragma unroll
    for (auto j = 0; j < N; ++j) sumlocal[j] += ylocal[j][i] * xi;
  }

#pragma unroll
  for (auto i = 0; i < N; ++i) shmem[tx + i * MDOT_WORKGROUP_SIZE] = sumlocal[i];

    // parallel reduction
#pragma unroll
  for (auto stride = bdx / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if (tx < stride) {
#pragma unroll
      //for (auto i = tx; i < N; i += MDOT_WORKGROUP_SIZE) shmem[i] += shmem[i+stride];
      for (auto i = 0; i < N; ++i) shmem[tx + i * MDOT_WORKGROUP_SIZE] += shmem[tx + stride + i * MDOT_WORKGROUP_SIZE];
    }
  }
  // bottom N threads per block write to global memory
  // REVIEW ME: I am ~pretty~ sure we don't need another __syncthreads() here since each thread
  // writes to the same sections in the above loop that it is about to read from below
  if (tx < N) results[bx + tx * gdx] = shmem[tx * MDOT_WORKGROUP_SIZE];
  return;
}

} // namespace kernels

template <Device::CUPM::DeviceType T>
template <int N>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_kernel_dispatch_(PetscDeviceContext dctx, cupmStream_t stream, const PetscScalar *xarr, const Vec yin[], PetscInt size, PetscScalar **device_y, PetscScalar *results, PetscInt *yidx)) {
  const auto   yidxt = *yidx;
  const auto   yint  = yin + yidxt;
  PetscScalar *host_y[N];

  PetscFunctionBegin;
  for (auto i = 0; i < N; ++i) host_y[i] = DeviceArrayRead(dctx, yint[i]);
  PetscCallCUPM(cupmMemcpyAsync(device_y, host_y, N * sizeof(*device_y), cupmMemcpyHostToDevice, stream));
  PetscCallCUPM(cupmLaunchKernel(kernels::mdot_kernel<N>, dim3(MDOT_WORKGROUP_NUM), dim3(MDOT_WORKGROUP_SIZE), 0, stream, xarr, device_y, size, results + (yidxt * MDOT_WORKGROUP_NUM)));
  *yidx += N;
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_async_(UseComplexTag<false>, Vec xin, PetscInt nv, const Vec yin[], PetscScalar *z)) {
  const auto         n      = xin->map->n;
  const auto         nv1    = ((nv % 4) == 1) ? nv - 1 : nv;
  const auto         nwork  = nv1 * MDOT_WORKGROUP_NUM;
  const auto         nbytes = nwork * sizeof(PetscScalar);
  PetscBool          device_mem;
  PetscScalar      **d_y;
  PetscScalar       *d_results;
  PetscDeviceContext dctx;
  cupmStream_t       stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx, &stream));
  // will hold all the device y pointers
  PetscCallCUPM(cupmMallocAsync(reinterpret_cast<void **>(&d_y), 8 * sizeof(*d_y), stream));
  // allocate scratchpad memory for the results of individual work groups
  PetscCallCUPM(cupmMallocAsync(reinterpret_cast<void **>(&d_results), nbytes, stream));
  {
    auto yidx = PetscInt{0};
    auto xptr = DeviceArrayRead(dctx, xin);

    PetscCall(PetscLogGpuTimeBegin());
    // REVIEW ME: Can fork-join here, but should probably only have a single-sized kernel then
    while (yidx < nv) {
      switch (nv - yidx) {
      case 7:
      case 6:
      case 5:
      case 4: PetscCall(mdot_kernel_dispatch_<4>(dctx, stream, xptr, yin, n, d_y, d_results, &yidx)); break;
      case 3: PetscCall(mdot_kernel_dispatch_<3>(dctx, stream, xptr, yin, n, d_y, d_results, &yidx)); break;
      case 2: PetscCall(mdot_kernel_dispatch_<2>(dctx, stream, xptr, yin, n, d_y, d_results, &yidx)); break;
      case 1: {
        cupmBlasHandle_t handle;

        PetscCall(PetscDeviceContextGetBLASHandle_Internal(dctx, &handle));
        PetscCall(cupmBlasSetPointerModeFromPointer(handle, z));
        PetscCallCUPMBLAS(cupmBlasXdot(handle, n, DeviceArrayRead(dctx, yin[yidx]), 1, xptr, 1, cupmScalarCast(z + yidx)));
        ++yidx;
      }
      case 0: break;
      default: // 8 or more
        PetscCall(mdot_kernel_dispatch_<8>(dctx, stream, xptr, yin, n, d_y, d_results, &yidx));
        break;
      }
    }
    PetscCall(PetscLogGpuTimeEnd());
  }
  PetscCall(IsDeviceMemory(z, &device_mem));
  // copy results to CPU
  if (device_mem) {
    // REVIEW ME: TODO
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Device memory MDOT implementation pending");
  } else {
    auto       stackarray = std::array<PetscScalar, PETSC_MAX_PATH_LEN>{};
    const auto allocate   = static_cast<decltype(stackarray.size())>(nwork) > stackarray.size();
    auto       h_results  = stackarray.data();

    if (allocate) PetscCall(PetscMalloc1(nwork, &h_results));
    PetscCallCUPM(cupmMemcpyAsync(h_results, d_results, nbytes, cupmMemcpyDeviceToHost, stream));
    // do these now while memcpy is in flight
    PetscCall(PetscLogFlops(nwork));
    PetscCall(PetscLogGpuToCpuScalar(nbytes));
    // REVIEW ME: need to hard sync here...
    PetscCall(PetscDeviceContextSynchronize(dctx));
    // REVIEW ME: it is likely faster to do this in a micro kernel rather than do it on the
    // host which requires synchronization and possibly an additional allocation
    // sum group results into z
    for (auto j = PetscInt{0}; j < nv1; ++j) {
      for (auto i = j * MDOT_WORKGROUP_NUM; i < (j + 1) * MDOT_WORKGROUP_NUM; ++i) z[j] += h_results[i];
    }
    if (allocate) PetscCall(PetscFree(h_results));
  }
  // free these here, we will already have synchronized
  PetscCallCUPM(cupmFreeAsync(d_results, stream));
  PetscCallCUPM(cupmFreeAsync(d_y, stream));
  PetscFunctionReturn(0);
}

#undef MDOT_WORKGROUP_NUM
#undef MDOT_WORKGROUP_SIZE

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_async_(UseComplexTag<true>, Vec xin, PetscInt nv, const Vec yin[], PetscScalar *z)) {
  const auto         n = static_cast<cupmBlasInt_t>(xin->map->n);
  PetscBool          device_mem;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx));
  PetscCall(IsDeviceMemory(z, &device_mem));
  {
    const auto          xptr  = DeviceArrayRead(dctx, xin);
    const auto          mode  = device_mem ? CUPMBLAS_POINTER_MODE_DEVICE : CUPMBLAS_POINTER_MODE_HOST;
    // probably not worth it to run more than 8 of these at a time?
    const auto          n_sub = PetscMin(nv, 8);
    PetscDeviceContext *subctx;

    PetscCall(PetscDeviceContextFork(dctx, n_sub, &subctx));
    PetscCall(PetscLogGpuTimeBegin());
    for (auto i = PetscInt{0}; i < nv; ++i) {
      cupmBlasHandle_t handle;

      PetscCall(PetscDeviceContextGetBLASHandle_Internal(subctx[i % n_sub], &handle));
      PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, mode));
      PetscCallCUPMBLAS(cupmBlasXdot(handle, n, DeviceArrayRead(dctx, yin[i]), 1, xptr, 1, cupmScalarCast(z + i)));
    }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscDeviceContextJoin(dctx, n_sub, PETSC_DEVICE_CONTEXT_JOIN_DESTROY, &subctx));
  }
  // REVIEW ME: flops?????
  PetscCall(PetscLogGpuToCpuScalar(nv * sizeof(*z)));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_async(Vec xin, PetscInt nv, const Vec yin[], PetscScalar *z)) {
  const auto n = xin->map->n;

  PetscFunctionBegin;
  PetscCheck(nv > 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Number of vectors provided to %s %" PetscInt_FMT " not positive", PETSC_FUNCTION_NAME, nv);
  if (PetscUnlikely(nv == 1)) {
    PetscCall(dot_async(xin, PetscRemoveConstCast(yin[0]), z));
    PetscFunctionReturn(0);
  }
  // z will always need to be zeroed first, either for a quick return or for summing later on
  PetscCall(PetscArrayzero(z, nv));
  // nothing to do if x has no entries
  if (n) {
    PetscCall(mdot_async_(UseComplexTag<PetscDefined(USE_COMPLEX)>{}, xin, nv, yin, z));
    // REVIEW ME: double count of flops??
    PetscCall(PetscLogGpuFlops(PetscMax(nv * (2.0 * n - 1), 0.0)));
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::set_async(Vec xin, PetscScalar alpha)) {
  const auto         n = xin->map->n;
  cupmStream_t       stream;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx, &stream));
  if (alpha == PetscScalar(0)) {
    PetscCallCUPM(cupmMemsetAsync(DeviceArrayWrite(dctx, xin).ptr, 0, n * sizeof(PetscScalar), stream));
  } else {
    CHKERRTHRUST(auto xptr = thrust::device_pointer_cast(DeviceArrayWrite(dctx, xin).ptr);

                 THRUST_CALL(thrust::fill, stream, xptr, xptr + n, alpha););
    PetscCall(PetscLogCpuToGpuScalar(sizeof(alpha)));
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::scale_async(Vec xin, PetscScalar alpha)) {
  PetscFunctionBegin;
  if (alpha == PetscScalar(1.0)) PetscFunctionReturn(0);
  else if (alpha == PetscScalar(0.0)) {
    PetscCall(set_async(xin, alpha));
  } else {
    const auto         n      = static_cast<cupmBlasInt_t>(xin->map->n);
    const auto         calpha = makeCupmScalar(alpha);
    PetscDeviceContext dctx;
    cupmBlasHandle_t   cupmBlasHandle;

    PetscCall(GetHandles_(&dctx, &cupmBlasHandle));
    PetscCallCUPMBLAS(cupmBlasSetPointerMode(cupmBlasHandle, CUPMBLAS_POINTER_MODE_HOST));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUPMBLAS(cupmBlasXscal(cupmBlasHandle, n, &calpha, DeviceArrayReadWrite(dctx, xin), 1));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogCpuToGpuScalar(sizeof(calpha)));
    PetscCall(PetscLogGpuFlops(n));
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::tdot_async(Vec xin, Vec yin, PetscScalar *z)) {
  const auto         n = static_cast<cupmBlasInt_t>(xin->map->n);
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx, &cupmBlasHandle));
  PetscCall(cupmBlasSetPointerModeFromPointer(cupmBlasHandle, z));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUPMBLAS(cupmBlasXdotu(cupmBlasHandle, n, DeviceArrayRead(dctx, xin), 1, DeviceArrayRead(dctx, yin), 1, cupmScalarCast(z)));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(PetscMax(2 * n - 1, 0)));
  PetscCall(PetscLogGpuToCpuScalar(sizeof(*z)));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::copy_async(Vec xin, Vec yin)) {
  PetscFunctionBegin;
  if (xin != yin) {
    const auto         n       = xin->map->n;
    const auto         nbytes  = n * sizeof(*VecIMPLCast(xin)->array);
    auto               yiscupm = PETSC_TRUE, xondevice = PETSC_TRUE; // assume we start on device
    // silence buggy gcc warning: ‘mode’ may be used uninitialized in this function
    cupmMemcpyKind_t   mode = cupmMemcpyDeviceToDevice;
    PetscDeviceContext dctx;
    cupmStream_t       stream;

    switch (xin->offloadmask) {
    case PETSC_OFFLOAD_KOKKOS:                       // technically an error
    case PETSC_OFFLOAD_UNALLOCATED:                  // technically an error
    case PETSC_OFFLOAD_CPU: xondevice = PETSC_FALSE; // we assumed partially wrong
    case PETSC_OFFLOAD_GPU:
    case PETSC_OFFLOAD_BOTH:
      break;
      // no default case so warnings are thrown for new offloadmasks
    }

    switch (yin->offloadmask) {
    case PETSC_OFFLOAD_KOKKOS:
    case PETSC_OFFLOAD_UNALLOCATED:
    case PETSC_OFFLOAD_CPU: PetscCall(PetscObjectTypeCompareAny(PetscObjectCast(yin), &yiscupm, VECSEQCUPM(), VECMPICUPM(), ""));
    case PETSC_OFFLOAD_GPU:
    case PETSC_OFFLOAD_BOTH:
      if (yiscupm) { // PETSC_TRUE by default (unless on the host)
        // even though y may be on the host, its a cupm vector, so it ought to be on the device
        mode = xondevice ? cupmMemcpyDeviceToDevice : cupmMemcpyHostToDevice;
      } else {
        // we assumed really wrong
        mode = xondevice ? cupmMemcpyDeviceToHost : cupmMemcpyHostToHost;
      }
      break;
    }

    PetscCall(GetHandles_(&dctx, &stream));
    switch (mode) {
    case cupmMemcpyDeviceToDevice:
      // the best case
      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUPM(cupmMemcpyAsync(DeviceArrayWrite(dctx, yin).ptr, DeviceArrayRead(dctx, xin).ptr, nbytes, mode, stream));
      PetscCall(PetscLogGpuTimeEnd());
      break;
    case cupmMemcpyHostToDevice:
      // not terrible
      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUPM(cupmMemcpyAsync(DeviceArrayWrite(dctx, yin).ptr, HostArrayRead(dctx, xin).ptr, nbytes, mode, stream));
      PetscCall(PetscLogGpuTimeEnd());
      break;
    case cupmMemcpyDeviceToHost: {
      // not great
      PetscScalar *yarray;

      PetscCall(VecGetArrayWrite(yin, &yarray));
      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUPM(cupmMemcpyAsync(yarray, DeviceArrayRead(dctx, xin).ptr, nbytes, mode, stream));
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(VecRestoreArrayWrite(yin, &yarray));
    } break;
    case cupmMemcpyHostToHost: {
      // the worst case
      PetscScalar *yarray;

      PetscCall(VecGetArrayWrite(yin, &yarray));
      PetscCall(PetscArraycpy(yarray, HostArrayRead(dctx, xin).ptr, n));
      PetscCall(VecRestoreArrayWrite(yin, &yarray));
    } break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU, "Unknown cupmMemcpyKind %d", mode);
    }
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::swap_async(Vec xin, Vec yin)) {
  PetscFunctionBegin;
  if (xin != yin) {
    const auto         n = static_cast<cupmBlasInt_t>(xin->map->n);
    PetscDeviceContext dctx;
    cupmBlasHandle_t   cupmBlasHandle;

    PetscCall(GetHandles_(&dctx, &cupmBlasHandle));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUPMBLAS(cupmBlasXswap(cupmBlasHandle, n, DeviceArrayReadWrite(dctx, xin), 1, DeviceArrayReadWrite(dctx, yin), 1));
    PetscCall(PetscLogGpuTimeEnd());
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::axpby_async(Vec yin, PetscScalar alpha, PetscScalar beta, Vec xin)) {
  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) {
    PetscCall(scale_async(yin, beta));
  } else if (beta == PetscScalar(1.0)) {
    PetscCall(axpy_async(yin, alpha, xin));
  } else if (alpha == PetscScalar(1.0)) {
    PetscCall(aypx_async(yin, beta, xin));
  } else {
    const auto         betaIsZero = beta == PetscScalar(0.0);
    const auto         n          = static_cast<cupmBlasInt_t>(yin->map->n);
    cupmBlasHandle_t   cupmBlasHandle;
    PetscDeviceContext dctx;

    PetscCall(GetHandles_(&dctx, &cupmBlasHandle));
    {
      const auto calpha = makeCupmScalar(alpha);
      const auto xarray = DeviceArrayRead(dctx, xin);

      PetscCallCUPMBLAS(cupmBlasSetPointerMode(cupmBlasHandle, CUPMBLAS_POINTER_MODE_HOST));
      PetscCall(PetscLogGpuTimeBegin());
      if (betaIsZero) {
        // here we can get away with purely write-only as we memcpy into it first
        const auto   yarray = DeviceArrayWrite(dctx, yin);
        const auto   nbytes = n * sizeof(PetscScalar);
        cupmStream_t stream;

        PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx, &stream));
        PetscCallCUPM(cupmMemcpyAsync(yarray.ptr, xarray.ptr, nbytes, cupmMemcpyDeviceToDevice, stream));
        PetscCallCUPMBLAS(cupmBlasXscal(cupmBlasHandle, n, &calpha, yarray, 1));
      } else {
        const auto cbeta  = makeCupmScalar(beta);
        const auto yarray = DeviceArrayReadWrite(dctx, yin);

        PetscCallCUPMBLAS(cupmBlasXscal(cupmBlasHandle, n, &cbeta, yarray, 1));
        PetscCallCUPMBLAS(cupmBlasXaxpy(cupmBlasHandle, n, &calpha, xarray, 1, yarray, 1));
      }
      PetscCall(PetscLogGpuTimeEnd());
    }
    PetscCall(PetscLogGpuFlops((betaIsZero ? 1 : 3) * n));
    PetscCall(PetscLogCpuToGpuScalar((betaIsZero ? 1 : 2) * sizeof(alpha)));
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::axpbypcz_async(Vec zin, PetscScalar alpha, PetscScalar beta, PetscScalar gamma, Vec xin, Vec yin)) {
  PetscFunctionBegin;
  if (gamma != PetscScalar(1.0)) PetscCall(scale_async(zin, gamma)); // z <- a*x + b*y + c*z
  PetscCall(axpy_async(zin, alpha, xin));
  PetscCall(axpy_async(zin, beta, yin));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::norm_async(Vec xin, NormType type, PetscReal *z)) {
  const auto         n = static_cast<cupmBlasInt_t>(xin->map->n);
  PetscBool          device_mem;
  PetscInt           flopCount = 0;
  cupmBlasHandle_t   cupmBlasHandle;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(IsDeviceMemory(z, &device_mem));
  if (!n) {
    const auto norm1and2 = type == NORM_1_AND_2;

    if (device_mem) {
      cupmStream_t stream;

      PetscCall(GetHandles_(&stream));
      PetscCallCUPM(cupmMemsetAsync(z, 0, (1 + norm1and2) * sizeof(*z), stream));
    } else {
      z[0]         = 0.0;
      // yes this technically sets z[0] = 0 again half the time
      z[norm1and2] = 0.0;
    }
    PetscFunctionReturn(0);
  }
  PetscCall(GetHandles_(&dctx, &cupmBlasHandle));
  {
    auto xarray = DeviceArrayRead(dctx, xin);

    // in case NORM_INFINITY we may technically undo this, but cleaner this way
    PetscCallCUPMBLAS(cupmBlasSetPointerMode(cupmBlasHandle, device_mem ? CUPMBLAS_POINTER_MODE_DEVICE : CUPMBLAS_POINTER_MODE_HOST));
    PetscCall(PetscLogGpuTimeBegin());
    switch (type) {
    case NORM_1_AND_2:
    case NORM_1:
      PetscCallCUPMBLAS(cupmBlasXasum(cupmBlasHandle, n, xarray, 1, cupmRealCast(z)));
      flopCount = PetscMax(n - 1, 0);
      if (type == NORM_1) break;
      ++z; // fall-through
    case NORM_2:
    case NORM_FROBENIUS:
      PetscCallCUPMBLAS(cupmBlasXnrm2(cupmBlasHandle, n, xarray, 1, cupmRealCast(z)));
      flopCount += PetscMax(2 * n - 1, 0); // += in case we've fallen through from NORM_1_AND_2
      break;
    case NORM_INFINITY: {
      cupmStream_t  stream;
      PetscScalar   zs;
      cupmBlasInt_t i;

      PetscCheck(!device_mem, PETSC_COMM_SELF, PETSC_ERR_SUP, "Device memory norm pending implementation");
      // REVIEW ME: this needs to be redone by hand
      PetscCallCUPMBLAS(cupmBlasSetPointerMode(cupmBlasHandle, CUPMBLAS_POINTER_MODE_HOST));
      PetscCallCUPMBLAS(cupmBlasXamax(cupmBlasHandle, n, xarray, 1, &i));
      PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx, &stream));
      PetscCallCUPM(cupmMemcpyAsync(&zs, xarray.ptr + i - 1, sizeof(zs), cupmMemcpyDeviceToHost, stream));
      PetscCall(PetscDeviceContextSynchronize(dctx));
      *z = PetscAbsScalar(zs);
      // REVIEW ME: flopCount = ???
    } break;
    }
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(flopCount));
  PetscCall(PetscLogGpuToCpuScalar(sizeof(*z)));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::dotnorm2_async(Vec s, Vec t, PetscScalar *dp, PetscScalar *nm)) {
  PetscFunctionBegin;
  PetscCall(dot_async(s, t, dp));
  PetscCall(dot_async(t, t, nm));
  PetscFunctionReturn(0);
}

namespace detail {

struct conjugate {
  PETSC_HOSTDEVICE_DECL PetscScalar operator()(PetscScalar x) const { return PetscConj(x); }
};

} // namespace detail

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::conjugate_async(Vec xin)) {
  PetscFunctionBegin;
  if (PetscDefined(USE_COMPLEX)) PetscCall(pointwiseunary_async_(detail::conjugate{}, xin));
  PetscFunctionReturn(0);
}

namespace detail {

struct real_part {
  PETSC_HOSTDEVICE_DECL
  thrust::tuple<PetscReal, PetscInt> operator()(const thrust::tuple<PetscScalar, PetscInt> &x) const { return {PetscRealPart(x.get<0>()), x.get<1>()}; }

  PETSC_HOSTDEVICE_DECL PetscReal operator()(PetscScalar x) const { return PetscRealPart(x); }
};

template <typename Operator>
struct tuple_compare {
  using tuple_type = thrust::tuple<PetscReal, PetscInt>;

  PETSC_HOSTDEVICE_DECL tuple_type operator()(const tuple_type &x, const tuple_type &y) const {
    if (Operator{}(y.get<0>(), x.get<0>())) {
      // if y is strictly greater/less than x, return y
      return y;
    } else if (y.get<0>() == x.get<0>()) {
      // if equal, prefer lower index
      return y.get<1>() < x.get<1>() ? y : x;
    } else {
      // otherwise return x
      return x;
    }
  }
};

} // namespace detail

template <Device::CUPM::DeviceType T>
template <typename TupleFuncT, typename UnaryFuncT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::minmax_async_(TupleFuncT &&tuple_ftr, UnaryFuncT &&unary_ftr, PetscReal init_val, Vec v, PetscInt *p, PetscReal *m)) {
  constexpr auto     init_ptr = PetscInt{-1};
  const auto         n        = v->map->n;
  PetscDeviceContext dctx;
  cupmStream_t       stream;

  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQCUPM(), VECMPICUPM());
  if (!n) {
    *m = init_val;
    if (p) *p = init_ptr;
    PetscFunctionReturn(0);
  }
  PetscCall(GetHandles_(&dctx, &stream));
  // REVIEW ME: why not cupmBlasIXamin()/cupmBlasIXamax()?
  CHKERRTHRUST(
    auto vptr = thrust::device_pointer_cast<PetscScalar>(DeviceArrayRead(dctx, v));

    if (p) {
      auto tup = thrust::make_tuple(init_val, init_ptr);
      auto zip = thrust::make_zip_iterator(thrust::make_tuple(vptr, thrust::make_counting_iterator(PetscInt{0})));

    // need to use preprocessor conditionals since otherwise thrust complains about not being
    // able to convert a thrust::device_reference<PetscScalar> to a PetscReal on complex
    // builds...
#if PetscDefined(USE_COMPLEX)
      thrust::tie(*m, *p) = THRUST_CALL(thrust::transform_reduce, stream, zip, zip + n, detail::real_part(), tup, std::forward<TupleFuncT>(tuple_ftr));
#else
      thrust::tie(*m, *p) = THRUST_CALL(thrust::reduce, stream, zip, zip + n, tup, std::forward<TupleFuncT>(tuple_ftr));
#endif
    } else {
#if PetscDefined(USE_COMPLEX)
      *m = THRUST_CALL(thrust::transform_reduce, stream, vptr, vptr + n, detail::real_part(), init_val, std::forward<UnaryFuncT>(unary_ftr));
#else
      *m                  = THRUST_CALL(thrust::reduce, stream, vptr, vptr + n, init_val, std::forward<UnaryFuncT>(unary_ftr));
#endif
    });
  // REVIEW ME: flops?
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::max_async(Vec v, PetscInt *p, PetscReal *m)) {
  using value_type    = util::remove_pointer_t<decltype(m)>;
  using tuple_functor = detail::tuple_compare<thrust::greater<value_type>>;
  using unary_functor = thrust::maximum<value_type>;

  PetscFunctionBegin;
  // use {} constructor syntax otherwise most vexing parse
  PetscCall(minmax_async_(tuple_functor{}, unary_functor{}, PETSC_MIN_REAL, v, p, m));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::min_async(Vec v, PetscInt *p, PetscReal *m)) {
  using value_type    = util::remove_pointer_t<decltype(m)>;
  using tuple_functor = detail::tuple_compare<thrust::less<value_type>>;
  using unary_functor = thrust::minimum<value_type>;

  PetscFunctionBegin;
  // use {} constructor syntax otherwise most vexing parse
  PetscCall(minmax_async_(tuple_functor{}, unary_functor{}, PETSC_MAX_REAL, v, p, m));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::sum_async(Vec v, PetscScalar *sum)) {
  const auto         n = v->map->n;
  PetscDeviceContext dctx;
  cupmStream_t       stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx, &stream));
  // REVIEW ME: why not cupmBlasXasum()?
  CHKERRTHRUST(auto dptr = thrust::device_pointer_cast(DeviceArrayRead(dctx, v).ptr);

               *sum = THRUST_CALL(thrust::reduce, stream, dptr, dptr + n, PetscScalar{0.0}););
  // REVIEW ME: must be at least n additions
  PetscCall(PetscLogGpuFlops(n));
  PetscFunctionReturn(0);
}

namespace detail {

struct shifter {
  const PetscScalar s;

  PETSC_HOSTDEVICE_DECL PetscScalar operator()(PetscScalar x) const { return x + s; }
};

} // namespace detail

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::shift_async(Vec v, PetscScalar shift)) {
  PetscFunctionBegin;
  PetscCall(pointwiseunary_async_(detail::shifter{shift}, v));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::setrandom_async(Vec v, PetscRandom rand)) {
  const auto         n = v->map->n;
  PetscDeviceContext dctx;
  PetscBool          iscurand;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(rand), PETSCCURAND, &iscurand));
  PetscCall(GetHandles_(&dctx));
  if (iscurand) PetscCall(PetscRandomGetValues(rand, n, DeviceArrayWrite(dctx, v)));
  else PetscCall(PetscRandomGetValues(rand, n, HostArrayWrite(dctx, v)));
  // REVIEW ME: flops????
  // REVIEW ME: Timing???
  PetscFunctionReturn(0);
}

// REVIEW ME: TODO figure out a way around gcc 9.3.1 linker bug to get these to
// work. Everything compiles but get linker error in optimized build for mdot_async_(). I think
// this is because it is overloaded function and since we __always__ only call one overload,
// compiler doesn't generate the object code for the other overload...
// /usr/bin/ld: arch-linux-c-opt/lib/libpetsc.so: undefined reference to
// `Petsc::Vector::CUPM::Impl::VecSeq_CUPM<(Petsc::Device::CUPM::DeviceType)0>::mdot_async_(Petsc::Vector::CUPM::Impl::(anonymousnamespace)::UseComplexTag<false>,
// _p_Vec*, int, _p_Vec* const*, double*)'

// declare the extern templates, each is explicitly instantiated in the respective
// implementation directories
#if PetscDefined(HAVE_CUDA)
//extern template struct VecSeq_CUPM<Device::CUPM::DeviceType::CUDA>;
#endif

#if PetscDefined(HAVE_HIP)
//extern template struct VecSeq_CUPM<Device::CUPM::DeviceType::HIP>;
#endif

} // namespace Impl

} // namespace CUPM

} // namespace Vector

} // namespace Petsc

#endif // PETSCVECSEQCUPM_HPP
