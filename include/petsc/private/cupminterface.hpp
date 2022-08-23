#ifndef PETSCCUPMINTERFACE_HPP
#define PETSCCUPMINTERFACE_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cpputil.hpp>
#include <petsc/private/petscadvancedmacros.h>
#include <petscdevice_cupm.h>

#if defined(__cplusplus)
#include <array>

namespace Petsc {

namespace Device {

namespace CUPM {

// enum describing available cupm devices, this is used as the template parameter to any
// class subclassing the Interface or using it as a member variable
enum class DeviceType : int {
  CUDA,
  HIP
};

static constexpr std::array<const char *const, 5> DeviceTypes = {"cuda", "hip", "Petsc::Device::CUPM::DeviceType", "Petsc::Device::CUPM::DeviceType::", nullptr};

namespace Impl {

// A backend agnostic PetscCallCUPM() function, this will only work inside the member
// functions of a class inheriting from CUPM::Interface. Thanks to __VA_ARGS__ templated
// functions can also be wrapped inline:
//
// PetscCallCUPM(foo<int,char,bool>());
#define PetscCallCUPM(...) \
  do { \
    const cupmError_t cerr_p_ = __VA_ARGS__; \
    PetscCheck(cerr_p_ == cupmSuccess, PETSC_COMM_SELF, PETSC_ERR_GPU, "%s error %d (%s) : %s", cupmName(), static_cast<PetscErrorCode>(cerr_p_), cupmGetErrorName(cerr_p_), cupmGetErrorString(cerr_p_)); \
  } while (0)

// PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT() - declaration to alias a cuda/hip integral constant
// value
//
// input params:
// our_prefix   - the prefix of the alias
// our_suffix   - the suffix of the alias
// their_prefix - the prefix of the variable being aliased
// their_suffix - the suffix of the variable being aliased
//
// example usage:
// PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(cupm,Success,cuda,AllGood); ->
// static const auto cupmSuccess = cudaAllGood;
//
// PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(cupm,Success,hip,AllRight); ->
// static const auto cupmSuccess = hipAllRight;
#define PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(our_prefix, our_suffix, their_prefix, their_suffix) static const auto PetscConcat(our_prefix, our_suffix) = PetscConcat(their_prefix, their_suffix)

// PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON() - declaration to alias a cuda/hip integral constant
// value
//
// input params:
// our_suffix   - the suffix of the alias
// their_suffix - the suffix of the variable being aliased
//
// notes:
// requires PETSC_CUPM_PREFIX_L to be defined to the specific prefix
//
// example usage:
// #define PETSC_CUPM_PREFIX_L cuda
// PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(Success,AllGood); ->
// static const auto cupmSuccess = cudaAllGood;
//
// #define PETSC_CUPM_PREFIX_L hip
// PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(Success,AllRight); ->
// static const auto cupmSuccess = hipAllRight;
#define PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(our_suffix, their_suffix) PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(cupm, our_suffix, PETSC_CUPM_PREFIX_L, their_suffix)

// PETSC_CUPM_ALIAS_INTEGRAL_VALUE() - declaration to alias a cuda/hip integral constant value
//
// input param:
// suffix - the common suffix shared between cuda, hip, and cupm
//
// notes:
// requires PETSC_CUPM_PREFIX_L to be defined to the specific prefix
//
// example usage:
// #define PETSC_CUPM_PREFIX_L cuda
// PETSC_CUPM_ALIAS_INTEGRAL_VALUE(Success); -> static const auto cupmSuccess = cudaSuccess;
//
// #define PETSC_CUPM_PREFIX_L hip
// PETSC_CUPM_ALIAS_INTEGRAL_VALUE(Success); -> static const auto cupmSuccess = hipSuccess;
#define PETSC_CUPM_ALIAS_INTEGRAL_VALUE(suffix) PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(suffix, suffix)

// PETSC_CUPM_ALIAS_FUNCTION_EXACT() - declaration to alias a cuda/hip function
//
// input params:
// our_prefix   - the prefix of the alias
// our_suffix   - the suffix of the alias
// their_prefix - the prefix of the function being aliased
// their_suffix - the suffix of the function being aliased
//
// notes:
// see PETSC_ALIAS_FUNCTION() for the exact nature of the expansion
//
// example usage:
// PETSC_CUPM_ALIAS_FUNCTION_EXACT(cupm,Malloc,cuda,Malloc) ->
// template <typename... T>
// static constexpr auto cupmMalloc(T&&... args) *noexcept and trailing return type deduction*
// {
//   return cudaMalloc(std::forward<T>(args)...);
// }
#define PETSC_CUPM_ALIAS_FUNCTION_EXACT(our_prefix, our_suffix, their_prefix, their_suffix) PETSC_ALIAS_FUNCTION(static PetscConcat(our_prefix, our_suffix), PetscConcat(their_prefix, their_suffix))

// PETSC_CUPM_ALIAS_FUNCTION_COMMON() - declaration to alias a cuda/hip function
//
// input params:
// our_suffix   - the suffix of the alias
// their_suffix - the common suffix of the cuda/hip function being aliased
//
// notes:
// requires PETSC_CUPM_PREFIX_L to be defined to the specific prefix of the function being
// aliased. see PETSC_ALIAS_FUNCTION() for the exact nature of the expansion
//
// example usage:
// #define PETSC_CUPM_PREFIX_L cuda
// PETSC_CUPM_ALIAS_FUNCTION_COMMON(MallocFancy,Malloc) ->
// template <typename... T>
// static constexpr auto cupmMallocFancy(T&&... args) *noexcept and trailing return type deduction*
// {
//   return cudaMalloc(std::forward<T>(args)...);
// }
//
// #define PETSC_CUPM_PREFIX_L hip
// PETSC_CUPM_ALIAS_FUNCTION_COMMON(MallocFancy,Malloc) ->
// template <typename... T>
// static constexpr auto cupmMallocFancy(T&&... args) *noexcept and trailing return type deduction*
// {
//   return hipMalloc(std::forward<T>(args)...);
// }
#define PETSC_CUPM_ALIAS_FUNCTION_COMMON(our_suffix, their_suffix) PETSC_CUPM_ALIAS_FUNCTION_EXACT(cupm, our_suffix, PETSC_CUPM_PREFIX_L, their_suffix)

// PETSC_CUPM_ALIAS_FUNCTION() - declaration to alias a cuda/hip function
//
// input param:
// suffix - the common suffix for hip, cuda and the alias
//
// notes:
// requires PETSC_CUPM_PREFIX_L to be defined to the specific prefix of the function being
// aliased. see PETSC_ALIAS_FUNCTION() for the exact nature of the expansion
//
// example usage:
// #define PETSC_CUPM_PREFIX_L cuda
// PETSC_CUPM_ALIAS_FUNCTION(Malloc) ->
// template <typename... T>
// static constexpr auto cupmMalloc(T&&... args) *noexcept and trailing return type deduction*
// {
//   return cudaMalloc(std::forward<T>(args)...);
// }
//
// #define PETSC_CUPM_PREFIX_L hip
// PETSC_CUPM_ALIAS_FUNCTION(Malloc) ->
// template <typename... T>
// static constexpr auto cupmMalloc(T&&... args) *noexcept and trailing return type deduction*
// {
//   return hipMalloc(std::forward<T>(args)...);
// }
#define PETSC_CUPM_ALIAS_FUNCTION(suffix) PETSC_CUPM_ALIAS_FUNCTION_COMMON(suffix, suffix)

// PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_EXACT() - declaration to alias a cuda/hip function but
// discard the last N arguments
//
// input params:
// our_prefix   - the prefix of the alias
// our_suffix   - the suffix of the alias
// their_prefix - the prefix of the function being aliased
// their_suffix - the suffix of the function being aliased
// N            - integer constant [0,INT_MAX) dictating how many arguments to chop off the end
//
// notes:
// see PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS() for the exact nature of the expansion
//
// example use:
// PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_EXACT(cupm,MallocAsync,cuda,Malloc,1) ->
// template <typename... T, typename Tend>
// static constexpr auto cupmMallocAsync(T&&... args, Tend argend) *noexcept and trailing
// return type deduction*
// {
//   (void)argend;
//   return cudaMalloc(std::forward<T>(args)...);
// }
#define PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_EXACT(our_prefix, our_suffix, their_prefix, their_suffix, N) PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS(static PetscConcat(our_prefix, our_suffix), PetscConcat(their_prefix, their_suffix), N)

// PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON() - declaration to alias a cuda/hip function but
// discard the last N arguments
//
// input params:
// our_suffix   - the suffix of the alias
// their_suffix - the suffix of the function being aliased
// N            - integer constant [0,INT_MAX) dictating how many arguments to chop off the end
//
// notes:
// requires PETSC_CUPM_PREFIX_L to be defined to the specific prefix of the function being
// aliased. see PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS() for the exact nature of the
// expansion
//
// example use:
// #define PETSC_CUPM_PREFIX_L cuda
// PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(MallocAsync,Malloc,1) ->
// template <typename... T, typename Tend>
// static constexpr auto cupmMallocAsync(T&&... args, Tend argend) *noexcept and trailing
// return type deduction*
// {
//   (void)argend;
//   return cudaMalloc(std::forward<T>(args)...);
// }
#define PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(our_suffix, their_suffix, N) PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_EXACT(cupm, our_suffix, PETSC_CUPM_PREFIX_L, their_suffix, N)

// Base class that holds functions and variables that don't require CUDA or HIP to be present
// on the system
template <DeviceType T>
struct InterfaceBase {
  static const DeviceType type = T;

  PETSC_CXX_COMPAT_DECL(constexpr const char *cupmName()) {
    static_assert(util::integral_value(DeviceType::CUDA) == 0, "");
    static_assert(util::integral_value(DeviceType::HIP) == 1, "");
    return std::get<util::integral_value(T)>(DeviceTypes);
  }

  PETSC_CXX_COMPAT_DECL(constexpr auto cupmDeviceTypeToPetscDeviceType())
  PETSC_DECLTYPE_AUTO_RETURNS(T == DeviceType::CUDA ? PETSC_DEVICE_CUDA : PETSC_DEVICE_HIP);

  PETSC_CXX_COMPAT_DECL(constexpr auto cupmDeviceTypeToPetscMemType())
  PETSC_DECLTYPE_AUTO_RETURNS(T == DeviceType::CUDA ? PETSC_MEMTYPE_CUDA : PETSC_MEMTYPE_HIP);
};

// declare the base class static member variables
template <DeviceType T>
const DeviceType InterfaceBase<T>::type;

#define PETSC_CUPM_BASE_CLASS_HEADER(DEVICE_TYPE) \
  using base_type = Petsc::Device::CUPM::Impl::InterfaceBase<DEVICE_TYPE>; \
  using base_type::type; \
  using base_type::cupmName; \
  using base_type::cupmDeviceTypeToPetscDeviceType; \
  using base_type::cupmDeviceTypeToPetscMemType

// A templated C++ struct that defines the entire CUPM interface. Use of templating vs
// preprocessor macros allows us to use both interfaces simultaneously as well as easily
// import them into classes.
template <DeviceType>
struct InterfaceImpl;

#if PetscDefined(HAVE_CUDA)
#define PETSC_CUPM_PREFIX_L cuda
#define PETSC_CUPM_PREFIX_U CUDA
template <>
struct InterfaceImpl<DeviceType::CUDA> : InterfaceBase<DeviceType::CUDA> {
  PETSC_CUPM_BASE_CLASS_HEADER(DeviceType::CUDA);

  // typedefs
  using cupmError_t             = cudaError_t;
  using cupmEvent_t             = cudaEvent_t;
  using cupmStream_t            = cudaStream_t;
  using cupmDeviceProp_t        = cudaDeviceProp;
  using cupmMemcpyKind_t        = cudaMemcpyKind;
  using cupmComplex_t           = util::conditional_t<PetscDefined(USE_REAL_SINGLE), cuComplex, cuDoubleComplex>;
  using cupmPointerAttributes_t = struct cudaPointerAttributes;
  using cupmMemoryType_t        = enum cudaMemoryType;
  using cupmDim3                = dim3;
  using cupmHostFn_t            = cudaHostFn_t;

  // values
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(Success);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorNotReady);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorDeviceAlreadyInUse);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorSetOnActiveProcess);
#if PETSC_PKG_CUDA_VERSION_GE(11, 1, 0)
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorStubLibrary);
#else
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(ErrorStubLibrary, ErrorInsufficientDriver);
#endif
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorNoDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(StreamNonBlocking);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(DeviceMapHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyHostToDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDeviceToHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDeviceToDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyHostToHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDefault);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemoryTypeHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemoryTypeDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemoryTypeManaged);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(EventDisableTiming);

  // error functions
  PETSC_CUPM_ALIAS_FUNCTION(GetErrorName)
  PETSC_CUPM_ALIAS_FUNCTION(GetErrorString)
  PETSC_CUPM_ALIAS_FUNCTION(GetLastError)

  // device management
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceCount)
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceProperties)
  PETSC_CUPM_ALIAS_FUNCTION(GetDevice)
  PETSC_CUPM_ALIAS_FUNCTION(SetDevice)
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceFlags)
  PETSC_CUPM_ALIAS_FUNCTION(SetDeviceFlags)
  PETSC_CUPM_ALIAS_FUNCTION(PointerGetAttributes)

  // stream management
  PETSC_CUPM_ALIAS_FUNCTION(EventCreate)
  PETSC_CUPM_ALIAS_FUNCTION(EventCreateWithFlags)
  PETSC_CUPM_ALIAS_FUNCTION(EventDestroy)
  PETSC_CUPM_ALIAS_FUNCTION(EventRecord)
  PETSC_CUPM_ALIAS_FUNCTION(EventSynchronize)
  PETSC_CUPM_ALIAS_FUNCTION(EventElapsedTime)
  PETSC_CUPM_ALIAS_FUNCTION(StreamCreate)
  PETSC_CUPM_ALIAS_FUNCTION(StreamCreateWithFlags)
  PETSC_CUPM_ALIAS_FUNCTION(StreamDestroy)
  PETSC_CUPM_ALIAS_FUNCTION(StreamWaitEvent)
  PETSC_CUPM_ALIAS_FUNCTION(StreamQuery)
  PETSC_CUPM_ALIAS_FUNCTION(StreamSynchronize)
  PETSC_CUPM_ALIAS_FUNCTION(DeviceSynchronize)
  PETSC_CUPM_ALIAS_FUNCTION(GetSymbolAddress)

  // memory management
  PETSC_CUPM_ALIAS_FUNCTION(Free)
  PETSC_CUPM_ALIAS_FUNCTION(Malloc)
#if PETSC_PKG_CUDA_VERSION_GE(11, 2, 0)
  PETSC_CUPM_ALIAS_FUNCTION(FreeAsync)
  PETSC_CUPM_ALIAS_FUNCTION(MallocAsync)
#else
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(FreeAsync, Free, 1)
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(MallocAsync, Malloc, 1)
#endif
  PETSC_CUPM_ALIAS_FUNCTION(Memcpy)
  PETSC_CUPM_ALIAS_FUNCTION(MemcpyAsync)
  PETSC_CUPM_ALIAS_FUNCTION(MallocHost)
  PETSC_CUPM_ALIAS_FUNCTION(FreeHost)
  PETSC_CUPM_ALIAS_FUNCTION(MemsetAsync)

  PETSC_CUPM_ALIAS_FUNCTION(LaunchHostFunc)

  template <typename FunctionT, typename... KernelArgsT>
  PETSC_CXX_COMPAT_DECL(cudaError_t cupmLaunchKernel(FunctionT &&func, dim3 gridDim, dim3 blockDim, std::size_t sharedMem, cudaStream_t stream, KernelArgsT &&...kernelArgs)) {
    void *args[] = {(void *)&kernelArgs...};
    return cudaLaunchKernel((void *)func, std::move(gridDim), std::move(blockDim), args, sharedMem, std::move(stream));
  }
};
#undef PETSC_CUPM_PREFIX_L
#undef PETSC_CUPM_PREFIX_U
#endif // PetscDefined(HAVE_CUDA)

#if PetscDefined(HAVE_HIP)
#define PETSC_CUPM_PREFIX_L hip
#define PETSC_CUPM_PREFIX_U HIP
template <>
struct InterfaceImpl<DeviceType::HIP> : InterfaceBase<DeviceType::HIP> {
  PETSC_CUPM_BASE_CLASS_HEADER(DeviceType::HIP);

  // typedefs
  using cupmError_t             = hipError_t;
  using cupmEvent_t             = hipEvent_t;
  using cupmStream_t            = hipStream_t;
  using cupmDeviceProp_t        = hipDeviceProp_t;
  using cupmMemcpyKind_t        = hipMemcpyKind;
  using cupmComplex_t           = util::conditional_t<PetscDefined(USE_REAL_SINGLE), hipComplex, hipDoubleComplex>;
  using cupmPointerAttributes_t = hipPointerAttribute_t;
  using cupmMemoryType_t        = enum hipMemoryType;
  using cupmDim3                = dim3;
#if PETSC_PKG_HIP_VERSION_GE(5, 2, 0)
  using cupmHostFn_t = hipHostFn_t;
#else
  using cupmHostFn_t = void (*)(void *);
#endif

  // values
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(Success);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorNotReady);
  // see https://github.com/ROCm-Developer-Tools/HIP/blob/develop/bin/hipify-perl
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(ErrorDeviceAlreadyInUse, ErrorContextAlreadyInUse);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorSetOnActiveProcess);
  // as of HIP v4.2 cudaErrorStubLibrary has no HIP equivalent
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(ErrorStubLibrary, ErrorInsufficientDriver);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorNoDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(StreamNonBlocking);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(DeviceMapHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyHostToDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDeviceToHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDeviceToDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyHostToHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDefault);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemoryTypeHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemoryTypeDevice);
  // see
  // https://github.com/ROCm-Developer-Tools/HIP/blob/develop/include/hip/hip_runtime_api.h#L156
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(MemoryTypeManaged, MemoryTypeUnified);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(EventDisableTiming);

  // error functions
  PETSC_CUPM_ALIAS_FUNCTION(GetErrorName)
  PETSC_CUPM_ALIAS_FUNCTION(GetErrorString)
  PETSC_CUPM_ALIAS_FUNCTION(GetLastError)

  // device management
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceCount)
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceProperties)
  PETSC_CUPM_ALIAS_FUNCTION(GetDevice)
  PETSC_CUPM_ALIAS_FUNCTION(SetDevice)
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceFlags)
  PETSC_CUPM_ALIAS_FUNCTION(SetDeviceFlags)
  PETSC_CUPM_ALIAS_FUNCTION(PointerGetAttributes)

  // stream management
  PETSC_CUPM_ALIAS_FUNCTION(EventCreate)
  PETSC_CUPM_ALIAS_FUNCTION(EventCreateWithFlags)
  PETSC_CUPM_ALIAS_FUNCTION(EventDestroy)
  PETSC_CUPM_ALIAS_FUNCTION(EventRecord)
  PETSC_CUPM_ALIAS_FUNCTION(EventSynchronize)
  PETSC_CUPM_ALIAS_FUNCTION(EventElapsedTime)
  PETSC_CUPM_ALIAS_FUNCTION(StreamCreate)
  PETSC_CUPM_ALIAS_FUNCTION(StreamCreateWithFlags)
  PETSC_CUPM_ALIAS_FUNCTION(StreamDestroy)
  PETSC_CUPM_ALIAS_FUNCTION(StreamWaitEvent)
  PETSC_CUPM_ALIAS_FUNCTION(StreamQuery)
  PETSC_CUPM_ALIAS_FUNCTION(StreamSynchronize)
  PETSC_CUPM_ALIAS_FUNCTION(DeviceSynchronize)
  PETSC_CUPM_ALIAS_FUNCTION(GetSymbolAddress)

  // memory management
  PETSC_CUPM_ALIAS_FUNCTION(Free)
  PETSC_CUPM_ALIAS_FUNCTION(Malloc)
  // HIP has no hipFreeAsync
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(FreeAsync, Free, 1)
  // HIP has no hipMallocAsync
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(MallocAsync, Malloc, 1)
  PETSC_CUPM_ALIAS_FUNCTION(Memcpy)
  PETSC_CUPM_ALIAS_FUNCTION(MemcpyAsync)
  // hipMallocHost is deprecated
  PETSC_CUPM_ALIAS_FUNCTION_COMMON(MallocHost, HostMalloc)
  // hipFreeHost is deprecated
  PETSC_CUPM_ALIAS_FUNCTION_COMMON(FreeHost, HostFree)
  PETSC_CUPM_ALIAS_FUNCTION(MemsetAsync)

  // HIP appears to only have hipLaunchHostFunc from 5.2.0 onwards
  // https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/CUDA_Runtime_API_functions_supported_by_HIP.md#7-execution-control=
  PETSC_CXX_COMPAT_DECL(hipError_t cupmLaunchHostFunc(hipStream_t stream, cupmHostFn_t fn, void *ctx)) {
#if PETSC_PKG_HIP_VERSION_GE(5, 2, 0)
    return hipLaunchHostFunc(stream, fn, ctx);
#else
    // the only correct way to spoof this function is to do it synchronously...
    if (auto ret = hipStreamSynchronize(stream)) return ret;
    fn(ctx);
    return hipSuccess;
#endif
  }

  // kernel launching
  template <typename FunctionT, typename... KernelArgsT>
  PETSC_CXX_COMPAT_DECL(hipError_t cupmLaunchKernel(FunctionT &&func, dim3 gridDim, dim3 blockDim, std::size_t sharedMem, hipStream_t stream, KernelArgsT &&...kernelArgs)) {
    void *args[] = {(void *)&kernelArgs...};
    return hipLaunchKernel((void *)func, std::move(gridDim), std::move(blockDim), args, sharedMem, std::move(stream));
  }
};
#undef PETSC_CUPM_PREFIX_L
#undef PETSC_CUPM_PREFIX_U
#endif // PetscDefined(HAVE_HIP)

#undef PETSC_CUPM_BASE_CLASS_HEADER

// shorthand for bringing all of the typedefs from the base Interface class into your own,
// it's annoying that c++ doesn't have a way to do this automatically
#define PETSC_CUPM_IMPL_CLASS_HEADER(base_name, T) \
  using base_name = Petsc::Device::CUPM::Impl::InterfaceImpl<T>; \
  /* introspection */ \
  using base_name::type; \
  using base_name::cupmName; \
  using base_name::cupmDeviceTypeToPetscDeviceType; \
  using base_name::cupmDeviceTypeToPetscMemType; \
  /* types */ \
  using typename base_name::cupmComplex_t; \
  using typename base_name::cupmError_t; \
  using typename base_name::cupmEvent_t; \
  using typename base_name::cupmStream_t; \
  using typename base_name::cupmDeviceProp_t; \
  using typename base_name::cupmMemcpyKind_t; \
  using typename base_name::cupmPointerAttributes_t; \
  using typename base_name::cupmMemoryType_t; \
  using typename base_name::cupmDim3; \
  /* variables */ \
  using base_name::cupmSuccess; \
  using base_name::cupmErrorNotReady; \
  using base_name::cupmErrorDeviceAlreadyInUse; \
  using base_name::cupmErrorSetOnActiveProcess; \
  using base_name::cupmErrorStubLibrary; \
  using base_name::cupmErrorNoDevice; \
  using base_name::cupmStreamNonBlocking; \
  using base_name::cupmDeviceMapHost; \
  using base_name::cupmMemcpyHostToDevice; \
  using base_name::cupmMemcpyDeviceToHost; \
  using base_name::cupmMemcpyDeviceToDevice; \
  using base_name::cupmMemcpyHostToHost; \
  using base_name::cupmMemcpyDefault; \
  using base_name::cupmMemoryTypeHost; \
  using base_name::cupmMemoryTypeDevice; \
  using base_name::cupmMemoryTypeManaged; \
  using base_name::cupmEventDisableTiming; \
  /* functions */ \
  using base_name::cupmGetErrorName; \
  using base_name::cupmGetErrorString; \
  using base_name::cupmGetLastError; \
  using base_name::cupmGetDeviceCount; \
  using base_name::cupmGetDeviceProperties; \
  using base_name::cupmGetDevice; \
  using base_name::cupmSetDevice; \
  using base_name::cupmGetDeviceFlags; \
  using base_name::cupmSetDeviceFlags; \
  using base_name::cupmPointerGetAttributes; \
  using base_name::cupmEventCreate; \
  using base_name::cupmEventCreateWithFlags; \
  using base_name::cupmEventDestroy; \
  using base_name::cupmEventRecord; \
  using base_name::cupmEventSynchronize; \
  using base_name::cupmEventElapsedTime; \
  using base_name::cupmStreamCreate; \
  using base_name::cupmStreamCreateWithFlags; \
  using base_name::cupmStreamDestroy; \
  using base_name::cupmStreamWaitEvent; \
  using base_name::cupmStreamQuery; \
  using base_name::cupmStreamSynchronize; \
  using base_name::cupmDeviceSynchronize; \
  using base_name::cupmGetSymbolAddress; \
  using base_name::cupmMalloc; \
  using base_name::cupmMallocAsync; \
  using base_name::cupmMemcpy; \
  using base_name::cupmMemcpyAsync; \
  using base_name::cupmMallocHost; \
  using base_name::cupmMemsetAsync; \
  using base_name::cupmLaunchHostFunc

template <DeviceType>
struct Interface;

// The actual interface class
template <DeviceType T>
struct Interface : InterfaceImpl<T> {
  PETSC_CUPM_IMPL_CLASS_HEADER(interface_type, T);

  using cupmReal_t   = util::conditional_t<PetscDefined(USE_REAL_SINGLE), float, double>;
  using cupmScalar_t = util::conditional_t<PetscDefined(USE_COMPLEX), cupmComplex_t, cupmReal_t>;

  // REVIEW ME: this needs to be cleaned up, it is unreadable
  PETSC_CXX_COMPAT_DECL(constexpr auto makeCupmScalar(PetscScalar s))
  PETSC_DECLTYPE_AUTO_RETURNS(PetscIfPetscDefined(USE_COMPLEX, (cupmComplex_t{PetscRealPart(s), PetscImaginaryPart(s)}), static_cast<cupmReal_t>(s)));

  PETSC_CXX_COMPAT_DECL(constexpr auto cupmScalarCast(const PetscScalar *s))
  PETSC_DECLTYPE_AUTO_RETURNS(reinterpret_cast<const cupmScalar_t *>(s));

  PETSC_CXX_COMPAT_DECL(constexpr auto cupmScalarCast(PetscScalar *s))
  PETSC_DECLTYPE_AUTO_RETURNS(reinterpret_cast<cupmScalar_t *>(s));

  PETSC_CXX_COMPAT_DECL(constexpr auto cupmRealCast(PetscReal *s))
  PETSC_DECLTYPE_AUTO_RETURNS(reinterpret_cast<cupmReal_t *>(s));

  PETSC_CXX_COMPAT_DECL(constexpr auto cupmRealCast(const PetscReal *s))
  PETSC_DECLTYPE_AUTO_RETURNS(reinterpret_cast<const cupmReal_t *>(s));

#if !defined(PETSC_PKG_CUDA_VERSION_GE)
#define PETSC_PKG_CUDA_VERSION_GE(...) 0
#define CUPM_DEFINED_PETSC_PKG_CUDA_VERSION_GE
#endif
  PETSC_CXX_COMPAT_DECL(PetscErrorCode cupmGetMemType(const void *data, PetscMemType *type)) {
    cupmPointerAttributes_t attr;
    cupmError_t             cerr;

    PetscFunctionBegin;
    PetscValidPointer(type, 2);
    // Do not check error, instead reset it via GetLastError() since before CUDA 11.0, passing
    // a host pointer returns cudaErrorInvalidValue
    cerr = cupmPointerGetAttributes(&attr, data);
    cerr = cupmGetLastError();
    // HIP seems to always have used memoryType though
#if (defined(CUDART_VERSION) && (CUDART_VERSION < 10000)) || defined(__HIP_PLATFORM_HCC__)
    const auto mtype = attr.memoryType;
#else
    if (PETSC_PKG_CUDA_VERSION_GE(11, 0, 0) && (T == DeviceType::CUDA)) PetscCallCUPM(cerr);
    const auto mtype = attr.type;
#endif // CUDART_VERSION && CUDART_VERSION < 10000 || __HIP_PLATFORM_HCC__
    *type = ((cerr == cupmSuccess) && (mtype == cupmMemoryTypeDevice)) ? cupmDeviceTypeToPetscMemType() : PETSC_MEMTYPE_HOST;
    PetscFunctionReturn(0);
  }
#if defined(CUPM_DEFINED_PETSC_PKG_CUDA_VERSION_GE)
#undef PETSC_PKG_CUDA_VERSION_GE
#endif

  PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 cupmMemcpyKind_t PetscDeviceCopyModeToCUPMMemcpyKind(PetscDeviceCopyMode mode)) {
    switch (mode) {
    case PETSC_DEVICE_COPY_HTOH: return cupmMemcpyHostToHost;
    case PETSC_DEVICE_COPY_HTOD: return cupmMemcpyHostToDevice;
    case PETSC_DEVICE_COPY_DTOD: return cupmMemcpyDeviceToDevice;
    case PETSC_DEVICE_COPY_DTOH: return cupmMemcpyDeviceToHost;
    case PETSC_DEVICE_COPY_AUTO: return cupmMemcpyDefault;
    }
    PetscUnreachable();
    return cupmMemcpyDefault;
  }

  // these change what the arguments mean, so need to namespace these
  template <typename M>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode PetscCUPMMallocAsync(M **ptr, std::size_t n, cupmStream_t stream = nullptr)) {
    PetscFunctionBegin;
    PetscValidPointer(ptr, 1);
    if (PetscLikely(n)) {
      PetscCallCUPM(cupmMallocAsync(reinterpret_cast<void **>(ptr), n * sizeof(M), stream));
    } else {
      *ptr = nullptr;
    }
    PetscFunctionReturn(0);
  }

  template <typename M>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode PetscCUPMMalloc(M **ptr, std::size_t n)) {
    PetscFunctionBegin;
    PetscCall(PetscCUPMMallocAsync(ptr, n));
    PetscFunctionReturn(0);
  }

  template <typename M>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode PetscCUPMMallocHost(M **ptr, std::size_t n)) {
    PetscFunctionBegin;
    PetscValidPointer(ptr, 1);
    if (PetscLikely(n)) {
      PetscCall(cupmMallocHost(reinterpret_cast<void **>(ptr), n * sizeof(M)));
    } else {
      *ptr = nullptr;
    }
    PetscFunctionReturn(0);
  }

  template <typename D, typename S>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode PetscCUPMMemcpyAsync(D *dest, const S *src, std::size_t n, cupmMemcpyKind_t kind, cupmStream_t stream = nullptr)) {
    static_assert(sizeof(D) == sizeof(S), "");
    static_assert(!std::is_void<D>::value && !std::is_void<S>::value, "");

    PetscFunctionBegin;
    if (PetscLikely(n)) {
      constexpr auto is_scalar = std::is_same<util::remove_cv_t<D>, PetscScalar>::value;
      const auto     size      = n * sizeof(D);
      // cannot dereference (i.e. cannot call PetscValidPointer() here)
      PetscCheck(dest, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to copy to a NULL pointer");
      PetscCheck(src, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to copy from a NULL pointer");
      PetscCallCUPM(cupmMemcpyAsync(dest, src, size, kind, stream));
      // do this with preprocessors, since if no log is used the functions below are macros and
      // hence the ternary is ill-formed
#if PetscDefined(USE_LOG) && PetscDefined(HAVE_DEVICE)
      // only the explicit HTOD or DTOH are handled, since we either don't log the other cases
      // (yet) or don't know the direction
      if (kind == cupmMemcpyDeviceToHost) {
        PetscCall((is_scalar ? PetscLogGpuToCpuScalar : PetscLogGpuToCpu)(size));
      } else if (kind == cupmMemcpyHostToDevice) {
        PetscCall((is_scalar ? PetscLogCpuToGpuScalar : PetscLogCpuToGpu)(size));
      }
#else
#if !defined(PetscLogGpuToCpu) // use PetscLogGpuToCpu as the canary
#error "PetscLogGpuToCpu() is no longer a macro when no logging or no device. PetscCUPMMemcpyAsync() should be updated"
#endif
#endif
    }
    PetscFunctionReturn(0);
  }

  template <typename D, typename S>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode PetscCUPMMemcpy(D *dest, const S *src, std::size_t n, cupmMemcpyKind_t kind)) {
    PetscFunctionBegin;
    PetscCall(PetscCUPMMemcpyAsync(dest, src, n, kind));
    PetscFunctionReturn(0);
  }

  template <typename M>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode PetscCUPMMemsetAsync(M *ptr, int value, std::size_t n, cupmStream_t stream = nullptr)) {
    PetscFunctionBegin;
    if (n) {
      PetscCheck(ptr, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to memset a NULL pointer with size %zu != 0", n);
      PetscCallCUPM(cupmMemsetAsync(ptr, value, n * sizeof(M), stream));
    }
    PetscFunctionReturn(0);
  }

  template <typename M>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode PetscCUPMMemset(M *ptr, int value, std::size_t n)) {
    PetscFunctionBegin;
    PetscCall(PetscCUPMMemsetAsync(ptr, value, n));
    PetscFunctionReturn(0);
  }

  // these we can transparently wrap, no need to namespace it to Petsc
  template <typename M>
  PETSC_CXX_COMPAT_DECL(cupmError_t cupmFreeAsync(M &&ptr, cupmStream_t stream = nullptr)) {
    static_assert(std::is_pointer<util::decay_t<M>>::value, "");
    auto cerr = cupmSuccess;
    if (ptr) {
      cerr = interface_type::cupmFreeAsync(std::forward<M>(ptr), stream);
      ptr  = nullptr;
    }
    return cerr;
  }

  PETSC_CXX_COMPAT_DECL(cupmError_t cupmFreeAsync(std::nullptr_t ptr, cupmStream_t stream = nullptr)) {
    return interface_type::cupmFreeAsync(ptr, stream);
  }

  template <typename M>
  PETSC_CXX_COMPAT_DECL(cupmError_t cupmFree(M &&ptr)) {
    static_assert(std::is_pointer<util::decay_t<M>>::value, "");
    return cupmFreeAsync(std::forward<M>(ptr));
  }

  PETSC_CXX_COMPAT_DECL(cupmError_t cupmFree(std::nullptr_t ptr)) {
    return cupmFreeAsync(ptr);
  }

  template <typename M>
  PETSC_CXX_COMPAT_DECL(cupmError_t cupmFreeHost(M &&ptr)) {
    static_assert(std::is_pointer<util::decay_t<M>>::value, "");
    const auto cerr = interface_type::cupmFreeHost(std::forward<M>(ptr));
    ptr             = nullptr;
    return cerr;
  }

  PETSC_CXX_COMPAT_DECL(cupmError_t cupmFreeHost(std::nullptr_t ptr)) {
    return interface_type::cupmFreeHost(ptr);
  }

  // specific wrapper for device launch function, as the real function is a C routine and
  // doesn't have variable arguments. The actual mechanics of this are a bit complicated but
  // boils down to the fact that ultimately we pass a
  //
  // void *args[] = {(void*)&kernel_args...};
  //
  // to the kernel launcher. Since we pass void* this means implicit conversion does **not**
  // happen to the kernel arguments so we must do it ourselves here. This function does this in
  // 3 stages:
  // 1. Enumerate the kernel arguments (cupmLaunchKernel)
  // 2. Deduce the signature of func() and static_cast the kernel arguments to the type
  //    expected by func() using the enumeration above (deduceKernelCall)
  // 3. Form the void* array with the converted arguments and call cuda/hipLaunchKernel with
  //    it. (interface_type::cupmLaunchKernel)
  template <typename F, typename... Args>
  PETSC_CXX_COMPAT_DECL(cupmError_t cupmLaunchKernel(F &&func, cupmDim3 gridDim, cupmDim3 blockDim, std::size_t sharedMem, cupmStream_t stream, Args &&...kernelArgs)) {
    return deduceKernelCall(util::index_sequence_for<Args...>{}, std::forward<F>(func), std::move(gridDim), std::move(blockDim), std::move(sharedMem), std::move(stream), std::forward<Args>(kernelArgs)...);
  }

  template <std::size_t block_size = 256, std::size_t warp_size = 32, typename F, typename... Args>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode PetscCUPMLaunchKernel1D(std::size_t n, std::size_t sharedMem, cupmStream_t stream, F &&func, Args &&...kernelArgs)) {
    static_assert(block_size > 0, "");
    static_assert(warp_size > 0, "");
    // want block_size to be a multiple of the warp_size
    static_assert(block_size % warp_size == 0, "");
    // the round-up algorithm below requires warp_size be a power of 2
    static_assert((warp_size & (warp_size - 1)) == 0, "");
    // round up to nearest multiple of warp_size if n < block_size
    const auto nthread = n >= block_size ? block_size : (n + warp_size - 1) & -warp_size;
    const auto nblock  = (n + block_size - 1) / block_size;

    PetscFunctionBegin;
    // if n = 0 then nthread = 0, which is not allowed. rather than letting the user try to
    // decipher cryptic 'cuda/hipErrorLaunchFailure' we explicitly check for zero here
    PetscAssert(n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Trying to launch kernel with grid/block size 0");
    PetscCallCUPM(cupmLaunchKernel(std::forward<F>(func), nblock, nthread, sharedMem, stream, std::forward<Args>(kernelArgs)...));
    PetscFunctionReturn(0);
  }

private:
  template <typename F, typename... Args, std::size_t... i>
  PETSC_CXX_COMPAT_DECL(cupmError_t deduceKernelCall(util::index_sequence<i...>, F &&func, cupmDim3 gridDim, cupmDim3 blockDim, std::size_t sharedMem, cupmStream_t stream, Args &&...kernelArgs)) {
    return interface_type::template cupmLaunchKernel(std::forward<F>(func), std::move(gridDim), std::move(blockDim), std::move(sharedMem), std::move(stream),
                                                     // can't static_cast() here since the function argument type may be cv-qualified, in
                                                     // which case we would need to const_cast(). But you can only const_cast()
                                                     // indirect types (pointers, references) and I don't want to add a
                                                     // static_cast_that_becomes_a_const_cast() SFINAE monster to this template mess. C-style
                                                     // casts luckily work here since it tries the following and uses the first one that
                                                     // succeeds:
                                                     // 1. const_cast()
                                                     // 2. static_cast()
                                                     // 3. static_cast() then const_cast()
                                                     // 4. reinterpret_cast()...
                                                     // hopefully we never get to reinterpret_cast() land
                                                     (typename util::func_traits<F>::template arg<i>::type)(kernelArgs)...);
  }
};

#define PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(base_name, T) \
  PETSC_CUPM_IMPL_CLASS_HEADER(PetscConcat(base_name, _impl), T); \
  using base_name = Petsc::Device::CUPM::Impl::Interface<T>; \
  using typename base_name::cupmReal_t; \
  using typename base_name::cupmScalar_t; \
  using base_name::makeCupmScalar; \
  using base_name::cupmScalarCast; \
  using base_name::cupmRealCast; \
  using base_name::cupmGetMemType; \
  using base_name::PetscCUPMMemset; \
  using base_name::PetscCUPMMemsetAsync; \
  using base_name::PetscCUPMMalloc; \
  using base_name::PetscCUPMMallocAsync; \
  using base_name::PetscCUPMMallocHost; \
  using base_name::PetscCUPMMemcpy; \
  using base_name::PetscCUPMMemcpyAsync; \
  using base_name::cupmFree; \
  using base_name::cupmFreeAsync; \
  using base_name::cupmFreeHost; \
  using base_name::cupmLaunchKernel; \
  using base_name::PetscCUPMLaunchKernel1D; \
  using base_name::PetscDeviceCopyModeToCUPMMemcpyKind

} // namespace Impl

} // namespace CUPM

} // namespace Device

} // namespace Petsc

#endif /* __cplusplus */

#endif /* PETSCCUPMINTERFACE_HPP */
