#include "sycldevice.hpp"
#include <CL/sycl.hpp>

namespace Petsc
{

namespace device
{

namespace sycl
{

namespace impl
{

class DeviceContext {
public:
  struct PetscDeviceContext_IMPLS {
    ::sycl::event event;
    ::sycl::event begin; // timer-only
    ::sycl::event end;   // timer-only
#if PetscDefined(USE_DEBUG)
    PetscBool timerInUse;
#endif
  };

private:
  static bool initialized_;

  PETSC_NODISCARD static PetscErrorCode finalize_() noexcept
  {
    PetscFunctionBegin;
    initialized_ = false;
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode initialize_(PetscInt id, DeviceContext *dci) noexcept
  {
    PetscFunctionBegin;
    PetscCall(PetscDeviceCheckDeviceCount_Internal(id));
    if (!initialized_) {
      initialized_ = true;
      PetscCall(PetscRegisterFinalize(finalize_));
    }
    PetscFunctionReturn(0);
  }

public:
  const struct _DeviceContextOps ops = {destroy, changeStreamType, setUp, query, waitForContext, synchronize, getBlasHandle, getSolverHandle, getStreamHandle, beginTimer, endTimer, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

  // default constructor
  DeviceContext() noexcept = default;

  // All of these functions MUST be static in order to be callable from C, otherwise they
  // get the implicit 'this' pointer tacked on
  PETSC_NODISCARD static PetscErrorCode destroy(PetscDeviceContext dctx) noexcept
  {
    PetscFunctionBegin;
    delete static_cast<PetscDeviceContext_IMPLS *>(dctx->data);
    dctx->data = nullptr;
    PetscFunctionReturn(0);
  };
  PETSC_NODISCARD static PetscErrorCode changeStreamType(PetscDeviceContext, PetscStreamType) noexcept { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); };
  PETSC_NODISCARD static PetscErrorCode setUp(PetscDeviceContext) noexcept { return 0; }; // Nothing to setup
  PETSC_NODISCARD static PetscErrorCode query(PetscDeviceContext, PetscBool *) noexcept { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); };
  PETSC_NODISCARD static PetscErrorCode waitForContext(PetscDeviceContext, PetscDeviceContext) noexcept { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); };
  PETSC_NODISCARD static PetscErrorCode synchronize(PetscDeviceContext) noexcept { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); };
  PETSC_NODISCARD static PetscErrorCode getBlasHandle(PetscDeviceContext, void *) noexcept { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); };
  PETSC_NODISCARD static PetscErrorCode getSolverHandle(PetscDeviceContext, void *) noexcept { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); };
  PETSC_NODISCARD static PetscErrorCode getStreamHandle(PetscDeviceContext, void *) noexcept { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); };
  PETSC_NODISCARD static PetscErrorCode beginTimer(PetscDeviceContext) noexcept { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); };
  PETSC_NODISCARD static PetscErrorCode endTimer(PetscDeviceContext, PetscLogDouble *) noexcept { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); };
};

} // namespace impl

} // namespace sycl

} // namespace device

} // namespace Petsc

PetscErrorCode PetscDeviceContextCreate_SYCL(PetscDeviceContext dctx)
{
  using namespace Petsc::device::sycl::impl;

  static const DeviceContext syclctx;

  PetscFunctionBegin;
  dctx->data = new DeviceContext::PetscDeviceContext_IMPLS();
  PetscCall(PetscMemcpy(dctx->ops, &syclctx.ops, sizeof(syclctx.ops)));
  PetscFunctionReturn(0);
}
