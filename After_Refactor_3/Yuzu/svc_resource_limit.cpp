// File: resource_limit.cpp

#include "common/scope_exit.h"
#include "core/core.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_resource_limit.h"
#include "core/hle/kernel/svc.h"

namespace Kernel::Svc {

// Constants
static constexpr int VALID_RESOURCE_TYPE = 0;
static constexpr int RESULT_INVALID_ENUM_VALUE = 1;
static constexpr int RESULT_OUT_OF_RESOURCE = 2;
static constexpr int RESULT_INVALID_HANDLE = 3;
static constexpr int RESULT_NOT_IMPLEMENTED = 4;

// Helper function to retrieve a resource limit object from the handle table.
static KScopedAutoObject GetResourceLimitObject(Core::System& system,
                                                Handle resource_limit_handle) {
  auto handle_table = GetCurrentProcess(system.Kernel()).GetHandleTable();
  return handle_table.GetObject<KResourceLimit>(resource_limit_handle);
}

// Create a new resource limit and add it to the handle table.
static Result AddResourceLimit(Core::System& system, KResourceLimit*& resource_limit,
                               Handle* out_handle) {
  auto& kernel = system.Kernel();

  // Create a new resource limit.
  resource_limit = KResourceLimit::Create(kernel);
  if (resource_limit == nullptr) {
    return ResultOutOfResource;
  }

  // Ensure we don't leak a reference to the limit.
  SCOPE_EXIT({ resource_limit->Close(); });

  // Initialize the resource limit.
  resource_limit->Initialize(std::addressof(system.CoreTiming()));

  // Register the limit.
  KResourceLimit::Register(kernel, resource_limit);

  // Add the limit to the handle table.
  auto handle_table = GetCurrentProcess(kernel).GetHandleTable();
  return handle_table.Add(out_handle, resource_limit);
}

Result CreateResourceLimit(Core::System& system, Handle* out_handle) {
  LOG_DEBUG(Kernel_SVC, "CreateResourceLimit called");

  KResourceLimit* resource_limit = nullptr;
  Result result = AddResourceLimit(system, resource_limit, out_handle);

  return result;
}

Result GetResourceLimitLimitValue(Core::System& system, s64* out_limit_value,
                                  Handle resource_limit_handle, LimitableResource which) {
  LOG_DEBUG(Kernel_SVC, "GetResourceLimitLimitValue called, handle={:08X}, which={}", resource_limit_handle, which);

  // Validate the resource.
  if (!IsValidResourceType(which)) {
    return ResultInvalidEnumValue;
  }

  // Get the resource limit.
  KScopedAutoObject resource_limit = GetResourceLimitObject(system, resource_limit_handle);
  if (!resource_limit.IsNotNull()) {
    return ResultInvalidHandle;
  }

  // Get the limit value.
  *out_limit_value = resource_limit->GetLimitValue(which);

  return ResultSuccess;
}

Result GetResourceLimitCurrentValue(Core::System& system, s64* out_current_value,
                                    Handle resource_limit_handle, LimitableResource which) {
  LOG_DEBUG(Kernel_SVC, "GetResourceLimitCurrentValue called, handle={:08X}, which={}", resource_limit_handle,
            which);

  // Validate the resource.
  if (!IsValidResourceType(which)) {
    return ResultInvalidEnumValue;
  }

  // Get the resource limit.
  KScopedAutoObject resource_limit = GetResourceLimitObject(system, resource_limit_handle);
  if (!resource_limit.IsNotNull()) {
    return ResultInvalidHandle;
  }

  // Get the current value.
  *out_current_value = resource_limit->GetCurrentValue(which);

  return ResultSuccess;
}

Result SetResourceLimitLimitValue(Core::System& system, Handle resource_limit_handle,
                                  LimitableResource which, s64 limit_value) {
  LOG_DEBUG(Kernel_SVC, "SetResourceLimitLimitValue called, handle={:08X}, which={}, limit_value={}",
            resource_limit_handle, which, limit_value);

  // Validate the resource.
  if (!IsValidResourceType(which)) {
    return ResultInvalidEnumValue;
  }

  // Get the resource limit.
  KScopedAutoObject resource_limit = GetResourceLimitObject(system, resource_limit_handle);
  if (!resource_limit.IsNotNull()) {
    return ResultInvalidHandle;
  }

  // Set the limit value.
  return resource_limit->SetLimitValue(which, limit_value);
}

Result GetResourceLimitPeakValue(Core::System& system, int64_t* out_peak_value,
                                 Handle resource_limit_handle, LimitableResource which) {
  LOG_DEBUG(Kernel_SVC, "GetResourceLimitPeakValue called, handle={:08X}, which={}", resource_limit_handle, which);

  // This function is not implemented yet.
  return ResultNotImplemented;
}

// 32-bit to 64-bit conversions
Result GetResourceLimitLimitValue64(Core::System& system, int64_t* out_limit_value,
                                    Handle resource_limit_handle, LimitableResource which) {
  return GetResourceLimitLimitValue(system, out_limit_value, resource_limit_handle, which);
}

Result GetResourceLimitCurrentValue64(Core::System& system, int64_t* out_current_value,
                                      Handle resource_limit_handle, LimitableResource which) {
  return GetResourceLimitCurrentValue(system, out_current_value, resource_limit_handle, which);
}

Result GetResourceLimitPeakValue64(Core::System& system, int64_t* out_peak_value,
                                   Handle resource_limit_handle, LimitableResource which) {
  return GetResourceLimitPeakValue(system, out_peak_value, resource_limit_handle, which);
}

Result CreateResourceLimit64(Core::System& system, Handle* out_handle) {
  return CreateResourceLimit(system, out_handle);
}

Result SetResourceLimitLimitValue64(Core::System& system, Handle resource_limit_handle,
                                    LimitableResource which, int64_t limit_value) {
  return SetResourceLimitLimitValue(system, resource_limit_handle, which, limit_value);
}

Result GetResourceLimitLimitValue64From32(Core::System& system, int64_t* out_limit_value,
                                          Handle resource_limit_handle, LimitableResource which) {
  return GetResourceLimitLimitValue(system, out_limit_value, resource_limit_handle, which);
}

Result GetResourceLimitCurrentValue64From32(Core::System& system, int64_t* out_current_value,
                                            Handle resource_limit_handle, LimitableResource which) {
  return GetResourceLimitCurrentValue(system, out_current_value, resource_limit_handle, which);
}

Result GetResourceLimitPeakValue64From32(Core::System& system, int64_t* out_peak_value,
                                         Handle resource_limit_handle, LimitableResource which) {
  return GetResourceLimitPeakValue(system, out_peak_value, resource_limit_handle, which);
}

Result CreateResourceLimit64From32(Core::System& system, Handle* out_handle) {
  return CreateResourceLimit(system, out_handle);
}

Result SetResourceLimitLimitValue64From32(Core::System& system, Handle resource_limit_handle,
                                          LimitableResource which, int64_t limit_value) {
  return SetResourceLimitLimitValue(system, resource_limit_handle, which, limit_value);
}

}  // namespace Kernel::Svc

