#include "common/scope_exit.h"
#include "core/core.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_resource_limit.h"
#include "core/hle/kernel/svc.h"

namespace Kernel::Svc {

// Helper function to get a resource limit object from a handle.
static KResourceLimit* getResourceLimit(Core::System& system, Handle resource_limit_handle) {
    auto& kernel = system.Kernel();
    KScopedAutoObject resource_limit =
        GetCurrentProcess(kernel).GetHandleTable().GetObject<KResourceLimit>(resource_limit_handle);
    return resource_limit.IsNotNull() ? resource_limit.Get() : nullptr;
}

// Helper function to get a resource value from a resource limit object.
template <typename TValue>
static Result getResourceValue(Core::System& system, TValue* out_value, Handle resource_limit_handle,
                                LimitableResource which, const char* function_name) {
    LOG_DEBUG(Kernel_SVC, "called, resource_limit_handle={:08X}, which={}", resource_limit_handle,
              which);

    // Validate the resource.
    R_UNLESS(IsValidResourceType(which), ResultInvalidEnumValue);

    // Get the resource limit.
    KResourceLimit* resource_limit = getResourceLimit(system, resource_limit_handle);
    R_UNLESS(resource_limit != nullptr, ResultInvalidHandle);

    // Get the value.
    *out_value = static_cast<TValue>(resource_limit->GetLimitValue(which));

    R_SUCCEED();
}

Result CreateResourceLimit(Core::System& system, Handle* out_handle) {
    LOG_DEBUG(Kernel_SVC, "called");

    // Create a new resource limit.
    auto& kernel = system.Kernel();
    KResourceLimit* resource_limit = KResourceLimit::Create(kernel);
    R_UNLESS(resource_limit != nullptr, ResultOutOfResource);

    // Ensure we don't leak a reference to the limit.
    SCOPE_EXIT({ resource_limit->Close(); });

    // Initialize the resource limit.
    resource_limit->Initialize(std::addressof(system.CoreTiming()));

    // Register the limit.
    KResourceLimit::Register(kernel, resource_limit);

    // Add the limit to the handle table.
    R_RETURN(GetCurrentProcess(kernel).GetHandleTable().Add(out_handle, resource_limit));
}

Result GetResourceLimitLimitValue(Core::System& system, s64* out_limit_value,
                                  Handle resource_limit_handle, LimitableResource which) {
    return getResourceValue(system, out_limit_value, resource_limit_handle, which,
                             __FUNCTION__);
}

Result GetResourceLimitCurrentValue(Core::System& system, s64* out_current_value,
                                    Handle resource_limit_handle, LimitableResource which) {
    return getResourceValue(system, out_current_value, resource_limit_handle, which,
                             __FUNCTION__);
}

Result SetResourceLimitLimitValue(Core::System& system, Handle resource_limit_handle,
                                  LimitableResource which, s64 limit_value) {
    LOG_DEBUG(Kernel_SVC, "called, resource_limit_handle={:08X}, which={}, limit_value={}",
              resource_limit_handle, which, limit_value);

    // Validate the resource.
    R_UNLESS(IsValidResourceType(which), ResultInvalidEnumValue);

    // Get the resource limit.
    KResourceLimit* resource_limit = getResourceLimit(system, resource_limit_handle);
    R_UNLESS(resource_limit != nullptr, ResultInvalidHandle);

    // Set the limit value.
    R_RETURN(resource_limit->SetLimitValue(which, limit_value));
}

Result GetResourceLimitPeakValue(Core::System& system, int64_t* out_peak_value,
                                 Handle resource_limit_handle, LimitableResource which) {
    UNIMPLEMENTED();
    R_THROW(ResultNotImplemented);
}

Result CreateResourceLimit64(Core::System& system, Handle* out_handle) {
    return CreateResourceLimit(system, out_handle);
}

Result SetResourceLimitLimitValue64(Core::System& system, Handle resource_limit_handle,
                                    LimitableResource which, int64_t limit_value) {
    return SetResourceLimitLimitValue(system, resource_limit_handle, which, limit_value);
}

// These 64-bit variants of the other functions call their corresponding 32-bit versions.
// Note: these functions are not strictly necessary, and they could be removed for simplicity.
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

Result CreateResourceLimit64From32(Core::System& system, Handle* out_handle) {
    return CreateResourceLimit(system, out_handle);
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

Result SetResourceLimitLimitValue64From32(Core::System& system, Handle resource_limit_handle,
                                          LimitableResource which, int64_t limit_value) {
    return SetResourceLimitLimitValue(system, resource_limit_handle, which, limit_value);
}

} // namespace Kernel::Svc

