#include "core/core.h"
#include "core/hle/kernel/handle_table.h"
#include "core/hle/kernel/k_resource_limit.h"
#include "core/hle/kernel/svc.h"

namespace Kernel::Svc {

Result CreateResourceLimit(Core::System& system, Handle* out_handle);
Result GetResourceLimitValue(Core::System& system, s64* out_value, Handle resource_limit_handle, LimitableResource which);
Result SetResourceLimitValue(Core::System& system, Handle resource_limit_handle, LimitableResource which, s64 value);
Result GetResourceLimitPeakValue(Core::System& system, s64* out_peak_value, Handle resource_limit_handle, LimitableResource which);

Result CreateResourceLimit64(Core::System& system, Handle* out_handle);
Result GetResourceLimitValue64(Core::System& system, int64_t* out_value, Handle resource_limit_handle, LimitableResource which);
Result SetResourceLimitValue64(Core::System& system, Handle resource_limit_handle, LimitableResource which, int64_t value);
Result GetResourceLimitPeakValue64(Core::System& system, int64_t* out_peak_value, Handle resource_limit_handle, LimitableResource which);

} // namespace Kernel::Svc
```

Source file:
```
#include <iostream>
#include <memory>

#include "common/scope_exit.h"
#include "core/hle/kernel/k_process.h"

namespace Kernel::Svc {

void log_debug(const std::string& message) {
    std::cout << message << std::endl;
}

std::unique_ptr<KResourceLimit> get_resource_limit(Core::System& system, Handle resource_limit_handle) {
    auto& kernel = system.Kernel();
    auto& handle_table = GetCurrentProcess(kernel).GetHandleTable();
    auto resource_limit = handle_table.GetObject<KResourceLimit>(resource_limit_handle);
    if (resource_limit.IsNull()) {
        return nullptr;
    }
    return std::make_unique<KResourceLimit>(*resource_limit);
}

Result CreateResourceLimit(Core::System& system, Handle* out_handle) {
    log_debug("CreateResourceLimit called");

    auto& kernel = system.Kernel();
    auto resource_limit = KResourceLimit::Create(kernel);
    R_UNLESS(resource_limit != nullptr, ResultOutOfResource);

    SCOPE_EXIT({ resource_limit->Close(); });

    resource_limit->Initialize(std::addressof(system.CoreTiming()));
    KResourceLimit::Register(kernel, resource_limit);
    R_RETURN(GetCurrentProcess(kernel).GetHandleTable().Add(out_handle, resource_limit));
}

Result GetResourceLimitValue(Core::System& system, s64* out_value, Handle resource_limit_handle, LimitableResource which) {
    log_debug("GetResourceLimitValue called");

    R_UNLESS(IsValidResourceType(which), ResultInvalidEnumValue);

    auto resource_limit = get_resource_limit(system, resource_limit_handle);
    R_UNLESS(resource_limit != nullptr, ResultInvalidHandle);

    *out_value = resource_limit->GetLimitValue(which);
    R_SUCCEED();
}

Result SetResourceLimitValue(Core::System& system, Handle resource_limit_handle, LimitableResource which, s64 value) {
    log_debug("SetResourceLimitValue called");

    R_UNLESS(IsValidResourceType(which), ResultInvalidEnumValue);

    auto resource_limit = get_resource_limit(system, resource_limit_handle);
    R_UNLESS(resource_limit != nullptr, ResultInvalidHandle);

    R_RETURN(resource_limit->SetLimitValue(which, value));
}

Result GetResourceLimitPeakValue(Core::System& system, s64* out_peak_value, Handle resource_limit_handle, LimitableResource which) {
    log_debug("GetResourceLimitPeakValue called");

    UNIMPLEMENTED();
    R_THROW(ResultNotImplemented);
}

Result CreateResourceLimit64(Core::System& system, Handle* out_handle) {
    log_debug("CreateResourceLimit64 called");

    return CreateResourceLimit(system, out_handle);
}

Result GetResourceLimitValue64(Core::System& system, int64_t* out_value, Handle resource_limit_handle, LimitableResource which) {
    log_debug("GetResourceLimitValue64 called");

    s64 value;
    R_RETURN(GetResourceLimitValue(system, &value, resource_limit_handle, which));

    *out_value = value;
}

Result SetResourceLimitValue64(Core::System& system, Handle resource_limit_handle, LimitableResource which, int64_t value) {
    log_debug("SetResourceLimitValue64 called");

    return SetResourceLimitValue(system, resource_limit_handle, which, value);
}

Result GetResourceLimitPeakValue64(Core::System& system, int64_t* out_peak_value, Handle resource_limit_handle, LimitableResource which) {
    log_debug("GetResourceLimitPeakValue64 called");

    s64 peak_value;
    R_RETURN(GetResourceLimitPeakValue(system, &peak_value, resource_limit_handle, which));

    *out_peak_value = peak_value;
}

} // namespace Kernel::Svc