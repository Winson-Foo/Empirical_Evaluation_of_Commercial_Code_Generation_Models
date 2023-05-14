#include <memory>
#include <string>
#include <vector>

#include "common/assert.h"
#include "core/core.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/kernel/k_readable_event.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/hle/service/nvdrv/core/container.h"
#include "core/hle/service/nvnflinger/buffer_item_consumer.h"
#include "core/hle/service/nvnflinger/buffer_queue_consumer.h"
#include "core/hle/service/nvnflinger/buffer_queue_core.h"
#include "core/hle/service/nvnflinger/buffer_queue_producer.h"
#include "core/hle/service/nvnflinger/hos_binder_driver_server.h"
#include "core/hle/service/vi/display/vi_display.h"
#include "core/hle/service/vi/layer/vi_layer.h"
#include "core/hle/service/vi/vi_results.h"

namespace Service::VI {

struct BufferQueue {
    std::shared_ptr<android::BufferQueueCore> core;
    std::unique_ptr<android::BufferQueueProducer> producer;
    std::unique_ptr<android::BufferQueueConsumer> consumer;
};

static BufferQueue CreateBufferQueue(KernelHelpers::ServiceContext& service_context,
                                     const Service::Nvidia::NvCore::NvMap& nvmap) {
    auto buffer_queue_core = std::make_shared<android::BufferQueueCore>();
    return {
        buffer_queue_core,
        std::make_unique<android::BufferQueueProducer>(service_context, buffer_queue_core, nvmap),
        std::make_unique<android::BufferQueueConsumer>(buffer_queue_core, nvmap)
    };
}

Display::Display(const u64 id, const std::string& name, Nvnflinger::HosBinderDriverServer& hos_binder_driver_server,
    KernelHelpers::ServiceContext& service_context, Core::System& system)
    : display_id{id}, name{name}, hos_binder_driver_server{hos_binder_driver_server}, service_context{service_context} {
    vsync_event = service_context.CreateEvent(fmt::format("Display VSync Event {}", id));
}

Display::~Display() {
    service_context.CloseEvent(vsync_event);
}

Layer& Display::GetLayer(const std::size_t index) {
    return *layers.at(index);
}

const Layer& Display::GetLayer(const std::size_t index) const {
    return *layers.at(index);
}

ResultVal<Kernel::KReadableEvent*> Display::GetVSyncEvent() {
    if (got_vsync_event) {
        return ResultPermissionDenied;
    }

    got_vsync_event = true;

    return GetVSyncEventUnchecked();
}

Kernel::KReadableEvent* Display::GetVSyncEventUnchecked() {
    return &vsync_event->GetReadableEvent();
}

void Display::SignalVSyncEvent() {
    vsync_event->Signal();
}

void Display::CreateLayer(const u64 layer_id, const u32 binder_id, Service::Nvidia::NvCore::Container& nv_core) {
    ASSERT_MSG(layers.empty(), "Only one layer is supported per display at the moment");

    auto [core, producer, consumer] = CreateBufferQueue(service_context, nv_core.GetNvMapFile());

    auto buffer_item_consumer = std::make_shared<android::BufferItemConsumer>(std::move(consumer));
    buffer_item_consumer->Connect(false);

    layers.emplace_back(std::make_unique<Layer>(layer_id, binder_id, *core, *producer, std::move(buffer_item_consumer)));

    hos_binder_driver_server.RegisterProducer(std::move(producer));
}

void Display::CloseLayer(const u64 layer_id) {
    std::erase_if(layers, [layer_id](const auto& layer) { return layer->GetLayerId() == layer_id; });
}

Layer* Display::FindLayer(const u64 layer_id) {
    for (auto& layer : layers) {
        if (layer->GetLayerId() == layer_id) {
            return layer.get();
        }
    }

    return nullptr;
}

const Layer* Display::FindLayer(const u64 layer_id) const {
    for (const auto& layer : layers) {
        if (layer->GetLayerId() == layer_id) {
            return layer.get();
        }
    }

    return nullptr;
}

} // namespace Service::VI

