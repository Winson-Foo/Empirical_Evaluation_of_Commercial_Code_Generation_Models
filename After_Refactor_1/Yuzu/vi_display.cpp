// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright 2019 yuzu Emulator Project

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <fmt/format.h>

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

// Constants
static constexpr ResultVal<Kernel::KReadableEvent*> RESULT_PERMISSION_DENIED =
    ResultVal<Kernel::KReadableEvent*>{nullptr, ResultPermissionDenied};

// Helper structs
struct BufferQueue {
    std::shared_ptr<android::BufferQueueCore> core;
    std::unique_ptr<android::BufferQueueProducer> producer;
    std::unique_ptr<android::BufferQueueConsumer> consumer;
};

// Helper function to create a buffer queue
static BufferQueue create_buffer_queue(KernelHelpers::ServiceContext& service_context,
                                        Service::Nvidia::NvCore::NvMap& nvmap) {
    auto buffer_queue_core = std::make_shared<android::BufferQueueCore>();
    return {buffer_queue_core,
            std::make_unique<android::BufferQueueProducer>(service_context, buffer_queue_core, nvmap),
            std::make_unique<android::BufferQueueConsumer>(buffer_queue_core, nvmap)};
}

// Class implementation
Display::Display(u64 display_id, std::string name, Nvnflinger::HosBinderDriverServer& hos_binder_driver_server,
                 KernelHelpers::ServiceContext& service_context, Core::System& system)
    : display_id{display_id}, name{std::move(name)}, hos_binder_driver_server{hos_binder_driver_server},
      service_context{service_context} {
    vsync_event = service_context.CreateEvent(fmt::format("Display VSync Event {}", display_id));
}

Display::~Display() {
    service_context.CloseEvent(vsync_event);
}

Layer& Display::get_layer(std::size_t index) {
    return *layers.at(index);
}

const Layer& Display::get_layer(std::size_t index) const {
    return *layers.at(index);
}

ResultVal<Kernel::KReadableEvent*> Display::get_vsync_event() {
    if (got_vsync_event) {
        return RESULT_PERMISSION_DENIED;
    }

    got_vsync_event = true;

    return get_vsync_event_unchecked();
}

Kernel::KReadableEvent* Display::get_vsync_event_unchecked() {
    return &vsync_event->GetReadableEvent();
}

void Display::signal_vsync_event() {
    vsync_event->Signal();
}

void Display::create_layer(u64 layer_id, u32 binder_id, Service::Nvidia::NvCore::Container& nv_core) {
    ASSERT_MSG(layers.empty(), "Only one layer is supported per display at the moment");

    auto [core, producer, consumer] = create_buffer_queue(service_context, nv_core.GetNvMapFile());

    auto buffer_item_consumer = std::make_shared<android::BufferItemConsumer>(std::move(consumer));
    buffer_item_consumer->Connect(false);

    layers.emplace_back(std::make_unique<Layer>(layer_id, binder_id, *core, *producer, std::move(buffer_item_consumer)));

    hos_binder_driver_server.RegisterProducer(std::move(producer));
}

void Display::close_layer(u64 layer_id) {
    layers.erase(
        std::remove_if(layers.begin(), layers.end(),
                       [layer_id](const auto& layer) { return layer->GetLayerId() == layer_id; }),
        layers.end());
}

Layer* Display::find_layer(u64 layer_id) {
    const auto itr = std::find_if(layers.begin(), layers.end(),
                                  [layer_id](const auto& layer) { return layer->GetLayerId() == layer_id; });

    if (itr == layers.end()) {
        return nullptr;
    }

    return itr->get();
}

const Layer* Display::find_layer(u64 layer_id) const {
    const auto itr = std::find_if(layers.begin(), layers.end(),
                                  [layer_id](const auto& layer) { return layer->GetLayerId() == layer_id; });

    if (itr == layers.end()) {
        return nullptr;
    }

    return itr->get();
}

}  // namespace Service::VI

