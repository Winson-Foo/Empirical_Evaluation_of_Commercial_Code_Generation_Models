#pragma once

#include <memory>
#include <string>
#include <vector>

#include "nvnflinger/hos_binder_driver_server.h"
#include "service/kernel_helpers.h"
#include "service/nvidia/nvcore/container.h"
#include "vi/layer/layer.h"
#include "vi/vi_results.h"

namespace Service::VI {

    class Display {
    public:
        explicit Display(u64 id, std::string name, Nvnflinger::HosBinderDriverServer& hos_binder_driver_server,
                          KernelHelpers::ServiceContext& service_context, Core::System& system);
        ~Display();

        Layer& GetLayer(std::size_t index);
        const Layer& GetLayer(std::size_t index) const;
        ResultVal<Kernel::KReadableEvent*> GetVSyncEvent();
        void SignalVSyncEvent();
        void CreateLayer(u64 layer_id, u32 binder_id, Service::Nvidia::NvCore::Container& nv_core);
        void CloseLayer(u64 layer_id);

    private:
        Kernel::KReadableEvent* GetVSyncEventUnchecked();
        Layer* FindLayer(u64 layer_id);
        const Layer* FindLayer(u64 layer_id) const;

        u64 display_id;
        std::string name;
        Nvnflinger::HosBinderDriverServer& hos_binder_driver_server;
        KernelHelpers::ServiceContext& service_context;
        std::vector<std::unique_ptr<Layer>> layers;
        std::unique_ptr<Kernel::KEvent> vsync_event;
        bool got_vsync_event = false;
    };

} // namespace Service::VI


// display.cpp

#include <algorithm>
#include <utility>

#include <fmt/format.h>

#include "assert.h"
#include "core.h"
#include "hle/kernel/k_event.h"
#include "hle/kernel/k_readable_event.h"
#include "nvnflinger/buffer_item_consumer.h"
#include "nvnflinger/buffer_queue_consumer.h"
#include "nvnflinger/buffer_queue_core.h"
#include "nvnflinger/buffer_queue_producer.h"
#include "vi/display.h"
#include "vi/vi_results.h"

namespace {

    Service::VI::BufferQueue CreateBufferQueue(KernelHelpers::ServiceContext& service_context,
                                               Service::Nvidia::NvCore::NvMap& nvmap) {
        auto buffer_queue_core = std::make_shared<android::BufferQueueCore>();
        return {
            buffer_queue_core,
            std::make_unique<android::BufferQueueProducer>(service_context, buffer_queue_core, nvmap),
            std::make_unique<android::BufferQueueConsumer>(buffer_queue_core, nvmap)};
    }

} // namespace

namespace Service::VI {

    Display::Display(u64 id, std::string name_, Nvnflinger::HosBinderDriverServer& hos_binder_driver_server_,
                     KernelHelpers::ServiceContext& service_context_, Core::System& system_)
        : display_id{id}, name{std::move(name_)}, hos_binder_driver_server{hos_binder_driver_server_},
          service_context{service_context_} {
        vsync_event = service_context.CreateEvent(fmt::format("Display VSync Event {}", id));
    }

    Display::~Display() {
        service_context.CloseEvent(vsync_event);
    }

    Layer& Display::GetLayer(std::size_t index) {
        return *layers.at(index);
    }

    const Layer& Display::GetLayer(std::size_t index) const {
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

    void Display::CreateLayer(u64 layer_id, u32 binder_id, Service::Nvidia::NvCore::Container& nv_core) {
        ASSERT_MSG(layers.empty(), "Only one layer is supported per display at the moment");

        auto [core, producer, consumer] = CreateBufferQueue(service_context, nv_core.GetNvMapFile());

        auto buffer_item_consumer = std::make_shared<android::BufferItemConsumer>(std::move(consumer));
        buffer_item_consumer->Connect(false);

        layers.emplace_back(std::make_unique<Layer>(layer_id, binder_id, *core, *producer,
                                                    std::move(buffer_item_consumer)));

        hos_binder_driver_server.RegisterProducer(std::move(producer));
    }

    void Display::CloseLayer(u64 layer_id) {
        layers.erase(std::remove_if(layers.begin(), layers.end(),
                                    [layer_id](const auto& layer) { return layer->GetLayerId() == layer_id; }),
                     layers.end());
    }

    Layer* Display::FindLayer(u64 layer_id) {
        for (auto& layer : layers) {
            if (layer->GetLayerId() == layer_id) {
                return layer.get();
            }
        }

        return nullptr;
    }

    const Layer* Display::FindLayer(u64 layer_id) const {
        for (const auto& layer : layers) {
            if (layer->GetLayerId() == layer_id) {
                return layer.get();
            }
        }

        return nullptr;
    }

} // namespace Service::VI

