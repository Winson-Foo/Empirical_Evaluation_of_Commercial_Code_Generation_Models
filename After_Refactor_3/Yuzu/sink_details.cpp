// sink_config.h
#pragma once

#include <string_view>
#include <vector>

#include "audio_core/sink/sink.h"

namespace AudioCore::Sink {

struct SinkConfig {
    using FactoryFn = std::unique_ptr<Sink> (*)(std::string_view);
    using ListDevicesFn = std::vector<std::string> (*)(bool);
    using LatencyFn = u32 (*)();

    /// Name for this sink.
    std::string_view id;
    /// A method to call to construct an instance of this type of sink.
    FactoryFn factory;
    /// A method to call to list available devices.
    ListDevicesFn list_devices;
    /// Method to get the latency of this backend.
    LatencyFn latency;
};

std::vector<SinkConfig> GetSinkConfigs();

} // namespace AudioCore::Sink

// sink_config.cpp
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "audio_core/sink/sink_config.h"
#include "audio_core/sink/cubeb_sink.h"
#include "audio_core/sink/null_sink.h"
#include "audio_core/sink/sdl2_sink.h"

#ifdef HAVE_CUBEB
#include "audio_core/sink/cubeb_sink.h"
#endif

#ifdef HAVE_SDL2
#include "audio_core/sink/sdl2_sink.h"
#endif

namespace AudioCore::Sink {

std::vector<SinkConfig> GetSinkConfigs() {
    std::vector<SinkConfig> configs;

#ifdef HAVE_CUBEB
    configs.push_back({
        "cubeb",
        [](std::string_view device_id) -> std::unique_ptr<Sink> {
            return std::make_unique<CubebSink>(device_id);
        },
        &ListCubebSinkDevices,
        &GetCubebLatency,
    });
#endif

#ifdef HAVE_SDL2
    configs.push_back({
        "sdl2",
        [](std::string_view device_id) -> std::unique_ptr<Sink> {
            return std::make_unique<SDLSink>(device_id);
        },
        &ListSDLSinkDevices,
        &GetSDLLatency,
    });
#endif

    configs.push_back({
        "null",
        [](std::string_view device_id) -> std::unique_ptr<Sink> {
            return std::make_unique<NullSink>(device_id);
        },
        [](bool capture) {
            return std::vector<std::string>{"null"};
        },
        []() {
            return 0u;
        }
    });

    return configs;
}

} // namespace AudioCore::Sink

// sink_manager.h
#pragma once

#include <string_view>
#include <vector>

#include "audio_core/sink/sink.h"

namespace AudioCore::Sink {

std::vector<std::string_view> GetSinkIDs();
std::vector<std::string> GetDeviceListForSink(std::string_view sink_id, bool capture);
std::unique_ptr<Sink> CreateSinkFromID(std::string_view sink_id, std::string_view device_id);

} // namespace AudioCore::Sink

// sink_manager.cpp
#include <algorithm>
#include <string_view>
#include <vector>

#include "audio_core/sink/sink_config.h"
#include "audio_core/sink/sink_manager.h"

#include "common/logging/log.h"

namespace AudioCore::Sink {

namespace {

const SinkConfig& GetOutputSinkConfig(std::string_view sink_id) {
    const auto& configs = GetSinkConfigs();

    auto iter = std::find_if(configs.begin(), configs.end(), [&sink_id](const auto& config) {
        return config.id == sink_id;
    });

    if (sink_id == "auto") {
        // Auto-select a backend. Prefer CubeB, but it may report a large minimum latency which
        // causes audio issues, in that case go with SDL.
#if defined(HAVE_CUBEB) && defined(HAVE_SDL2)
        iter = std::find_if(configs.begin(), configs.end(), [](const auto& config) {
            return config.id == "cubeb";
        });

        if (iter->latency() > TargetSampleCount * 3) {
            iter = std::find_if(configs.begin(), configs.end(), [](const auto& config) {
                return config.id == "sdl2";
            });
        }
#else
        iter = configs.begin();
#endif
        LOG_INFO(Service_Audio, "Auto-selecting the {} backend", iter->id);
    }

    if (iter == configs.end()) {
        LOG_ERROR(Audio, "Invalid sink_id {}", sink_id);
        iter = std::find_if(configs.begin(), configs.end(), [](const auto& config) {
            return config.id == "null";
        });
    }

    return *iter;
}

} // Anonymous namespace

std::vector<std::string_view> GetSinkIDs() {
    std::vector<std::string_view> sink_ids;
    const auto& configs = GetSinkConfigs();

    sink_ids.reserve(configs.size());

    for (const auto& config : configs) {
        sink_ids.push_back(config.id);
    }

    return sink_ids;
}

std::vector<std::string> GetDeviceListForSink(std::string_view sink_id, bool capture) {
    return GetOutputSinkConfig(sink_id).list_devices(capture);
}

std::unique_ptr<Sink> CreateSinkFromID(std::string_view sink_id, std::string_view device_id) {
    return GetOutputSinkConfig(sink_id).factory(device_id);
}

} // namespace AudioCore::Sink

