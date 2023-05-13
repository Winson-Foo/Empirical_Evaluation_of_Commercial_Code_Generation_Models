#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "audio_core/sink/sink_details.h"
#ifdef HAVE_CUBEB
#include "audio_core/sink/cubeb_sink.h"
#endif
#ifdef HAVE_SDL2
#include "audio_core/sink/sdl2_sink.h"
#endif
#include "audio_core/sink/null_sink.h"
#include "common/logging/log.h"

namespace AudioCore::Sink {
namespace {
using FactoryFn = std::unique_ptr<Sink> (*)(std::string_view);
using ListDevicesFn = std::vector<std::string> (*)(bool);
using LatencyFn = u32 (*)();

struct SinkFactory {
    std::string_view id;
    FactoryFn factory;
    ListDevicesFn list_devices;
    LatencyFn latency;
};

// sink_factories is ordered in terms of desirability, with the best choice at the top.
constexpr SinkFactory sink_factories[] = {
#ifdef HAVE_CUBEB
    {
        "cubeb",
        [](std::string_view device_id) -> std::unique_ptr<Sink> {
            return std::make_unique<CubebSink>(device_id);
        },
        &ListCubebSinkDevices,
        &GetCubebLatency,
    },
#endif
#ifdef HAVE_SDL2
    {
        "sdl2",
        [](std::string_view device_id) -> std::unique_ptr<Sink> {
            return std::make_unique<SDLSink>(device_id);
        },
        &ListSDLSinkDevices,
        &GetSDLLatency,
    },
#endif
    {
        "null",
        [](std::string_view device_id) -> std::unique_ptr<Sink> {
            return std::make_unique<NullSink>(device_id);
        },
        [](bool capture) { return std::vector<std::string>{"null"}; },
        []() { return 0u; },
    },
};

const size_t NUM_SINK_FACTORIES = sizeof(sink_factories) / sizeof(SinkFactory);
const std::string AUTO_SELECT_STRING = "auto";

const SinkFactory& GetOutputSinkFactory(std::string_view sink_id) {
  const auto find_backend = [](std::string_view id) {
      return std::find_if(std::begin(sink_factories), std::end(sink_factories),
                          [&id](const auto& sink_factory) { return sink_factory.id == id; });
  };

  size_t index = NUM_SINK_FACTORIES - 1;
  if (sink_id == AUTO_SELECT_STRING) {
    // Auto-select a backend. Prefer CubeB, but it may report a large minimum latency which
    // causes audio issues, in that case go with SDL.
    for (size_t i = 0; i < NUM_SINK_FACTORIES; i++) {
      const auto& sink_factory = sink_factories[i];
      if (sink_factory.latency() <= TargetSampleCount * 3) {
        index = i;
        break;
      }
    }
    LOG_INFO(Service_Audio, "Auto-selecting the {} backend", sink_factories[index].id);
  }
  else {
    for (size_t i = 0; i < NUM_SINK_FACTORIES; i++) {
      const auto& sink_factory = sink_factories[i];
      if (sink_factory.id == sink_id) {
        index = i;
        break;
      }
    }
    if (index == NUM_SINK_FACTORIES - 1) {
      LOG_ERROR(Audio, "Invalid sink_id {}", sink_id);
    }
  }

  return sink_factories[index];
}
} // Anonymous namespace

std::vector<std::string_view> GetSinkIDs() {
    std::vector<std::string_view> sink_ids;
    sink_ids.reserve(NUM_SINK_FACTORIES);

    std::transform(std::begin(sink_factories), std::end(sink_factories), std::back_inserter(sink_ids),
                   [](const auto& sink_factory) { return sink_factory.id; });

    return sink_ids;
}

std::vector<std::string> GetDeviceListForSink(std::string_view sink_id, bool capture) {
    return GetOutputSinkFactory(sink_id).list_devices(capture);
}

std::unique_ptr<Sink> CreateSinkFromID(std::string_view sink_id, std::string_view device_id) {
    return GetOutputSinkFactory(sink_id).factory(device_id);
}

} // namespace AudioCore::Sink