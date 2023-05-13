#ifndef SINK_DETAILS_H
#define SINK_DETAILS_H

#include <string>
#include <vector>
#include <memory>

#include "audio_core/sink/sink.h"

namespace AudioCore::Sink {
    struct SinkDetails {
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

    const SinkDetails& GetOutputSinkDetails(std::string_view sink_id);

    std::vector<std::string_view> GetSinkIDs();

    std::vector<std::string> GetDeviceListForSink(std::string_view sink_id, bool capture);

    std::unique_ptr<Sink> CreateSinkFromID(std::string_view sink_id, std::string_view device_id);
}

#endif // SINK_DETAILS_H

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
    // sink_details is ordered in terms of desirability, with the best choice at the top.
    static constexpr SinkDetails sink_details[] = {
#ifdef HAVE_CUBEB
        SinkDetails{
            "cubeb",
            [](std::string_view device_id) -> std::unique_ptr<Sink> {
                return std::make_unique<CubebSink>(device_id);
            },
            &ListCubebSinkDevices,
            &GetCubebLatency,
        },
#endif
#ifdef HAVE_SDL2
        SinkDetails{
            "sdl2",
            [](std::string_view device_id) -> std::unique_ptr<Sink> {
                return std::make_unique<SDLSink>(device_id);
            },
            &ListSDLSinkDevices,
            &GetSDLLatency,
        },
#endif
        SinkDetails{"null",
                    [](std::string_view device_id) -> std::unique_ptr<Sink> {
                        return std::make_unique<NullSink>(device_id);
                    },
                    [](bool capture) { return std::vector<std::string>{"null"}; }, []() { return 0u; }},
    };

    const SinkDetails& GetOutputSinkDetails(std::string_view sink_id) {
        const auto find_backend{[](std::string_view id) {
            return std::find_if(std::begin(sink_details), std::end(sink_details),
                                [&id](const auto& sink_detail) { return sink_detail.id == id; });
        }};

        auto iter = find_backend(sink_id);

        if (sink_id == "auto") {
            // Auto-select a backend. Prefer CubeB, but it may report a large minimum latency which
            // causes audio issues, in that case go with SDL.
#if defined(HAVE_CUBEB) && defined(HAVE_SDL2)
            iter = find_backend("cubeb");
            if (iter->latency() > TargetSampleCount * 3) {
                iter = find_backend("sdl2");
            }
#else
            iter = std::begin(sink_details);
#endif
            LOG_INFO(Service_Audio, "Auto-selecting the {} backend", iter->id);
        }

        if (iter == std::end(sink_details)) {
            LOG_ERROR(Audio, "Invalid sink_id {}", sink_id);
            iter = find_backend("null");
        }

        return *iter;
    }
} // Anonymous namespace

std::vector<std::string_view> GetSinkIDs() {
    std::vector<std::string_view> sink_ids(std::size(sink_details));

    std::transform(std::begin(sink_details), std::end(sink_details), std::begin(sink_ids),
                   [](const auto& sink) { return sink.id; });

    return sink_ids;
}

std::vector<std::string> GetDeviceListForSink(std::string_view sink_id, bool capture) {
    return GetOutputSinkDetails(sink_id).list_devices(capture);
}

std::unique_ptr<Sink> CreateSinkFromID(std::string_view sink_id, std::string_view device_id) {
    return GetOutputSinkDetails(sink_id).factory(device_id);
}

#include "audio_core/sink/null_sink.h"
#include "audio_core/sink/sink_details.h"

namespace AudioCore::Sink {
    NullSink::NullSink(std::string_view device_id) : Sink{device_id} {
    }

    void NullSink::SubmitBufferImpl(f32, const s16*, u32) {}

    u32 NullSink::GetLatency() const {
        return GetOutputSinkDetails("null").latency();
    }
}

#include "audio_core/sink/cubeb_sink.h"
#ifdef HAVE_CUBEB
#include "audio_core/sink/sink_details.h"

namespace AudioCore::Sink {
    std::vector<std::string> ListCubebSinkDevices(bool capture) {
        std::vector<std::string> device_list;
        auto ctx = CubebContextPtr{};

        if (auto result = cubeb_init(&ctx, "yuzu"); result != CUBEB_OK) {
            LOG_ERROR(Audio, "Failed to init cubeb: {}", cubeb_error_string(result));
            return device_list;
        }

        cubeb_device_type device_type = capture ? CUBEB_DEVICE_TYPE_INPUT : CUBEB_DEVICE_TYPE_OUTPUT;
        cubeb_device_info** info;

        if (const auto result = cubeb_enumerate_devices(ctx, device_type, &info); result != CUBEB_OK) {
            LOG_ERROR(Audio, "Failed to enumerate cubeb devices: {}", cubeb_error_string(result));
            return device_list;
        }

        for (auto i = 0;; ++i) {
            auto* candidate_info = info[i];
            if (!candidate_info) {
                break;
            }
            // Filter out non-matching devices.
            if ((capture && candidate_info->input_channels < TargetChannelCount) ||
                (!capture && candidate_info->output_channels < TargetChannelCount)) {
                continue;
            }
            device_list.emplace_back(candidate_info->name);
        }

        cubeb_device_collection_destroy(ctx, info);

        if (device_list.empty()) {
            // Category still fails, list all devices.
            if (const auto result = cubeb_enumerate_devices(ctx, CUBEB_DEVICE_TYPE_OUTPUT, &info);
                result != CUBEB_OK) {
                LOG_ERROR(Audio, "Failed to enumerate cubeb devices: {}", cubeb_error_string(result));
                return device_list;
            }

            for (auto i = 0;; ++i) {
                auto* candidate_info = info[i];
                if (!candidate_info) {
                    break;
                }
                device_list.emplace_back(candidate_info->name);
            }

            cubeb_device_collection_destroy(ctx, info);
        }

        return device_list;
    }

    CubebSink::CubebSink(std::string_view device_id) : Sink{device_id} {
        constexpr const char* const kCubebStreamName = "yuzu audio";

        cubeb_stream_params params = {};
        params.prefs = CUBEB_STREAM_PREF_NONE;
        params.format = CUBEB_SAMPLE_S16LE;
        params.rate = TargetSampleRate;
        params.channels = TargetChannelCount;
        params.layout = CUBEB_LAYOUT_UNDEFINED;

        const auto init_result = cubeb_init_stream(
            CubebContextPtr{}, &m_stream, device_id.data(), CUBEB_STREAM_OUTPUT, &params, nullptr,
            nullptr, nullptr);

        if (init_result != CUBEB_OK) {
            LOG_ERROR(Audio, "Failed to initialize cubeb stream: {}", cubeb_error_string(init_result));
        } else {
            if (const auto start_result = cubeb_stream_start(m_stream); start_result != CUBEB_OK) {
                LOG_ERROR(Audio, "Failed to start cubeb stream: {}", cubeb_error_string(start_result));
            } else {
                LOG_INFO(Service_Audio, "Using cubeb audio backend with device {}", device_id);
            }
        }
    }

    void CubebSink::SubmitBufferImpl(const f32, const s16* buffer, const u32 buffer_size) {
        const auto buffer_in_frames = buffer_size / TargetChannelCount / sizeof(*buffer);

        const auto write_result = cubeb_stream_write(m_stream, buffer, buffer_in_frames);
        if (write_result != buffer_in_frames) {
            LOG_ERROR(Audio, "Failed to write audio buffer to cubeb stream: expected {}, wrote {}", buffer_in_frames, write_result);
        }
    }

    u32 GetCubebLatency() {
        return std::max(u32{TargetSampleCount}, cubeb_get_min_latency(nullptr, 48000u));
    }
}
#endif

#include "audio_core/sink/sdl2_sink.h"
#ifdef HAVE_SDL2
#include "audio_core/sink/sink_details.h"

#include <algorithm>

namespace AudioCore::Sink {
    std::vector<std::string> ListSDLSinkDevices(bool capture) {
        // Ensure that SDL is initialized and that it can request device IDs.
        SDL_InitSubSystem(SDL_INIT_AUDIO);

        const auto device_count = SDL_GetNumAudioDevices(capture ? SDL_TRUE : SDL_FALSE);
        std::vector<std::string> device_list;

        // Add every valid output device as an option.
        for (auto device_idx = 0; device_idx < device_count; ++device_idx) {
            const auto device_name = SDL_GetAudioDeviceName(device_idx, capture ? SDL_TRUE : SDL_FALSE);

            SDL_AudioSpec want = {};
            want.freq = static_cast<int>(TargetSampleRate);
            want.channels = TargetChannelCount;
            want.format = AUDIO_S16;
            want.samples = static_cast<uint16_t>(TargetSampleCount);

            const auto valid = SDL_OpenAudioDevice(device_name, capture ? SDL_TRUE : SDL_FALSE, &want, nullptr, 0);
            if (valid > 0) {
                device_list.push_back(device_name);
                SDL_CloseAudioDevice(valid);
            }
        }

        return device_list;
    }

    SDLSink::SDLSink(std::string_view device_id) : Sink{device_id} {
        SDL_AudioSpec want = {};
        want.freq = static_cast<int>(TargetSampleRate);
        want.channels = TargetChannelCount;
        want.format = AUDIO_S16;
        want.samples = static_cast<uint16_t>(TargetSampleCount);
        want.callback = nullptr;

        if (device_id.empty()) {
            m_device_id = nullptr;
        } else {
            m_device_id = device_id.data();
        }

        m_device = SDL_OpenAudioDevice(m_device_id, false, &want, &m_audioSpec, SDL_AUDIO_ALLOW_FORMAT_CHANGE);
        if (m_device == 0) {
            const auto err = SDL_GetError();
            LOG_ERROR(Audio, "Failed to open SDL audio device {}: {}", m_device_id, err);
        } else {
            SDL_PauseAudioDevice(m_device, 0);
            LOG_INFO(Service_Audio, "Opened SDL audio device {}: {}Hz, {} channels, {}Hz buffer, {} samples per buffer",
                     m_device_id, m_audioSpec.freq, m_audioSpec.channels, m_audioSpec.samples,
                     m_audioSpec.samples * kTargetSampleRate / m_audioSpec.freq);
        }
    }

    void SDLSink::SubmitBufferImpl(const f32, const s16* buffer, const u32 buffer_size) {
        const auto buffer_in_frames = buffer_size / TargetChannelCount / sizeof(*buffer);

        if (SDL_QueueAudio(m_device, buffer, buffer_size) != 0) {
            LOG_ERROR(Audio, "Failed to queue audio to SDL device: {}", SDL_GetError());
        }
    }

    u32 GetSDLLatency() {
        return SDL_GetQueuedAudioSize(0) / (sizeof(s16) * TargetChannelCount);
    }
}
#endif