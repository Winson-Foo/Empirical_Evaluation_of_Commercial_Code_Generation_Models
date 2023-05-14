// File: audio_in_system.h

#pragma once

#include <string_view>
#include <span>
#include <memory>
#include "audio_buffer.h"
#include "result.h"

namespace AudioCore {

class System;
class KEvent;

namespace AudioIn {

class IDeviceSession {
public:
    virtual ~IDeviceSession() {}
    virtual void Finalize() = 0;
    virtual void Start() = 0;
    virtual void Stop() = 0;
    virtual void Initialize(std::string_view name, SampleFormat format,
                             u16 channel_count, size_t session_id,
                             u32 handle, u64 applet_resource_user_id,
                             Sink::StreamType type) = 0;
    virtual void SetVolume(f32 volume) = 0;
    virtual void AppendBuffers(const std::vector<AudioBuffer>& buffers) = 0;
    virtual void ClearBuffers() = 0;
    virtual void SetRingSize(u32 size) = 0;
    virtual u64 GetPlayedSampleCount() const = 0;
};

class System {
public:
    System(System& system, KEvent* event, size_t session_id);
    ~System();

    Result Initialize(std::string_view device_name, const AudioInParameter& in_params,
                      u32 handle, u64 applet_resource_user_id);
    Result IsConfigValid(std::string_view device_name, const AudioInParameter& in_params) const;
    Result Start();
    Result Stop();
    Result FlushBuffers();
    size_t GetSessionId() const;
    u16 GetChannelCount() const;
    u32 GetSampleRate() const;
    SampleFormat GetSampleFormat() const;
    State GetState() const;
    std::string GetName() const;
    f32 GetVolume() const;
    void SetVolume(f32 volume);
    bool ContainsBuffer(u64 tag) const;
    u32 GetBufferCount() const;
    u64 GetPlayedSampleCount() const;

    bool AppendBuffer(const AudioInBuffer& buffer, u64 tag);

private:
    enum class State { Stopped, Started };
    static constexpr size_t kDefaultBufferSize = 8;

    System& system_;
    KEvent* event_;
    size_t session_id_;
    std::unique_ptr<IDeviceSession> session_;
    State state_;
    std::string name_;
    u32 sample_rate_;
    SampleFormat sample_format_;
    u16 channel_count_;
    bool is_uac_;
    f32 volume_;
    AudioBufferList buffers_;

    std::string_view GetDefaultDeviceName() const;
    std::string_view GetDefaultUacDeviceName() const;
    void RegisterBuffers();
    void ReleaseBuffers(bool signal);
};

} // namespace AudioIn
} // namespace AudioCore

// File: audio_in_system.cpp

#include "audio_in_system.h"
#include "audio_manager.h"
#include "core.h"
#include "common/logging/log.h"

namespace AudioCore {
namespace AudioIn {

System::System(System& system, KEvent* event, size_t session_id)
    : system_(system), event_(event), session_id_(session_id),
      buffers_(kDefaultBufferSize), state_(State::Stopped) {}

System::~System() {
    Stop();
}

Result System::Initialize(std::string_view device_name, const AudioInParameter& in_params,
                          u32 handle, u64 applet_resource_user_id) {
    auto result{ IsConfigValid(device_name, in_params) };
    if (result.IsError()) {
        return result;
    }

    name_ = device_name.empty() ? GetDefaultDeviceName() : device_name;
    sample_rate_ = TargetSampleRate;
    sample_format_ = SampleFormat::PcmInt16;
    channel_count_ = in_params.channel_count <= 2 ? 2 : 6;
    volume_ = 1.0f;
    is_uac_ = name_ == GetDefaultUacDeviceName() || name_ == "Uac";

    session_ = AudioManager::CreateDeviceSession(system_, name_);
    session_->Initialize(name_, sample_format_, channel_count_, session_id_,
                          handle, applet_resource_user_id, Sink::StreamType::In);
    session_->SetVolume(volume_);
    session_->Start();

    std::vector<AudioBuffer> buffers_to_flush;
    buffers_.RegisterBuffers(buffers_to_flush);
    session_->AppendBuffers(buffers_to_flush);
    session_->SetRingSize(static_cast<u32>(buffers_to_flush.size()));

    return ResultSuccess;
}

Result System::IsConfigValid(std::string_view device_name, const AudioInParameter& in_params) const {
    if (device_name.size() > 0 && device_name != GetDefaultDeviceName() &&
        device_name != GetDefaultUacDeviceName()) {
        return ResultNotFound;
    }

    if (in_params.sample_rate != TargetSampleRate && in_params.sample_rate > 0) {
        return ResultInvalidSampleRate;
    }

    return ResultSuccess;
}

Result System::Start() {
    if (state_ != State::Stopped) {
        return ResultOperationFailed;
    }

    session_->Start();
    state_ = State::Started;

    std::vector<AudioBuffer> buffers_to_flush;
    buffers_.RegisterBuffers(buffers_to_flush);
    session_->AppendBuffers(buffers_to_flush);
    session_->SetRingSize(static_cast<u32>(buffers_to_flush.size()));

    return ResultSuccess;
}

Result System::Stop() {
    if (state_ == State::Started) {
        session_->SetVolume(0.0f);
        session_->Stop();
        buffers_.Clear();
        if (buffers_.ReleaseBuffers(system_.CoreTiming(), *session_, true)) {
            event_->Signal();
        }
        state_ = State::Stopped;
    }

    return ResultSuccess;
}

Result System::FlushBuffers() {
    if (state_ != State::Started) {
        return ResultOperationFailed;
    }

    u32 buffers_released{};
    buffers_.FlushBuffers(buffers_released);

    if (buffers_released > 0) {
        event_->Signal();
    }
    return ResultSuccess;
}

size_t System::GetSessionId() const {
    return session_id_;
}

u16 System::GetChannelCount() const {
    return channel_count_;
}

u32 System::GetSampleRate() const {
    return sample_rate_;
}

SampleFormat System::GetSampleFormat() const {
    return sample_format_;
}

State System::GetState() const {
    return state_;
}

std::string_view System::GetDefaultDeviceName() const {
    return "BuiltInHeadset";
}

std::string_view System::GetDefaultUacDeviceName() const {
    return "Uac";
}

std::string System::GetName() const {
    return name_;
}

f32 System::GetVolume() const {
    return volume_;
}

void System::SetVolume(f32 volume) {
    volume_ = volume;
    session_->SetVolume(volume);
}

bool System::ContainsBuffer(u64 tag) const {
    return buffers_.ContainsBuffer(tag);
}

u32 System::GetBufferCount() const {
    return buffers_.GetAppendedRegisteredCount();
}

u64 System::GetPlayedSampleCount() const {
    return session_->GetPlayedSampleCount();
}

bool System::AppendBuffer(const AudioInBuffer& buffer, u64 tag) {
    if (buffers_.GetTotalBufferCount() == BufferCount) {
        return false;
    }

    const auto timestamp{ buffers_.GetNextTimestamp() };
    const AudioBuffer new_buffer{
        .start_timestamp = timestamp,
        .end_timestamp = timestamp + buffer.size / (channel_count_ * sizeof(s16)),
        .played_timestamp = 0,
        .samples = buffer.samples,
        .tag = tag,
        .size = buffer.size,
        .metadata = {}
    };

    buffers_.AppendBuffer(new_buffer);
    RegisterBuffers();

    return true;
}

void System::RegisterBuffers() {
    if (state_ == State::Started) {
        std::vector<AudioBuffer> registered_buffers;
        buffers_.RegisterBuffers(registered_buffers);
        session_->AppendBuffers(registered_buffers);
    }
}

void System::ReleaseBuffers(bool signal) {
    bool released_buffers{ buffers_.ReleaseBuffers(system_.CoreTiming(), *session_, false) };

    if (signal && released_buffers) {
        event_->Signal();
    }
}

} // namespace AudioIn
} // namespace AudioCore

