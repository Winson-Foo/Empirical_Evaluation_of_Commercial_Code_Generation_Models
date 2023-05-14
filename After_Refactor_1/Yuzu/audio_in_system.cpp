#include <mutex>
#include <string_view>
#include <vector>
#include <span>
#include "audio_core/audio_event.h"
#include "audio_core/audio_manager.h"
#include "audio_core/in/audio_in_system.h"
#include "common/logging/log.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/kernel/k_event.h"

namespace AudioCore::AudioIn {

System::System(Core::System& system, Kernel::KEvent* event, const size_t sessionId)
    : system_{system},
      bufferEvent_{event},
      sessionId_{sessionId},
      session_{std::make_unique<DeviceSession>(system)} {}

System::~System() {
    Finalize();
}

void System::Finalize() {
    Stop();
    session_->Finalize();
}

void System::StartSession() {
    session_->Start();
}

size_t System::GetSessionId() const {
    return sessionId_;
}

std::string_view System::GetDefaultDeviceName() const {
    return "BuiltInHeadset";
}

std::string_view System::GetDefaultUacDeviceName() const {
    return "Uac";
}

Result System::IsConfigValid(const std::string_view deviceName, const AudioInParameter& inParams) const {
    if ((deviceName.size() > 0) &&
        (deviceName != GetDefaultDeviceName() && deviceName != GetDefaultUacDeviceName())) {
        return Service::Audio::ResultNotFound;
    }

    if (inParams.sample_rate != TargetSampleRate && inParams.sample_rate > 0) {
        return Service::Audio::ResultInvalidSampleRate;
    }

    return ResultSuccess;
}

Result System::Initialize(const std::string& deviceName, const AudioInParameter& inParams,
                          const u32 handle, const u64 appletResourceUserId) {
    const auto result = IsConfigValid(deviceName, inParams);
    if (result.IsError()) {
        return result;
    }

    handle_ = handle;
    appletResourceUserId_ = appletResourceUserId;
    if (deviceName.empty() || deviceName[0] == '\0') {
        name_ = std::string{GetDefaultDeviceName()};
    } else {
        name_ = deviceName;
    }

    sampleRate_ = TargetSampleRate;
    sampleFormat_ = SampleFormat::PcmInt16;
    channelCount_ = inParams.channel_count <= 2 ? 2 : 6;
    volume_ = 1.0f;
    isUac_ = (name_ == "Uac");

    return ResultSuccess;
}

Result System::Start() {
    if (state_ != State::Stopped) {
        return Service::Audio::ResultOperationFailed;
    }

    session_->Initialize(name_, sampleFormat_, channelCount_, sessionId_, handle_,
                          appletResourceUserId_, Sink::StreamType::In);
    session_->SetVolume(volume_);
    session_->Start();
    state_ = State::Started;

    std::vector<AudioBuffer> buffersToFlush{};
    buffers_.RegisterBuffers(buffersToFlush);
    session_->AppendBuffers(buffersToFlush);
    session_->SetRingSize(static_cast<u32>(buffersToFlush.size()));

    return ResultSuccess;
}

Result System::Stop() {
    if (state_ == State::Started) {
        session_->Stop();
        session_->SetVolume(0.0f);
        session_->ClearBuffers();
        if (buffers_.ReleaseBuffers(system_.CoreTiming(), *session_, true)) {
            bufferEvent_->Signal();
        }
        state_ = State::Stopped;
    }

    return ResultSuccess;
}

bool System::AppendBuffer(const AudioInBuffer& buffer, const u64 tag) {
    if (buffers_.GetTotalBufferCount() == BufferCount) {
        return false;
    }

    const auto timestamp { buffers_.GetNextTimestamp() };
    const AudioBuffer newBuffer {
        .start_timestamp = timestamp,
        .end_timestamp = timestamp + buffer.size / (channelCount_ * sizeof(s16)),
        .played_timestamp = 0,
        .samples = buffer.samples,
        .tag = tag,
        .size = buffer.size,
    };

    buffers_.AppendBuffer(newBuffer);
    RegisterBuffers();

    return true;
}

void System::RegisterBuffers() {
    if (state_ == State::Started) {
        std::vector<AudioBuffer> registeredBuffers {};
        buffers_.RegisterBuffers(registeredBuffers);
        session_->AppendBuffers(registeredBuffers);
    }
}

void System::ReleaseBuffers() {
    const bool signal { buffers_.ReleaseBuffers(system_.CoreTiming(), *session_, false) };

    if (signal) {
        // Signal if any buffer was released, or if none are registered, we need more.
        bufferEvent_->Signal();
    }
}

u32 System::GetReleasedBuffers(std::span<u64> tags) {
    return buffers_.GetReleasedBuffers(tags);
}

bool System::FlushAudioInBuffers() {
    if (state_ != State::Started) {
        return false;
    }

    u32 buffersReleased {};
    buffers_.FlushBuffers(buffersReleased);

    if (buffersReleased > 0) {
        bufferEvent_->Signal();
    }
    return true;
}

u16 System::GetChannelCount() const {
    return channelCount_;
}

u32 System::GetSampleRate() const {
    return sampleRate_;
}

SampleFormat System::GetSampleFormat() const {
    return sampleFormat_;
}

State System::GetState() {
    switch (state_) {
    case State::Started:
    case State::Stopped:
        return state_;
    default:
        LOG_ERROR(Service_Audio, "AudioIn invalid state!");
        state_ = State::Stopped;
        break;
    }
    return state_;
}

std::string System::GetName() const {
    return name_;
}

f32 System::GetVolume() const {
    return volume_;
}

void System::SetVolume(const f32 volume) {
    volume_ = volume;
    session_->SetVolume(volume_);
}

bool System::ContainsAudioBuffer(const u64 tag) const {
    return buffers_.ContainsBuffer(tag);
}

u32 System::GetBufferCount() const {
    return buffers_.GetAppendedRegisteredCount();
}

u64 System::GetPlayedSampleCount() const {
    return session_->GetPlayedSampleCount();
}

bool System::IsUac() const {
    return isUac_;
}

} // namespace AudioCore::AudioIn

