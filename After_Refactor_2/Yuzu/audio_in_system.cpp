namespace AudioCore::AudioIn {

    using namespace Core;
    using namespace HLE::Kernel;
    using namespace Logging;
    using namespace Service::Audio;

    static constexpr u32 TargetSampleRate{48000};
    static constexpr u32 BufferCount{16};

    System::System(SystemCore& system, KEvent* event, const size_t sessionId)
        : mSystem{system},
          mBufferEvent{event},
          mSessionId{sessionId},
          mSession{std::make_unique<DeviceSession>(system)} {}

    System::~System() {
        Finalize();
    }

    void System::Finalize() {
        Stop();
        mSession->Finalize();
    }

    void System::StartSession() {
        mSession->Start();
    }

    size_t System::GetSessionId() const {
        return mSessionId;
    }

    std::string_view System::GetDefaultDeviceName() const {
        return "BuiltInHeadset";
    }

    std::string_view System::GetDefaultUacDeviceName() const {
        return "Uac";
    }

    Result System::IsConfigValid(const std::string_view deviceName, const AudioInParameter& inParams) const {
        if ((deviceName.size() > 0) && (deviceName != GetDefaultDeviceName() && deviceName != GetDefaultUacDeviceName())) {
            return ResultNotFound;
        }

        if (inParams.sample_rate != TargetSampleRate && inParams.sample_rate > 0) {
            return ResultInvalidSampleRate;
        }

        return ResultSuccess;
    }

    Result System::Initialize(const std::string& deviceName, const AudioInParameter& inParams, const u32 handle, const u64 appletResourceUserId) {
        auto result = IsConfigValid(deviceName, inParams);
        if (result.IsError()) {
            return result;
        }

        mHandle = handle;
        mAppletResourceUserId = appletResourceUserId;
        if (deviceName.empty() || deviceName[0] == '\0') {
            mName = std::string(GetDefaultDeviceName());
        } else {
            mName = deviceName;
        }

        mSampleRate = TargetSampleRate;
        mSampleFormat = SampleFormat::PcmInt16;
        mChannelCount = inParams.channel_count <= 2 ? 2 : 6;
        mVolume = 1.0f;
        mIsUac = mName == "Uac";
        return ResultSuccess;
    }

    Result System::Start() {
        if (mState != State::Stopped) {
            return ResultOperationFailed;
        }

        mSession->Initialize(mName, mSampleFormat, mChannelCount, mSessionId, mHandle, mAppletResourceUserId, Sink::StreamType::In);
        mSession->SetVolume(mVolume);
        mSession->Start();
        mState = State::Started;

        std::vector<AudioBuffer> buffersToFlush{};
        mBuffers.RegisterBuffers(buffersToFlush);
        mSession->AppendBuffers(buffersToFlush);
        mSession->SetRingSize(static_cast<u32>(buffersToFlush.size()));

        return ResultSuccess;
    }

    Result System::Stop() {
        if (mState == State::Started) {
            mSession->Stop();
            mSession->SetVolume(0.0f);
            mSession->ClearBuffers();
            if (mBuffers.ReleaseBuffers(mSystem.CoreTiming(), *mSession, true)) {
                mBufferEvent->Signal();
            }
            mState = State::Stopped;
        }

        return ResultSuccess;
    }

    bool System::AppendBuffer(const AudioInBuffer& buffer, const u64 tag) {
        if (mBuffers.GetTotalBufferCount() == BufferCount) {
            return false;
        }

        const auto timestamp{mBuffers.GetNextTimestamp()};
        const AudioBuffer newBuffer{
            .startTimestamp = timestamp,
            .endTimestamp = timestamp + buffer.size / (mChannelCount * sizeof(s16)),
            .playedTimestamp = 0,
            .samples = buffer.samples,
            .tag = tag,
            .size = buffer.size,
        };

        mBuffers.AppendBuffer(newBuffer);
        RegisterBuffers();

        return true;
    }

    void System::RegisterBuffers() {
        if (mState == State::Started) {
            std::vector<AudioBuffer> registeredBuffers{};
            mBuffers.RegisterBuffers(registeredBuffers);
            mSession->AppendBuffers(registeredBuffers);
        }
    }

    void System::ReleaseBuffers() {
        bool signal{mBuffers.ReleaseBuffers(mSystem.CoreTiming(), *mSession, false)};

        if (signal) {
            // Signal if any buffer was released, or if none are registered, we need more.
            mBufferEvent->Signal();
        }
    }

    u32 System::GetReleasedBuffers(std::span<u64> tags) {
        return mBuffers.GetReleasedBuffers(tags);
    }

    bool System::FlushAudioInBuffers() {
        if (mState != State::Started) {
            return false;
        }

        u32 buffersReleased{};
        mBuffers.FlushBuffers(buffersReleased);

        if (buffersReleased > 0) {
            mBufferEvent->Signal();
        }
        return true;
    }

    u16 System::GetChannelCount() const {
        return mChannelCount;
    }

    u32 System::GetSampleRate() const {
        return mSampleRate;
    }

    SampleFormat System::GetSampleFormat() const {
        return mSampleFormat;
    }

    State System::GetState() {
        switch (mState) {
        case State::Started:
        case State::Stopped:
            return mState;
        default:
            LOG_ERROR(Service_Audio, "AudioIn invalid state!");
            mState = State::Stopped;
            break;
        }
        return mState;
    }

    std::string System::GetName() const {
        return mName;
    }

    f32 System::GetVolume() const {
        return mVolume;
    }

    void System::SetVolume(const f32 volume) {
        mVolume = volume;
        mSession->SetVolume(volume);
    }

    bool System::ContainsAudioBuffer(const u64 tag) const {
        return mBuffers.ContainsBuffer(tag);
    }

    u32 System::GetBufferCount() const {
        return mBuffers.GetAppendedRegisteredCount();
    }

    u64 System::GetPlayedSampleCount() const {
        return mSession->GetPlayedSampleCount();
    }

    bool System::IsUac() const {
        return mIsUac;
    }
} // namespace AudioCore::AudioIn