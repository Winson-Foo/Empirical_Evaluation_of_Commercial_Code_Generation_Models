namespace Tegra {

class DmaPusher {
public:
    DmaPusher(Core::System& system, GPU& gpu, MemoryManager& memoryManager, Control::ChannelState& channelState);
    ~DmaPusher() = default;

    void DispatchCalls();

private:
    void ProcessNextCommandList();
    void SetDmaState(const CommandHeader& commandHeader);
    void ProcessCommands(std::span<const CommandHeader> commands);
    void CallMethod(u32 argument) const;
    void CallMultiMethod(std::span<const u32> methodArgs) const;

    Core::System& system_;
    GPU& gpu_;
    MemoryManager& memoryManager_;
    Control::ChannelState& channelState_;
    Maxwell3D puller_;
    std::queue<CommandList> dmaPushbuffer_;
    std::size_t dmaPushbufferSubindex_ = 0;
    std::vector<CommandHeader> commandHeaders_;
    DmaState dmaState_;
    std::vector<Subchannel*> subchannels_;
    bool ibEnable_ = false;
    bool dmaIncrementOnce_ = false;

    static constexpr std::size_t nonPullerMethods = Engines::Puller::NumMethods;
    static constexpr std::size_t maxNumSubchannels = 32;
};

} // namespace Tegra

namespace {

enum class SubmissionMode {
    Increasing,
    NonIncreasing,
    Inline,
    IncreaseOnce
};

struct CommandHeader {
    u32 mode : 2;
    u32 method : 12;
    u32 subchannel : 5;
    u32 arg_count : 13;
};

static_assert(sizeof(CommandHeader) == sizeof(u32));

struct CommandListHeader {
    u64 addr;
    u32 size;
    u32 flags;
};

static_assert(sizeof(CommandListHeader) == 16);

} // namespace

namespace Tegra {

DmaPusher::DmaPusher(Core::System& system, GPU& gpu, MemoryManager& memoryManager, Control::ChannelState& channelState)
    : system_(system)
    , gpu_(gpu)
    , memoryManager_(memoryManager)
    , channelState_(channelState)
    , puller_(gpu, memoryManager, *this, channelState)
    , dmaState_()
    , subchannels_(maxNumSubchannels, nullptr)
{}

void DmaPusher::DispatchCalls() {
    constexpr auto MP_RGB_128_128_192 = MP_RGB(128, 128, 192);
    MICROPROFILE_DEFINE(DispatchCalls, "GPU", "Execute command buffer", MP_RGB_128_128_192);
    MICROPROFILE_SCOPE(DispatchCalls);

    dmaPushbufferSubindex_ = 0;
    dmaState_.is_last_call = true;

    while (system_.IsPoweredOn()) {
        if (!ibEnable_ || dmaPushbuffer_.empty()) {
            // Pushbuffer empty and IB empty or nonexistent - nothing to do
            break;
        }
        ProcessNextCommandList();
    }
    gpu_.FlushCommands();
    gpu_.OnCommandListEnd();
}

void DmaPusher::ProcessNextCommandList() {
    CommandList& commandList = dmaPushbuffer_.front();

    ASSERT_OR_EXECUTE(
        commandList.commandLists.size() || commandList.prefetchCommandList.size(), {
            // Somehow the command_list is empty, in order to avoid a crash
            // We ignore it and assume its size is 0.
            dmaPushbuffer_.pop();
            dmaPushbufferSubindex_ = 0;
            return;
        });

    if (!commandList.prefetchCommandList.empty()) {
        // Prefetched command list from nvdrv, used for things like synchronization
        ProcessCommands(commandList.prefetchCommandList);
        dmaPushbuffer_.pop();
    } else {
        const CommandListHeader commandListHeader = commandList.commandLists[dmaPushbufferSubindex_++];
        dmaState_.dma_get = commandListHeader.addr;

        if (dmaPushbufferSubindex_ >= commandList.commandLists.size()) {
            // We've gone through the current list, remove it from the queue
            dmaPushbuffer_.pop();
            dmaPushbufferSubindex_ = 0;
        }

        if (commandListHeader.size == 0) {
            return;
        }

        // Push buffer non-empty, read a word
        commandHeaders_.resize_destructive(commandListHeader.size);
        constexpr u32 MacroRegistersStart = 0xE00;
        if (dmaState_.method < MacroRegistersStart) {
            if (Settings::IsGPULevelHigh()) {
                memoryManager_.ReadBlock(dmaState_.dma_get, commandHeaders_.data(), commandListHeader.size * sizeof(u32));
            } else {
                memoryManager_.ReadBlockUnsafe(dmaState_.dma_get, commandHeaders_.data(), commandListHeader.size * sizeof(u32));
            }
        } else {
            const std::size_t copySize = commandListHeader.size * sizeof(u32);
            const auto subchannel = subchannels_[dmaState_.subchannel];
            if (subchannel) {
                subchannel->currentDirty_ = memoryManager_.IsMemoryDirty(dmaState_.dma_get, copySize);
            }
            memoryManager_.ReadBlockUnsafe(dmaState_.dma_get, commandHeaders_.data(), copySize);
        }
        ProcessCommands(commandHeaders_);
    }
}

void DmaPusher::SetDmaState(const CommandHeader& commandHeader) {
    dmaState_.method = commandHeader.method;
    dmaState_.subchannel = commandHeader.subchannel;
    dmaState_.method_count = commandHeader.arg_count;
}

void DmaPusher::ProcessCommands(std::span<const CommandHeader> commands) {
    for (std::size_t index = 0; index < commands.size();) {
        const CommandHeader& commandHeader = commands[index];

        if (dmaState_.method_count) {
            // Data word of methods command
            dmaState_.dma_word_offset = static_cast<u32>(index * sizeof(u32));
            if (dmaState_.non_incrementing) {
                const u32 maxWrite = static_cast<u32>(
                    std::min<std::size_t>(index + dmaState_.method_count, commands.size()) - index);
                CallMultiMethod(std::span(&commandHeader.argument, maxWrite));
                dmaState_.method_count -= maxWrite;
                dmaState_.is_last_call = true;
                index += maxWrite;
                continue;
            } else {
                dmaState_.is_last_call = dmaState_.method_count <= 1;
                CallMethod(commandHeader.argument);
            }

            if (!dmaState_.non_incrementing) {
                dmaState_.method++;
            }

            if (dmaIncrementOnce_) {
                dmaState_.non_incrementing = true;
            }

            dmaState_.method_count--;
        } else {
            // No command active - this is the first word of a new one
            switch (commandHeader.mode) {
            case SubmissionMode::Increasing:
                SetDmaState(commandHeader);
                dmaState_.non_incrementing = false;
                dmaIncrementOnce_ = false;
                break;
            case SubmissionMode::NonIncreasing:
                SetDmaState(commandHeader);
                dmaState_.non_incrementing = true;
                dmaIncrementOnce_ = false;
                break;
            case SubmissionMode::Inline:
                dmaState_.method = commandHeader.method;
                dmaState_.subchannel = commandHeader.subchannel;
                dmaState_.dma_word_offset = static_cast<u64>(
                    -static_cast<s64>(dmaState_.dma_get)); // negate to set address as 0
                CallMethod(commandHeader.arg_count);
                dmaState_.non_incrementing = true;
                dmaIncrementOnce_ = false;
                break;
            case SubmissionMode::IncreaseOnce:
                SetDmaState(commandHeader);
                dmaState_.non_incrementing = false;
                dmaIncrementOnce_ = true;
                break;
            default:
                break;
            }
        }
        index++;
    }
}

void DmaPusher::CallMethod(u32 argument) const {
    if (dmaState_.method < nonPullerMethods) {
        puller_.CallPullerMethod(Engines::Puller::MethodCall{
            dmaState_.method,
            argument,
            dmaState_.subchannel,
            dmaState_.method_count,
        });
    } else {
        auto subchannel = subchannels_[dmaState_.subchannel];
        if (!subchannel || !subchannel->execution_mask_[dmaState_.method]) {
            subchannel->methodSink_.emplace_back(dmaState_.method, argument);
            return;
        }
        subchannel->ConsumeSink();
        subchannel->currentDmaSegment_ = dmaState_.dma_get + dmaState_.dma_word_offset;
        subchannel->CallMethod(dmaState_.method, argument, dmaState_.is_last_call);
    }
}

void DmaPusher::CallMultiMethod(std::span<const u32> methodArgs) const {
    if (dmaState_.method < nonPullerMethods) {
        puller_.CallMultiMethod(dmaState_.method, dmaState_.subchannel, methodArgs, dmaState_.method_count);
    } else {
        auto subchannel = subchannels_[dmaState_.subchannel];
        subchannel->ConsumeSink();
        subchannel->currentDmaSegment_ = dmaState_.dma_get + dmaState_.dma_word_offset;
        subchannel->CallMultiMethod(dmaState_.method, methodArgs, dmaState_.method_count);
    }
}

void DmaPusher::BindRasterizer(VideoCore::RasterizerInterface& rasterizer) {
    puller_.BindRasterizer(rasterizer);
}

} // namespace Tegra

