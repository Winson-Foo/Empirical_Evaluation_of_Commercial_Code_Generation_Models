namespace Tegra {

constexpr u32 kMacroRegistersStart = 0xE00;

class DmaPusher {
public:
    DmaPusher(Core::System& system, GPU& gpu, MemoryManager& memory_manager,
              Control::ChannelState& channel_state)
        : gpu_(gpu), system_(system), memory_manager_(memory_manager),
          puller_(gpu_, memory_manager_, *this, channel_state) {}

    ~DmaPusher() = default;

    void DispatchCalls();

private:
    GPU& gpu_;
    Core::System& system_;
    MemoryManager& memory_manager_;
    Engines::Puller puller_;

    struct DmaState {
        u32 dma_get;
        u32 method;
        u32 method_count;
        bool non_incrementing;
        u64 dma_word_offset;
        u32 subchannel;
        bool is_last_call;
    } dma_state;

    std::size_t dma_pushbuffer_subindex;
    bool ib_enable;
    bool dma_increment_once;

    std::vector<CommandHeader> command_headers;
    Queue<CommandList> dma_pushbuffer;

    std::array<std::unique_ptr<Subchannel>, kNumSubchannels> subchannels{};

    void ProcessCommands(std::span<const CommandHeader> commands);

    void SetDmaState(const CommandHeader& command_header);

    void CallMethod(u32 argument) const;
    void CallMultiMethod(const u32* base_start, u32 num_methods) const;

    bool Step();
};

void DmaPusher::DispatchCalls() {
    MICROPROFILE_SCOPE(DispatchCalls);

    dma_pushbuffer_subindex = 0;

    dma_state.is_last_call = true;

    while (system_.IsPoweredOn()) {
        if (!Step()) {
            break;
        }
    }

    gpu_.FlushCommands();
    gpu_.OnCommandListEnd();
}

bool DmaPusher::Step() {
    if (!ib_enable || dma_pushbuffer.empty()) {
        return false;
    }

    CommandList& command_list{dma_pushbuffer.front()};

    ASSERT_OR_EXECUTE(
        command_list.command_lists.size() || command_list.prefetch_command_list.size(), {
            dma_pushbuffer.pop();
            dma_pushbuffer_subindex = 0;
            return true;
        });

    if (command_list.prefetch_command_list.size()) {
        ProcessCommands(command_list.prefetch_command_list);
        dma_pushbuffer.pop();
    } else {
        const CommandListHeader command_list_header{
            command_list.command_lists[dma_pushbuffer_subindex++]};
        dma_state.dma_get = command_list_header.addr;

        if (dma_pushbuffer_subindex >= command_list.command_lists.size()) {
            dma_pushbuffer.pop();
            dma_pushbuffer_subindex = 0;
        }

        if (command_list_header.size == 0) {
            return true;
        }

        command_headers.resize_destructive(command_list_header.size);

        if (dma_state.method < kMacroRegistersStart) {
            const u64 copy_size = command_list_header.size * sizeof(u32);

            if (Settings::IsGPULevelHigh()) {
                memory_manager_.ReadBlock(dma_state.dma_get, command_headers.data(), copy_size);
            } else {
                memory_manager_.ReadBlockUnsafe(dma_state.dma_get, command_headers.data(), copy_size);
            }
        } else {
            const u64 copy_size = command_list_header.size * sizeof(u32);
            auto& subchannel = subchannels[dma_state.subchannel];

            if (subchannel) {
                subchannel->current_dirty =
                    memory_manager_.IsMemoryDirty(dma_state.dma_get, copy_size);
            }

            memory_manager_.ReadBlockUnsafe(dma_state.dma_get, command_headers.data(), copy_size);
        }

        ProcessCommands(command_headers);
    }

    return true;
}

void DmaPusher::ProcessCommands(std::span<const CommandHeader> commands) {
    for (std::size_t index = 0; index < commands.size();) {
        const CommandHeader& command_header = commands[index];

        if (dma_state.method_count) {
            dma_state.dma_word_offset = static_cast<u32>(index * sizeof(u32));

            if (dma_state.non_incrementing) {
                const u32 max_write =
                    static_cast<u32>(std::min(std::size_t(index + dma_state.method_count), commands.size()) - index);

                CallMultiMethod(&command_header.argument, max_write);
                dma_state.method_count -= max_write;
                dma_state.is_last_call = true;
                index += max_write;

                continue;
            } else {
                dma_state.is_last_call = dma_state.method_count <= 1;
                CallMethod(command_header.argument);
            }

            if (!dma_state.non_incrementing) {
                dma_state.method++;
            }

            if (dma_increment_once) {
                dma_state.non_incrementing = true;
            }

            dma_state.method_count--;
        } else {
            switch (command_header.mode) {
            case SubmissionMode::Increasing:
                SetDmaState(command_header);
                dma_state.non_incrementing = false;
                dma_increment_once = false;
                break;

            case SubmissionMode::NonIncreasing:
                SetDmaState(command_header);
                dma_state.non_incrementing = true;
                dma_increment_once = false;
                break;

            case SubmissionMode::Inline:
                dma_state.method = command_header.method;
                dma_state.subchannel = command_header.subchannel;
                dma_state.dma_word_offset = static_cast<u64>(-static_cast<s64>(dma_state.dma_get));
                CallMethod(command_header.arg_count);
                dma_state.non_incrementing = true;
                dma_increment_once = false;
                break;

            case SubmissionMode::IncreaseOnce:
                SetDmaState(command_header);
                dma_state.non_incrementing = false;
                dma_increment_once = true;
                break;

            default:
                break;
            }

            index++;
        }
    }
}

void DmaPusher::SetDmaState(const CommandHeader& command_header) {
    dma_state.method = command_header.method;
    dma_state.subchannel = command_header.subchannel;
    dma_state.method_count = command_header.method_count;
}

void DmaPusher::CallMethod(u32 argument) const {
    if (dma_state.method < non_puller_methods) {
        puller_.CallPullerMethod(Engines::Puller::MethodCall{
            dma_state.method,
            argument,
            dma_state.subchannel,
            dma_state.method_count,
        });
    } else {
        auto& subchannel = subchannels[dma_state.subchannel];

        if (!subchannel) {
            return;
        }

        if (!subchannel->execution_mask[dma_state.method]) [[likely]] {
            subchannel->method_sink.emplace_back(dma_state.method, argument);
            return;
        }

        subchannel->ConsumeSink();
        subchannel->current_dma_segment = dma_state.dma_get + dma_state.dma_word_offset;
        subchannel->CallMethod(dma_state.method, argument, dma_state.is_last_call);
    }
}

void DmaPusher::CallMultiMethod(const u32* base_start, u32 num_methods) const {
    if (dma_state.method < non_puller_methods) {
        puller_.CallMultiMethod(dma_state.method, dma_state.subchannel, base_start, num_methods, dma_state.method_count);
    } else {
        auto& subchannel = subchannels[dma_state.subchannel];

        if (!subchannel) {
            return;
        }

        subchannel->ConsumeSink();
        subchannel->current_dma_segment = dma_state.dma_get + dma_state.dma_word_offset;
        subchannel->CallMultiMethod(dma_state.method, base_start, num_methods, dma_state.method_count);
    }
}

void DmaPusher::BindRasterizer(VideoCore::RasterizerInterface* rasterizer) {
    puller_.BindRasterizer(rasterizer);
}

} // namespace Tegra