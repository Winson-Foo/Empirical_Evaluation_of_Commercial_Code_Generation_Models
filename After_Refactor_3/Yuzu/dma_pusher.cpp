#include "DmaPusher.h"
#include "CommandBuffer.h"
#include "Puller.h"
#include "RasterizerInterface.h"

namespace Tegra {

  class DmaCommandProcessor {
  public:
    DmaCommandProcessor(GPU& gpu_, MemoryManager& memory_manager_, Control::ChannelState& channel_state_)
      : gpu(gpu_), memory_manager(memory_manager_), puller(gpu_, memory_manager_, channel_state_) {}

    void ProcessCommands(const CommandBuffer& command_buffer) {
      for (const auto& command_header : command_buffer.GetCommands()) {
        if (state.method_count) {
          dma_word_offset = static_cast<u32>(index * sizeof(u32));
          if (state.non_incrementing) {
            const u32 max_write = static_cast<u32>(std::min<std::size_t>(index + state.method_count, command_buffer.GetCommands().size()) - index);
            CallMultiMethod(&command_header.argument, max_write);
            state.method_count -= max_write;
            state.is_last_call = true;
            index += max_write;
            continue;
          } else {
            state.is_last_call = state.method_count <= 1;
            CallMethod(command_header.argument);
          }
          if (!state.non_incrementing) {
            state.method++;
          }
          if (increment_once) {
            state.non_incrementing = true;
          }
          state.method_count--;
        } else {
          switch (command_header.mode) {
            case SubmissionMode::Increasing:
              SetState(command_header);
              state.non_incrementing = false;
              increment_once = false;
              break;
            case SubmissionMode::NonIncreasing:
              SetState(command_header);
              state.non_incrementing = true;
              increment_once = false;
              break;
            case SubmissionMode::Inline:
              state.method = command_header.method;
              state.subchannel = command_header.subchannel;
              dma_word_offset = static_cast<u64>( -static_cast<s64>(state.dma_get)); // negate to set address as 0
              CallMethod(command_header.arg_count);
              state.non_incrementing = true;
              increment_once = false;
              break;
            case SubmissionMode::IncreaseOnce:
              SetState(command_header);
              state.non_incrementing = false;
              increment_once = true;
            default:
              break;
          }
        }
        index++;
      }
    }

  private:
    void SetState(const CommandHeader& command_header) {
      state.method = command_header.method;
      state.subchannel = command_header.subchannel;
      state.method_count = command_header.method_count;
    }

    void CallMethod(u32 argument) const {
      if (state.method < non_puller_methods) {
        puller.CallPullerMethod({
          state.method,
          argument,
          state.subchannel,
          state.method_count,
        });
      } else {
        auto subchannel = subchannels[state.subchannel];
        if (!subchannel->execution_mask[state.method]) [[likely]] {
          subchannel->method_sink.emplace_back(state.method, argument);
          return;
        }
        subchannel->ConsumeSink();
        subchannel->current_dma_segment = state.dma_get + dma_word_offset;
        subchannel->CallMethod(state.method, argument, state.is_last_call);
      }
    }

    void CallMultiMethod(const u32* base_start, u32 num_methods) const {
      if (state.method < non_puller_methods) {
        puller.CallMultiMethod(
          state.method,
          state.subchannel,
          base_start,
          num_methods,
          state.method_count);
      } else {
        auto subchannel = subchannels[state.subchannel];
        subchannel->ConsumeSink();
        subchannel->current_dma_segment = state.dma_get + dma_word_offset;
        subchannel->CallMultiMethod(state.method, base_start, num_methods, state.method_count);
      }
    }

    GPU& gpu;
    MemoryManager& memory_manager;
    CommandState state;
    bool increment_once = false;
    std::vector<Maxwell3D*> subchannels;
    Puller puller;
    u32 index = 0;
    u32 dma_word_offset = 0;
    constexpr u32 non_puller_methods = 0x88;
  };

  class DmaPusherImpl {
  public:
    DmaPusherImpl(Core::System& system_, GPU& gpu_, MemoryManager& memory_manager_, Control::ChannelState& channel_state_)
      : system(system_), gpu(gpu_), memory_manager(memory_manager_), channel_state(channel_state_), puller(gpu_, memory_manager_, channel_state_) {}

    void DispatchCalls() {
      MICROPROFILE_SCOPE(DispatchCalls);

      dma_pushbuffer_subindex = 0;

      dma_state.is_last_call = true;

      while (system.IsPoweredOn()) {
        if (!Step()) {
          break;
        }
      }
      gpu.FlushCommands();
      gpu.OnCommandListEnd();
    }

    bool Step() {
      if (!ib_enable || dma_pushbuffer.empty()) {
        return false;
      }

      CommandBuffer& command_buffer{ dma_pushbuffer.front() };

      ASSERT_OR_EXECUTE(
        command_buffer.GetCommandLists().size() || command_buffer.GetPrefetchCommandList().size(),
        {
          dma_pushbuffer.pop();
          dma_pushbuffer_subindex = 0;
          return true;
        });

      if (command_buffer.GetPrefetchCommandList().size()) {
        command_processor.ProcessCommands(command_buffer.GetPrefetchCommandList());
        dma_pushbuffer.pop();
      } else {
        const CommandListHeader& command_list_header{ command_buffer.GetCommandLists()[dma_pushbuffer_subindex++] };
        dma_state.dma_get = command_list_header.addr;
        if (dma_pushbuffer_subindex >= command_buffer.GetCommandLists().size()) {
          dma_pushbuffer.pop();
          dma_pushbuffer_subindex = 0;
        }
        if (command_list_header.size == 0) {
          return true;
        }
        command_headers.resize_destructive(command_list_header.size);
        constexpr u32 MacroRegistersStart = 0xE00;
        if (dma_state.method < MacroRegistersStart) {
          if (Settings::IsGPULevelHigh()) {
            memory_manager.ReadBlock(dma_state.dma_get, command_headers.data(), command_list_header.size * sizeof(u32));
          } else {
            memory_manager.ReadBlockUnsafe(dma_state.dma_get, command_headers.data(), command_list_header.size * sizeof(u32));
          }
        } else {
          const size_t copy_size = command_list_header.size * sizeof(u32);
          if (subchannels[dma_state.subchannel]) {
            subchannels[dma_state.subchannel]->current_dirty = memory_manager.IsMemoryDirty(dma_state.dma_get, copy_size);
          }
          memory_manager.ReadBlockUnsafe(dma_state.dma_get, command_headers.data(), copy_size);
        }
        command_processor.ProcessCommands(CommandBuffer{std::move(command_headers)});
      }

      return true;
    }

    void BindRasterizer(VideoCore::RasterizerInterface* rasterizer) {
      puller.BindRasterizer(rasterizer);
    }

  private:
    Core::System& system;
    GPU& gpu;
    MemoryManager& memory_manager;
    Control::ChannelState& channel_state;

    CommandState dma_state;
    std::vector<CommandHeader> command_headers;
    std::vector<CommandBuffer> dma_pushbuffer;
    bool ib_enable = true;
    int dma_pushbuffer_subindex = 0;

    DmaCommandProcessor command_processor = DmaCommandProcessor(gpu, memory_manager, channel_state);
    Puller puller;
  };

  DmaPusher::DmaPusher(Core::System& system_, GPU& gpu_, MemoryManager& memory_manager_, Control::ChannelState& channel_state_)
    : impl(std::make_unique<DmaPusherImpl>(system_, gpu_, memory_manager_, channel_state_)) {}

  DmaPusher::~DmaPusher() = default;

  void DmaPusher::DispatchCalls() {
    impl->DispatchCalls();
  }

  void DmaPusher::BindRasterizer(VideoCore::RasterizerInterface* rasterizer) {
    impl->BindRasterizer(rasterizer);
  }

} // namespace Tegra