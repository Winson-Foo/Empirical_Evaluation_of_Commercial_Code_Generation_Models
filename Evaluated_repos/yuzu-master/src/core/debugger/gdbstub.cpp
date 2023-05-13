// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <numeric>
#include <optional>
#include <thread>

#include <boost/algorithm/string.hpp>

#include "common/hex_util.h"
#include "common/logging/log.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/arm/arm_interface.h"
#include "core/core.h"
#include "core/debugger/gdbstub.h"
#include "core/debugger/gdbstub_arch.h"
#include "core/hle/kernel/k_page_table.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/loader/loader.h"
#include "core/memory.h"

namespace Core {

constexpr char GDB_STUB_START = '$';
constexpr char GDB_STUB_END = '#';
constexpr char GDB_STUB_ACK = '+';
constexpr char GDB_STUB_NACK = '-';
constexpr char GDB_STUB_INT3 = 0x03;
constexpr int GDB_STUB_SIGTRAP = 5;

constexpr char GDB_STUB_REPLY_ERR[] = "E01";
constexpr char GDB_STUB_REPLY_OK[] = "OK";
constexpr char GDB_STUB_REPLY_EMPTY[] = "";

static u8 CalculateChecksum(std::string_view data) {
    return std::accumulate(data.begin(), data.end(), u8{0},
                           [](u8 lhs, u8 rhs) { return static_cast<u8>(lhs + rhs); });
}

static std::string EscapeGDB(std::string_view data) {
    std::string escaped;
    escaped.reserve(data.size());

    for (char c : data) {
        switch (c) {
        case '#':
            escaped += "}\x03";
            break;
        case '$':
            escaped += "}\x04";
            break;
        case '*':
            escaped += "}\x0a";
            break;
        case '}':
            escaped += "}\x5d";
            break;
        default:
            escaped += c;
            break;
        }
    }

    return escaped;
}

static std::string EscapeXML(std::string_view data) {
    std::string escaped;
    escaped.reserve(data.size());

    for (char c : data) {
        switch (c) {
        case '&':
            escaped += "&amp;";
            break;
        case '"':
            escaped += "&quot;";
            break;
        case '<':
            escaped += "&lt;";
            break;
        case '>':
            escaped += "&gt;";
            break;
        default:
            escaped += c;
            break;
        }
    }

    return escaped;
}

GDBStub::GDBStub(DebuggerBackend& backend_, Core::System& system_)
    : DebuggerFrontend(backend_), system{system_} {
    if (system.ApplicationProcess()->Is64BitProcess()) {
        arch = std::make_unique<GDBStubA64>();
    } else {
        arch = std::make_unique<GDBStubA32>();
    }
}

GDBStub::~GDBStub() = default;

void GDBStub::Connected() {}

void GDBStub::ShuttingDown() {}

void GDBStub::Stopped(Kernel::KThread* thread) {
    SendReply(arch->ThreadStatus(thread, GDB_STUB_SIGTRAP));
}

void GDBStub::Watchpoint(Kernel::KThread* thread, const Kernel::DebugWatchpoint& watch) {
    const auto status{arch->ThreadStatus(thread, GDB_STUB_SIGTRAP)};

    switch (watch.type) {
    case Kernel::DebugWatchpointType::Read:
        SendReply(fmt::format("{}rwatch:{:x};", status, GetInteger(watch.start_address)));
        break;
    case Kernel::DebugWatchpointType::Write:
        SendReply(fmt::format("{}watch:{:x};", status, GetInteger(watch.start_address)));
        break;
    case Kernel::DebugWatchpointType::ReadOrWrite:
    default:
        SendReply(fmt::format("{}awatch:{:x};", status, GetInteger(watch.start_address)));
        break;
    }
}

std::vector<DebuggerAction> GDBStub::ClientData(std::span<const u8> data) {
    std::vector<DebuggerAction> actions;
    current_command.insert(current_command.end(), data.begin(), data.end());

    while (current_command.size() != 0) {
        ProcessData(actions);
    }

    return actions;
}

void GDBStub::ProcessData(std::vector<DebuggerAction>& actions) {
    const char c{current_command[0]};

    // Acknowledgement
    if (c == GDB_STUB_ACK || c == GDB_STUB_NACK) {
        current_command.erase(current_command.begin());
        return;
    }

    // Interrupt
    if (c == GDB_STUB_INT3) {
        LOG_INFO(Debug_GDBStub, "Received interrupt");
        current_command.erase(current_command.begin());
        actions.push_back(DebuggerAction::Interrupt);
        SendStatus(GDB_STUB_ACK);
        return;
    }

    // Otherwise, require the data to be the start of a command
    if (c != GDB_STUB_START) {
        LOG_ERROR(Debug_GDBStub, "Invalid command buffer contents: {}", current_command.data());
        current_command.clear();
        SendStatus(GDB_STUB_NACK);
        return;
    }

    // Continue reading until command is complete
    while (CommandEnd() == current_command.end()) {
        const auto new_data{backend.ReadFromClient()};
        current_command.insert(current_command.end(), new_data.begin(), new_data.end());
    }

    // Execute and respond to GDB
    const auto command{DetachCommand()};

    if (command) {
        SendStatus(GDB_STUB_ACK);
        ExecuteCommand(*command, actions);
    } else {
        SendStatus(GDB_STUB_NACK);
    }
}

void GDBStub::ExecuteCommand(std::string_view packet, std::vector<DebuggerAction>& actions) {
    LOG_TRACE(Debug_GDBStub, "Executing command: {}", packet);

    if (packet.length() == 0) {
        SendReply(GDB_STUB_REPLY_ERR);
        return;
    }

    if (packet.starts_with("vCont")) {
        HandleVCont(packet.substr(5), actions);
        return;
    }

    std::string_view command{packet.substr(1, packet.size())};

    switch (packet[0]) {
    case 'H': {
        Kernel::KThread* thread{nullptr};
        s64 thread_id{strtoll(command.data() + 1, nullptr, 16)};
        if (thread_id >= 1) {
            thread = GetThreadByID(thread_id);
        } else {
            thread = backend.GetActiveThread();
        }

        if (thread) {
            SendReply(GDB_STUB_REPLY_OK);
            backend.SetActiveThread(thread);
        } else {
            SendReply(GDB_STUB_REPLY_ERR);
        }
        break;
    }
    case 'T': {
        s64 thread_id{strtoll(command.data(), nullptr, 16)};
        if (GetThreadByID(thread_id)) {
            SendReply(GDB_STUB_REPLY_OK);
        } else {
            SendReply(GDB_STUB_REPLY_ERR);
        }
        break;
    }
    case 'Q':
    case 'q':
        HandleQuery(command);
        break;
    case '?':
        SendReply(arch->ThreadStatus(backend.GetActiveThread(), GDB_STUB_SIGTRAP));
        break;
    case 'k':
        LOG_INFO(Debug_GDBStub, "Shutting down emulation");
        actions.push_back(DebuggerAction::ShutdownEmulation);
        break;
    case 'g':
        SendReply(arch->ReadRegisters(backend.GetActiveThread()));
        break;
    case 'G':
        arch->WriteRegisters(backend.GetActiveThread(), command);
        SendReply(GDB_STUB_REPLY_OK);
        break;
    case 'p': {
        const size_t reg{static_cast<size_t>(strtoll(command.data(), nullptr, 16))};
        SendReply(arch->RegRead(backend.GetActiveThread(), reg));
        break;
    }
    case 'P': {
        const auto sep{std::find(command.begin(), command.end(), '=') - command.begin() + 1};
        const size_t reg{static_cast<size_t>(strtoll(command.data(), nullptr, 16))};
        arch->RegWrite(backend.GetActiveThread(), reg, std::string_view(command).substr(sep));
        SendReply(GDB_STUB_REPLY_OK);
        break;
    }
    case 'm': {
        const auto sep{std::find(command.begin(), command.end(), ',') - command.begin() + 1};
        const size_t addr{static_cast<size_t>(strtoll(command.data(), nullptr, 16))};
        const size_t size{static_cast<size_t>(strtoll(command.data() + sep, nullptr, 16))};

        if (system.ApplicationMemory().IsValidVirtualAddressRange(addr, size)) {
            std::vector<u8> mem(size);
            system.ApplicationMemory().ReadBlock(addr, mem.data(), size);

            SendReply(Common::HexToString(mem));
        } else {
            SendReply(GDB_STUB_REPLY_ERR);
        }
        break;
    }
    case 'M': {
        const auto size_sep{std::find(command.begin(), command.end(), ',') - command.begin() + 1};
        const auto mem_sep{std::find(command.begin(), command.end(), ':') - command.begin() + 1};

        const size_t addr{static_cast<size_t>(strtoll(command.data(), nullptr, 16))};
        const size_t size{static_cast<size_t>(strtoll(command.data() + size_sep, nullptr, 16))};

        const auto mem_substr{std::string_view(command).substr(mem_sep)};
        const auto mem{Common::HexStringToVector(mem_substr, false)};

        if (system.ApplicationMemory().IsValidVirtualAddressRange(addr, size)) {
            system.ApplicationMemory().WriteBlock(addr, mem.data(), size);
            system.InvalidateCpuInstructionCacheRange(addr, size);
            SendReply(GDB_STUB_REPLY_OK);
        } else {
            SendReply(GDB_STUB_REPLY_ERR);
        }
        break;
    }
    case 's':
        actions.push_back(DebuggerAction::StepThreadLocked);
        break;
    case 'C':
    case 'c':
        actions.push_back(DebuggerAction::Continue);
        break;
    case 'Z':
        HandleBreakpointInsert(command);
        break;
    case 'z':
        HandleBreakpointRemove(command);
        break;
    default:
        SendReply(GDB_STUB_REPLY_EMPTY);
        break;
    }
}

enum class BreakpointType {
    Software = 0,
    Hardware = 1,
    WriteWatch = 2,
    ReadWatch = 3,
    AccessWatch = 4,
};

void GDBStub::HandleBreakpointInsert(std::string_view command) {
    const auto type{static_cast<BreakpointType>(strtoll(command.data(), nullptr, 16))};
    const auto addr_sep{std::find(command.begin(), command.end(), ',') - command.begin() + 1};
    const auto size_sep{std::find(command.begin() + addr_sep, command.end(), ',') -
                        command.begin() + 1};
    const size_t addr{static_cast<size_t>(strtoll(command.data() + addr_sep, nullptr, 16))};
    const size_t size{static_cast<size_t>(strtoll(command.data() + size_sep, nullptr, 16))};

    if (!system.ApplicationMemory().IsValidVirtualAddressRange(addr, size)) {
        SendReply(GDB_STUB_REPLY_ERR);
        return;
    }

    bool success{};

    switch (type) {
    case BreakpointType::Software:
        replaced_instructions[addr] = system.ApplicationMemory().Read32(addr);
        system.ApplicationMemory().Write32(addr, arch->BreakpointInstruction());
        system.InvalidateCpuInstructionCacheRange(addr, sizeof(u32));
        success = true;
        break;
    case BreakpointType::WriteWatch:
        success = system.ApplicationProcess()->InsertWatchpoint(addr, size,
                                                                Kernel::DebugWatchpointType::Write);
        break;
    case BreakpointType::ReadWatch:
        success = system.ApplicationProcess()->InsertWatchpoint(addr, size,
                                                                Kernel::DebugWatchpointType::Read);
        break;
    case BreakpointType::AccessWatch:
        success = system.ApplicationProcess()->InsertWatchpoint(
            addr, size, Kernel::DebugWatchpointType::ReadOrWrite);
        break;
    case BreakpointType::Hardware:
    default:
        SendReply(GDB_STUB_REPLY_EMPTY);
        return;
    }

    if (success) {
        SendReply(GDB_STUB_REPLY_OK);
    } else {
        SendReply(GDB_STUB_REPLY_ERR);
    }
}

void GDBStub::HandleBreakpointRemove(std::string_view command) {
    const auto type{static_cast<BreakpointType>(strtoll(command.data(), nullptr, 16))};
    const auto addr_sep{std::find(command.begin(), command.end(), ',') - command.begin() + 1};
    const auto size_sep{std::find(command.begin() + addr_sep, command.end(), ',') -
                        command.begin() + 1};
    const size_t addr{static_cast<size_t>(strtoll(command.data() + addr_sep, nullptr, 16))};
    const size_t size{static_cast<size_t>(strtoll(command.data() + size_sep, nullptr, 16))};

    if (!system.ApplicationMemory().IsValidVirtualAddressRange(addr, size)) {
        SendReply(GDB_STUB_REPLY_ERR);
        return;
    }

    bool success{};

    switch (type) {
    case BreakpointType::Software: {
        const auto orig_insn{replaced_instructions.find(addr)};
        if (orig_insn != replaced_instructions.end()) {
            system.ApplicationMemory().Write32(addr, orig_insn->second);
            system.InvalidateCpuInstructionCacheRange(addr, sizeof(u32));
            replaced_instructions.erase(addr);
            success = true;
        }
        break;
    }
    case BreakpointType::WriteWatch:
        success = system.ApplicationProcess()->RemoveWatchpoint(addr, size,
                                                                Kernel::DebugWatchpointType::Write);
        break;
    case BreakpointType::ReadWatch:
        success = system.ApplicationProcess()->RemoveWatchpoint(addr, size,
                                                                Kernel::DebugWatchpointType::Read);
        break;
    case BreakpointType::AccessWatch:
        success = system.ApplicationProcess()->RemoveWatchpoint(
            addr, size, Kernel::DebugWatchpointType::ReadOrWrite);
        break;
    case BreakpointType::Hardware:
    default:
        SendReply(GDB_STUB_REPLY_EMPTY);
        return;
    }

    if (success) {
        SendReply(GDB_STUB_REPLY_OK);
    } else {
        SendReply(GDB_STUB_REPLY_ERR);
    }
}

// Structure offsets are from Atmosphere
// See osdbg_thread_local_region.os.horizon.hpp and osdbg_thread_type.os.horizon.hpp

static std::optional<std::string> GetNameFromThreadType32(Core::Memory::Memory& memory,
                                                          const Kernel::KThread* thread) {
    // Read thread type from TLS
    const VAddr tls_thread_type{memory.Read32(thread->GetTlsAddress() + 0x1fc)};
    const VAddr argument_thread_type{thread->GetArgument()};

    if (argument_thread_type && tls_thread_type != argument_thread_type) {
        // Probably not created by nnsdk, no name available.
        return std::nullopt;
    }

    if (!tls_thread_type) {
        return std::nullopt;
    }

    const u16 version{memory.Read16(tls_thread_type + 0x26)};
    VAddr name_pointer{};
    if (version == 1) {
        name_pointer = memory.Read32(tls_thread_type + 0xe4);
    } else {
        name_pointer = memory.Read32(tls_thread_type + 0xe8);
    }

    if (!name_pointer) {
        // No name provided.
        return std::nullopt;
    }

    return memory.ReadCString(name_pointer, 256);
}

static std::optional<std::string> GetNameFromThreadType64(Core::Memory::Memory& memory,
                                                          const Kernel::KThread* thread) {
    // Read thread type from TLS
    const VAddr tls_thread_type{memory.Read64(thread->GetTlsAddress() + 0x1f8)};
    const VAddr argument_thread_type{thread->GetArgument()};

    if (argument_thread_type && tls_thread_type != argument_thread_type) {
        // Probably not created by nnsdk, no name available.
        return std::nullopt;
    }

    if (!tls_thread_type) {
        return std::nullopt;
    }

    const u16 version{memory.Read16(tls_thread_type + 0x46)};
    VAddr name_pointer{};
    if (version == 1) {
        name_pointer = memory.Read64(tls_thread_type + 0x1a0);
    } else {
        name_pointer = memory.Read64(tls_thread_type + 0x1a8);
    }

    if (!name_pointer) {
        // No name provided.
        return std::nullopt;
    }

    return memory.ReadCString(name_pointer, 256);
}

static std::optional<std::string> GetThreadName(Core::System& system,
                                                const Kernel::KThread* thread) {
    if (system.ApplicationProcess()->Is64BitProcess()) {
        return GetNameFromThreadType64(system.ApplicationMemory(), thread);
    } else {
        return GetNameFromThreadType32(system.ApplicationMemory(), thread);
    }
}

static std::string_view GetThreadWaitReason(const Kernel::KThread* thread) {
    switch (thread->GetWaitReasonForDebugging()) {
    case Kernel::ThreadWaitReasonForDebugging::Sleep:
        return "Sleep";
    case Kernel::ThreadWaitReasonForDebugging::IPC:
        return "IPC";
    case Kernel::ThreadWaitReasonForDebugging::Synchronization:
        return "Synchronization";
    case Kernel::ThreadWaitReasonForDebugging::ConditionVar:
        return "ConditionVar";
    case Kernel::ThreadWaitReasonForDebugging::Arbitration:
        return "Arbitration";
    case Kernel::ThreadWaitReasonForDebugging::Suspended:
        return "Suspended";
    default:
        return "Unknown";
    }
}

static std::string GetThreadState(const Kernel::KThread* thread) {
    switch (thread->GetState()) {
    case Kernel::ThreadState::Initialized:
        return "Initialized";
    case Kernel::ThreadState::Waiting:
        return fmt::format("Waiting ({})", GetThreadWaitReason(thread));
    case Kernel::ThreadState::Runnable:
        return "Runnable";
    case Kernel::ThreadState::Terminated:
        return "Terminated";
    default:
        return "Unknown";
    }
}

static std::string PaginateBuffer(std::string_view buffer, std::string_view request) {
    const auto amount{request.substr(request.find(',') + 1)};
    const auto offset_val{static_cast<u64>(strtoll(request.data(), nullptr, 16))};
    const auto amount_val{static_cast<u64>(strtoll(amount.data(), nullptr, 16))};

    if (offset_val + amount_val > buffer.size()) {
        return fmt::format("l{}", buffer.substr(offset_val));
    } else {
        return fmt::format("m{}", buffer.substr(offset_val, amount_val));
    }
}

void GDBStub::HandleQuery(std::string_view command) {
    if (command.starts_with("TStatus")) {
        // no tracepoint support
        SendReply("T0");
    } else if (command.starts_with("Supported")) {
        SendReply("PacketSize=4000;qXfer:features:read+;qXfer:threads:read+;qXfer:libraries:read+;"
                  "vContSupported+;QStartNoAckMode+");
    } else if (command.starts_with("Xfer:features:read:target.xml:")) {
        const auto target_xml{arch->GetTargetXML()};
        SendReply(PaginateBuffer(target_xml, command.substr(30)));
    } else if (command.starts_with("Offsets")) {
        Loader::AppLoader::Modules modules;
        system.GetAppLoader().ReadNSOModules(modules);

        const auto main = std::find_if(modules.begin(), modules.end(),
                                       [](const auto& key) { return key.second == "main"; });
        if (main != modules.end()) {
            SendReply(fmt::format("TextSeg={:x}", main->first));
        } else {
            SendReply(fmt::format(
                "TextSeg={:x}",
                GetInteger(system.ApplicationProcess()->PageTable().GetCodeRegionStart())));
        }
    } else if (command.starts_with("Xfer:libraries:read::")) {
        Loader::AppLoader::Modules modules;
        system.GetAppLoader().ReadNSOModules(modules);

        std::string buffer;
        buffer += R"(<?xml version="1.0"?>)";
        buffer += "<library-list>";
        for (const auto& [base, name] : modules) {
            buffer += fmt::format(R"(<library name="{}"><segment address="{:#x}"/></library>)",
                                  EscapeXML(name), base);
        }
        buffer += "</library-list>";

        SendReply(PaginateBuffer(buffer, command.substr(21)));
    } else if (command.starts_with("fThreadInfo")) {
        // beginning of list
        const auto& threads = system.ApplicationProcess()->GetThreadList();
        std::vector<std::string> thread_ids;
        for (const auto& thread : threads) {
            thread_ids.push_back(fmt::format("{:x}", thread->GetThreadId()));
        }
        SendReply(fmt::format("m{}", fmt::join(thread_ids, ",")));
    } else if (command.starts_with("sThreadInfo")) {
        // end of list
        SendReply("l");
    } else if (command.starts_with("Xfer:threads:read::")) {
        std::string buffer;
        buffer += R"(<?xml version="1.0"?>)";
        buffer += "<threads>";

        const auto& threads = system.ApplicationProcess()->GetThreadList();
        for (const auto* thread : threads) {
            auto thread_name{GetThreadName(system, thread)};
            if (!thread_name) {
                thread_name = fmt::format("Thread {:d}", thread->GetThreadId());
            }

            buffer += fmt::format(R"(<thread id="{:x}" core="{:d}" name="{}">{}</thread>)",
                                  thread->GetThreadId(), thread->GetActiveCore(),
                                  EscapeXML(*thread_name), GetThreadState(thread));
        }

        buffer += "</threads>";

        SendReply(PaginateBuffer(buffer, command.substr(19)));
    } else if (command.starts_with("Attached")) {
        SendReply("0");
    } else if (command.starts_with("StartNoAckMode")) {
        no_ack = true;
        SendReply(GDB_STUB_REPLY_OK);
    } else if (command.starts_with("Rcmd,")) {
        HandleRcmd(Common::HexStringToVector(command.substr(5), false));
    } else {
        SendReply(GDB_STUB_REPLY_EMPTY);
    }
}

void GDBStub::HandleVCont(std::string_view command, std::vector<DebuggerAction>& actions) {
    if (command == "?") {
        // Continuing and stepping are supported
        // (signal is ignored, but required for GDB to use vCont)
        SendReply("vCont;c;C;s;S");
        return;
    }

    Kernel::KThread* stepped_thread{nullptr};
    bool lock_execution{true};

    std::vector<std::string> entries;
    boost::split(entries, command.substr(1), boost::is_any_of(";"));
    for (const auto& thread_action : entries) {
        std::vector<std::string> parts;
        boost::split(parts, thread_action, boost::is_any_of(":"));

        if (parts.size() == 1 && (parts[0] == "c" || parts[0].starts_with("C"))) {
            lock_execution = false;
        }
        if (parts.size() == 2 && (parts[0] == "s" || parts[0].starts_with("S"))) {
            stepped_thread = GetThreadByID(strtoll(parts[1].data(), nullptr, 16));
        }
    }

    if (stepped_thread) {
        backend.SetActiveThread(stepped_thread);
        actions.push_back(lock_execution ? DebuggerAction::StepThreadLocked
                                         : DebuggerAction::StepThreadUnlocked);
    } else {
        actions.push_back(DebuggerAction::Continue);
    }
}

constexpr std::array<std::pair<const char*, Kernel::Svc::MemoryState>, 22> MemoryStateNames{{
    {"----- Free -----", Kernel::Svc::MemoryState::Free},
    {"Io              ", Kernel::Svc::MemoryState::Io},
    {"Static          ", Kernel::Svc::MemoryState::Static},
    {"Code            ", Kernel::Svc::MemoryState::Code},
    {"CodeData        ", Kernel::Svc::MemoryState::CodeData},
    {"Normal          ", Kernel::Svc::MemoryState::Normal},
    {"Shared          ", Kernel::Svc::MemoryState::Shared},
    {"AliasCode       ", Kernel::Svc::MemoryState::AliasCode},
    {"AliasCodeData   ", Kernel::Svc::MemoryState::AliasCodeData},
    {"Ipc             ", Kernel::Svc::MemoryState::Ipc},
    {"Stack           ", Kernel::Svc::MemoryState::Stack},
    {"ThreadLocal     ", Kernel::Svc::MemoryState::ThreadLocal},
    {"Transfered      ", Kernel::Svc::MemoryState::Transfered},
    {"SharedTransfered", Kernel::Svc::MemoryState::SharedTransfered},
    {"SharedCode      ", Kernel::Svc::MemoryState::SharedCode},
    {"Inaccessible    ", Kernel::Svc::MemoryState::Inaccessible},
    {"NonSecureIpc    ", Kernel::Svc::MemoryState::NonSecureIpc},
    {"NonDeviceIpc    ", Kernel::Svc::MemoryState::NonDeviceIpc},
    {"Kernel          ", Kernel::Svc::MemoryState::Kernel},
    {"GeneratedCode   ", Kernel::Svc::MemoryState::GeneratedCode},
    {"CodeOut         ", Kernel::Svc::MemoryState::CodeOut},
    {"Coverage        ", Kernel::Svc::MemoryState::Coverage},
}};

static constexpr const char* GetMemoryStateName(Kernel::Svc::MemoryState state) {
    for (size_t i = 0; i < MemoryStateNames.size(); i++) {
        if (std::get<1>(MemoryStateNames[i]) == state) {
            return std::get<0>(MemoryStateNames[i]);
        }
    }
    return "Unknown         ";
}

static constexpr const char* GetMemoryPermissionString(const Kernel::Svc::MemoryInfo& info) {
    if (info.state == Kernel::Svc::MemoryState::Free) {
        return "   ";
    }

    switch (info.permission) {
    case Kernel::Svc::MemoryPermission::ReadExecute:
        return "r-x";
    case Kernel::Svc::MemoryPermission::Read:
        return "r--";
    case Kernel::Svc::MemoryPermission::ReadWrite:
        return "rw-";
    default:
        return "---";
    }
}

static VAddr GetModuleEnd(Kernel::KPageTable& page_table, VAddr base) {
    Kernel::Svc::MemoryInfo mem_info;
    VAddr cur_addr{base};

    // Expect: r-x Code (.text)
    mem_info = page_table.QueryInfo(cur_addr).GetSvcMemoryInfo();
    cur_addr = mem_info.base_address + mem_info.size;
    if (mem_info.state != Kernel::Svc::MemoryState::Code ||
        mem_info.permission != Kernel::Svc::MemoryPermission::ReadExecute) {
        return cur_addr - 1;
    }

    // Expect: r-- Code (.rodata)
    mem_info = page_table.QueryInfo(cur_addr).GetSvcMemoryInfo();
    cur_addr = mem_info.base_address + mem_info.size;
    if (mem_info.state != Kernel::Svc::MemoryState::Code ||
        mem_info.permission != Kernel::Svc::MemoryPermission::Read) {
        return cur_addr - 1;
    }

    // Expect: rw- CodeData (.data)
    mem_info = page_table.QueryInfo(cur_addr).GetSvcMemoryInfo();
    cur_addr = mem_info.base_address + mem_info.size;
    return cur_addr - 1;
}

void GDBStub::HandleRcmd(const std::vector<u8>& command) {
    std::string_view command_str{reinterpret_cast<const char*>(&command[0]), command.size()};
    std::string reply;

    auto* process = system.ApplicationProcess();
    auto& page_table = process->PageTable();

    const char* commands = "Commands:\n"
                           "  get fastmem\n"
                           "  get info\n"
                           "  get mappings\n";

    if (command_str == "get fastmem") {
        if (Settings::IsFastmemEnabled()) {
            const auto& impl = page_table.PageTableImpl();
            const auto region = reinterpret_cast<uintptr_t>(impl.fastmem_arena);
            const auto region_bits = impl.current_address_space_width_in_bits;
            const auto region_size = 1ULL << region_bits;

            reply = fmt::format("Region bits:  {}\n"
                                "Host address: {:#x} - {:#x}\n",
                                region_bits, region, region + region_size - 1);
        } else {
            reply = "Fastmem is not enabled.\n";
        }
    } else if (command_str == "get info") {
        Loader::AppLoader::Modules modules;
        system.GetAppLoader().ReadNSOModules(modules);

        reply = fmt::format("Process:     {:#x} ({})\n"
                            "Program Id:  {:#018x}\n",
                            process->GetProcessId(), process->GetName(), process->GetProgramId());
        reply += fmt::format("Layout:\n"
                             "  Alias: {:#012x} - {:#012x}\n"
                             "  Heap:  {:#012x} - {:#012x}\n"
                             "  Aslr:  {:#012x} - {:#012x}\n"
                             "  Stack: {:#012x} - {:#012x}\n"
                             "Modules:\n",
                             GetInteger(page_table.GetAliasRegionStart()),
                             GetInteger(page_table.GetAliasRegionEnd()),
                             GetInteger(page_table.GetHeapRegionStart()),
                             GetInteger(page_table.GetHeapRegionEnd()),
                             GetInteger(page_table.GetAliasCodeRegionStart()),
                             GetInteger(page_table.GetAliasCodeRegionEnd()),
                             GetInteger(page_table.GetStackRegionStart()),
                             GetInteger(page_table.GetStackRegionEnd()));

        for (const auto& [vaddr, name] : modules) {
            reply += fmt::format("  {:#012x} - {:#012x} {}\n", vaddr,
                                 GetModuleEnd(page_table, vaddr), name);
        }
    } else if (command_str == "get mappings") {
        reply = "Mappings:\n";
        VAddr cur_addr = 0;

        while (true) {
            using MemoryAttribute = Kernel::Svc::MemoryAttribute;

            auto mem_info = page_table.QueryInfo(cur_addr).GetSvcMemoryInfo();

            if (mem_info.state != Kernel::Svc::MemoryState::Inaccessible ||
                mem_info.base_address + mem_info.size - 1 != std::numeric_limits<u64>::max()) {
                const char* state = GetMemoryStateName(mem_info.state);
                const char* perm = GetMemoryPermissionString(mem_info);

                const char l = True(mem_info.attribute & MemoryAttribute::Locked) ? 'L' : '-';
                const char i = True(mem_info.attribute & MemoryAttribute::IpcLocked) ? 'I' : '-';
                const char d = True(mem_info.attribute & MemoryAttribute::DeviceShared) ? 'D' : '-';
                const char u = True(mem_info.attribute & MemoryAttribute::Uncached) ? 'U' : '-';

                reply +=
                    fmt::format("  {:#012x} - {:#012x} {} {} {}{}{}{} [{}, {}]\n",
                                mem_info.base_address, mem_info.base_address + mem_info.size - 1,
                                perm, state, l, i, d, u, mem_info.ipc_count, mem_info.device_count);
            }

            const uintptr_t next_address = mem_info.base_address + mem_info.size;
            if (next_address <= cur_addr) {
                break;
            }

            cur_addr = next_address;
        }
    } else if (command_str == "help") {
        reply = commands;
    } else {
        reply = "Unknown command.\n";
        reply += commands;
    }

    std::span<const u8> reply_span{reinterpret_cast<u8*>(&reply.front()), reply.size()};
    SendReply(Common::HexToString(reply_span, false));
}

Kernel::KThread* GDBStub::GetThreadByID(u64 thread_id) {
    const auto& threads{system.ApplicationProcess()->GetThreadList()};
    for (auto* thread : threads) {
        if (thread->GetThreadId() == thread_id) {
            return thread;
        }
    }

    return nullptr;
}

std::vector<char>::const_iterator GDBStub::CommandEnd() const {
    // Find the end marker
    const auto end{std::find(current_command.begin(), current_command.end(), GDB_STUB_END)};

    // Require the checksum to be present
    return std::min(end + 2, current_command.end());
}

std::optional<std::string> GDBStub::DetachCommand() {
    // Slice the string part from the beginning to the end marker
    const auto end{CommandEnd()};

    // Extract possible command data
    std::string data(current_command.data(), end - current_command.begin() + 1);

    // Shift over the remaining contents
    current_command.erase(current_command.begin(), end + 1);

    // Validate received command
    if (data[0] != GDB_STUB_START) {
        LOG_ERROR(Debug_GDBStub, "Invalid start data: {}", data[0]);
        return std::nullopt;
    }

    u8 calculated = CalculateChecksum(std::string_view(data).substr(1, data.size() - 4));
    u8 received = static_cast<u8>(strtoll(data.data() + data.size() - 2, nullptr, 16));

    // Verify checksum
    if (calculated != received) {
        LOG_ERROR(Debug_GDBStub, "Checksum mismatch: calculated {:02x}, received {:02x}",
                  calculated, received);
        return std::nullopt;
    }

    return data.substr(1, data.size() - 4);
}

void GDBStub::SendReply(std::string_view data) {
    const auto escaped{EscapeGDB(data)};
    const auto output{fmt::format("{}{}{}{:02x}", GDB_STUB_START, escaped, GDB_STUB_END,
                                  CalculateChecksum(escaped))};
    LOG_TRACE(Debug_GDBStub, "Writing reply: {}", output);

    // C++ string support is complete rubbish
    const u8* output_begin = reinterpret_cast<const u8*>(output.data());
    const u8* output_end = output_begin + output.size();
    backend.WriteToClient(std::span<const u8>(output_begin, output_end));
}

void GDBStub::SendStatus(char status) {
    if (no_ack) {
        return;
    }

    std::array<u8, 1> buf = {static_cast<u8>(status)};
    LOG_TRACE(Debug_GDBStub, "Writing status: {}", status);
    backend.WriteToClient(buf);
}

} // namespace Core
