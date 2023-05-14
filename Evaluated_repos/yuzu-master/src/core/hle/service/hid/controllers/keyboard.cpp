// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include "common/common_types.h"
#include "common/settings.h"
#include "core/core_timing.h"
#include "core/hid/emulated_devices.h"
#include "core/hid/hid_core.h"
#include "core/hle/service/hid/controllers/keyboard.h"

namespace Service::HID {
constexpr std::size_t SHARED_MEMORY_OFFSET = 0x3800;

Controller_Keyboard::Controller_Keyboard(Core::HID::HIDCore& hid_core_, u8* raw_shared_memory_)
    : ControllerBase{hid_core_} {
    static_assert(SHARED_MEMORY_OFFSET + sizeof(KeyboardSharedMemory) < shared_memory_size,
                  "KeyboardSharedMemory is bigger than the shared memory");
    shared_memory = std::construct_at(
        reinterpret_cast<KeyboardSharedMemory*>(raw_shared_memory_ + SHARED_MEMORY_OFFSET));
    emulated_devices = hid_core.GetEmulatedDevices();
}

Controller_Keyboard::~Controller_Keyboard() = default;

void Controller_Keyboard::OnInit() {}

void Controller_Keyboard::OnRelease() {}

void Controller_Keyboard::OnUpdate(const Core::Timing::CoreTiming& core_timing) {
    if (!IsControllerActivated()) {
        shared_memory->keyboard_lifo.buffer_count = 0;
        shared_memory->keyboard_lifo.buffer_tail = 0;
        return;
    }

    const auto& last_entry = shared_memory->keyboard_lifo.ReadCurrentEntry().state;
    next_state.sampling_number = last_entry.sampling_number + 1;

    if (Settings::values.keyboard_enabled) {
        const auto& keyboard_state = emulated_devices->GetKeyboard();
        const auto& keyboard_modifier_state = emulated_devices->GetKeyboardModifier();

        next_state.key = keyboard_state;
        next_state.modifier = keyboard_modifier_state;
        next_state.attribute.is_connected.Assign(1);
    }

    shared_memory->keyboard_lifo.WriteNextEntry(next_state);
}

} // namespace Service::HID
