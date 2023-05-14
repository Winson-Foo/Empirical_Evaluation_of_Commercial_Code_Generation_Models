/*
 * Copyright 2021 yuzu Emulator Project
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include <fmt/format.h>

#include "shader_recompiler/backend/glasm/reg_alloc.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/value.h"

namespace Shader::Backend::GLASM {

const int INVALID_ID = -1;

// Helper functions

template <typename T>
inline void set_bit(T& value, int bit, bool set) {
    if (set) {
        value |= 1 << bit;
    } else {
        value &= ~(1 << bit);
    }
}

inline bool get_bit(const u32 value, int bit) {
    return (value >> bit) & 1;
}

inline void set_id_field(Id& id, int field, bool value) {
    switch (field) {
        case 0: set_bit(id.is_null, 0, value); break;
        case 1: set_bit(id.is_valid, 0, value); break;
        case 2: set_bit(id.is_spill, 0, value); break;
        case 3: set_bit(id.is_long, 0, value); break;
        case 4: set_bit(id.is_condition_code, 0, value); break;
        case 5: set_bit(id.index, 0, value); break;
    }
}

inline bool get_id_field(const Id& id, int field) {
    switch (field) {
        case 0: return get_bit(id.is_null, 0);
        case 1: return get_bit(id.is_valid, 0);
        case 2: return get_bit(id.is_spill, 0);
        case 3: return get_bit(id.is_long, 0);
        case 4: return get_bit(id.is_condition_code, 0);
        case 5: return get_bit(id.index, 0);
    }
    return false;
}

// RegAlloc functions

Register RegAlloc::Define(IR::Inst& inst) {
    return Define(inst, false);
}

Register RegAlloc::LongDefine(IR::Inst& inst) {
    return Define(inst, true);
}

Value RegAlloc::Peek(const IR::Value& value) {
    if (value.IsImmediate()) {
        return MakeImm(value);
    } else {
        return PeekInst(*value.Inst());
    }
}

Value RegAlloc::Consume(const IR::Value& value) {
    if (value.IsImmediate()) {
        return MakeImm(value);
    } else {
        return ConsumeInst(*value.Inst());
    }
}

void RegAlloc::Unref(IR::Inst& inst) {
    IR::Inst& value_inst{AliasInst(inst)};
    value_inst.DestructiveRemoveUsage();
    if (!value_inst.HasUses()) {
        Free(value_inst.Definition<Id>());
    }
}

Register RegAlloc::AllocReg(bool is_long) {
    Register ret;
    ret.type = Type::Register;
    ret.id = Alloc(is_long);
    return ret;
}

void RegAlloc::FreeReg(Register reg) {
    Free(reg.id);
}

Value RegAlloc::MakeImm(const IR::Value& value) {
    Value ret;
    switch (value.Type()) {
        case IR::Type::Void:
            ret.type = Type::Void;
            break;
        case IR::Type::U1:
            ret.type = Type::U32;
            ret.imm_u32 = value.U1() ? 0xffffffff : 0;
            break;
        case IR::Type::U32:
            ret.type = Type::U32;
            ret.imm_u32 = value.U32();
            break;
        case IR::Type::F32:
            ret.type = Type::U32;
            ret.imm_u32 = Common::BitCast<u32>(value.F32());
            break;
        case IR::Type::U64:
            ret.type = Type::U64;
            ret.imm_u64 = value.U64();
            break;
        case IR::Type::F64:
            ret.type = Type::U64;
            ret.imm_u64 = Common::BitCast<u64>(value.F64());
            break;
        default:
            throw NotImplementedException("Immediate type {}", value.Type());
    }
    return ret;
}

Register RegAlloc::Define(IR::Inst& inst, bool is_long) {
    if (inst.HasUses()) {
        inst.SetDefinition<Id>(Alloc(is_long));
    } else {
        Id id{};
        set_id_field(id, 3, is_long);
        inst.SetDefinition<Id>(id);
    }
    return Register{PeekInst(inst)};
}

Value RegAlloc::PeekInst(IR::Inst& inst) {
    Value ret;
    ret.type = Type::Register;
    ret.id = inst.Definition<Id>();
    return ret;
}

Value RegAlloc::ConsumeInst(IR::Inst& inst) {
    Unref(inst);
    return PeekInst(inst);
}

int RegAlloc::Alloc(bool is_long) {
    int& num_regs{(is_long ? num_used_long_registers : num_used_registers)};
    std::vector<int>& reg_use{(is_long ? long_register_use : register_use)};
    if (num_regs < NUM_REGS) {
        for (int reg = 0; reg < NUM_REGS; ++reg) {
            if (reg_use[reg] != INVALID_ID) {
                continue;
            }
            num_regs++;
            reg_use[reg] = num_regs;
            Id ret{};
            set_id_field(ret, 1, true);
            set_id_field(ret, 3, is_long);
            set_id_field(ret, 5, true);
            set_id_field(ret, 5, reg);
            return num_regs;
        }
    }
    throw NotImplementedException("Register spilling");
}

void RegAlloc::Free(Id id) {
    if (id.is_valid == 0) {
        throw LogicError("Freeing invalid register");
    }
    if (id.is_spill != 0) {
        throw NotImplementedException("Free spill");
    }
    if (id.is_long != 0) {
        long_register_use[id.index] = INVALID_ID;
    } else {
        register_use[id.index] = INVALID_ID;
    }
}

bool RegAlloc::IsAliased(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
        case IR::Opcode::Identity:
        case IR::Opcode::BitCastU16F16:
        case IR::Opcode::BitCastU32F32:
        case IR::Opcode::BitCastU64F64:
        case IR::Opcode::BitCastF16U16:
        case IR::Opcode::BitCastF32U32:
        case IR::Opcode::BitCastF64U64:
            return true;
        default:
            return false;
    }
}

IR::Inst& RegAlloc::AliasInst(IR::Inst& inst) {
    IR::Inst* it{&inst};
    while (IsAliased(*it)) {
        const IR::Value arg{it->Arg(0)};
        if (arg.IsImmediate()) {
            break;
        }
        it = arg.InstRecursive();
    }
    return *it;
}

} // namespace Shader::Backend::GLASM