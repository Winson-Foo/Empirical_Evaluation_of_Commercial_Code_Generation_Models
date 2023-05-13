// SPDX-License-Identifier: GPL-2.0-or-later

#include <fmt/format.h>

#include "shader_recompiler/backend/glasm/reg_alloc.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/value.h"

namespace Shader::Backend::GLASM {

// Define a register to hold the result of an instruction
Register RegAlloc::Define(IR::Inst& inst) {
    return Define(inst, false);
}

// Define a long register to hold the result of an instruction
Register RegAlloc::LongDefine(IR::Inst& inst) {
    return Define(inst, true);
}

// Peek at the value of a given operand
Value RegAlloc::Peek(const IR::Value& value) {
    if (value.IsImmediate()) {
        return MakeImm(value);
    } else {
        return PeekInst(*value.Inst());
    }
}

// Consume the value of a given operand (used for freeing registers)
Value RegAlloc::Consume(const IR::Value& value) {
    if (value.IsImmediate()) {
        return MakeImm(value);
    } else {
        return ConsumeInst(*value.Inst());
    }
}

// Remove a reference to an instruction and free the associated register if it has no more uses
void RegAlloc::Unref(IR::Inst& inst) {
    IR::Inst& value_inst{AliasInst(inst)};
    value_inst.DestructiveRemoveUsage();
    if (!value_inst.HasUses()) {
        Free(value_inst.Definition<Id>());
    }
}

// Allocate a register of the default type
Register RegAlloc::AllocReg() {
    Register ret;
    ret.type = Type::Register;
    ret.id = Alloc(false);
    return ret;
}

// Allocate a long register
Register RegAlloc::AllocLongReg() {
    Register ret;
    ret.type = Type::Register;
    ret.id = Alloc(true);
    return ret;
}

// Free a register
void RegAlloc::FreeReg(Register reg) {
    Free(reg.id);
}

// Make an immediate value
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

// Define a register to hold the result of an instruction and mark it as long if necessary
Register RegAlloc::Define(IR::Inst& inst, bool is_long) {
    if (inst.HasUses()) {
        inst.SetDefinition<Id>(Alloc(is_long));
    } else {
        Id id{};
        id.is_long.Assign(is_long ? 1 : 0);
        id.is_null.Assign(1);
        inst.SetDefinition<Id>(id);
    }
    return Register{PeekInst(inst)};
}

// Peek at the value held by an instruction's result register
Value RegAlloc::PeekInst(IR::Inst& inst) {
    Value ret;
    ret.type = Type::Register;
    ret.id = inst.Definition<Id>();
    return ret;
}

// Consume the value held by an instruction's result register
Value RegAlloc::ConsumeInst(IR::Inst& inst) {
    Unref(inst);
    return PeekInst(inst);
}

// Allocate a register
Id RegAlloc::Alloc(bool is_long) {
    size_t& num_regs{is_long ? num_used_long_registers : num_used_registers};
    std::bitset<NUM_REGS>& use{is_long ? long_register_use : register_use};

    // Find the next available register
    if (num_used_registers + num_used_long_registers < NUM_REGS) {
        for (size_t reg = 0; reg < NUM_REGS; ++reg) {
            if (use[reg]) {
                continue;
            }
            num_regs = std::max(num_regs, reg + 1);
            use[reg] = true;
            Id ret{};
            ret.is_valid.Assign(1);
            ret.is_long.Assign(is_long ? 1 : 0);
            ret.is_spill.Assign(0);
            ret.is_condition_code.Assign(0);
            ret.is_null.Assign(0);
            ret.index.Assign(static_cast<u32>(reg));
            return ret;
        }
    }

    // TODO: handle register spilling
    throw NotImplementedException("Register spilling");
}

// Free a register
void RegAlloc::Free(Id id) {
    if (id.is_valid == 0) {
        throw LogicError("Freeing invalid register");
    }
    if (id.is_spill != 0) {
        // TODO: handle spill freeing
        throw NotImplementedException("Free spill");
    }
    if (id.is_long != 0) {
        long_register_use[id.index] = false;
    } else {
        register_use[id.index] = false;
    }
}

// Check if an instruction is an alias of another instruction (e.g. a bitcast)
/*static*/ bool RegAlloc::IsAliased(const IR::Inst& inst) {
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

// Get the instruction that an alias refers to
/*static*/ IR::Inst& RegAlloc::AliasInst(IR::Inst& inst) {
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

