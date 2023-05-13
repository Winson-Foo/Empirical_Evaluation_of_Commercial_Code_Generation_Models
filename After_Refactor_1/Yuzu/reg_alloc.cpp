// SPDX-License-Identifier: GPL-2.0-or-later

#include <bitset>
#include <fmt/format.h>

#include "shader_recompiler/backend/glasm/reg_alloc.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/value.h"

namespace Shader::Backend::GLASM {

const size_t NUM_REGS = 1024;

// Represents a register identifier.
struct RegId {
    bool is_valid = false;
    bool is_long = false;
    bool is_spill = false;
    bool is_condition_code = false;
    bool is_null = false;
    u32 index = 0;
};

// Represents a register or an immediate value.
struct RegValue {
    enum class Type {
        Void,
        U32,
        U64,
    };
    Type type = Type::Void;
    u64 value = 0;
};

// Performs register allocation for GLASM shaders.
class RegAllocator {
public:
    // Defines a new register for the given instruction and returns its register identifier.
    RegId DefineRegister(IR::Inst& inst);

    // Defines a new long register for the given instruction and returns its register identifier.
    RegId DefineLongRegister(IR::Inst& inst);

    // Returns the register or immediate value corresponding to the given IR value.
    RegValue PeekValue(const IR::Value& value) const;

    // Consumes the register corresponding to the given IR value and returns its value.
    RegValue ConsumeValue(IR::Value& value);

    // Removes the reference to the given instruction and frees its register if it has no uses.
    void UnrefInst(IR::Inst& inst);

    // Allocates a new register and returns its identifier.
    RegId AllocRegister();

    // Allocates a new long register and returns its identifier.
    RegId AllocLongRegister();

    // Frees the register corresponding to the given register identifier.
    void FreeRegister(RegId id);

private:
    size_t num_used_registers = 0;
    size_t num_used_long_registers = 0;
    std::bitset<NUM_REGS> register_use;
    std::bitset<NUM_REGS> long_register_use;

    // Frees the register corresponding to the given register identifier.
    void Free(RegId id);

    // Returns the register corresponding to the given IR instruction.
    RegValue PeekInst(IR::Inst& inst) const;

    // Consumes the register corresponding to the given IR instruction and returns its value.
    RegValue ConsumeInst(IR::Inst& inst);

    // Returns the alias instruction for the given IR instruction, if any.
    IR::Inst& GetAliasInst(IR::Inst& inst) const;

    // Determines if the given IR instruction is an alias instruction.
    static bool IsAliasInst(const IR::Inst& inst);

    // Determines if the given register identifier is valid.
    static bool IsValidId(RegId id) {
        return id.is_valid && !id.is_null;
    }

    // Allocates a new register and returns its identifier.
    RegId Alloc(bool is_long);
};

RegId RegAllocator::DefineRegister(IR::Inst& inst) {
    return DefineRegister(inst, false);
}

RegId RegAllocator::DefineLongRegister(IR::Inst& inst) {
    return DefineRegister(inst, true);
}

RegValue RegAllocator::PeekValue(const IR::Value& value) const {
    if (value.IsImmediate()) {
        const IR::Immediate& imm = value.Imm();
        switch (imm.Type()) {
        case IR::ImmediateType::U1:
            return { RegValue::Type::U32, imm.U1() ? UINT32_MAX : 0 };
        case IR::ImmediateType::U32:
            return { RegValue::Type::U32, imm.U32() };
        case IR::ImmediateType::F32:
            return { RegValue::Type::U32, Common::BitCast<u32>(imm.F32()) };
        case IR::ImmediateType::U64:
            return { RegValue::Type::U64, imm.U64() };
        case IR::ImmediateType::F64:
            return { RegValue::Type::U64, Common::BitCast<u64>(imm.F64()) };
        default:
            throw NotImplementedException("Immediate type {}", imm.Type());
        }
    } else {
        return PeekInst(*value.Inst());
    }
}

RegValue RegAllocator::ConsumeValue(IR::Value& value) {
    if (value.IsImmediate()) {
        return PeekValue(value);
    } else {
        return ConsumeInst(*value.Inst());
    }
}

void RegAllocator::UnrefInst(IR::Inst& inst) {
    IR::Inst& value_inst = GetAliasInst(inst);
    value_inst.DestructiveRemoveUsage();
    if (!value_inst.HasUses()) {
        Free(value_inst.Definition<RegId>());
    }
}

RegId RegAllocator::AllocRegister() {
    RegId id;
    id.is_valid = true;
    id.is_long = false;
    id.is_spill = false;
    id.is_condition_code = false;
    id.is_null = false;
    for (size_t i = 0; i < NUM_REGS; i++) {
        if (!register_use[i]) {
            register_use[i] = true;
            id.index = static_cast<u32>(i);
            num_used_registers = std::max(num_used_registers, i + 1);
            return id;
        }
    }
    throw NotImplementedException("Register spilling");
}

RegId RegAllocator::AllocLongRegister() {
    RegId id;
    id.is_valid = true;
    id.is_long = true;
    id.is_spill = false;
    id.is_condition_code = false;
    id.is_null = false;
    for (size_t i = 0; i < NUM_REGS; i++) {
        if (!long_register_use[i]) {
            long_register_use[i] = true;
            id.index = static_cast<u32>(i);
            num_used_long_registers = std::max(num_used_long_registers, i + 1);
            return id;
        }
    }
    throw NotImplementedException("Register spilling");
}

void RegAllocator::FreeRegister(RegId id) {
    if (!IsValidId(id)) {
        throw LogicError("Freeing invalid register");
    }
    Free(id);
}

void RegAllocator::Free(RegId id) {
    if (id.is_spill) {
        throw NotImplementedException("Free spill");
    }
    if (id.is_long) {
        long_register_use[id.index] = false;
    } else {
        register_use[id.index] = false;
    }
}

RegValue RegAllocator::PeekInst(IR::Inst& inst) const {
    RegValue ret;
    ret.type = RegValue::Type::U32;
    ret.value = inst.Definition<RegId>().index;
    return ret;
}

RegValue RegAllocator::ConsumeInst(IR::Inst& inst) {
    UnrefInst(inst);
    return PeekInst(inst);
}

IR::Inst& RegAllocator::GetAliasInst(IR::Inst& inst) const {
    IR::Inst* it = &inst;
    while (IsAliasInst(*it)) {
        const IR::Value arg = it->Arg(0);
        if (arg.IsImmediate()) {
            break;
        }
        it = arg.InstRecursive();
    }
    return *it;
}

bool RegAllocator::IsAliasInst(const IR::Inst& inst) {
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

RegId RegAllocator::DefineRegister(IR::Inst& inst, bool is_long) {
    if (inst.HasUses()) {
        inst.SetDefinition<RegId>(Alloc(is_long));
    } else {
        RegId id;
        id.is_valid = true;
        id.is_long = is_long;
        id.is_spill = false;
        id.is_condition_code = false;
        id.is_null = true;
        inst.SetDefinition<RegId>(id);
    }
    return inst.Definition<RegId>();
}

RegId RegAllocator::Alloc(bool is_long) {
    size_t& num_regs = is_long ? num_used_long_registers : num_used_registers;
    std::bitset<NUM_REGS>& use = is_long ? long_register_use : register_use;
    for (size_t i = 0; i < NUM_REGS; i++) {
        if (!use[i]) {
            use[i] = true;
            num_regs = std::max(num_regs, i + 1);
            RegId id;
            id.is_valid = true;
            id.is_long = is_long;
            id.is_spill = false;
            id.is_condition_code = false;
            id.is_null = false;
            id.index = static_cast<u32>(i);
            return id;
        }
    }
    throw NotImplementedException("Register spilling");
}

} // namespace Shader::Backend::GLASM