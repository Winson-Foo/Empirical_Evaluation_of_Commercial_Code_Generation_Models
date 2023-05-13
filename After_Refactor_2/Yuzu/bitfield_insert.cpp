#include "common/bit_field.h"
#include "common/common_types.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"

namespace Shader::Maxwell {
namespace {

// Define constants
const int kMaxSize = 32;

// Helper function to extract and return offset and count values from a given instruction
std::pair<IR::U32, IR::U32> GetOffsetAndCount(TranslatorVisitor& v, const IR::U32& src_a) {
    const IR::U32 zero{v.ir.Imm32(0)};
    const IR::U32 offset{v.ir.BitFieldExtract(src_a, zero, v.ir.Imm32(8), false)};
    const IR::U32 count{v.ir.BitFieldExtract(src_a, v.ir.Imm32(8), v.ir.Imm32(8), false)};
    return std::make_pair(offset, count);
}

// Helper function to handle edge cases and return safe count value
IR::U32 GetSafeCount(TranslatorVisitor& v, const IR::U32& offset, const IR::U32& count) {
    const IR::U1 exceed_offset{v.ir.IGreaterThanEqual(offset, kMaxSize, false)};
    const IR::U1 exceed_count{v.ir.IGreaterThan(count, kMaxSize, false)};

    const IR::U32 remaining_size{v.ir.ISub(kMaxSize, offset)};
    const IR::U32 safe_count{v.ir.Select(exceed_count, remaining_size, count)};

    return v.ir.Select(exceed_offset, v.ir.Imm32(0), safe_count);
}

// Helper function to handle BitFieldInsert operation and return result
IR::U32 GetBFIResult(TranslatorVisitor& v, const IR::U32& base, const IR::U32& insert, const IR::U32& offset, const IR::U32& count) {
    return v.ir.BitFieldInsert(base, insert, offset, count);
}

// Helper function to handle flags in case of conditional code execution
void HandleFlags(TranslatorVisitor& v, const IR::U32& result, const IR::U32& zero) {
    v.SetZFlag(v.ir.IEqual(result, zero));
    v.SetSFlag(v.ir.ILessThan(result, zero, true));
    v.ResetCFlag();
    v.ResetOFlag();
}

void BFI(TranslatorVisitor& v, u64 insn, const IR::U32& src_a, const IR::U32& base) {
    // Extract fields from instruction
    union {
        u64 insn;
        BitField<0, 8, IR::Reg> dest_reg;
        BitField<8, 8, IR::Reg> insert_reg;
        BitField<47, 1, u64> cc;
    } const bfi{insn};

    // Extract offset and count values
    const auto [offset, count] = GetOffsetAndCount(v, src_a);

    // Get safe count value
    const IR::U32 safe_count{GetSafeCount(v, offset, count)};

    // Get BFI result
    const IR::U32 insert{v.X(bfi.insert_reg)};
    IR::U32 result{GetBFIResult(v, base, insert, offset, safe_count)};

    // Handle edge case of exceeding max size
    result = v.ir.Select(v.ir.IGreaterThanEqual(offset, kMaxSize, false), base, result);

    // Write result to destination register
    v.X(bfi.dest_reg, result);

    // Handle flags in case of conditional code execution
    if (bfi.cc != 0) {
        HandleFlags(v, result, v.ir.Imm32(0));
    }
}
} // Anonymous namespace

void TranslatorVisitor::BFI_reg(u64 insn) {
    BFI(*this, insn, GetReg20(insn), GetReg39(insn));
}

void TranslatorVisitor::BFI_rc(u64 insn) {
    BFI(*this, insn, GetReg39(insn), GetCbuf(insn));
}

void TranslatorVisitor::BFI_cr(u64 insn) {
    BFI(*this, insn, GetCbuf(insn), GetReg39(insn));
}

void TranslatorVisitor::BFI_imm(u64 insn) {
    BFI(*this, insn, GetImm20(insn), GetReg39(insn));
}

} // namespace Shader::Maxwell

