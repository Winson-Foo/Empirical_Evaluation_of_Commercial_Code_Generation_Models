#include "common/bit_field.h"
#include "common/common_types.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"

namespace Shader::Maxwell {

namespace {

const IR::U32 kMaxSize = v.ir.Imm32(32);
const IR::U32 kZero = v.ir.Imm32(0);

void BFI(TranslatorVisitor& v, u64 insn, const IR::U32& src_a, const IR::U32& base) {
    union {
        u64 insn;
        BitField<0, 8, IR::Reg> dest_reg;
        BitField<8, 8, IR::Reg> insert_reg;
        BitField<47, 1, u64> cc;
    } const bfi { insn };

    const IR::U32 offset = v.ir.BitFieldExtract(src_a, kZero, v.ir.Imm32(8), false);
    const IR::U32 unsafe_count = v.ir.BitFieldExtract(src_a, v.ir.Imm32(8), v.ir.Imm32(8), false);
 
   // Handle edge cases
    const bool exceed_offset = v.ir.IGreaterThanEqual(offset, kMaxSize, false);
    const IR::U32 remaining_size = v.ir.ISub(kMaxSize, offset);

    const IR::U32 insert = v.X(bfi.insert_reg);
    const IR::U32 safe_count = v.ir.IGreaterThan(unsafe_count, kMaxSize, false) ? remaining_size : unsafe_count;
    IR::U32 result = v.ir.BitFieldInsert(base, insert, offset, safe_count);
    result = IR::U32{v.ir.Select(exceed_offset, base, result)};

    v.X(bfi.dest_reg, result);

    if (bfi.cc != 0) {
        v.SetZFlag(v.ir.IEqual(result, kZero));
        v.SetSFlag(v.ir.ILessThan(result, kZero, true));
        v.ResetCFlag();
        v.ResetOFlag();
    }
}
} // Anonymous namespace

void TranslatorVisitor::BFI_reg(u64 insn) {
    const IR::U32 src_a = GetReg20(insn);
    const IR::U32 base = GetReg39(insn);
    BFI(*this, insn, src_a, base);
}

void TranslatorVisitor::BFI_rc(u64 insn) {
    const IR::U32 src_a = GetReg39(insn);
    const IR::U32 base = GetCbuf(insn);
    BFI(*this, insn, src_a, base);
}

void TranslatorVisitor::BFI_cr(u64 insn) {
    const IR::U32 src_a = GetCbuf(insn);
    const IR::U32 base = GetReg39(insn);
    BFI(*this, insn, src_a, base);
}

void TranslatorVisitor::BFI_imm(u64 insn) {
    const IR::U32 src_a = GetImm20(insn);
    const IR::U32 base = GetReg39(insn);
    BFI(*this, insn, src_a, base);
}

} // namespace Shader::Maxwell