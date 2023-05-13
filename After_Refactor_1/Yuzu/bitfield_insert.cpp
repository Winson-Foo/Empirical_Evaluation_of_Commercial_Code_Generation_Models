namespace Shader::Maxwell {

// BFI instruction helpers
namespace BFI {
    // Constants
    constexpr u32 kMaxSize = 32;

    // Extracts operands and performs Bit Field Insert
    void Execute(TranslatorVisitor& v, u64 insn, const IR::U32& src_a, const IR::U32& base) {
        // Extract instruction fields using BitField utility class
        const auto [dest_reg, insert_reg, cc] = BitField<0, 8, IR::Reg, 8, IR::Reg, 47, u64>::FromValue(insn);

        // Extract offset and count operands
        const auto zero = v.ir.Imm32(0);
        const auto offset = v.ir.BitFieldExtract(src_a, zero, v.ir.Imm32(8), false);
        const auto unsafe_count = v.ir.BitFieldExtract(src_a, v.ir.Imm32(8), v.ir.Imm32(8), false);

        // Calculate safe count and handle edge cases
        const auto remaining_size = v.ir.ISub(kMaxSize, offset);
        const auto safe_count = v.ir.Select(
            v.ir.IGreaterThan(kMaxSize, unsafe_count, false),
            unsafe_count,
            remaining_size);

        // Perform Bit Field Insert and handle edge case where offset exceeds max size
        auto result = v.ir.BitFieldInsert(base, v.X(insert_reg), offset, safe_count);
        result = v.ir.Select(v.ir.IGreaterThanEqual(offset, kMaxSize, false), base, result);

        v.X(dest_reg, result);
        if (cc) {
            // Set flags if cc bit is set
            const auto zero = v.ir.Imm32(0);
            v.SetZFlag(v.ir.IEqual(result, zero));
            v.SetSFlag(v.ir.ILessThan(result, zero, true));
            v.ResetCFlag();
            v.ResetOFlag();
        }
    }
}

// BFI instructions in various operand combinations
void TranslatorVisitor::BFI_reg(u64 insn) {
    BFI::Execute(*this, insn, GetReg20(insn), GetReg39(insn));
}

void TranslatorVisitor::BFI_rc(u64 insn) {
    BFI::Execute(*this, insn, GetReg39(insn), GetCbuf(insn));
}

void TranslatorVisitor::BFI_cr(u64 insn) {
    BFI::Execute(*this, insn, GetCbuf(insn), GetReg39(insn));
}

void TranslatorVisitor::BFI_imm(u64 insn) {
    BFI::Execute(*this, insn, GetImm20(insn), GetReg39(insn));
}

} // namespace Shader::Maxwell