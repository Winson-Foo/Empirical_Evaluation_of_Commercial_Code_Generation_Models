#include "shader_recompiler/frontend/maxwell/translate/impl/common_funcs.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/half_floating_point_helper.h"

namespace Shader::Maxwell {

using namespace std::bitfield_ops;

namespace {

struct HSetP2Bits {
    u64 insn;
    std::bitfield< 8, 8, IR::Reg> src_a_reg;
    std::bitfield< 3, 3, IR::Pred> dest_pred_a;
    std::bitfield< 0, 3, IR::Pred> dest_pred_b;
    std::bitfield<39, 3, IR::Pred> pred;
    std::bitfield<42, 1, u64> neg_pred;
    std::bitfield<43, 1, u64> neg_a;
    std::bitfield<45, 2, BooleanOp> bop;
    std::bitfield<44, 1, u64> abs_a;
    std::bitfield< 6, 1, u64> ftz;
    std::bitfield<47, 2, Swizzle> swizzle_a;
};

void ConvertOperandsIfNeeded(IR::InstBuilder& ir, IR::Value& lhs, IR::Value& rhs) {
    if (lhs.Type() == rhs.Type()) {
        return;
    }

    if (lhs.Type() == IR::Type::F16) {
        lhs = ir.FPConvert(32, lhs);
        rhs = ir.FPConvert(32, rhs);
    }
    else if (lhs.Type() == IR::Type::F32 && rhs.Type() == IR::Type::F16) {
        rhs = ir.FPConvert(32, rhs);
    }
    else if (lhs.Type() == IR::Type::F16 && rhs.Type() == IR::Type::F32) {
        lhs = ir.FPConvert(32, lhs);
    }
}

void HSETP2(TranslatorVisitor& visitor, u64 insn, const IR::U32& src_b, bool neg_b,
            bool abs_b, Swizzle swizzle_b, FPCompareOp compare_op, bool h_and) {
    IR::InstBuilder& ir(visitor.ir);

    HSetP2Bits bits{insn};

    IR::Value lhs_a, rhs_a;
    std::tie(lhs_a, rhs_a) = Extract(ir, visitor.X(bits.src_a_reg), bits.swizzle_a);

    IR::Value lhs_b, rhs_b;
    std::tie(lhs_b, rhs_b) = Extract(ir, src_b, swizzle_b);

    ConvertOperandsIfNeeded(ir, lhs_a, lhs_b);
    ConvertOperandsIfNeeded(ir, rhs_a, rhs_b);

    lhs_a = ir.FPAbsNeg(lhs_a, bits.abs_a != 0, bits.neg_a != 0);
    rhs_a = ir.FPAbsNeg(rhs_a, bits.abs_a != 0, bits.neg_a != 0);

    lhs_b = ir.FPAbsNeg(lhs_b, abs_b, neg_b);
    rhs_b = ir.FPAbsNeg(rhs_b, abs_b, neg_b);

    const IR::FpControl control{
        .no_contraction = false,
        .rounding = IR::FpRounding::DontCare,
        .fmz_mode = (bits.ftz != 0 ? IR::FmzMode::FTZ : IR::FmzMode::None),
    };

    IR::U1 pred{ir.GetPred(bits.pred)};
    if (bits.neg_pred != 0) {
        pred = ir.LogicalNot(pred);
    }
    const IR::U1 cmp_result_lhs{FloatingPointCompare(ir, lhs_a, lhs_b, compare_op, control)};
    const IR::U1 cmp_result_rhs{FloatingPointCompare(ir, rhs_a, rhs_b, compare_op, control)};
    const IR::U1 bop_result_lhs{PredicateCombine(ir, cmp_result_lhs, pred, bits.bop)};
    const IR::U1 bop_result_rhs{PredicateCombine(ir, cmp_result_rhs, pred, bits.bop)};

    IR::U1 result;
    if (h_and) {
        result = ir.LogicalAnd(bop_result_lhs, bop_result_rhs);
    } else {
        result = bop_result_lhs;
        ir.SetPred(bits.dest_pred_b, bop_result_rhs);
    }
    ir.SetPred(bits.dest_pred_a, result);
}

} // namespace

void TranslatorVisitor::HSETP2_reg(u64 insn) {
    HSetP2Bits bits{insn};

    const IR::U32 src_b{GetReg20(insn)};

    HSETP2(*this, insn, src_b, bits.neg_b != 0, bits.abs_b != 0, bits.swizzle_b,
           bits.compare_op, bits.h_and != 0);
}

void TranslatorVisitor::HSETP2_cbuf(u64 insn) {
    HSetP2Bits bits{insn};

    const IR::U32 src_b{GetCbuf(insn)};

    HSETP2(*this, insn, src_b, bits.neg_b != 0, bits.abs_b != 0, Swizzle::F32,
           bits.compare_op, bits.h_and != 0);
}

void TranslatorVisitor::HSETP2_imm(u64 insn) {
    HSetP2Bits bits{insn};

    const u32 imm{static_cast<u32>(bits.low << 6) |
                  static_cast<u32>((bits.neg_low != 0 ? 1 : 0) << 15) |
                  static_cast<u32>(bits.high << 22) |
                  static_cast<u32>((bits.neg_high != 0 ? 1 : 0) << 31)};
    const IR::U32 src_b{ir.Imm32(imm)};

    HSETP2(*this, insn, src_b, false, false, Swizzle::H1_H0, bits.compare_op, bits.h_and != 0);
}

} // namespace Shader::Maxwell