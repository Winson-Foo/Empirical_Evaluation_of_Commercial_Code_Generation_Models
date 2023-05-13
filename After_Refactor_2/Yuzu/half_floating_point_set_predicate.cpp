// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright 2021 yuzu Emulator Project

#include "shader_recompiler/frontend/maxwell/translate/impl/common_funcs.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/half_floating_point_helper.h"

namespace Shader::Maxwell {

namespace {

// Extracts values based on the provided swizzle index
void extract_values(TranslatorVisitor& visitor, IR::U32& value_a, IR::U32& value_b,
                    const IR::U32& register_value, const Swizzle swizzle_a,
                    const Swizzle swizzle_b) {
    std::tie(value_a, value_b) = Extract(visitor.ir, register_value, swizzle_a, swizzle_b);
}

// Applies absolute value and negation based on the provided flags
IR::U32 apply_abs_neg(TranslatorVisitor& visitor, IR::U32& value, const bool use_abs,
                       const bool use_neg) {
    return visitor.ir.FPAbsNeg(value, use_abs, use_neg);
}

// Combines the two predicate results based on the provided boolean operation flag
void set_predicate_result(TranslatorVisitor& visitor, const IR::U1& predicate_result_lhs,
                          const IR::U1& predicate_result_rhs, const IR::Pred dest_pred_a,
                          const IR::Pred dest_pred_b, const BooleanOp boolean_op) {
    if (boolean_op == BooleanOp::And) {
        auto result = visitor.ir.LogicalAnd(predicate_result_lhs, predicate_result_rhs);
        visitor.ir.SetPred(dest_pred_a, result);
        visitor.ir.SetPred(dest_pred_b, visitor.ir.LogicalNot(result));
    } else {
        visitor.ir.SetPred(dest_pred_a, predicate_result_lhs);
        visitor.ir.SetPred(dest_pred_b, predicate_result_rhs);
    }
}

}

void TranslatorVisitor::HSETP2_reg(const u64 insn) {
    const auto hsetp2_reg =
        bit_cast<HSETP2Reg>(insn).get();  // struct that contains fields from the instruction
    IR::U32 src_b_value{};
    extract_values(*this, hsetp2_reg.lhs_b_value, src_b_value, GetReg20(insn),
                   hsetp2_reg.swizzle_a, hsetp2_reg.swizzle_b);
    if (hsetp2_reg.lhs_a_value.Type() != src_b_value.Type()) {
        if (hsetp2_reg.lhs_a_value.Type() == IR::Type::F16) {
            hsetp2_reg.lhs_a_value = ir.FPConvert(32, hsetp2_reg.lhs_a_value);
            hsetp2_reg.rhs_a_value = ir.FPConvert(32, hsetp2_reg.rhs_a_value);
        }
        if (src_b_value.Type() == IR::Type::F16) {
            src_b_value = ir.FPConvert(32, src_b_value);
        }
    }
    hsetp2_reg.lhs_a_value = apply_abs_neg(*this, hsetp2_reg.lhs_a_value, hsetp2_reg.use_abs_a,
                                           hsetp2_reg.use_neg_a);
    hsetp2_reg.rhs_a_value = apply_abs_neg(*this, hsetp2_reg.rhs_a_value, hsetp2_reg.use_abs_a,
                                           hsetp2_reg.use_neg_a);
    src_b_value = apply_abs_neg(*this, src_b_value, hsetp2_reg.use_abs_b, hsetp2_reg.use_neg_b);
    const IR::FpControl control{
        .no_contraction = false,
        .rounding = IR::FpRounding::DontCare,
        .fmz_mode = (hsetp2_reg.fma_zero_mode ? IR::FmzMode::FTZ : IR::FmzMode::None),
    };
    const IR::U1 predicate_result_lhs = FloatingPointCompare(
        ir, hsetp2_reg.lhs_a_value, src_b_value, hsetp2_reg.compare_op, control);
    const IR::U1 predicate_result_rhs = FloatingPointCompare(
        ir, hsetp2_reg.rhs_a_value, src_b_value, hsetp2_reg.compare_op, control);
    set_predicate_result(*this, predicate_result_lhs, predicate_result_rhs, hsetp2_reg.dest_pred_a,
                          hsetp2_reg.dest_pred_b, hsetp2_reg.boolean_op);
}

void TranslatorVisitor::HSETP2_cbuf(const u64 insn) {
    const auto hsetp2_cbuf = bit_cast<HSETP2CBuf>(insn).get();
    const IR::U32 src_b_value = GetCbuf(insn);
    HSETP2_reg(CreateHSETP2Reg(hsetp2_cbuf.lhs_a_value, hsetp2_cbuf.rhs_a_value, false, false,
                               hsetp2_cbuf.use_abs_a, hsetp2_cbuf.use_neg_a, Swizzle::F32,
                               hsetp2_cbuf.use_abs_b, hsetp2_cbuf.use_neg_b,
                               hsetp2_cbuf.swizzle_b, hsetp2_cbuf.compare_op,
                               hsetp2_cbuf.boolean_op, hsetp2_cbuf.fma_zero_mode,
                               hsetp2_cbuf.dest_pred_a, hsetp2_cbuf.dest_pred_b,
                               hsetp2_cbuf.h_and_mode));
}

void TranslatorVisitor::HSETP2_imm(const u64 insn) {
    const auto hsetp2_imm = bit_cast<HSETP2Imm>(insn).get();
    const std::uint32_t imm = (hsetp2_imm.high << 22) | (hsetp2_imm.low << 6) |
                              (hsetp2_imm.neg_low << 15) | (hsetp2_imm.neg_high << 31);
    const IR::U32 src_b_value = ir.Imm32(imm, IR::Type::U32);
    HSETP2_reg(CreateHSETP2Reg(hsetp2_imm.lhs_a_value, hsetp2_imm.rhs_a_value, false, false, false,
                               false, Swizzle::H1_H0, false, false, Swizzle::F32,
                               hsetp2_imm.compare_op, hsetp2_imm.boolean_op,
                               hsetp2_imm.fma_zero_mode, hsetp2_imm.dest_pred_a,
                               hsetp2_imm.dest_pred_b, hsetp2_imm.h_and_mode));
}

}  // namespace Shader::Maxwell

