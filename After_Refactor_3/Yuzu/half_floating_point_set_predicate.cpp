// Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/frontend/maxwell/translate/impl/common_funcs.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/half_floating_point_helper.h"

namespace Shader::Maxwell {

namespace {

// Extract source operands and convert them if needed
void ExtractAndConvertOperands(TranslatorVisitor& v, const IR::U32& src_a,
                               const IR::U32& src_b, Swizzle swizzle_a, Swizzle swizzle_b,
                               IR::FpControl& control, IR::FpAbsNeg& abs_neg_a,
                               IR::FpAbsNeg& abs_neg_b, IR::FpDoublePair& operands) {
    auto [lhs_a, rhs_a]{Extract(v.ir, src_a, swizzle_a)};
    auto [lhs_b, rhs_b]{Extract(v.ir, src_b, swizzle_b)};
    const IR::Type lhs_type{lhs_a.Type()};
    const IR::Type rhs_type{rhs_a.Type()};
    if (lhs_type != rhs_type) {
        if (lhs_type == IR::Type::F16) {
            lhs_a = v.ir.FPConvert(32, lhs_a);
            rhs_a = v.ir.FPConvert(32, rhs_a);
        }
        if (rhs_type == IR::Type::F16) {
            lhs_b = v.ir.FPConvert(32, lhs_b);
            rhs_b = v.ir.FPConvert(32, rhs_b);
        }
    }

    abs_neg_a = {lhs_a, rhs_a, false, false};
    abs_neg_b = {lhs_b, rhs_b, false, false};

    operands = {lhs_a, rhs_a, lhs_b, rhs_b};
}

// Combine predicate result with compare result using boolean operation
IR::U1 CombinePredicateAndCompareResult(TranslatorVisitor& v, const IR::U1& predicate_result,
                                        const IR::U1& cmp_result, BooleanOp bop) {
    const IR::U1 combined_predicate_result{PredicateCombine(v.ir, cmp_result, predicate_result, bop)};
    return combined_predicate_result;
}

// Compare two operands and return the comparison result
IR::U1 CompareOperands(TranslatorVisitor& v, const IR::FpDoublePair& operands,
                       const FPCompareOp& compare_op, const IR::FpControl& control) {
    const IR::U1 cmp_result{FloatingPointCompare(v.ir, operands.lhs_a, operands.rhs_a, compare_op, control)};
    return cmp_result;
}

// Apply absolute or negation to source operands
void AbsNegOperands(IR::FpAbsNeg& abs_neg_a, const IR::FpAbsNeg& abs_neg_b) {
    abs_neg_a.ApplyAbsNeg(abs_neg_b.abs, abs_neg_b.neg);
}

// Set the destination predicate according to the comparison results and boolean operations
void SetDestinationPredicate(TranslatorVisitor& v, const IR::U1& combined_predicate_result_a,
                             const IR::U1& combined_predicate_result_b, const IR::Pred& pred_a,
                             const IR::Pred& pred_b) {
    v.ir.SetPred(pred_a, combined_predicate_result_a);
    v.ir.SetPred(pred_b, combined_predicate_result_b);
}

// Apply Float to Zero (FTZ) mode
void ApplyFTZMode(IR::FpControl& control, bool ftz_mode) {
    control.fmz_mode = (ftz_mode ? IR::FmzMode::FTZ : IR::FmzMode::None);
}

// Perform HSETP2 operation on the given source operands and arguments
void PerformHSETP2Operation(TranslatorVisitor& v, u64 insn, const IR::U32& src_b, bool neg_b,
                            bool abs_b, Swizzle swizzle_b, FPCompareOp compare_op, bool h_and) {
    union {
        u64 insn;
        BitField<8, 8, IR::Reg> src_a_reg;
        BitField<3, 3, IR::Pred> dest_pred_a;
        BitField<0, 3, IR::Pred> dest_pred_b;
        BitField<39, 3, IR::Pred> pred;
        BitField<42, 1, u64> neg_pred;
        BitField<43, 1, u64> neg_a;
        BitField<45, 2, BooleanOp> bop;
        BitField<44, 1, u64> abs_a;
        BitField<6, 1, u64> ftz;
        BitField<47, 2, Swizzle> swizzle_a;
    } const hsetp2{insn};

    IR::FpControl control{};
    ApplyFTZMode(control, hsetp2.ftz != 0);

    IR::FpAbsNeg abs_neg_a{}, abs_neg_b{};
    IR::FpDoublePair operands{};
    ExtractAndConvertOperands(v, v.X(hsetp2.src_a_reg), src_b, hsetp2.swizzle_a, swizzle_b,
                              control, abs_neg_a, abs_neg_b, operands);

    AbsNegOperands(abs_neg_a, abs_neg_b);

    IR::U1 cmp_result_lhs{CompareOperands(v, {abs_neg_a.lhs_a, abs_neg_a.rhs_a}, compare_op, control)};
    IR::U1 cmp_result_rhs{CompareOperands(v, {abs_neg_a.lhs_b, abs_neg_a.rhs_b}, compare_op, control)};

    IR::U1 combined_predicate_result_a{
        CombinePredicateAndCompareResult(v, v.ir.GetPred(hsetp2.pred), cmp_result_lhs, hsetp2.bop)};
    IR::U1 combined_predicate_result_b{
        CombinePredicateAndCompareResult(v, v.ir.GetPred(hsetp2.pred), cmp_result_rhs, hsetp2.bop)};

    SetDestinationPredicate(v, combined_predicate_result_a, combined_predicate_result_b,
                            hsetp2.dest_pred_a, hsetp2.dest_pred_b);
    if (h_and) {
        IR::U1 result{v.ir.LogicalAnd(combined_predicate_result_a, combined_predicate_result_b)};
        v.ir.SetPred(hsetp2.dest_pred_a, result);
        v.ir.SetPred(hsetp2.dest_pred_b, v.ir.LogicalNot(result));
    }
}

} // Anonymous namespace

void TranslatorVisitor::HSETP2_reg(u64 insn) {
    union {
        u64 insn;
        BitField<30, 1, u64> abs_b;
        BitField<49, 1, u64> h_and;
        BitField<31, 1, u64> neg_b;
        BitField<35, 4, FPCompareOp> compare_op;
        BitField<28, 2, Swizzle> swizzle_b;
    } const hsetp2{insn};

    PerformHSETP2Operation(*this, insn, GetReg20(insn), hsetp2.neg_b != 0, hsetp2.abs_b != 0,
                           hsetp2.swizzle_b, hsetp2.compare_op, hsetp2.h_and != 0);
}

void TranslatorVisitor::HSETP2_cbuf(u64 insn) {
    union {
        u64 insn;
        BitField<53, 1, u64> h_and;
        BitField<54, 1, u64> abs_b;
        BitField<56, 1, u64> neg_b;
        BitField<49, 4, FPCompareOp> compare_op;
    } const hsetp2{insn};

    PerformHSETP2Operation(*this, insn, GetCbuf(insn), hsetp2.neg_b != 0, hsetp2.abs_b != 0, Swizzle::F32,
                           hsetp2.compare_op, hsetp2.h_and != 0);
}

void TranslatorVisitor::HSETP2_imm(u64 insn) {
    union {
        u64 insn;
        BitField<53, 1, u64> h_and;
        BitField<54, 1, u64> ftz;
        BitField<49, 4, FPCompareOp> compare_op;
        BitField<56, 1, u64> neg_high;
        BitField<30, 9, u64> high;
        BitField<29, 1, u64> neg_low;
        BitField<20, 9, u64> low;
    } const hsetp2{insn};

    const u32 imm{static_cast<u32>(hsetp2.low << 6) |
                  static_cast<u32>((hsetp2.neg_low != 0 ? 1 : 0) << 15) |
                  static_cast<u32>(hsetp2.high << 22) |
                  static_cast<u32>((hsetp2.neg_high != 0 ? 1 : 0) << 31)};
    PerformHSETP2Operation(*this, insn, ir.Imm32(imm), false, false, Swizzle::H1_H0, hsetp2.compare_op,
                           hsetp2.h_and != 0);
}

} // namespace Shader::Maxwell