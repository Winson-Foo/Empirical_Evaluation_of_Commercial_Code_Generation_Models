#include "shader_recompiler/backend/glasm/emit_glasm_instructions.h"
#include "shader_recompiler/backend/glasm/glasm_emit_context.h"
#include "shader_recompiler/frontend/ir/value.h"

#include <unordered_map>

namespace Shader::Backend::GLASM {

namespace {

void ForwardDefinePhiIfNeeded(EmitContext& ctx, IR::Inst& phi) {
  if (!phi.Definition<Id>().is_valid) {
    // The phi node wasn't forward defined
    switch (phi.Type()) {
    case IR::Type::U1:
    case IR::Type::U32:
    case IR::Type::F32:
      ctx.reg_alloc.Define(phi);
      break;
    case IR::Type::U64:
    case IR::Type::F64:
      ctx.reg_alloc.LongDefine(phi);
      break;
    default:
      throw NotImplementedException("Phi node type {}", phi.Type());
    }
  }
}

void MoveValueToPhiReg(EmitContext& ctx, IR::Inst& phi, const IR::Value& value) {
  const Register phi_reg = ctx.reg_alloc.Consume(IR::Value{&phi});
  const Value eval_value = ctx.reg_alloc.Consume(value);

  if (phi_reg == eval_value) {
    return;
  }

  switch (phi.Flags<IR::Type>()) {
  case IR::Type::U1:
  case IR::Type::U32:
  case IR::Type::F32:
    ctx.Add("MOV.S {}.x,{};", phi_reg, ScalarS32{eval_value});
    break;
  case IR::Type::U64:
  case IR::Type::F64:
    ctx.Add("MOV.U64 {}.x,{};", phi_reg, ScalarRegister{eval_value});
    break;
  default:
    throw NotImplementedException("Phi node type {}", phi.Type());
  }
}

std::string Format_EMIT(const IR::Value& stream) {
  if (stream.IsImmediate()) {
    const ScalarS32 stream_val = ScalarS32{stream};
    return stream_val.type == Type::U32 && stream_val.imm_u32 == 0 ? "EMIT;" : fmt::format("EMITS {};", stream_val);
  }

  // The stream value is not immediate - issue a warning and ignore it.
  LOG_WARNING(Shader_GLASM, "Stream is not immediate");
  return "EMIT;";
}

std::string Format_ENDPRIM(const IR::Value& stream) {
  if (!stream.IsImmediate()) {
    LOG_WARNING(Shader_GLASM, "Stream is not immediate");
  }

  return "ENDPRIM;";
}

}  // namespace

void EmitPhi(EmitContext& ctx, IR::Inst& phi) {
  const size_t num_args = phi.NumArgs();

  for (size_t i = 0; i < num_args; ++i) {
    ctx.reg_alloc.Consume(phi.Arg(i));
  }

  ForwardDefinePhiIfNeeded(ctx, phi);
}

void EmitVoid(EmitContext&) {}

void EmitReference(EmitContext& ctx, const IR::Value& value) {
  ctx.reg_alloc.Consume(value);
}

void EmitPhiMove(EmitContext& ctx, const IR::Value& phi_value, const IR::Value& value) {
  IR::Inst& phi = RegAlloc::AliasInst(*phi_value.Inst());

  ForwardDefinePhiIfNeeded(ctx, phi);
  MoveValueToPhiReg(ctx, phi, value);
}

void EmitPrologue(EmitContext&) {}

void EmitEpilogue(EmitContext&) {}

void EmitEmitVertex(EmitContext& ctx, ScalarS32 stream) {
  ctx.Add(Format_EMIT(IR::Value{stream}));
}

void EmitEndPrimitive(EmitContext& ctx, const IR::Value& stream) {
  ctx.reg_alloc.Consume(stream);
  ctx.Add(Format_ENDPRIM(stream));
}

}  // namespace Shader::Backend::GLASM