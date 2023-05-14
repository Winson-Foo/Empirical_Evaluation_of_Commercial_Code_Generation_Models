#include "shader_recompiler/backend/glasm/emit_glasm_instructions.h"
#include "shader_recompiler/backend/glasm/glasm_emit_context.h"
#include "shader_recompiler/frontend/ir/value.h"

namespace Shader::Backend::GLASM {

enum class NodeType {
  Bool,
  Signed32,
  Unsigned32,
  Float32,
  Signed64,
  Unsigned64,
  Float64
};

void define_phi_node(EmitContext& ctx, IR::Inst& phi) {
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

void consume_phi_args(EmitContext& ctx, IR::Inst& phi) {
  const size_t num_args = phi.NumArgs();
  for (size_t i = 0; i < num_args; ++i) {
    ctx.reg_alloc.Consume(phi.Arg(i));
  }
}

void emit_phi_node(EmitContext& ctx, IR::Inst& phi) {
  consume_phi_args(ctx, phi);
  if (!phi.Definition<Id>().is_valid) {
    // The phi node wasn't forward defined
    define_phi_node(ctx, phi);
  }
}

void emit_void(EmitContext&) {}

void consume_value(EmitContext& ctx, const IR::Value& value) {
  ctx.reg_alloc.Consume(value);
}

void check_phi_definition(EmitContext& ctx, const IR::Value& phi_value) {
  IR::Inst& phi = RegAlloc::AliasInst(*phi_value.Inst());
  if (!phi.Definition<Id>().is_valid) {
    define_phi_node(ctx, phi);
  }
}

void emit_phi_move(EmitContext& ctx, const IR::Value& phi_value, const IR::Value& value) {
  check_phi_definition(ctx, phi_value);
  const Register phi_reg = ctx.reg_alloc.Consume(IR::Value{&RegAlloc::AliasInst(*phi_value.Inst())});
  const Value eval_value = ctx.reg_alloc.Consume(value);

  if (phi_reg == eval_value) {
    return;
  }

  NodeType node_type;
  switch (phi.Flags<IR::Type>()) {
    case IR::Type::U1:
      node_type = NodeType::Bool;
      break;
    case IR::Type::U32:
      node_type = NodeType::Unsigned32;
      break;
    case IR::Type::F32:
      node_type = NodeType::Float32;
      break;
    case IR::Type::U64:
      node_type = NodeType::Unsigned64;
      break;
    case IR::Type::F64:
      node_type = NodeType::Float64;
      break;
    default:
      throw NotImplementedException("Phi node type {}", phi.Type());
  }

  switch (node_type) {
    case NodeType::Bool:
    case NodeType::Unsigned32:
    case NodeType::Float32:
      ctx.Add("MOV.S {}.x,{};", phi_reg, ScalarS32{eval_value});
      break;
    case NodeType::Unsigned64:
    case NodeType::Float64:
      ctx.Add("MOV.U64 {}.x,{};", phi_reg, ScalarRegister{eval_value});
      break;
  }
}

void emit_prologue(EmitContext&) {
  // TODO
}

void emit_epilogue(EmitContext&) {
  // TODO
}

void emit_emit_vertex(EmitContext& ctx, ScalarS32 stream) {
  if (stream.type == Type::U32 && stream.imm_u32 == 0) {
    ctx.Add("EMIT;");
  } else {
    ctx.Add("EMITS {};", stream);
  }
}

void emit_end_primitive(EmitContext& ctx, const IR::Value& stream) {
  if (!stream.IsImmediate()) {
    throw std::runtime_error("Stream is not immediate");
  }
  ctx.reg_alloc.Consume(stream);
  ctx.Add("ENDPRIM;");
}

} // namespace Shader::Backend::GLASM