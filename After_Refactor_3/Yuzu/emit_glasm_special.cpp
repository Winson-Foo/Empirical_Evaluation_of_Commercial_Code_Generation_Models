#include "shader_recompiler/backend/glasm/emit_glasm_instructions.h"
#include "shader_recompiler/backend/glasm/glasm_emit_context.h"
#include "shader_recompiler/frontend/ir/value.h"

namespace Shader::Backend::GLASM {

static void define_phi_node(EmitContext& context, IR::Inst& phi_node) {
    switch (phi_node.Type()) {
    case IR::Type::U1:
    case IR::Type::U32:
    case IR::Type::F32:
        context.reg_alloc.Define(phi_node);
        break;
    case IR::Type::U64:
    case IR::Type::F64:
        context.reg_alloc.LongDefine(phi_node);
        break;
    default:
        throw NotImplementedException("Phi node type {}", phi_node.Type());
    }
}

void emit_phi_node(EmitContext& context, IR::Inst& phi_node) {
    const size_t num_args{phi_node.NumArgs()};
    for (size_t i = 0; i < num_args; ++i) {
        context.reg_alloc.Consume(phi_node.Arg(i));
    }

    if (!phi_node.Definition<Id>().is_valid) {
        // The phi node wasn't forward defined
        define_phi_node(context, phi_node);
    }
}

void emit_void(EmitContext&) {}

void emit_reference_to_value(EmitContext& context, const IR::Value& value) {
    context.reg_alloc.Consume(value);
}

void emit_phi_move(EmitContext& context, const IR::Value& source_phi_node, const IR::Value& dest_value) {
    IR::Inst& phi_node{RegAlloc::AliasInst(*source_phi_node.Inst())};
    if (!phi_node.Definition<Id>().is_valid) {
        // The phi node wasn't forward defined
        define_phi_node(context, phi_node);
    }

    const Register phi_reg{context.reg_alloc.Consume(IR::Value{&phi_node})};
    const Value eval_value{context.reg_alloc.Consume(dest_value)};

    if (phi_reg == eval_value) {
        return;
    }

    switch (phi_node.Flags<IR::Type>()) {
    case IR::Type::U1:
    case IR::Type::U32:
    case IR::Type::F32:
        context.Add("MOV.S {}.x,{};", phi_reg, ScalarS32{eval_value});
        break;
    case IR::Type::U64:
    case IR::Type::F64:
        context.Add("MOV.U64 {}.x,{};", phi_reg, ScalarRegister{eval_value});
        break;
    default:
        throw NotImplementedException("Phi node type {}", phi_node.Type());
    }
}

void emit_prologue(EmitContext&) {
    // TODO
}

void emit_epilogue(EmitContext&) {
    // TODO
}

void emit_emit_vertex(EmitContext& context, ScalarS32 stream) {
    if (stream.type == Type::U32 && stream.imm_u32 == 0) {
        context.Add("EMIT;");
    } else {
        context.Add("EMITS {};", stream);
    }
}

void emit_end_primitive(EmitContext& context, const IR::Value& stream) {
    if (!stream.IsImmediate()) {
        LOG_WARNING(Shader_GLASM, "Stream is not immediate");
    }
    context.reg_alloc.Consume(stream);
    context.Add("ENDPRIM;");
}

} // namespace Shader::Backend::GLASM