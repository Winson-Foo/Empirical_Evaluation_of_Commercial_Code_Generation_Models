// Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace ShaderBackend {
namespace GLSL {

std::string_view GetOutputVertexIndex(EmitContext& context) {
    return context.stage == Stage::TessellationControl ? "[gl_InvocationID]" : "";
}

void InitializeOutputVaryings(EmitContext& context) {
    if (context.uses_geometry_passthrough) {
        return;
    }
    if (context.stage == Stage::VertexB || context.stage == Stage::Geometry) {
        context.Add("gl_Position=vec4(0,0,0,1);");
    }
    for (size_t index = 0; index < IR::NUM_GENERICS; ++index) {
        if (!context.info.stores.Generic(index)) {
            continue;
        }
        const auto& info_array{context.output_generics.at(index)};
        const auto output_decorator{GetOutputVertexIndex(context)};
        size_t element{};
        while (element < info_array.size()) {
            const auto& info{info_array.at(element)};
            const auto varying_name{fmt::format("{}{}", info.name, output_decorator)};
            switch (info.num_components) {
            case 1: {
                const char value{element == 3 ? '1' : '0'};
                context.Add("{}={}.f;", varying_name, value);
                break;
            }
            case 2:
            case 3:
                if (element + info.num_components < 4) {
                    context.Add("{}=vec{}(0);", varying_name, info.num_components);
                } else {
                    // Last element is the w component, must be initialized to 1
                    const auto zeros{
                        info.num_components == 3 ? "0,0," : "0,"
                    };
                    context.Add("{}=vec{}({}1);", varying_name, info.num_components, zeros);
                }
                break;
            case 4:
                context.Add("{}=vec4(0,0,0,1);", varying_name);
                break;
            default:
                break;
            }
            element += info.num_components;
        }
    }
}

void EmitPhi(EmitContext& context, IR::Inst& phi) {
    const size_t num_args{phi.NumArgs()};
    for (size_t i = 0; i < num_args; ++i) {
        context.var_alloc.Consume(phi.Arg(i));
    }
    if (!phi.Definition<Id>().is_valid) {
        // The phi node wasn't forward defined
        context.var_alloc.PhiDefine(phi, phi.Type());
    }
}

void EmitVoid(EmitContext&) {}

void EmitReference(EmitContext& context, const IR::Value& value) {
    context.var_alloc.Consume(value);
}

void MovePhiValue(EmitContext& context, const IR::Value& phi_value, 
                  const IR::Value& value, const IR::Type& phi_type) {
    IR::Inst& phi{*phi_value.InstRecursive()};
    if (!phi.Definition<Id>().is_valid) {
        // The phi node wasn't forward defined
        context.var_alloc.PhiDefine(phi, phi_type);
    }
    const auto phi_reg{context.var_alloc.Consume(IR::Value{&phi})};
    const auto val_reg{context.var_alloc.Consume(value)};
    if (phi_reg == val_reg) {
        return;
    }
    const bool needs_workaround{
        context.profile.has_gl_bool_ref_bug && phi_type == IR::Type::U1
    };
    const auto suffix{needs_workaround ? "?true:false" : ""};
    context.Add("{}={}{};", phi_reg, val_reg, suffix);
}

void EmitPrologue(EmitContext& context) {
    InitializeOutputVaryings(context);
}

void EmitEpilogue(EmitContext&) {}

void EmitEmitVertex(EmitContext& context, const IR::Value& stream) {
    context.Add("EmitStreamVertex(int({}));", context.var_alloc.Consume(stream));
    InitializeOutputVaryings(context);
}

void EmitEndPrimitive(EmitContext& context, const IR::Value& stream) {
    context.Add("EndStreamPrimitive(int({}));", context.var_alloc.Consume(stream));
}

} // namespace GLSL
} // namespace ShaderBackend

