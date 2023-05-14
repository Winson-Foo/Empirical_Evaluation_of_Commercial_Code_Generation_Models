namespace Shader::Backend::GLSL {

namespace {

// Initializes output varyings for the current stage.
void InitializeOutputVaryings(EmitContext& ctx) {
  if (ctx.uses_geometry_passthrough) {
    return;
  }

  if (ctx.stage == Stage::VertexB || ctx.stage == Stage::Geometry) {
    // Set output position to the origin.
    ctx.Add("gl_Position = vec4(0, 0, 0, 1);");
  }

  for (size_t generic_index = 0; generic_index < IR::NUM_GENERICS; ++generic_index) {
    if (!ctx.info.stores.Generic(generic_index)) {
      continue;
    }

    const auto& info_array = ctx.output_generics.at(generic_index);

    // Add the output decorator only for the tessellation control shader stage.
    const auto output_decorator = ctx.stage == Stage::TessellationControl ? "[gl_InvocationID]" : "";

    size_t element_index = 0;
    while (element_index < info_array.size()) {
      const auto& info = info_array.at(element_index);
      const auto varying_name = fmt::format("{}{}", info.name, output_decorator);

      switch (info.num_components) {
        case 1: {
          // Set scalar output to 0, except the last element which is set to 1.
          const char value = element_index == 3 ? '1' : '0';
          ctx.Add("{} = {}.f;", varying_name, value);
          break;
        }

        case 2:
        case 3:
          if (element_index + info.num_components < 4) {
            // Set vector output to the origin.
            ctx.Add("{} = vec{}(0);", varying_name, info.num_components);
          } else {
            // Set the last element as 1, as it represents the w component.
            const auto zero_values = info.num_components == 3 ? "0, 0," : "0,";
            ctx.Add("{} = vec{}({}1);", varying_name, info.num_components, zero_values);
          }
          break;

        case 4:
          // Set output to the origin with w component set to 1.
          ctx.Add("{} = vec4(0, 0, 0, 1);", varying_name);
          break;
      }

      element_index += info.num_components;
    }
  }
}

} // Anonymous namespace

void EmitPhi(EmitContext& ctx, IR::Inst& phi) {
  // Consume the phi node's arguments.
  const size_t num_args = phi.NumArgs();
  for (size_t i = 0; i < num_args; ++i) {
    ctx.var_alloc.Consume(phi.Arg(i));
  }

  if (!phi.Definition<Id>().is_valid) {
    // The phi node wasn't forward defined, so define it now.
    ctx.var_alloc.PhiDefine(phi, phi.Type());
  }
}

// Emits code for void value.
void EmitVoid(EmitContext&) {}

void EmitReference(EmitContext& ctx, const IR::Value& value) {
  // Consume and allocate a register for the value.
  ctx.var_alloc.Consume(value);
}

// Emits code for a phi move instruction.
void EmitPhiMove(EmitContext& ctx, const IR::Value& phi_value, const IR::Value& value) {
  IR::Inst& phi = *phi_value.InstRecursive();
  const auto phi_type = phi.Type();

  if (!phi.Definition<Id>().is_valid) {
    // The phi node wasn't forward defined, so define it now.
    ctx.var_alloc.PhiDefine(phi, phi_type);
  }

  const auto phi_reg = ctx.var_alloc.Consume(IR::Value{&phi});
  const auto val_reg = ctx.var_alloc.Consume(value);

  if (phi_reg == val_reg) {
    // The register for the phi node and value are the same, so no further action is needed.
    return;
  }

  const bool needs_workaround = ctx.profile.has_gl_bool_ref_bug && phi_type == IR::Type::U1;
  const auto suffix = needs_workaround ? "?true:false" : "";

  // Move value to phi register.
  ctx.Add("{} = {}{};", phi_reg, val_reg, suffix);
}

// Emits the prologue for a shader.
void EmitPrologue(EmitContext& ctx) {
  InitializeOutputVaryings(ctx);
}

// Emits the epilogue for a shader.
void EmitEpilogue(EmitContext&) {}

// Emits an emit-vertex instruction.
void EmitEmitVertex(EmitContext& ctx, const IR::Value& stream) {
  // Emit a vertex for the stream.
  ctx.Add("EmitStreamVertex(int({}));", ctx.var_alloc.Consume(stream));

  // Reinitialize output varyings.
  InitializeOutputVaryings(ctx);
}

// Emits an end-primitive instruction.
void EmitEndPrimitive(EmitContext& ctx, const IR::Value& stream) {
  // End the primitive for the stream.
  ctx.Add("EndStreamPrimitive(int({}));", ctx.var_alloc.Consume(stream));
}

} // namespace Shader::Backend::GLSL