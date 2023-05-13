namespace Shader::IR {

// Maps a block to its index for printing.
static std::string BlockToIndex(const std::map<const Block*, size_t>& block_to_index,
                                Block* block) {
    if (const auto block_index{block_to_index.find(block)}; block_index != block_to_index.end()) {
        return fmt::format("{{Block ${}}}", block_index->second);
    }
    return fmt::format("$<unknown block {:016x}>", reinterpret_cast<u64>(block));
}

// Maps an instruction to its index for printing.
static size_t InstIndex(std::map<const Inst*, size_t>& inst_to_index, size_t& index,
                        const Inst* inst) {
    const auto [it, is_inserted]{inst_to_index.emplace(inst, index + 1)};
    if (is_inserted) {
        ++index;
    }
    return it->second;
}

// Maps an argument to its index for printing.
static std::string ArgToIndex(std::map<const Inst*, size_t>& inst_to_index, size_t& index,
                              const Value& arg) {
    if (arg.IsEmpty()) {
        return "<null>";
    }
    if (!arg.IsImmediate() || arg.IsIdentity()) {
        return fmt::format("%{}", InstIndex(inst_to_index, index, arg.Inst()));
    }
    switch (arg.Type()) {
    case Type::U1:
        return fmt::format("#{}", arg.U1() ? "true" : "false");
    case Type::U8:
        return fmt::format("#{}", arg.U8());
    case Type::U16:
        return fmt::format("#{}", arg.U16());
    case Type::U32:
        return fmt::format("#{}", arg.U32());
    case Type::U64:
        return fmt::format("#{}", arg.U64());
    case Type::F32:
        return fmt::format("#{}", arg.F32());
    case Type::Reg:
        return fmt::format("{}", arg.Reg());
    case Type::Pred:
        return fmt::format("{}", arg.Pred());
    case Type::Attribute:
        return fmt::format("{}", arg.Attribute());
    default:
        return "<unknown immediate type>";
    }
}

Block::Block(ObjectPool<Inst>& inst_pool) : inst_pool{&inst_pool} {}

Block::~Block() = default;

void Block::AppendNewInst(Opcode op, std::initializer_list<Value> args) {
    PrependNewInst(end(), op, args);
}

// Adds a new instruction to the beginning of the block, before the given iterator.
Block::iterator Block::PrependNewInst(iterator insertion_point, const Inst& base_inst) {
    Inst* inst{inst_pool->Create(base_inst)};
    return instructions.insert(insertion_point, *inst);
}

// Adds a new instruction with the given opcode and arguments to the beginning of the block,
// before the given iterator.
Block::iterator Block::PrependNewInst(iterator insertion_point, Opcode op,
                                      std::initializer_list<Value> args, u32 flags) {
    Inst* inst{inst_pool->Create(op, flags)};
    const auto result_it{instructions.insert(insertion_point, *inst)};

    const size_t arg_count{inst->NumArgs()};
    if (arg_count != args.size()) {
        throw InvalidArgument("Invalid number of arguments {} in {}", args.size(), op);
    }

    // Set the arguments of the new instruction based on the initializer list.
    size_t index{0};
    for (const auto& arg : args) {
        inst->SetArg(index, arg);
        ++index;
    }
    return result_it;
}

// Adds a successor block to this block.
void Block::AddBranch(Block* block) {
    if (std::ranges::find(imm_successors, block) != imm_successors.end()) {
        throw LogicError("Successor already inserted");
    }
    if (std::ranges::find(block->imm_predecessors, this) != block->imm_predecessors.end()) {
        throw LogicError("Predecessor already inserted");
    }
    imm_successors.push_back(block);
    block->imm_predecessors.push_back(this);
}

// Dumps the contents of the block and its instructions as a string.
std::string DumpBlock(const Block& block) {
    size_t inst_index{0};
    std::map<const Inst*, size_t> inst_to_index;
    return DumpBlock(block, {}, inst_to_index, inst_index);
}

// Dumps the contents of the block and its instructions as a string, using the given block and
// instruction indices.
std::string DumpBlock(const Block& block, const std::map<const Block*, size_t>& block_to_index,
                      std::map<const Inst*, size_t>& inst_to_index, size_t& inst_index) {
    std::string ret{"Block"};

    // Add the block index if it is defined.
    if (const auto block_index{block_to_index.find(&block)}; block_index != block_to_index.end()) {
        ret += fmt::format(" ${}", block_index->second);
    }

    ret += '\n';
    // Loop through each instruction in the block.
    for (const Inst& inst : block) {
        const Opcode op{inst.GetOpcode()};
        ret += fmt::format("[{:016x}] ", reinterpret_cast<u64>(&inst));
        if (TypeOf(op) != Type::Void) {
            ret += fmt::format("%{:<5} = {}", InstIndex(inst_to_index, inst_index, &inst), op);
        } else {
            // Only print the opcode if the instruction is void.
            ret += fmt::format("         {}", op); // '%00000 = ' -> 1 + 5 + 3 = 9 spaces
        }

        // Loop through each argument in the instruction.
        const size_t arg_count{inst.NumArgs()};
        for (size_t arg_index{0}; arg_index < arg_count; ++arg_index) {
            const Value arg{inst.Arg(arg_index)};
            const std::string arg_str{ArgToIndex(inst_to_index, inst_index, arg)};
            ret += arg_index != 0 ? ", " : " ";

            // If the opcode is Phi, print the argument and block indices.
            if (op == Opcode::Phi) {
                ret += fmt::format("[ {}, {} ]", arg_str,
                                   BlockToIndex(block_to_index, inst.PhiBlock(arg_index)));
            } else {
                ret += arg_str;
            }

            // Check if the expected and actual types of the argument are compatible.
            if (op != Opcode::Phi) {
                const Type actual_type{arg.Type()};
                const Type expected_type{ArgTypeOf(op, arg_index)};
                if (!AreTypesCompatible(actual_type, expected_type)) {
                    ret += fmt::format("<type error: {} != {}>", actual_type, expected_type);
                }
            }
        }

        // If the instruction is not void, print the number of uses.
        if (TypeOf(op) != Type::Void) {
            ret += fmt::format(" (uses: {})\n", inst.UseCount());
        } else {
            ret += '\n';
        }
    }

    return ret;
}

} // namespace Shader::IR