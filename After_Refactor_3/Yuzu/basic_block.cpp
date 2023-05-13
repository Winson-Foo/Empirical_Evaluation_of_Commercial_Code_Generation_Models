#include <algorithm>
#include <initializer_list>
#include <map>

#include "common/common_types.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/value.h"

namespace Shader::IR {

Block::Block(ObjectPool<Inst>& inst_pool) : inst_pool(&inst_pool) {}

Block::~Block() = default;

void Block::AppendNewInst(Opcode op, const std::initializer_list<const Value>& args) {
    PrependNewInst(end(), op, args, 0);
}

auto Block::PrependNewInst(iterator insertion_point, const Inst& base_inst) -> iterator {
    const auto inst = inst_pool->Create(base_inst);
    return instructions.insert(insertion_point, *inst);
}

auto Block::PrependNewInst(iterator insertion_point, const Opcode op,
                           const std::initializer_list<const Value>& args, const u32 flags) -> iterator {
    const auto inst = inst_pool->Create(op, flags);
    const auto result_it = instructions.insert(insertion_point, *inst);

    if (inst->NumArgs() != args.size()) {
        throw InvalidArgument("Invalid number of arguments {} in {}", args.size(), op);
    }
    std::ranges::for_each(args, [inst, index=0u](const auto& arg) mutable {
        inst->SetArg(index++, arg);
    });
    return result_it;
}

void Block::AddBranch(Block* block) {
    const auto already_inserted = [](const auto& container, auto element) {
        return std::ranges::find(container, element) != container.end();
    };
    if (already_inserted(imm_successors, block)) {
        throw LogicError("Successor already inserted");
    }
    if (already_inserted(block->imm_predecessors, this)) {
        throw LogicError("Predecessor already inserted");
    }
    imm_successors.push_back(block);
    block->imm_predecessors.push_back(this);
}

static auto BlockToIndex(const std::map<const Block*, size_t>& block_to_index, const Block* block) -> std::string {
    const auto it = block_to_index.find(block);
    if (it != block_to_index.end()) {
        return fmt::format("{{Block ${}}}", it->second);
    }
    return fmt::format("$<unknown block {:016x}>", reinterpret_cast<u64>(block));
}

static auto InstIndex(std::map<const Inst*, size_t>& inst_to_index, size_t& inst_index,
                      const Inst* inst) -> size_t {
    return inst_to_index.emplace(inst, ++inst_index).first->second;
}

static auto ArgToIndex(std::map<const Inst*, size_t>& inst_to_index, size_t& inst_index,
                       const Value& arg) -> std::string {
    if (arg.IsEmpty()) {
        return "<null>";
    }
    if (!arg.IsImmediate() || arg.IsIdentity()) {
        return fmt::format("%{}", InstIndex(inst_to_index, inst_index, arg.Inst()));
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

auto DumpBlock(const Block& block) -> std::string {
    std::map<const Inst*, size_t> inst_to_index;
    size_t inst_index = 0;
    return DumpBlock(block, {}, inst_to_index, inst_index);
}

auto DumpBlock(const Block& block, const std::map<const Block*, size_t>& block_to_index,
               std::map<const Inst*, size_t>& inst_to_index, size_t& inst_index) -> std::string {
    std::string ret = "Block";
    const auto block_index = block_to_index.find(&block);
    if (block_index != block_to_index.end()) {
        ret += fmt::format(" ${}", block_index->second);
    }
    ret += '\n';
    for (const auto& inst : block) {
        const Opcode op = inst.GetOpcode();
        ret += fmt::format("[{:016x}] ", reinterpret_cast<u64>(&inst));
        if (TypeOf(op) != Type::Void) {
            ret += fmt::format("%{:<5} = {}", InstIndex(inst_to_index, inst_index, &inst), op);
        } else {
            ret += fmt::format("         {}", op);
        }
        const auto arg_count = inst.NumArgs();
        for (size_t arg_index = 0; arg_index < arg_count; ++arg_index) {
            const auto arg = inst.Arg(arg_index);
            const auto arg_str = ArgToIndex(inst_to_index, inst_index, arg);
            ret += arg_index != 0 ? ", " : " ";
            if (op == Opcode::Phi) {
                const auto phi_block = inst.PhiBlock(arg_index);
                ret += fmt::format("[ {}, {} ]", arg_str, BlockToIndex(block_to_index, phi_block));
            } else {
                ret += arg_str;
            }
            if (op != Opcode::Phi) {
                const auto actual_type = arg.Type();
                const auto expected_type = ArgTypeOf(op, arg_index);
                if (!AreTypesCompatible(actual_type, expected_type)) {
                    ret += fmt::format("<type error: {} != {}>", actual_type, expected_type);
                }
            }
        }
        if (TypeOf(op) != Type::Void) {
            ret += fmt::format(" (uses: {})\n", inst.UseCount());
        } else {
            ret += '\n';
        }
    }
    return ret;
}

} // namespace Shader::IR

