// Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/common_types.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/value.h"

namespace Shader::IR {

Block::Block(ObjectPool<Inst>& inst_pool)
    : instruction_pool_(&inst_pool)
{}

Block::~Block() = default;

void Block::AppendNewInstruction(Opcode operation, std::initializer_list<Value> arguments)
{
    PrependNewInstruction(end(), operation, arguments);
}

Block::iterator Block::PrependNewInstruction(iterator insertion_point, const Inst& base_inst)
{
    Inst* const instruction{instruction_pool_->Create(base_inst)};
    return instructions_.insert(insertion_point, *instruction);
}

Block::iterator Block::PrependNewInstruction(iterator insertion_point, Opcode operation,
                                              std::initializer_list<Value> arguments, u32 flags)
{
    Inst* const instruction{instruction_pool_->Create(operation, flags)};
    const auto result_iterator{instructions_.insert(insertion_point, *instruction)};

    if (instruction->NumArgs() != arguments.size()) {
        throw InvalidArgument(
            "Invalid number of arguments {} in {}", arguments.size(), operation);
    }
    std::ranges::for_each(arguments, [instruction, index = size_t{0}](const Value& arg) mutable {
        instruction->SetArg(index, arg);
        ++index;
    });
    return result_iterator;
}

void Block::AddBranch(Block* block)
{
    if (std::ranges::find(immediate_successors_, block) != immediate_successors_.end()) {
        throw LogicError("Successor already inserted");
    }
    if (std::ranges::find(block->immediate_predecessors_, this) != block->immediate_predecessors_.end()) {
        throw LogicError("Predecessor already inserted");
    }
    immediate_successors_.push_back(block);
    block->immediate_predecessors_.push_back(this);
}

static std::string BlockToIndex(const std::map<const Block*, size_t>& block_to_index,
                                Block* block)
{
    if (const auto it{block_to_index.find(block)}; it != block_to_index.end()) {
        return fmt::format("{{Block ${}}}", it->second);
    }
    return fmt::format("$<unknown block {:016x}>", reinterpret_cast<u64>(block));
}

static size_t InstructionIndex(std::map<const Inst*, size_t>& inst_to_index,
                                size_t& inst_index,
                                const Inst* instruction)
{
    const auto [it, inserted]{inst_to_index.emplace(instruction, inst_index + 1)};
    if (inserted) {
        ++inst_index;
    }
    return it->second;
}

static std::string ArgumentToIndex(std::map<const Inst*, size_t>& inst_to_index,
                                    size_t& inst_index,
                                    const Value& argument)
{
    if (argument.IsEmpty()) {
        return "<null>";
    }
    if (!argument.IsImmediate() || argument.IsIdentity()) {
        return fmt::format("%{}", InstructionIndex(inst_to_index, inst_index, argument.Inst()));
    }
    switch (argument.Type()) {
        case Type::U1:
            return fmt::format("#{}", argument.U1() ? "true" : "false");
        case Type::U8:
            return fmt::format("#{}", argument.U8());
        case Type::U16:
            return fmt::format("#{}", argument.U16());
        case Type::U32:
            return fmt::format("#{}", argument.U32());
        case Type::U64:
            return fmt::format("#{}", argument.U64());
        case Type::F32:
            return fmt::format("#{}", argument.F32());
        case Type::Reg:
            return fmt::format("{}", argument.Reg());
        case Type::Pred:
            return fmt::format("{}", argument.Pred());
        case Type::Attribute:
            return fmt::format("{}", argument.Attribute());
        default:
            return "<unknown immediate type>";
    }
}

std::string DumpBlock(const Block& block)
{
    size_t inst_index{0};
    std::map<const Inst*, size_t> inst_to_index;
    return DumpBlock(block, {}, inst_to_index, inst_index);
}

std::string DumpBlock(const Block& block, const std::map<const Block*, size_t>& block_to_index,
                      std::map<const Inst*, size_t>& inst_to_index, size_t& inst_index)
{
    std::string result{"Block"};
    if (const auto it{block_to_index.find(&block)}; it != block_to_index.end()) {
        result += fmt::format(" ${}", it->second);
    }
    result += '\n';
    for (const Inst& instruction : block) {
        const Opcode operation{instruction.GetOpcode()};
        result += fmt::format("[{:016x}] ", reinterpret_cast<u64>(&instruction));
        if (TypeOf(operation) != Type::Void) {
            result += fmt::format("%{:<5} = {}", InstructionIndex(inst_to_index, inst_index, &instruction), operation);
        } else {
            result += fmt::format("         {}", operation); 
        }
        const size_t argument_count{instruction.NumArgs()};
        for (size_t argument_index = 0; argument_index < argument_count; ++argument_index) {
            const Value argument{instruction.Arg(argument_index)};
            const std::string arg_str{ArgumentToIndex(inst_to_index, inst_index, argument)};
            result += argument_index != 0 ? ", " : " ";
            if (operation == Opcode::Phi) {
                result += fmt::format("[ {}, {} ]", arg_str,
                                       BlockToIndex(block_to_index, instruction.PhiBlock(argument_index)));
            } else {
                result += arg_str;
            }
            if (operation != Opcode::Phi) {
                const Type actual_type{argument.Type()};
                const Type expected_type{ArgTypeOf(operation, argument_index)};
                if (!AreTypesCompatible(actual_type, expected_type)) {
                    result += fmt::format("<type error: {} != {}>", actual_type, expected_type);
                }
            }
        }
        if (TypeOf(operation) != Type::Void) {
            result += fmt::format(" (uses: {})\n", instruction.UseCount());
        } else {
            result += '\n';
        }
    }
    return result;
}

} // namespace Shader::IR

