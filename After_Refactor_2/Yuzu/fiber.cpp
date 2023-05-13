// Fiber.hpp

#pragma once

#include <functional>
#include <memory>

namespace Common {

class Fiber {
public:
    Fiber(std::function<void()> entry_point_func);
    Fiber();
    ~Fiber();

    void SetRewindPoint(std::function<void()> const& rewind_func);
    void YieldTo(std::weak_ptr<Fiber> const& from, Fiber& to);
    void Rewind();
    static std::shared_ptr<Fiber> ThreadToFiber();
    void Exit();

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Common

// Fiber.cpp

#include "Fiber.hpp"

namespace Common {

class Fiber::Impl {
public:
    Impl(std::function<void()> entry_point_func) : entry_point{std::move(entry_point_func)} {
        stack.resize(default_stack_size);
        rewind_stack.resize(default_stack_size);
        stack_limit = stack.data();
        rewind_stack_limit = rewind_stack.data();
        u8* stack_base = stack_limit + default_stack_size;
        context = boost::context::detail::make_fcontext(stack_base, stack.size(), FiberStartFunc);
    }

    std::function<void()> entry_point;
    std::function<void()> rewind_point;
    VirtualBuffer<u8> stack;
    VirtualBuffer<u8> rewind_stack;
    u8* stack_limit{};
    u8* rewind_stack_limit{};
    std::shared_ptr<Fiber> previous_fiber;
    bool is_thread_fiber{};
    bool released{};
    std::mutex guard;
    boost::context::detail::fcontext_t context{};
    boost::context::detail::fcontext_t rewind_context{};
};

constexpr std::size_t default_stack_size = 512 * 1024;

Fiber::Fiber(std::function<void()> entry_point_func) : impl{std::make_unique<Impl>(std::move(entry_point_func))} {}

Fiber::Fiber() : impl{std::make_unique<Impl>(nullptr)} {}

Fiber::~Fiber() {
    if (impl->released) {
        return;
    }
    // Make sure the Fiber is not being used
    bool locked = impl->guard.try_lock();
    assert(locked && "Destroying a fiber that's still running");
    if (locked) {
        impl->guard.unlock();
    }
}

void Fiber::Exit() {
    assert(impl->is_thread_fiber && "Exiting non main thread fiber");
    if (!impl->is_thread_fiber) {
        return;
    }
    impl->guard.unlock();
    impl->released = true;
}

void Fiber::SetRewindPoint(std::function<void()> const& rewind_func) {
    impl->rewind_point = rewind_func;
}

void Fiber::YieldTo(std::weak_ptr<Fiber> const& weak_from, Fiber& to) {
    to.impl->guard.lock();
    to.impl->previous_fiber = weak_from.lock();
    auto transfer = boost::context::detail::jump_fcontext(to.impl->context, &to);
    auto from = weak_from.lock();
    if (from) {
        assert(from->impl->previous_fiber && "previous_fiber is nullptr!");
        from->impl->previous_fiber->impl->context = transfer.fctx;
        from->impl->previous_fiber->impl->guard.unlock();
        from->impl->previous_fiber.reset();
    }
}

void Fiber::Rewind() {
    assert(impl->rewind_point);
    assert(impl->rewind_context == nullptr);
    u8* stack_base = impl->rewind_stack_limit + default_stack_size;
    impl->rewind_context = boost::context::detail::make_fcontext(stack_base, impl->stack.size(), RewindStartFunc);
    boost::context::detail::jump_fcontext(impl->rewind_context, this);
}

std::shared_ptr<Fiber> Fiber::ThreadToFiber() {
    auto fiber = std::make_shared<Fiber>();
    fiber->impl->guard.lock();
    fiber->impl->is_thread_fiber = true;
    return fiber;
}

void Fiber::FiberStartFunc(boost::context::detail::transfer_t transfer) {
    auto* fiber = static_cast<Fiber*>(transfer.data);
    fiber->Start(transfer);
}

void Fiber::RewindStartFunc(boost::context::detail::transfer_t transfer) {
    auto* fiber = static_cast<Fiber*>(transfer.data);
    fiber->OnRewind(transfer);
}

void Fiber::Start(boost::context::detail::transfer_t& transfer) {
    assert(impl->previous_fiber != nullptr);
    impl->previous_fiber->impl->context = transfer.fctx;
    impl->previous_fiber->impl->guard.unlock();
    impl->previous_fiber.reset();
    impl->entry_point();
    assert(false && "Unreachable code");
}

void Fiber::OnRewind(boost::context::detail::transfer_t& transfer) {
    assert(impl->context != nullptr);
    impl->context = impl->rewind_context;
    impl->rewind_context = nullptr;
    u8* tmp = impl->stack_limit;
    impl->stack_limit = impl->rewind_stack_limit;
    impl->rewind_stack_limit = tmp;
    impl->rewind_point();
    assert(false && "Unreachable code");
}

} // namespace Common

