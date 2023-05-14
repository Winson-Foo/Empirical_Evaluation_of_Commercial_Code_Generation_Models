#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include "common/assert.h"

namespace Common {

class FContext {
public:
    FContext() = default;
    FContext(void* sp, std::size_t size, std::function<void(void*)> transfer_func)
        : fctx_{boost::context::detail::make_fcontext(sp, size, &FContext::TransferFunc)} {
        transfer_data_ = std::make_unique<TransferData>(TransferData{this, std::move(transfer_func)});
    }

    void Jump(void* data) {
        boost::context::detail::jump_fcontext(fctx_, data);
    }

    bool Valid() const {
        return fctx_ != nullptr;
    }

    void* Get() {
        return fctx_;
    }

private:
    struct TransferData {
        FContext* context = nullptr;
        std::function<void(void*)> transfer_func;
    };

    static void TransferFunc(boost::context::detail::transfer_t transfer) {
        auto* data = static_cast<TransferData*>(transfer.data);
        data->transfer_func(data->context);
    }

    boost::context::detail::fcontext_t fctx_{};
    std::unique_ptr<TransferData> transfer_data_{};
};

class Fiber {
public:
    Fiber() = default;
    explicit Fiber(std::function<void()> entry_func);
    ~Fiber();

    void SetRewindPoint(std::function<void()> rewind_func);
    void YieldTo(std::weak_ptr<Fiber> from, Fiber& to);

    static std::shared_ptr<Fiber> ThreadToFiber();

    Fiber(const Fiber&) = delete;
    Fiber& operator=(const Fiber&) = delete;

private:
    void Start();
    void OnRewind();
    static void TransferFunc(void* data);
    static void ThreadFunc(std::weak_ptr<Fiber> fiber_weak);

    std::unique_ptr<FContext> context_;
    std::unique_ptr<FContext> rewind_context_;

    std::function<void()> entry_func_;
    std::function<void()> rewind_func_;
    std::mutex guard_;
    std::shared_ptr<Fiber> previous_fiber_;
    bool is_thread_fiber_ = false;
    bool released_ = false;
};

Fiber::Fiber(std::function<void()> entry_func)
    : context_{std::make_unique<FContext>()}, entry_func_{std::move(entry_func)} {
    // Allocate the stacks
    constexpr std::size_t default_stack_size = 512 * 1024;
    auto stack = std::make_unique<char[]>(default_stack_size);
    auto rewind_stack = std::make_unique<char[]>(default_stack_size);

    // Initialize the contexts
    rewind_context_ = std::make_unique<FContext>(
        reinterpret_cast<void*>(rewind_stack.get() + default_stack_size), default_stack_size,
        [this](void* data) { OnRewind(); });
    context_ = std::make_unique<FContext>(
        reinterpret_cast<void*>(stack.get() + default_stack_size), default_stack_size,
        [](void* data) { TransferFunc(data); });
}

Fiber::~Fiber() {
    if (released_) {
        return;
    }
    // Make sure the Fiber is not being used
    const bool locked = guard_.try_lock();
    ASSERT_MSG(locked, "Destroying a fiber that's still running");
    if (locked) {
        guard_.unlock();
    }
}

void Fiber::SetRewindPoint(std::function<void()> rewind_func) {
    rewind_func_ = std::move(rewind_func);
}

void Fiber::Start() {
    ASSERT(previous_fiber_ != nullptr);
    previous_fiber_->context_->Jump(context_.get());
    previous_fiber_.reset();
    entry_func_();
    UNREACHABLE();
}

void Fiber::OnRewind() {
    ASSERT(context_->Valid());
    std::swap(context_, rewind_context_);
    rewind_func_();
    UNREACHABLE();
}

void Fiber::TransferFunc(void* data) {
    auto* fiber = static_cast<Fiber*>(data);
    fiber->Start();
}

void Fiber::YieldTo(std::weak_ptr<Fiber> from, Fiber& to) {
    std::unique_lock lock{to.guard_};
    to.previous_fiber_ = from.lock();
    to.previous_fiber_->context_->Jump(to.context_.get());
    if (auto from_shared = from.lock()) {
        if (from_shared->previous_fiber_ == nullptr) {
            ASSERT_MSG(false, "previous_fiber_ is nullptr!");
            return;
        }
        from_shared->previous_fiber_->context_->Jump(from_shared->context_.get());
        from_shared->previous_fiber_.reset();
    }
}

std::shared_ptr<Fiber> Fiber::ThreadToFiber() {
    auto fiber = std::make_shared<Fiber>();
    fiber->is_thread_fiber_ = true;

    std::thread thread{[fiber_weak = std::weak_ptr<Fiber>{fiber}]() { ThreadFunc(fiber_weak); }};
    thread.detach();
    return fiber;
}

void Fiber::ThreadFunc(std::weak_ptr<Fiber> fiber_weak) {
    auto fiber = fiber_weak.lock();
    if (fiber == nullptr) {
        return;
    }
    fiber->context_ = std::make_unique<FContext>(
        nullptr, 0,
        [fiber_weak](void* data) {
            auto fiber = fiber_weak.lock();
            if (fiber == nullptr) {
                return;
            }
            fiber->released_ = true;
            fiber->guard_.unlock();
        });
    fiber->guard_.lock();
    fiber->previous_fiber_ = nullptr;
    fiber->entry_func_();
    fiber->released_ = true;
    fiber->guard_.unlock();
    fiber.reset();
}

} // namespace Common