#include <string>
#include <utility>
#include <catch2/catch_test_macros.hpp>
#include "common/unique_function.h"

namespace {

struct Noisy {
    Noisy() : state{"Default constructed"} {}
    Noisy(Noisy&& rhs) noexcept : state{"Move constructed"} {
        rhs.state = "Moved away";
    }
    Noisy& operator=(Noisy&& rhs) noexcept {
        state = "Move assigned";
        rhs.state = "Moved away";
        return *this;
    }
    Noisy(const Noisy&) : state{"Copied constructed"} {}
    Noisy& operator=(const Noisy&) {
        state = "Copied assigned";
        return *this;
    }

    std::string state;
};

template<typename Func>
void test_capturing(Func func) {
    int value = 0;
    func([&value] { value = 5; });
    REQUIRE(value == 5);
}

template<typename Func>
void test_constructing(Func func) {
    int value = 0;
    func([&value] { value = 5; });
    REQUIRE(value == 5);
}

template<typename Func>
void test_assigning(Func func) {
    int value = 0;
    Common::UniqueFunction<void> handler;
    handler = [&value] { value = 5; };
    func(handler);
    REQUIRE(value == 5);
}

template<typename Func>
void test_arguments(Func func) {
    int result = 0;
    Common::UniqueFunction<void, int, int> handler([&result](int a, int b) { result = a + b; });
    func(handler);
    REQUIRE(result == 9);
}

template<typename Func>
void test_return_value(Func func) {
    Common::UniqueFunction<int, int, int> handler([](int a, int b) { return a + b; });
    REQUIRE(func(handler) == 9);
}

template<typename Func>
void test_destructor(Func func) {
    int num_destroyed = 0;
    struct Foo {
        Foo(int* num_) : num{num_} {}
        Foo(Foo&& rhs) : num{std::exchange(rhs.num, nullptr)} {}
        Foo(const Foo&) = delete;

        ~Foo() {
            if (num) {
                ++*num;
            }
        }

        int* num = nullptr;
    };
    Foo object{&num_destroyed};
    {
        Common::UniqueFunction<void> handler([object = std::move(object)] {});
        func(handler);
        REQUIRE(num_destroyed == 0);
    }
    REQUIRE(num_destroyed == 1);
}

} // Anonymous namespace

TEST_CASE("UniqueFunction", "[common]") {
    SECTION("Capture reference") {
        test_capturing([](auto func) {
            Common::UniqueFunction<void> handler(func);
            handler();
        });
    }
    SECTION("Capture pointer") {
        test_capturing([](auto func) {
            int value = 0;
            int* pointer = &value;
            Common::UniqueFunction<void> handler([pointer] { *pointer = 5; });
            func(handler);
            REQUIRE(value == 5);
        });
    }
    SECTION("Move object") {
        Noisy noisy;
        REQUIRE(noisy.state == "Default constructed");

        Common::UniqueFunction<void> handler([noisy = std::move(noisy)] {
            REQUIRE(noisy.state == "Move constructed");
        });
        REQUIRE(noisy.state == "Moved away");
        handler();
    }
    SECTION("Move construct function") {
        test_constructing([](auto func) {
            Common::UniqueFunction<void> handler([&func] {
                Common::UniqueFunction<void> new_handler = std::move(func);
                new_handler();
            });
            handler();
        });
    }
    SECTION("Move assign function") {
        test_assigning([](auto func) {
            Common::UniqueFunction<void> handler;
            func([&handler](auto new_handler) {
                handler = std::move(new_handler);
            });
            handler();
        });
    }
    SECTION("Default construct then assign function") {
        test_assigning([](auto func) {
            Common::UniqueFunction<void> handler;
            func([&handler](auto new_handler) {
                handler = std::move(new_handler);
            });
            handler();
        });
    }
    SECTION("Pass arguments") {
        test_arguments([](auto func) {
            Common::UniqueFunction<void, int, int> handler;
            func([&handler](auto new_handler) {
                handler = std::move(new_handler);
            });
            handler(5, 4);
        });
    }
    SECTION("Pass arguments and return value") {
        REQUIRE(test_return_value([](auto& func) {
            Common::UniqueFunction<int, int, int> handler;
            func([&handler](auto new_handler){ handler = std::move(new_handler); });
            return handler(5, 4);
        }) == 9);
    }
    SECTION("Destructor") {
        test_destructor([](auto func) {
            Common::UniqueFunction<void> handler;
            func([&handler](auto new_handler) {
                handler = std::move(new_handler);
            });
        });
    }
}