// SPDX-License-Identifier: GPL-2.0-or-later

#include <string>

#include <catch2/catch_test_macros.hpp>

#include "common/unique_function.h"

namespace yuzu {
namespace tests {

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

using VoidFunction = Common::UniqueFunction<void>;
template<typename Ret, typename... Args>
using Function = Common::UniqueFunction<Ret, Args...>;

template<typename T>
void test_capture_reference() {
    T value = 0;
    VoidFunction func = [&value] { value = 5; };
    func();
    REQUIRE(value == 5);
}

template<typename T>
void test_capture_pointer() {
    T value = 0;
    T* pointer = &value;
    VoidFunction func = [pointer] { *pointer = 5; };
    func();
    REQUIRE(value == 5);
}

void test_move_object() {
    Noisy noisy;
    REQUIRE(noisy.state == "Default constructed");

    VoidFunction func = [noisy = std::move(noisy)] {
        REQUIRE(noisy.state == "Move constructed");
    };
    REQUIRE(noisy.state == "Moved away");
    func();
}

void test_move_construct_function() {
    int value = 0;
    VoidFunction func = [&value] { value = 5; };
    VoidFunction new_func = std::move(func);
    new_func();
    REQUIRE(value == 5);
}

void test_move_assign_function() {
    int value = 0;
    VoidFunction func = [&value] { value = 5; };
    VoidFunction new_func;
    new_func = std::move(func);
    new_func();
    REQUIRE(value == 5);
}

void test_default_construct_then_assign_function() {
    int value = 0;
    VoidFunction func;
    func = [&value] { value = 5; };
    func();
    REQUIRE(value == 5);
}

void test_pass_arguments() {
    int result = 0;
    Function<void, int, int> func = [&result](int a, int b) { result = a + b; };
    func(5, 4);
    REQUIRE(result == 9);
}

void test_pass_arguments_and_return_value() {
    Function<int, int, int> func = [](int a, int b) { return a + b; };
    REQUIRE(func(5, 4) == 9);
}

void test_destructor() {
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
        VoidFunction func = [object = std::move(object)] {};
        REQUIRE(num_destroyed == 0);
    }
    REQUIRE(num_destroyed == 1);
}

} // Anonymous namespace

TEST_CASE("UniqueFunction", "[common]") {
    SECTION("Capture reference - int") {
        test_capture_reference<int>();
    }
    SECTION("Capture reference - long") {
        test_capture_reference<long>();
    }
    SECTION("Capture pointer - int") {
        test_capture_pointer<int>();
    }
    SECTION("Capture pointer - long") {
        test_capture_pointer<long>();
    }
    SECTION("Move object") {
        test_move_object();
    }
    SECTION("Move construct function") {
        test_move_construct_function();
    }
    SECTION("Move assign function") {
        test_move_assign_function();
    }
    SECTION("Default construct then assign function") {
        test_default_construct_then_assign_function();
    }
    SECTION("Pass arguments") {
        test_pass_arguments();
    }
    SECTION("Pass arguments and return value") {
        test_pass_arguments_and_return_value();
    }
    SECTION("Destructor") {
        test_destructor();
    }
}

} // namespace tests
} // namespace yuzu