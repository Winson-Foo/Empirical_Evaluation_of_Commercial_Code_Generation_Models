// SPDX-License-Identifier: GPL-2.0-or-later

#include <string>

#include <catch2/catch_test_macros.hpp>

#include "common/unique_function.h"

namespace yuzu {

struct ObjectWithState {
    ObjectWithState()
        : state{"Default constructed"}
    {}

    ObjectWithState(ObjectWithState&& rhs) noexcept
        : state{"Move constructed"}
    {
        rhs.state = "Moved away";
    }

    ObjectWithState& operator=(ObjectWithState&& rhs) noexcept {
        state = "Move assigned";
        rhs.state = "Moved away";
        return *this;
    }

    ObjectWithState(const ObjectWithState&)
        : state{"Copied constructed"}
    {}

    ObjectWithState& operator=(const ObjectWithState&) {
        state = "Copied assigned";
        return *this;
    }

    std::string state;
};

namespace unique_function_tests {

TEST_CASE("UniqueFunction") {
    SECTION("Capture reference") {
        int value = 0;

        Common::UniqueFunction<void> func = [&value] {
            value = 5;
        };

        func();
        REQUIRE(value == 5);
    }

    SECTION("Capture pointer") {
        int value = 0;
        int* pointer = &value;

        Common::UniqueFunction<void> func = [pointer] {
            *pointer = 5;
        };

        func();
        REQUIRE(value == 5);
    }

    SECTION("Move object") {
        ObjectWithState object;
        REQUIRE(object.state == "Default constructed");

        Common::UniqueFunction<void> func = [object = std::move(object)] {
            REQUIRE(object.state == "Move constructed");
        };

        REQUIRE(object.state == "Moved away");
        func();
    }

    SECTION("Move construct function") {
        int value = 0;

        Common::UniqueFunction<void> func = [&value] {
            value = 5;
        };

        Common::UniqueFunction<void> new_func = std::move(func);
        new_func();

        REQUIRE(value == 5);
    }

    SECTION("Move assign function") {
        int value = 0;

        Common::UniqueFunction<void> func = [&value] {
            value = 5;
        };

        Common::UniqueFunction<void> new_func;
        new_func = std::move(func);
        new_func();

        REQUIRE(value == 5);
    }

    SECTION("Default construct then assign function") {
        int value = 0;

        Common::UniqueFunction<void> func;
        func = [&value] {
            value = 5;
        };

        func();
        REQUIRE(value == 5);
    }

    SECTION("Pass arguments") {
        int result = 0;

        Common::UniqueFunction<void, int, int> func = [&result](int a, int b) {
            result = a + b;
        };

        func(5, 4);
        REQUIRE(result == 9);
    }

    SECTION("Pass arguments and return value") {
        Common::UniqueFunction<int, int, int> func = [](int a, int b) {
            return a + b;
        };

        REQUIRE(func(5, 4) == 9);
    }

    SECTION("Destructor") {
        int num_destroyed = 0;

        struct Foo {
            Foo(int* num_)
                : num{num_}
            {}

            Foo(Foo&& rhs)
                : num{std::exchange(rhs.num, nullptr)}
            {}

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
            Common::UniqueFunction<void> func = [object = std::move(object)] {};

            REQUIRE(num_destroyed == 0);
        }

        REQUIRE(num_destroyed == 1);
    }
}

} // namespace unique_function_tests

} // namespace yuzu