# Copyright 2023 The T5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Testing utilities for the evaluation package."""

from absl.testing import absltest


class BaseMetricsTest(absltest.TestCase):

    def assertDictClose(self, actual, expected, delta=None, places=None):
        """
        Asserts if two dictionaries are equal with a given delta or places tolerance for float values.

        Args:
            actual: The actual dictionary.
            expected: The expected dictionary.
            delta: The maximum difference between the actual and expected float values.
            places: The maximum number of decimal places difference between the actual and expected float values.

        Raises:
            AssertionError: If the dictionaries are not equal.
        """
        self.assertCountEqual(actual.keys(), expected.keys())

        for key in actual:
            try:
                self.assertAlmostEqual(actual[key], expected[key], delta=delta, places=places)
            except AssertionError as e:
                raise AssertionError(str(e) + f" for key '{key}'")