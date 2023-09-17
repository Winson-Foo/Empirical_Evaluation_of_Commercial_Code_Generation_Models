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

"""Tests for dump_task."""

from unittest import mock
from absl.testing import flagsaver
from absl.testing import parameterized
import gin
import seqio
from t5.scripts import dump_task
import tensorflow as tf


def get_mocked_task(name: str):
    task = mock.Mock()
    task.name = name
    task.postprocess_fn.return_value = ["test"]
    task.get_dataset.return_value = [{
        "inputs": tf.constant([111, 222, 333, 444]),
        "negative_inputs": tf.constant([[111, 222, 333, 444], [111, 222, 333, 444]]),
        "targets": tf.constant([222])
    }]
    task.mock_vocab = mock.Mock()
    task.mock_vocab.decode_tf.return_value = ["test"]
    task.output_features = {
        "inputs": seqio.dataset_providers.Feature(task.mock_vocab),
        "negative_inputs": seqio.dataset_providers.Feature(task.mock_vocab, rank=2),
        "targets": seqio.dataset_providers.Feature(task.mock_vocab)
    }
    return task


class DumpTaskTest(parameterized.TestCase, tf.test.TestCase):

    def tearDown(self):
        gin.clear_config(clear_constants=True)
        super().tearDown()

    def test_sequence_length(self):
        # Test sequence_length function
        sequence_length_512 = dump_task.sequence_length(512)
        sequence_length_256 = dump_task.sequence_length(256)
        self.assertEqual({"inputs": 512, "targets": 512}, sequence_length_512)
        self.assertEqual({"inputs": 256, "targets": 256}, sequence_length_256)

    @flagsaver.flagsaver(detokenize=True, pretty=True, delimiters=["\\[[A-Z]+\\]"])
    def test_pretty(self):
        # Test pretty function
        value_1 = "[CLS] some input"
        value_2 = "[CLS] some input [SEP] other input"

        prettied_1 = dump_task.pretty(value_1)
        prettied_2 = dump_task.pretty(value_2)

        self.assertGreaterEqual(prettied_1.find("\u001b[1m"), 0)  # Bold applied
        self.assertGreaterEqual(prettied_1.find("\u001b[0m"), 0)  # Reset applied

        self.assertGreaterEqual(prettied_2.find("\u001b[1m"), 0)  # Bold applied
        self.assertGreaterEqual(prettied_2.find("\u001b[0m"), 0)  # Reset applied

    @flagsaver.flagsaver(
        format_string="inputs: {inputs}, negatives: {negative_inputs}, targets: {targets}",
        module_import=[])
    def test_main(self):
        # Test main function
        mock_task = get_mocked_task("test_task")
        self.enter_context(mock.patch.object(seqio.TaskRegistry, "get", return_value=mock_task))
        self.enter_context(mock.patch.object(seqio.MixtureRegistry, "get", return_value=mock_task))

        flags = {
            "task": "test_task",
            "detokenize": True,
            "apply_postprocess_fn": True
        }

        with flagsaver.flagsaver(**flags):
            dump_task.main(None)

            seqio.TaskRegistry.get.assert_called_once_with("test_task")
            mock_task.get_dataset.assert_called_once()
            self.assertEqual(mock_task.mock_vocab.decode_tf.call_count, 3)
            mock_task.postprocess_fn.assert_called_once()


if __name__ == "__main__":
    tf.test.main()