from unittest import mock
import re
import absl.testing.parameterized as parameterized
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


def clear_gin_configs():
    gin.clear_config(clear_constants=True)


class DumpTaskTest(parameterized.TestCase, tf.test.TestCase):

    def tearDown(self):
        clear_gin_configs()
        super().tearDown()

    @parameterized.named_parameters(
        dict(testcase_name="512Tokens", value=512),
        dict(testcase_name="256Tokens", value=256))
    def test_sequence_length(self, value: int):
        sequence_lengths = dump_task.sequence_length(value)
        self.assertEqual({"inputs": value, "targets": value}, sequence_lengths)

    @parameterized.named_parameters(
        dict(testcase_name="Single delimiter", value="[CLS] some input"),
        dict(testcase_name="Multiple delimiter", value="[CLS] some input [SEP] other input"))
    def test_pretty(self, value: str):
        prettied = dump_task.pretty(value)
        self.assertGreaterEqual(prettied.find("\u001b[1m"), 0)  # Bold applied
        self.assertGreaterEqual(prettied.find("\u001b[0m"), 0)  # Reset applied

    @parameterized.named_parameters(
        dict(testcase_name="task", detokenize=True, task="test_task"),
        dict(testcase_name="detokenize_task", detokenize=True, task="test_task"),
        dict(testcase_name="postprocess", detokenize=True, apply_postprocess_fn=True, task="test_task"),
        dict(testcase_name="mixture", detokenize=True, apply_postprocess_fn=True, mixture="test_mixture"))
    def test_main(self, **flags):
        mock_task = get_mocked_task(flags.get("task", flags.get("mixture")))
        self.enter_context(mock.patch.object(seqio.TaskRegistry, "get", return_value=mock_task))
        self.enter_context(mock.patch.object(seqio.MixtureRegistry, "get", return_value=mock_task))
        with dump_task.flagsaver(**flags):
            dump_task.main(None)
        if "task" in flags:
            seqio.TaskRegistry.get.assert_called_once_with(flags["task"])
        if "mixture" in flags:
            seqio.MixtureRegistry.get.assert_called_once_with(flags["mixture"])
        mock_task.get_dataset.assert_called_once()
        if "detokenize" in flags:
            self.assertEqual(mock_task.mock_vocab.decode_tf.call_count, 3)
            if "apply_postprocess_fn" in flags:
                mock_task.postprocess_fn.assert_called_once()


if __name__ == "__main__":
    tf.test.main()