"""Tests for dump_task."""

from unittest import mock

import gin
import seqio
import tensorflow as tf

from t5.scripts import dump_task

class DumpTaskSequenceLengthTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    gin.clear_config(clear_constants=True)
    super().setUpClass()

  def test_sequence_length_512Tokens(self):
    sequence_lengths = dump_task.sequence_length(512)
    self.assertEqual({"inputs": 512, "targets": 512}, sequence_lengths)

  def test_sequence_length_256Tokens(self):
    sequence_lengths = dump_task.sequence_length(256)
    self.assertEqual({"inputs": 256, "targets": 256}, sequence_lengths)

class DumpTaskPrettyTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    gin.clear_config(clear_constants=True)
    super().setUpClass()

  def test_pretty_single_delimiter(self):
    with self.flags(
        detokenize=True, pretty=True, delimiters=["\\[[A-Z]+\\]"]):
      prettied = dump_task.pretty("[CLS] some input")
      self.assertGreaterEqual(prettied.find("\u001b[1m"), 0)  # Bold applied
      self.assertGreaterEqual(prettied.find("\u001b[0m"), 0)  # Reset applied

  def test_pretty_multiple_delimiters(self):
    with self.flags(
        detokenize=True, pretty=True, delimiters=["\\[[A-Z]+\\]"]):
      prettied = dump_task.pretty("[CLS] some input [SEP] other input")
      self.assertGreaterEqual(prettied.find("\u001b[1m"), 0)  # Bold applied
      self.assertGreaterEqual(prettied.find("\u001b[0m"), 0)  # Reset applied

class DumpTaskMainTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    gin.clear_config(clear_constants=True)
    super().setUpClass()

  def setUp(self):
    super().setUp()
    self.mock_task = self.get_mocked_task()

  def get_mocked_task(self):
    task = mock.Mock()
    task.name = "test_task"
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

  def tearDown(self):
    super().tearDown()
    self.mock_task = None

  def test_main_with_task(self):
    with mock.patch.object(
        seqio.TaskRegistry, "get", return_value=self.mock_task) as mock_get:
      with self.flags(task="test_task"):
        dump_task.main(None)
        mock_get.assert_called_once_with("test_task")
        self.mock_task.get_dataset.assert_called_once()

  def test_main_with_mixture(self):
    with mock.patch.object(
        seqio.MixtureRegistry, "get", return_value=self.mock_task) as mock_get:
      with self.flags(mixture="test_mixture"):
        dump_task.main(None)
        mock_get.assert_called_once_with("test_mixture")
        self.mock_task.get_dataset.assert_called_once()

  def test_main_with_detokenize_task(self):
    with mock.patch.object(
        seqio.TaskRegistry, "get", return_value=self.mock_task) as mock_get:
      with self.flags(detokenize=True, task="test_task"):
        dump_task.main(None)
        mock_get.assert_called_once_with("test_task")
        self.assertEqual(self.mock_task.mock_vocab.decode_tf.call_count, 3)
        self.mock_task.postprocess_fn.assert_not_called()

  def test_main_with_postprocess(self):
    with mock.patch.object(
        seqio.TaskRegistry, "get", return_value=self.mock_task) as mock_get:
      with self.flags(detokenize=True, apply_postprocess_fn=True, task="test_task"):
        dump_task.main(None)
        mock_get.assert_called_once_with("test_task")
        self.assertEqual(self.mock_task.mock_vocab.decode_tf.call_count, 3)
        self.mock_task.postprocess_fn.assert_called_once()

if __name__ == "__main__":
  tf.test.main()