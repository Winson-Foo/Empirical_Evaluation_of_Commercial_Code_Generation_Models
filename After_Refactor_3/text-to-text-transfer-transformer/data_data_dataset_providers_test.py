# Copyright 2023 The T5 Authors.

from absl.testing import absltest
import immutabledict
import os
from seqio import Feature, MixtureRegistry, TaskRegistry
from seqio import test_utils
from t5.data import dataset_providers
import tensorflow.compat.v2 as tf

tf.compat.v1.enable_eager_execution()

def add_t5_task(
    name, cls, text_preprocessor=(test_utils.test_text_preprocessor,),
    output_features=None, **kwargs):
  
  output_features = output_features or {
      "inputs": Feature(test_utils.sentencepiece_vocab()),
      "targets": Feature(test_utils.sentencepiece_vocab())
  }

  return TaskRegistry.add(
      name,
      cls,
      text_preprocessor=text_preprocessor,
      metric_fns=[],
      output_features=output_features,
      **kwargs
  )

class TasksTest(test_utils.FakeTaskTest):

  def test_tfds_task(self):
    add_t5_task(
        "t5_tfds_task", dataset_providers.TfdsTask, tfds_name="fake:0.0.0")
    self.verify_task_matches_fake_datasets("t5_tfds_task", use_cached=False)

  def test_immutabledict_features(self):
    add_t5_task(
        "t5_tfds_task", dataset_providers.TfdsTask, tfds_name="fake:0.0.0",
        output_features=immutabledict.immutabledict({
            "inputs": Feature(test_utils.sentencepiece_vocab()),
            "targets": Feature(test_utils.sentencepiece_vocab())
        }))
    self.verify_task_matches_fake_datasets("t5_tfds_task", use_cached=False)

  def test_function_task(self):
    add_t5_task(
        "t5_fn_task",
        dataset_providers.FunctionTask,
        splits=("train", "validation"),
        dataset_fn=test_utils.get_fake_dataset)
    self.verify_task_matches_fake_datasets("t5_fn_task", use_cached=False)

  def test_text_line_task(self):
    add_t5_task(
        "t5_text_line_task",
        dataset_providers.TextLineTask,
        split_to_filepattern={
            "train": os.path.join(self.test_data_dir, "train.tsv*"),
        },
        skip_header_lines=1,
        text_preprocessor=(test_utils.split_tsv_preprocessor,
                           test_utils.test_text_preprocessor))
    self.verify_task_matches_fake_datasets(
        "t5_text_line_task", use_cached=False, splits=["train"])

  def test_tf_example_task(self):
    self.verify_task_matches_fake_datasets(
        "tf_example_task", use_cached=False, splits=["train"])

  def test_cached_task(self):
    TaskRegistry.remove("cached_task")
    add_t5_task(
        "cached_task", dataset_providers.TfdsTask, tfds_name="fake:0.0.0")
    self.verify_task_matches_fake_datasets("cached_task", use_cached=True)

  def test_token_preprocessor(self):
    TaskRegistry.remove("cached_task")
    add_t5_task(
        "cached_task",
        dataset_providers.TfdsTask,
        tfds_name="fake:0.0.0",
        token_preprocessor=(test_utils.test_token_preprocessor,))

    self.verify_task_matches_fake_datasets(
        "cached_task", use_cached=False, token_preprocessed=True)
    self.verify_task_matches_fake_datasets(
        "cached_task", use_cached=True, token_preprocessed=True)

  def test_optional_features():

    def _dummy_preprocessor(output):
      return lambda _: tf.data.Dataset.from_tensors(output)

    default_vocab = test_utils.sentencepiece_vocab()
    features = {
        "inputs": Feature(vocabulary=default_vocab, required=False),
        "targets": Feature(vocabulary=default_vocab, required=True),
    }

    task = add_t5_task(
        "task_missing_optional_feature",
        dataset_providers.TfdsTask,
        tfds_name="fake:0.0.0",
        output_features=features,
        text_preprocessor=_dummy_preprocessor({"targets": "a"}))
    task.get_dataset({"targets": 13}, "train", use_cached=False)

    task = add_t5_task(
        "task_missing_required_feature",
        dataset_providers.TfdsTask,
        tfds_name="fake:0.0.0",
        output_features=features,
        text_preprocessor=_dummy_preprocessor({"inputs": "a"}))
    with self.assertRaisesRegex(
        ValueError,
        "Task dataset is missing expected output feature after preprocessing: "
        "targets"):
      task.get_dataset({"inputs": 13}, "train", use_cached=False)

  def test_no_eos(self):
    default_vocab = test_utils.sentencepiece_vocab()
    features = {
        "inputs": Feature(add_eos=True, vocabulary=default_vocab),
        "targets": Feature(add_eos=False, vocabulary=default_vocab),
    }
    add_t5_task(
        "task_no_eos",
        dataset_providers.TfdsTask,
        tfds_name="fake:0.0.0",
        output_features=features)
    self.verify_task_matches_fake_datasets("task_no_eos", use_cached=False)

  def test_task_registry_reset(self):
    TaskRegistry.add(
        "t5_task_before_reset",
        dataset_providers.TFExampleTask,
        split_to_filepattern={},
        feature_description={}
    )
    # Assert that task was added to both t5.data.TaskRegistry and
    # seqio.TaskRegistry.
    self.assertSameElements(TaskRegistry.names(), seqio.TaskRegistry.names())
    TaskRegistry.reset()
    TaskRegistry.add(
        "t5_task_after_reset",
        dataset_providers.TFExampleTask,
        split_to_filepattern={},
        feature_description={}
    )
    # Assert that task was added to both t5.data.TaskRegistry and
    # seqio.TaskRegistry so that they don't diverge after reset() call.
    self.assertSameElements(TaskRegistry.names(), seqio.TaskRegistry.names())


if __name__ == "__main__":
  absltest.main()