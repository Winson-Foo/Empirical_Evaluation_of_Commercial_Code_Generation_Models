import collections
import functools
import json

from t5.data import postprocessors
from t5.data import preprocessors
from t5.evaluation import metrics


class TaskConfig:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        self.super_glue_weight_mapping = config.get("SUPER_GLUE_WEIGHT_MAPPING", {})
        self.super_glue_weight_mapping_sentinel = config.get("SUPER_GLUE_WEIGHT_MAPPING_SENTINEL", {})
        self.glue_weight_mapping = config.get("GLUE_WEIGHT_MAPPING", {})


TASK_CONFIG = TaskConfig("config.json")


def get_glue_weight_mapping():
    return TASK_CONFIG.glue_weight_mapping


def get_super_glue_weight_mapping():
    return TASK_CONFIG.super_glue_weight_mapping


def get_super_glue_weight_mapping_sentinel():
    return TASK_CONFIG.super_glue_weight_mapping_sentinel


class GLUEPreprocessor:
    def __init__(self, builder_config):
        self.builder_config = builder_config
    
    def __call__(self):
        if self.builder_config.name == "stsb":
            return preprocessors.stsb
        elif self.builder_config.name == "wsc.fixed":
            return preprocessors.wsc
        elif self.builder_config.name == "record":
            return preprocessors.record
        else:
            if "mnli" in self.builder_config.name or self.builder_config.name == "ax":
                benchmark_name = "mnli"
            elif self.builder_config.name in ["axb", "axg"]:
                benchmark_name = "rte"
            else:
                benchmark_name = self.builder_config.name
            if self.builder_config.name == "multirc":
                feature_names = ("question", "answer", "paragraph")
            elif self.builder_config.name == "wic":
                feature_names = ("sentence1", "sentence2", "word")
            else:
                feature_names = None
            return functools.partial(
                preprocessors.glue,
                benchmark_name=benchmark_name,
                label_names=self.builder_config.label_classes,
                feature_names=feature_names
            )


class GLUEPostprocessor:
    def __init__(self, builder_config):
        self.builder_config = builder_config
    
    def __call__(self):
        if self.builder_config.name == "stsb":
            return postprocessors.string_to_float
        elif self.builder_config.name == "multirc":
            return postprocessors.multirc
        elif self.builder_config.name == "record":
            return postprocessors.record
        else:
            return functools.partial(
                postprocessors.string_label_to_class_id,
                label_classes=self.builder_config.label_classes
            )


class GLUEMetrics:
    def __init__(self):
        self.metrics = collections.OrderedDict([
            ("cola", [metrics.sklearn_metrics_wrapper(
                "matthews_corrcoef", metric_post_process_fn=lambda x: 100 * x)]),
            ("sst2", [metrics.accuracy]),
            ("mrpc", [metrics.f1_score_with_invalid, metrics.accuracy]),
            ("stsb", [metrics.pearson_corrcoef, metrics.spearman_corrcoef]),
            ("qqp", [metrics.f1_score_with_invalid, metrics.accuracy]),
            ("mnli", [metrics.accuracy]),
            ("mnli_matched", [metrics.accuracy]),
            ("mnli_mismatched", [metrics.accuracy]),
            ("qnli", [metrics.accuracy]),
            ("rte", [metrics.accuracy]),
            ("wnli", [metrics.accuracy]),
            ("ax", []),  # Only test set available.
        ])
    
    def __getitem__(self, task_name: str):
        return self.metrics[task_name]


class SuperGLUEMetrics:
    def __init__(self):
        self.metrics = collections.OrderedDict([
            ("boolq", [metrics.accuracy]),
            ("cb", [metrics.mean_multiclass_f1(num_classes=3), metrics.accuracy]),
            ("copa", [metrics.accuracy]),
            ("multirc", [
                metrics.multirc_f1_over_all_answers,
                metrics.mean_group_metric(metrics.all_match)
            ]),
            ("record", [metrics.deduplicate_metric(metrics.squad)]),
            ("rte", [metrics.accuracy]),
            ("wic", [metrics.accuracy]),
            ("axb", []),  # Only test set available.
            ("axg", []),  # Only test set available.
        ])
    
    def __getitem__(self, task_name: str):
        return self.metrics[task_name]


GLUE_METRICS = GLUEMetrics()
SUPERGLUE_METRICS = SuperGLUEMetrics()


def get_glue_metric(task_name):
    return GLUE_METRICS[task_name]


def get_super_glue_metric(task_name):
    return SUPERGLUE_METRICS[task_name]