class WeightMappings:
    SUPER_GLUE_WEIGHT_MAPPING = {
        "dpr_v001_simple": 1_322.,
        "super_glue_wsc_v102_simple_train": 259.,
        "super_glue_wsc_v102_simple_eval": 0.,
        "super_glue_boolq_v102": 9_427.,
        "super_glue_cb_v102": 250.,
        "super_glue_copa_v102": 400.,
        "super_glue_multirc_v102": 27_243.,
        "super_glue_record_v102": 138_854.,
        "super_glue_rte_v102": 2_490.,
        "super_glue_wic_v102": 5_428.,
        "super_glue_axb_v102": 0.,
        "super_glue_axg_v102": 0.,
    }

    SUPER_GLUE_WEIGHT_MAPPING_SENTINEL = {
        "dpr_v001_simple_1_sentinel": 1_322.,
        "super_glue_wsc_v102_simple_1_sentinel_train": 259.,
        "super_glue_wsc_v102_simple_1_sentinel_eval": 0.,
        "super_glue_boolq_v102_1_sentinel": 9_427.,
        "super_glue_cb_v102_1_sentinel": 250.,
        "super_glue_copa_v102_1_sentinel": 400.,
        "super_glue_multirc_v102_1_sentinel": 27_243.,
        "super_glue_record_v102_1_sentinel": 138_854.,
        "super_glue_rte_v102_1_sentinel": 2_490.,
        "super_glue_wic_v102_1_sentinel": 5_428.,
        "super_glue_axb_v102_1_sentinel": 0.,
        "super_glue_axg_v102_1_sentinel": 0.,
    }

    GLUE_WEIGHT_MAPPING = {
        "glue_cola_v002": 8_551.,
        "glue_sst2_v002": 67_349.,
        "glue_mrpc_v002": 3_668.,
        "glue_qqp_v002": 363_849.,
        "glue_stsb_v002": 5_749.,
        "glue_mnli_v002": 392_702.,
        "glue_qnli_v002": 104_743.,
        "glue_rte_v002": 2_490.,
        "glue_mnli_mismatched_v002": 0.,
        "glue_mnli_matched_v002": 0.,
        "glue_ax_v002": 0.,
    }

def get_glue_weight_mapping():
    return WeightMappings.GLUE_WEIGHT_MAPPING


def get_super_glue_weight_mapping():
    return WeightMappings.SUPER_GLUE_WEIGHT_MAPPING


def get_super_glue_weight_mapping_sentinel():
    return WeightMappings.SUPER_GLUE_WEIGHT_MAPPING_SENTINEL


def get_glue_metric(task_name):
    return glue_metrics.get(task_name, [])


def get_super_glue_metric(task_name):
    return superglue_metrics.get(task_name, [])

from t5.data.preprocessors import stsb, wsc, record, glue as glue_preprocessor
from t5.data.postprocessors import string_to_float, multirc, record, string_label_to_class_id

# Example usage:
glue_preprocessor = functools.partial(
    glue_preprocessor,
    benchmark_name=benchmark_name,
    label_names=builder_config.label_classes,
    feature_names=feature_names
)

GLUE_METRICS = {
    "cola": [
        metrics.sklearn_metrics_wrapper(
            "matthews_corrcoef",
            metric_post_process_fn=lambda x: 100 * x
        )
    ],
    "sst2": [metrics.accuracy],
    "mrpc": [metrics.f1_score_with_invalid, metrics.accuracy],
    "stsb": [metrics.pearson_corrcoef, metrics.spearman_corrcoef],
    "qqp": [metrics.f1_score_with_invalid, metrics.accuracy],
    "mnli": [metrics.accuracy],
    "mnli_matched": [metrics.accuracy],
    "mnli_mismatched": [metrics.accuracy],
    "qnli": [metrics.accuracy],
    "rte": [metrics.accuracy],
    "wnli": [metrics.accuracy],
    "ax": []  # Only test set available.
}

SUPERGLUE_METRICS = {
    "boolq": [metrics.accuracy],
    "cb": [metrics.mean_multiclass_f1(num_classes=3), metrics.accuracy],
    "copa": [metrics.accuracy],
    "multirc": [
        metrics.multirc_f1_over_all_answers,
        metrics.mean_group_metric(metrics.all_match)
    ],
    "record": [metrics.deduplicate_metric(metrics.squad)],
    "rte": [metrics.accuracy],
    "wic": [metrics.accuracy],
    "axb": [],  # Only test set available.
    "axg": []  # Only test set available.
}