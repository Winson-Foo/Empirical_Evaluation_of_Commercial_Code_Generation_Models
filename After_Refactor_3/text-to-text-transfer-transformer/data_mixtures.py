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

"""Add Mixtures to the registry.

This module contains different mixtures for training T5 models.
"""
from t5.data.glue_utils import get_glue_weight_mapping, get_super_glue_weight_mapping, get_super_glue_weight_mapping_sentinel
import seqio
import t5.data

MixtureRegistry = seqio.MixtureRegistry

_GLUE_WEIGHT_MAPPING = get_glue_weight_mapping()
_SUPER_GLUE_WEIGHT_MAPPING = get_super_glue_weight_mapping()
_SUPER_GLUE_WEIGHT_MAPPING_SENTINEL = get_super_glue_weight_mapping_sentinel()

_glue_tasks = list(_GLUE_WEIGHT_MAPPING.keys())
_glue_tasks_with_weight = list(_GLUE_WEIGHT_MAPPING.items())

_wsc_dpr_tasks = [
    "dpr_v001_simple",
    "super_glue_wsc_v102_simple_train",
    "super_glue_wsc_v102_simple_eval",
]

_super_glue_tasks = list(_SUPER_GLUE_WEIGHT_MAPPING.keys())
_super_glue_tasks_with_weight = list(_SUPER_GLUE_WEIGHT_MAPPING.items())
_super_glue_tasks_with_weight_sentinel = list(_SUPER_GLUE_WEIGHT_MAPPING_SENTINEL.items())

_supervised_tasks = _glue_tasks + _super_glue_tasks + [
    "cnn_dailymail_v002",
    "squad_v010_allanswers",
    "wmt_t2t_ende_v003",
    "wmt15_enfr_v003",
    "wmt16_enro_v003"
]

_finetune_tasks = [
    "glue_v002_proportional",
    "super_glue_v102_proportional",
    "cnn_dailymail_v002",
    "squad_v010_allanswers",
    "wmt_t2t_ende_v003",
    "wmt15_enfr_v003",
    "wmt16_enro_v003"
]

MixtureRegistry.add(
    "glue_v002_proportional",
    _glue_tasks_with_weight
)

MixtureRegistry.add(
    "super_glue_v102_proportional",
    _super_glue_tasks_with_weight
)

MixtureRegistry.add(
    "super_glue_v102_proportional_sentinel",
    _super_glue_tasks_with_weight_sentinel
)

MixtureRegistry.add(
    "glue_mnli_and_dev_v002",
    [t for t in _glue_tasks if "mnli" in t],
    default_rate=1.0
)

MixtureRegistry.add(
    "en_mix",
    [("c4_v020_unsupervised", t5.data.rate_unsupervised)] +
    _glue_tasks + _super_glue_tasks +
    ["squad_v010_allanswers"],
    default_rate=t5.data.rate_num_examples
)

MixtureRegistry.add(
    "all_equal",
    _supervised_tasks + ["c4_v020_unsupervised"],
    default_rate=1.,
)

def _dedupe(name):
    rate = None
    if name in _GLUE_WEIGHT_MAPPING:
        rate = _GLUE_WEIGHT_MAPPING[name]
    elif name in _SUPER_GLUE_WEIGHT_MAPPING:
        rate = _SUPER_GLUE_WEIGHT_MAPPING[name]
    if rate is None:
        return t5.data.rate_num_examples
    if "glue" in name and "rte" in name:
        rate *= 0.5
    return rate

MixtureRegistry.add(
    "all_proportional",
    [(t, _dedupe(t)) for t in _supervised_tasks + ["c4_v020_unsupervised"]]
)

MixtureRegistry.add(
    "all_mix",
    [("c4_v020_unsupervised", t5.data.rate_unsupervised)] +
    [(t, _dedupe(t)) for t in _supervised_tasks]
)

for task_name in _finetune_tasks:
    task_names = set(_supervised_tasks + ["c4_v020_unsupervised"])

    if task_name == "glue_v002_proportional":
        task_names -= set(_glue_tasks)
        tasks = [(t, _dedupe(t)) for t in task_names]
    elif task_name == "super_glue_v102_proportional":
        task_names -= set(_super_glue_tasks)
        tasks = [(t, _dedupe(t)) for t in task_names]
    else:
        task_names -= {task_name}
        tasks = [(t, _dedupe(t)) for t in task_names]

    MixtureRegistry.add("leave_one_out_{}".format(task_name), tasks)

_large_translation_tasks = ["wmt_t2t_ende_v003", "wmt15_enfr_v003"]
_large_supervised_tasks = _large_translation_tasks + ["cnn_dailymail_v002"]

MixtureRegistry.add(
    "large_supervised_equal",
    _large_supervised_tasks,
    default_rate=1.0)

MixtureRegistry.add(
    "large_supervised_proportional",
    _large_supervised_tasks,
    default_rate=t5.data.rate_num_examples)

MixtureRegistry.add(
    "large_translation_equal",
    _large_translation_tasks,
    default_rate=1.0)

MixtureRegistry.add(
    "squad_trivia_qa_equal",
    ["squad_v010_allanswers", "trivia_qa_v010"],
    default_rate=1.0)

MixtureRegistry.add(
    "wsc_dpr_simple_proportional",
    [(name, _SUPER_GLUE_WEIGHT_MAPPING[name]) for name in _wsc_dpr_tasks]
)