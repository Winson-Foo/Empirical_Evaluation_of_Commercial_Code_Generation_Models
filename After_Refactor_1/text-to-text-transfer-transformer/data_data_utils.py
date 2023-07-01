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

"""Utilities for data loading and processing."""

import gin
import seqio

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS
DEFAULT_EXTRA_IDS = 100


def get_default_vocabulary() -> seqio.SentencePieceVocabulary:
    return seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS)


# ========================= Mixing Rate Functions ==============================


@gin.configurable
def rate_num_examples(
    task: seqio.Task,
    maximum: int = None,
    temperature: float = 1.0,
    scale: float = 1.0,
    fallback: bool = True,
) -> seqio.MixingRate:
    """Mixing rate equal to the number of examples for the task."""
    return seqio.mixing_rate_num_examples(
        task=task,
        maximum=maximum,
        scale=scale,
        temperature=temperature,
        fallback=fallback,
    )


@gin.configurable
def rate_unsupervised(task: seqio.Task, value: float = 1e6) -> seqio.MixingRate:
    """Gin-configurable mixing rate for the unsupervised co-training task."""
    return value