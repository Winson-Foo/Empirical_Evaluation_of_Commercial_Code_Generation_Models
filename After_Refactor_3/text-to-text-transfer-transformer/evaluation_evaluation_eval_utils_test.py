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

"""Utils for evaluation."""

import collections
import os

import numpy as np
import pandas as pd
import seqio
import tensorflow.compat.v1 as tf


def parse_events_files(summary_dir):
    """Parse summary events from a directory."""
    tb_summary_dir = os.path.join(summary_dir, "tb")
    event_files = tf.gfile.Glob(os.path.join(tb_summary_dir, "events.out.tfevents.*"))

    events = collections.defaultdict(list)
    for event_file in event_files:
        for event in tf.train.summary_iterator(event_file):
            for value in event.summary.value:
                events[value.tag].append((event.step, value.simple_value))

    return events


def get_eval_metric_values(events, task_name=None):
    """Extract task-specific eval metric values from events."""
    eval_values = {}
    for tag, values in events.items():
        if task_name is not None:
            if not tag.startswith("eval/") or task_name not in tag:
                continue
            tag = tag[len("eval/"):]
        eval_values[tag] = values
    return eval_values


def compute_avg_glue(df):
    """Compute average GLUE score."""
    glue_metric_names = [metric.name for metric in METRIC_NAMES.values() if metric.name.startswith("glue")]
    super_glue_metric_names = [metric.name for metric in METRIC_NAMES.values() if metric.name.startswith("super")]

    df["Average GLUE Score"] = df[glue_metric_names].mean(axis=1)
    df["Average SuperGLUE Score"] = df[super_glue_metric_names].mean(axis=1)
    df.drop(columns=["CoLA", "Average GLUE Score"], inplace=True, errors="ignore")

    return df


def metric_group_max(df, metric_names):
    """Compute maximum metric value and corresponding step for each metric group."""
    metric_max = df[metric_names].max()
    metric_max_step = df[metric_names].idxmax().astype(int)
    return metric_max, metric_max_step


def log_csv(df, output_file):
    """Log dataframe to a CSV file."""
    df.to_csv(output_file)