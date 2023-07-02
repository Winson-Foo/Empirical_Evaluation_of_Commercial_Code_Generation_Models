import seqio
import t5.data

from t5.data.glue_utils import get_glue_weight_mapping
from t5.data.glue_utils import get_super_glue_weight_mapping
from t5.data.glue_utils import get_super_glue_weight_mapping_sentinel

MixtureRegistry = seqio.MixtureRegistry

_GLUE_WEIGHT_MAPPING = get_glue_weight_mapping()
_SUPER_GLUE_WEIGHT_MAPPING = get_super_glue_weight_mapping()
_SUPER_GLUE_WEIGHT_MAPPING_SENTINEL = get_super_glue_weight_mapping_sentinel()

def _assign_weight_or_rate_num_examples(name):
    if name in _GLUE_WEIGHT_MAPPING:
        return _GLUE_WEIGHT_MAPPING[name]
    elif name in _SUPER_GLUE_WEIGHT_MAPPING:
        return _SUPER_GLUE_WEIGHT_MAPPING[name]
    else:
        return t5.data.rate_num_examples

def _assign_supervised_tasks():
    glue_tasks = list(_GLUE_WEIGHT_MAPPING.keys())
    super_glue_tasks = list(_SUPER_GLUE_WEIGHT_MAPPING.keys())

    supervised_tasks = glue_tasks + super_glue_tasks + [
        "cnn_dailymail_v002",
        "squad_v010_allanswers",
        "wmt_t2t_ende_v003",
        "wmt15_enfr_v003",
        "wmt16_enro_v003"
    ]

    return supervised_tasks

def _prepare_glue_mixture():
    glue_tasks_with_weight = list(_GLUE_WEIGHT_MAPPING.items())

    MixtureRegistry.add(
        "glue_v002_proportional",
        glue_tasks_with_weight
    )

def _prepare_super_glue_mixture():
    super_glue_tasks_with_weight = list(_SUPER_GLUE_WEIGHT_MAPPING.items())
    super_glue_tasks_with_weight_sentinel = list(
        _SUPER_GLUE_WEIGHT_MAPPING_SENTINEL.items())

    MixtureRegistry.add(
        "super_glue_v102_proportional",
        super_glue_tasks_with_weight
    )

    MixtureRegistry.add(
        "super_glue_v102_proportional_sentinel",
        super_glue_tasks_with_weight_sentinel
    )


def _prepare_glue_mnli_and_dev_mixture():
    glue_tasks = _assign_supervised_tasks()
    mnli_tasks = [task for task in glue_tasks if "mnli" in task]

    MixtureRegistry.add(
        "glue_mnli_and_dev_v002",
        mnli_tasks,
        default_rate=1.0
    )


def _prepare_cotraining_mixture():
    supervised_tasks = _assign_supervised_tasks()

    MixtureRegistry.add(
        "en_mix",
        [("c4_v020_unsupervised", t5.data.rate_unsupervised)] +
        supervised_tasks + ["squad_v010_allanswers"],
        default_rate=t5.data.rate_num_examples
    )

    MixtureRegistry.add(
        "all_equal",
        supervised_tasks + ["c4_v020_unsupervised"],
        default_rate=1.,
    )


def _dedupe_tasks(name):
    rate = _assign_weight_or_rate_num_examples(name)

    if name in _GLUE_WEIGHT_MAPPING and "glue" in name and "rte" in name:
        rate *= 0.5

    return rate


def _prepare_all_proportional_mixture():
    supervised_tasks = _assign_supervised_tasks()
    all_proportional_tasks = [(task, _dedupe_tasks(task)) for task in supervised_tasks + ["c4_v020_unsupervised"]]

    MixtureRegistry.add(
        "all_proportional",
        all_proportional_tasks,
    )


def _prepare_leave_one_out_cotrain_mixture():
    supervised_tasks = _assign_supervised_tasks()

    for task_name in supervised_tasks:
        task_names = set(supervised_tasks + ["c4_v020_unsupervised"])

        if task_name == "glue_v002_proportional":
            task_names -= set(glue_tasks)
            tasks = [(task, _assign_weight_or_rate_num_examples(task)) for task in task_names]
        elif task_name == "super_glue_v102_proportional":
            task_names -= set(super_glue_tasks)
            tasks = [(task, _assign_weight_or_rate_num_examples(task)) for task in task_names]
        else:
            task_names -= {task_name}
            tasks = [(task, _dedupe_tasks(task)) for task in task_names]

        MixtureRegistry.add(
            "leave_one_out_{}".format(task_name),
            tasks
        )

def _prepare_pretrain_supervised_mixture():
    large_translation_tasks = ["wmt_t2t_ende_v003", "wmt15_enfr_v003"]
    large_supervised_tasks = large_translation_tasks + ["cnn_dailymail_v002"]

    MixtureRegistry.add(
        "large_supervised_equal",
        large_supervised_tasks,
        default_rate=1.0
    )

    MixtureRegistry.add(
        "large_supervised_proportional",
        large_supervised_tasks,
        default_rate=t5.data.rate_num_examples
    )

    MixtureRegistry.add(
        "large_translation_equal",
        large_translation_tasks,
        default_rate=1.0
    )

def _prepare_squad_trivia_qa_mixture():
    MixtureRegistry.add(
        "squad_trivia_qa_equal",
        ["squad_v010_allanswers", "trivia_qa_v010"],
        default_rate=1.0
    )

def _prepare_wsc_dpr_mixture():
    wsc_dpr_tasks = [
        "dpr_v001_simple",
        "super_glue_wsc_v102_simple_train",
        "super_glue_wsc_v102_simple_eval",
    ]

    MixtureRegistry.add(
        "wsc_dpr_simple_proportional",
        [(name, _SUPER_GLUE_WEIGHT_MAPPING[name]) for name in wsc_dpr_tasks]
    )

def _prepare_mixtures():
    _prepare_glue_mixture()
    _prepare_super_glue_mixture()
    _prepare_glue_mnli_and_dev_mixture()
    _prepare_cotraining_mixture()
    _prepare_all_proportional_mixture()
    _prepare_leave_one_out_cotrain_mixture()
    _prepare_pretrain_supervised_mixture()
    _prepare_squad_trivia_qa_mixture()
    _prepare_wsc_dpr_mixture()

_prepare_mixtures()
