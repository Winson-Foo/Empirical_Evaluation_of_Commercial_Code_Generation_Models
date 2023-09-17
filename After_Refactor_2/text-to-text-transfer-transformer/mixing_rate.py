import gin
import seqio
from utils import Vocabulary

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS
DEFAULT_EXTRA_IDS = 100


@gin.configurable
def rate_num_examples(
    task, maximum=None, temperature=1.0, scale=1.0,
    fallback_to_num_input_examples=True):
    """Mixing rate equal to the number of examples for the task."""
    return seqio.mixing_rate_num_examples(
        task=task, maximum=maximum, scale=scale, temperature=temperature,
        fallback_to_num_input_examples=fallback_to_num_input_examples)


@gin.configurable
def rate_unsupervised(task, value=1e6):
    """Gin-configurable mixing rate for the unsupervised co-training task."""
    del task
    return value