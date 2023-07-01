import seqio

@gin.configurable
def rate_num_examples(
    task, maximum=None, temperature=1.0, scale=1.0,
    fallback_to_num_input_examples=True):
    return seqio.mixing_rate_num_examples(
        task=task, maximum=maximum, scale=scale, temperature=temperature,
        fallback_to_num_input_examples=fallback_to_num_input_examples)


@gin.configurable
def rate_unsupervised(task, value=1e6):
    del task
    return value