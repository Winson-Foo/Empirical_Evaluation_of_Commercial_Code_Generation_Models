import seqio
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

tf.enable_eager_execution()


def check_dataset_shape(ds, sequence_length):
    for k, v in tf.data.get_output_shapes(ds).items():
        feat = k.split("_")[0]
        if len(v) == 0:  # pylint:disable=g-explicit-length-test
            expected_shape = []
        elif feat in sequence_length:
            expected_shape = [sequence_length[feat]]
        else:
            expected_shape = [None]
        assert expected_shape == v.as_list()


def create_mesh_train_dataset(mixture_name, sequence_length):
    dataset_fn = mesh_transformer.mesh_train_dataset_fn
    split = tfds.Split.TRAIN
    ds = dataset_fn(
        mixture_name,
        sequence_length=sequence_length,
        dataset_split=split,
        use_cached=True
    )
    check_dataset_shape(ds, sequence_length)
    return ds


def create_mesh_eval_dataset(mixture_name, sequence_length, use_cached):
    dataset_fn = mesh_transformer.mesh_eval_dataset_fn
    split = tfds.Split.VALIDATION
    ds = dataset_fn(
        mixture_name,
        sequence_length=sequence_length,
        dataset_split=split,
        use_cached=use_cached
    )
    check_dataset_shape(ds, sequence_length)
    return ds