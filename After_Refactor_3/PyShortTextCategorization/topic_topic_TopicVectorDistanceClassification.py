from shorttext.utils import textpreprocessing as txt_prep
from shorttext.generators import (
    AutoencodingTopicModeler,
    GensimTopicModeler,
    LatentTopicModeler,
    load_autoencoder_topicmodel,
    load_gensimtopicmodel,
)


class TopicVecCosineDistanceClassifier:
    """
    A classifier that performs classification based on the cosine similarity between the topic vectors
    of the user-input short texts and various classes.
    """

    def __init__(self, topic_modeler):
        """ Initialize the classifier.

        Args:
            topic_modeler (LatentTopicModeler): A topic modeler that generates topic vectors for short texts.
        """
        self.topic_modeler = topic_modeler

    def score(self, short_text):
        """ Calculate the cosine similarity scores between the topic vector of the short text and
        the topic vectors of all classes.

        Args:
            short_text (str): A short text for classification.

        Returns:
            A dictionary of the cosine similarity scores of the text to all classes.
        """
        return self.topic_modeler.get_batch_cos_similarities(short_text)

    def load_model(self, name_prefix):
        """ Load a topic model with the given prefix of the file paths.

        Given the prefix of the file paths, load the corresponding topic model. The files include a JSON (.json) file
        that specifies various parameters, a gensim dictionary (.gensimdict), and a topic model (.gensimmodel).
        If weighing is applied, load also the tf-idf model (.gensimtfidf).

        Args:
            name_prefix (str): The prefix of the file paths.
        """
        self.topic_modeler.load_model(name_prefix)

    def save_model(self, name_prefix):
        """ Save the model with the given prefix of the file paths.

        Given the prefix of the file paths, save the corresponding topic model. The files include a JSON (.json) file
        that specifies various parameters, a gensim dictionary (.gensimdict), and a topic model (.gensimmodel).
        If weighing is applied, load also the tf-idf model (.gensimtfidf).

        Args:
            name_prefix (str): The prefix of the file paths.

        Raises:
            ModelNotTrainedException: If neither the `train` method nor the `load_model` method was run.
        """
        self.topic_modeler.save_model(name_prefix)

    def load_compact_model(self, name):
        """ Load a compact topic model from files.

        The files for a compact topic model include a gensim dictionary (.gensimdict) and a topic model
        (.gensimmodel).

        Args:
            name (str): The name of the file paths.
        """
        self.topic_modeler.load_compact_model(name)

    def save_compact_model(self, name):
        """ Save the compact topic model to files.

        The files for a compact topic model include a gensim dictionary (.gensimdict) and a topic model
        (.gensimmodel).

        Args:
            name (str): The name of the file paths.
        """
        self.topic_modeler.save_compact_model(name)


def train_gensim_topic_vec_cosine_classifier(
    class_dict,
    num_topics,
    preprocessor=txt_prep.standard_text_preprocessor_1(),
    algorithm="lda",
    to_weigh=True,
    to_normalize=True,
    *args,
    **kwargs,
):
    """ Train a gensim topic model and return a cosine distance classifier.

    Args:
        class_dict (dict): The training data.
        num_topics (int): The number of latent topics.
        preprocessor (function, optional): A function that preprocesses the text.
            Defaults to `shorttext.utils.textpreprocessing.standard_text_preprocessor_1`.
        algorithm (str, optional): The algorithm for topic modeling. Options: lda, lsi, rp.
            Defaults to "lda".
        to_weigh (bool, optional): Whether to weigh the words using tf-idf. Defaults to True.
        to_normalize (bool, optional): Whether the retrieved topic vectors are normalized. Defaults to True.
        args: Arguments to pass to the `train` method for gensim topic models.
        kwargs: Arguments to pass to the `train` method for gensim topic models.

    Returns:
        A classifier that scores the short text based on the topic model.
    """
    # Train the topic model.
    topic_modeler = GensimTopicModeler(
        preprocessor=preprocessor,
        algorithm=algorithm,
        to_weigh=to_weigh,
        to_normalize=to_normalize,
    )
    topic_modeler.train(class_dict, num_topics, *args, **kwargs)

    # Cosine distance classifier.
    return TopicVecCosineDistanceClassifier(topic_modeler)


def load_gensim_topic_vec_cosine_classifier(
    name,
    preprocessor=txt_prep.standard_text_preprocessor_1(),
    is_compact=True,
):
    """ Load a gensim topic model from files and return a cosine distance classifier.

    Given the name or prefix of the files for the topic model, load the corresponding gensim topic model and return
    a classifier that scores the short text based on the model.

    Args:
        name (str): The name (if `is_compact=True`) or prefix (if `is_compact=False`) of the file paths.
        preprocessor (function, optional): A function that preprocesses the text.
            Defaults to `shorttext.utils.textpreprocessing.standard_text_preprocessor_1`.
        is_compact (bool, optional): Whether the model files are compact. Defaults to True.

    Returns:
        A classifier that scores the short text based on the topic model.
    """
    topic_modeler = load_gensimtopicmodel(
        name, preprocessor=preprocessor, compact=is_compact
    )
    return TopicVecCosineDistanceClassifier(topic_modeler)


def train_autoencoder_cosine_classifier(
    class_dict,
    num_topics,
    preprocessor=txt_prep.standard_text_preprocessor_1(),
    to_normalize=True,
    *args,
    **kwargs,
):
    """ Train an autoencoder as a topic model and return a cosine distance classifier.

    Args:
        class_dict (dict): The training data.
        num_topics (int): The number of topics, i.e., number of encoding dimensions.
        preprocessor (function, optional): A function that preprocesses the text.
            Defaults to `shorttext.utils.textpreprocessing.standard_text_preprocessor_1`.
        to_normalize (bool, optional): Whether the retrieved topic vectors are normalized. Defaults to True.
        args: Arguments to be passed to the keras model fitting.
        kwargs: Arguments to be passed to the keras model fitting.

    Returns:
        A classifier that scores the short text based on the autoencoder.
    """
    # Train the autoencoder.
    autoencoder = AutoencodingTopicModeler(
        preprocessor=preprocessor, to_normalize=to_normalize
    )
    autoencoder.train(class_dict, num_topics, *args, **kwargs)

    # Cosine distance classifier.
    return TopicVecCosineDistanceClassifier(autoencoder)


def load_autoencoder_cosine_classifier(
    name, preprocessor=txt_prep.standard_text_preprocessor_1(), is_compact=True
):
    """ Load an autoencoder from files and return a cosine distance classifier.

    Given the name or prefix of the files for the autoencoder, load the corresponding autoencoder and
    return a classifier that scores the short text based on the autoencoder.

    Args:
        name (str): The name (if `is_compact=True`) or prefix (if `is_compact=False`) of the file paths.
        preprocessor (function, optional): A function that preprocesses the text.
            Defaults to `shorttext.utils.textpreprocessing.standard_text_preprocessor_1`.
        is_compact (bool, optional): Whether the model files are compact. Defaults to True.

    Returns:
        A classifier that scores the short text based on the autoencoder.
    """
    autoencoder = load_autoencoder_topicmodel(
        name, preprocessor=preprocessor, compact=is_compact
    )
    return TopicVecCosineDistanceClassifier(autoencoder)