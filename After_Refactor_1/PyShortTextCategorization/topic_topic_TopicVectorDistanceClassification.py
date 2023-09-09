import shorttext.utils.textpreprocessing as textpreprocess
from shorttext.generators import (
    LatentTopicModeler,
    GensimTopicModeler,
    AutoencodingTopicModeler,
    load_autoencoder_topicmodel,
    load_gensimtopicmodel,
)

class TopicVecCosineDistanceClassifier:
    """
    This is a class that implements a classifier that perform classification based on
    the cosine similarity between the topic vectors of the user-input short texts and various classes.
    The topic vectors are calculated using :class:`LatentTopicModeler`.
    """
    def __init__(self, topicmodeler):
        """ Initialize the classifier.

        :param topicmodeler: topic modeler
        :type topicmodeler: LatentTopicModeler
        """
        self.topic_modeler = topicmodeler

    def score(self, short_text):
        """ Calculate the score, which is the cosine similarity with the topic vector of the model,
        of the short text against each class labels.

        :param short_text: short text
        :return: dictionary of scores of the text to all classes
        :type short_text: str
        :rtype: dict
        """
        return self.topic_modeler.get_batch_cos_similarities(short_text)

    def load_model(self, name_prefix):
        """ Load the topic model with the given prefix of the file paths.

        Given the prefix of the file paths, load the corresponding topic model. The files
        include a JSON (.json) file that specifies various parameters, a gensim dictionary (.gensimdict),
        and a topic model (.gensimmodel). If weighing is applied, load also the tf-idf model (.gensimtfidf).

        This is essentially loading the topic modeler :class:`LatentTopicModeler`.

        :param name_prefix: prefix of the file paths
        :return: None
        :type name_prefix: str
        """
        self.topic_modeler.loadmodel(name_prefix)

    def save_model(self, name_prefix):
        """ Save the model with names according to the prefix.

        Given the prefix of the file paths, save the corresponding topic model. The files
        include a JSON (.json) file that specifies various parameters, a gensim dictionary (.gensimdict),
        and a topic model (.gensimmodel). If weighing is applied, load also the tf-idf model (.gensimtfidf).

        If neither :func:`~train` nor :func:`~loadmodel` was run, it will raise `ModelNotTrainedException`.

        This is essentially saving the topic modeler :class:`LatentTopicModeler`.

        :param name_prefix: prefix of the file paths
        :return: None
        :raise: ModelNotTrainedException
        :type name_prefix: str
        """
        self.topic_modeler.savemodel(name_prefix)

    def load_compact_model(self, name):
        self.topic_modeler.load_compact_model(name)

    def save_compact_model(self, name):
        self.topic_modeler.save_compact_model(name)


def train_gensim_topic_vec_cosine_classifier(training_data,
                                             num_topics,
                                             preprocessor=textpreprocess.standard_text_preprocessor_1(),
                                             algorithm='lda',
                                             to_weigh=True,
                                             normalize=True,
                                             *args, **kwargs):
    """ Return a cosine distance classifier, i.e., :class:`TopicVecCosineDistanceClassifier`, while
    training a gensim topic model in between.

    :param training_data: training data
    :param num_topics: number of latent topics
    :param preprocessor: function that preprocesses the text. (Default: `utils.textpreprocess.standard_text_preprocessor_1`)
    :param algorithm: algorithm for topic modeling. Options: lda, lsi, rp. (Default: lda)
    :param to_weigh: whether to weigh the words using tf-idf. (Default: True)
    :param normalize: whether the retrieved topic vectors are normalized. (Default: True)
    :param args: arguments to pass to the `train` method for gensim topic models
    :param kwargs: arguments to pass to the `train` method for gensim topic models
    :return: a classifier that scores the short text based on the topic model
    :type training_data: dict
    :type num_topics: int
    :type preprocessor: function
    :type algorithm: str
    :type to_weigh: bool
    :type normalize: bool
    :rtype: TopicVecCosineDistanceClassifier
    """
    # train topic model
    topic_modeler = GensimTopicModeler(
        preprocessor=preprocessor,
        algorithm=algorithm,
        toweigh=to_weigh,
        normalize=normalize
    )
    topic_modeler.train(training_data, num_topics, *args, **kwargs)

    # cosine distance classifier
    return TopicVecCosineDistanceClassifier(topic_modeler)


def load_gensim_topic_vec_cosine_classifier(name,
                                            preprocessor=textpreprocess.standard_text_preprocessor_1(),
                                            compact=True):
    """ Load a gensim topic model from files and return a cosine distance classifier.

    Given the prefix of the files of the topic model, return a cosine distance classifier
    based on this model, i.e., :class:`TopicVecCosineDistanceClassifier`.

    The files include a JSON (.json) file that specifies various parameters, a gensim dictionary (.gensimdict),
    and a topic model (.gensimmodel). If weighing is applied, load also the tf-idf model (.gensimtfidf).

    :param name: name (if compact=True) or prefix (if compact=False) of the file paths
    :param preprocessor: function that preprocesses the text. (Default: `utils.textpreprocess.standard_text_preprocessor_1`)
    :param compact: whether model file is compact (Default: True)
    :return: a classifier that scores the short text based on the topic model
    :type name: str
    :type preprocessor: function
    :type compact: bool
    :rtype: TopicVecCosineDistanceClassifier
    """
    topic_modeler = load_gensimtopicmodel(name, preprocessor=preprocessor, compact=compact)
    return TopicVecCosineDistanceClassifier(topic_modeler)


def train_autoencoder_cosine_classifier(training_data,
                                        num_topics,
                                        preprocessor=textpreprocess.standard_text_preprocessor_1(),
                                        normalize=True,
                                        *args, **kwargs):
    """ Return a cosine distance classifier, i.e., :class:`TopicVecCosineDistanceClassifier`, while
    training an autoencoder as a topic model in between.

    :param classdict: training data
    :param num_topics: number of topics, i.e., number of encoding dimensions
    :param preprocessor: function that preprocesses the text. (Default: `utils.textpreprocess.standard_text_preprocessor_1`)
    :param normalize: whether the retrieved topic vectors are normalized. (Default: True)
    :param args: arguments to be passed to keras model fitting
    :param kwargs: arguments to be passed to keras model fitting
    :return: a classifier that scores the short text based on the autoencoder
    :type training_data: dict
    :type num_topics: int
    :type preprocessor: function
    :type normalize: bool
    :rtype: TopicVecCosineDistanceClassifier
    """
    # train the autoencoder
    autoencoder = AutoencodingTopicModeler(
        preprocessor=preprocessor,
        normalize=normalize
    )
    autoencoder.train(training_data, num_topics, *args, **kwargs)

    # cosine distance classifier
    return TopicVecCosineDistanceClassifier(autoencoder)


def load_autoencoder_cosine_classifier(name,
                                       preprocessor=textpreprocess.standard_text_preprocessor_1(),
                                       compact=True):
    """ Load an autoencoder from files for topic modeling, and return a cosine classifier.

    Given the prefix of the file paths, load the model into files, with name given by the prefix.
    There are files with names ending with "_encoder.json" and "_encoder.h5", which are
    the JSON and HDF5 files for the encoder respectively.
    They also include a gensim dictionary (.gensimdict).

    :param name: name (if compact=True) or prefix (if compact=False) of the file paths
    :param preprocessor: function that preprocesses the text. (Default: `utils.textpreprocess.standard_text_preprocessor_1`)
    :param compact: whether model file is compact (Default: True)
    :return: a classifier that scores the short text based on the autoencoder
    :type name: str
    :type preprocessor: function
    :type compact: bool
    :rtype: TopicVecCosineDistanceClassifier
    """
    autoencoder = load_autoencoder_topicmodel(name, preprocessor=preprocessor, compact=compact)
    return TopicVecCosineDistanceClassifier(autoencoder)