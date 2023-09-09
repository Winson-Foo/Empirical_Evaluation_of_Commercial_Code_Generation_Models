from shorttext.utils import textpreprocessing as textpreprocess
from shorttext.generators import LatentTopicModeler, GensimTopicModeler, AutoencodingTopicModeler
from shorttext.generators import load_autoencoder_topicmodel, load_gensimtopicmodel


class TopicVecCosineDistanceClassifier:
    """
    This class implements a classifier that performs classification based on the cosine similarity between the topic
    vectors of the user-input short texts and various classes. The topic vectors are calculated using the LatentTopicModeler.
    """

    def __init__(self, topic_modeler):
        """
        Initialize the classifier.

        :param topic_modeler: the topic modeler to use for calculating the topic vectors.
        :type topic_modeler: LatentTopicModeler
        """
        self.topic_modeler = topic_modeler

    def score(self, short_text):
        """
        Calculate the score of the short text against each class labels using cosine similarity.

        :param short_text: the input text to be classified.
        :type short_text: str
        :return: dictionary of scores of the text to all classes
        :rtype: dict
        """
        return self.topic_modeler.get_batch_cos_similarities(short_text)

    def load_model(self, prefix):
        """
        Load the topic model with the given filename prefix.

        Given the filename prefix of the topic model, load the corresponding model.

        :param prefix: prefix of the file paths
        :type prefix: str
        """
        self.topic_modeler.load_model(prefix)

    def save_model(self, prefix):
        """
        Save the model with names according to the prefix.

        Given the filename prefix, save the topic model.

        :param prefix: prefix of the file paths
        :type prefix: str
        """
        self.topic_modeler.save_model(prefix)

    def load_compact_model(self, name):
        self.topic_modeler.load_compact_model(name)

    def save_compact_model(self, name):
        self.topic_modeler.save_compact_model(name)


def train_gensim_topic_vector_cosine_classifier(
        class_dict,
        num_topics,
        preprocessor=textpreprocess.standard_text_preprocessor_1(),
        algorithm='lda',
        to_weigh=True,
        normalize=True,
        *args, **kwargs):
    """
    Train a gensim topic model and return a cosine distance classifier.

    :param class_dict: The dictionary of classes and their instance texts.
    :type class_dict: dict
    :param num_topics: The number of latent topics.
    :type num_topics: int
    :param preprocessor: The function that preprocesses the text.
    :type preprocessor: callable
    :param algorithm: The algorithm for topic modeling.
    :type algorithm: str
    :param to_weigh: Whether to weigh the words using tf-idf.
    :type to_weigh: bool
    :param normalize: Whether the retrieved topic vectors are normalized.
    :type normalize: bool
    :param args: Additional arguments to be passed to the gensim topic model training function.
    :type args: args
    :param kwargs: Additional arguments to be passed to the gensim topic model training function.
    :type kwargs: dict
    :return: A classifier that scores the short text based on the topic model.
    :rtype: TopicVecCosineDistanceClassifier
    """
    # train topic model
    topic_modeler = GensimTopicModeler(preprocessor=preprocessor,
                                       algorithm=algorithm,
                                       toweigh=to_weigh,
                                       normalize=normalize)
    topic_modeler.train(class_dict, num_topics, *args, **kwargs)

    # cosine distance classifier
    return TopicVecCosineDistanceClassifier(topic_modeler)


def load_gensim_topic_vector_cosine_classifier(
        name_prefix,
        preprocessor=textpreprocess.standard_text_preprocessor_1(),
        compact=True):
    """
    Load a gensim topic model from files and return a cosine distance classifier.

    Given the filename prefix of the files of the topic model, return a cosine distance classifier
    based on this model.

    :param name_prefix: The filename prefix of the model files.
    :type name_prefix: str
    :param preprocessor: The function that preprocesses the text.
    :type preprocessor: callable
    :param compact: Whether the model file is compact.
    :type compact: bool
    :return: A classifier that scores the short text based on the topic model.
    :rtype: TopicVecCosineDistanceClassifier
    """
    topic_modeler = load_gensimtopicmodel(name_prefix, preprocessor=preprocessor, compact=compact)
    return TopicVecCosineDistanceClassifier(topic_modeler)


def train_autoencoder_cosine_classifier(
        class_dict,
        num_topics,
        preprocessor=textpreprocess.standard_text_preprocessor_1(),
        normalize=True,
        *args, **kwargs):
    """
    Train an autoencoder as a topic model and return a cosine distance classifier.

    :param class_dict: The dictionary of classes and their instance texts.
    :type class_dict: dict
    :param num_topics: The number of topics, i.e., number of encoding dimensions.
    :type num_topics: int
    :param preprocessor: The function that preprocesses the text.
    :type preprocessor: callable
    :param normalize: Whether the retrieved topic vectors are normalized.
    :type normalize: bool
    :param args: Additional arguments to be passed to the autoencoder model fitting function.
    :type args: args
    :param kwargs: Additional arguments to be passed to the autoencoder model fitting function.
    :type kwargs: dict
    :return: A classifier that scores the short text based on the autoencoder.
    :rtype: TopicVecCosineDistanceClassifier
    """
    # train the autoencoder
    autoencoder = AutoencodingTopicModeler(preprocessor=preprocessor, normalize=normalize)
    autoencoder.train(class_dict, num_topics, *args, **kwargs)

    # cosine distance classifier
    return TopicVecCosineDistanceClassifier(autoencoder)


def load_autoencoder_cosine_classifier(
        name_prefix,
        preprocessor=textpreprocess.standard_text_preprocessor_1(),
        compact=True):
    """
    Load the autoencoder from files for topic modeling, and return a cosine classifier.

    Given the filename prefix of the model files, load the model.

    :param name_prefix: The filename prefix of the model files.
    :type name_prefix: str
    :param preprocessor: The function that preprocesses the text.
    :type preprocessor: callable
    :param compact: Whether the model file is compact.
    :type compact: bool
    :return: A classifier that scores the short text based on the autoencoder.
    :rtype: TopicVecCosineDistanceClassifier
    """
    autoencoder = load_autoencoder_topicmodel(name_prefix, preprocessor=preprocessor, compact=compact)
    return TopicVecCosineDistanceClassifier(autoencoder)