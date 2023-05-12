from .utils import standard_text_preprocessor_1
from .utils import compactmodel_io as cio
from .utils import classification_exceptions as e
from .utils import load_DocumentTermMatrix
from .classifiers import (
    load_varnnlibvec_classifier,
    load_sumword2vec_classifier,
    load_autoencoder_topic_sklearnclassifier,
    load_gensim_topicvec_sklearnclassifier,
    load_maxent_classifier
)
from .generators import (
    load_autoencoder_topicmodel,
    load_gensimtopicmodel,
    loadSeq2SeqWithKeras,
    loadCharBasedSeq2SeqGenerator
)
from .spell import loadSCRNNSpellCorrector


def smartload_compact_model(filename, wvmodel=None, preprocessor=standard_text_preprocessor_1(), vecsize=None):
    """
    Load appropriate classifier or model from the binary model.

    :param filename: str, path of the compact model file
    :param wvmodel: gensim.models.keyedvectors.KeyedVectors, Word2Vec model (Default: None)
    :param preprocessor: function, text preprocessor (Default: `shorttext.utils.textpreprocess.standard_text_preprocessor_1`)
    :param vecsize: int, length of embedded vectors in the model (Default: None, extracted directly from the word-embedding model)
    :return: appropriate classifier or model
    :raise: e.AlgorithmNotExistException
    """
    classifier_name = cio.get_model_classifier_name(filename)
    topic_models = ['ldatopic', 'lsitopic', 'rptopic']
    keras_autoencoder = 'kerasautoencoder'
    topic_sklearn = 'topic_sklearn'
    nnlibvec = 'nnlibvec'
    sumvec = 'sumvec'
    maxent = 'maxent'
    dtm = 'dtm'
    keras_seq2seq = 'kerasseq2seq'
    charbase_seq2seq = 'charbases2s'
    scrnn_spell = 'scrnn_spell'

    if classifier_name in topic_models:
        return load_gensimtopicmodel(filename, preprocessor=preprocessor, compact=True)
    elif classifier_name == keras_autoencoder:
        return load_autoencoder_topicmodel(filename, preprocessor=preprocessor, compact=True)
    elif classifier_name == topic_sklearn:
        topicmodel = cio.get_model_config_field(filename, 'topicmodel')
        if topicmodel in topic_models:
            return load_gensim_topicvec_sklearnclassifier(filename, preprocessor=preprocessor, compact=True)
        elif topicmodel == keras_autoencoder:
            return load_autoencoder_topic_sklearnclassifier(filename, preprocessor=preprocessor, compact=True)
        else:
            raise e.AlgorithmNotExistException(topicmodel)
    elif classifier_name == nnlibvec:
        return load_varnnlibvec_classifier(wvmodel, filename, compact=True, vecsize=vecsize)
    elif classifier_name == sumvec:
        return load_sumword2vec_classifier(wvmodel, filename, compact=True, vecsize=vecsize)
    elif classifier_name == maxent:
        return load_maxent_classifier(filename, compact=True)
    elif classifier_name == dtm:
        return load_DocumentTermMatrix(filename, compact=True)
    elif classifier_name == keras_seq2seq:
        return loadSeq2SeqWithKeras(filename, compact=True)
    elif classifier_name == charbase_seq2seq:
        return loadCharBasedSeq2SeqGenerator(filename, compact=True)
    elif classifier_name == scrnn_spell:
        return loadSCRNNSpellCorrector(filename, compact=True)
    else:
        raise e.AlgorithmNotExistException(classifier_name)