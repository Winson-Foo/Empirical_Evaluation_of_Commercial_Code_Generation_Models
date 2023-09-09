from .utils import (
    standard_text_preprocessor_1,
    compactmodel_io as cio,
    classification_exceptions as e,
    load_DocumentTermMatrix,
)
from .classifiers import (
    load_varnnlibvec_classifier,
    load_sumword2vec_classifier,
    load_autoencoder_topic_sklearnclassifier,
    load_gensim_topicvec_sklearnclassifier,
    load_maxent_classifier,
)
from .generators import (
    load_autoencoder_topicmodel,
    load_gensimtopicmodel,
    loadSeq2SeqWithKeras,
    loadCharBasedSeq2SeqGenerator,
)
from .spell import loadSCRNNSpellCorrector


def smartload_compact_model(filename, word2vec_model, preprocessor=standard_text_preprocessor_1(), vecsize=None):
    """
    Load appropriate classifier or model from the binary model.

    :param filename: path of the compact model file
    :param word2vec_model: Word2Vec model, can be set to None if not needed
    :param preprocessor: text preprocessor (Default: `shorttext.utils.textpreprocess.standard_text_preprocessor_1`)
    :param vecsize: length of embedded vectors in the model (Default: None, extracted directly from the word-embedding model)
    :return: appropriate classifier or model
    :raise: AlgorithmNotExistException
    """
    model_classifier = cio.get_model_classifier_name(filename)

    # Load appropriate classifier or model based on the model/classifier name
    if model_classifier == 'ldatopic' or model_classifier == 'lsitopic' or model_classifier == 'rptopic':
        return load_gensimtopicmodel(filename, preprocessor=preprocessor, compact=True)
    elif model_classifier == 'kerasautoencoder':
        return load_autoencoder_topicmodel(filename, preprocessor=preprocessor, compact=True)
    elif model_classifier == 'topic_sklearn':
        topic_model = cio.get_model_config_field(filename, 'topicmodel')
        if topic_model == 'ldatopic' or topic_model == 'lsitopic' or topic_model == 'rptopic':
            return load_gensim_topicvec_sklearnclassifier(filename, preprocessor=preprocessor, compact=True)
        elif topic_model == 'kerasautoencoder':
            return load_autoencoder_topic_sklearnclassifier(filename, preprocessor=preprocessor, compact=True)
        else:
            raise e.AlgorithmNotExistException(topic_model)
    elif model_classifier == 'nnlibvec':
        return load_varnnlibvec_classifier(word2vec_model, filename, compact=True, vecsize=vecsize)
    elif model_classifier == 'sumvec':
        return load_sumword2vec_classifier(word2vec_model, filename, compact=True, vecsize=vecsize)
    elif model_classifier == 'maxent':
        return load_maxent_classifier(filename, compact=True)
    elif model_classifier == 'dtm':
        return load_DocumentTermMatrix(filename, compact=True)
    elif model_classifier == 'kerasseq2seq':
        return loadSeq2SeqWithKeras(filename, compact=True)
    elif model_classifier == 'charbases2s':
        return loadCharBasedSeq2SeqGenerator(filename, compact=True)
    elif model_classifier == 'scrnn_spell':
        return loadSCRNNSpellCorrector(filename, compact=True)
    else:
        raise e.AlgorithmNotExistException(model_classifier)