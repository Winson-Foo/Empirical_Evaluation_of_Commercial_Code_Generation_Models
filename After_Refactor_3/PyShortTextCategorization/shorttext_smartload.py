from .generators import (
    load_autoencoder_topicmodel,
    load_gensimtopicmodel,
    loadSeq2SeqWithKeras,
    loadCharBasedSeq2SeqGenerator
)
from .classifiers import (
    load_varnnlibvec_classifier,
    load_sumword2vec_classifier,
    load_autoencoder_topic_sklearnclassifier,
    load_gensim_topicvec_sklearnclassifier,
    load_maxent_classifier
)
from .utils import (
    load_DocumentTermMatrix,
    compactmodel_io as cio,
    classification_exceptions as e,
    standard_text_preprocessor_1 as preprocessor
)
from .spell import loadSCRNNSpellCorrector


def smartload_compact_model(filename, wvmodel=None, preprocessor=preprocessor, vecsize=None):
    """
    Load appropriate classifier or model from the binary model.

    :param filename: path of the compact model file
    :param wvmodel: Word2Vec model (Default: None)
    :param preprocessor: text preprocessor (Default: shorttext.utils.textpreprocess.standard_text_preprocessor_1)
    :param vecsize: length of embedded vectors in the model (Default: None, extracted directly from the word-embedding model)
    :return: appropriate classifier or model
    :raise: AlgorithmNotExistException
    """
    classifier_name = cio.get_model_classifier_name(filename)
    try:
        if classifier_name == 'ldatopic' or classifier_name == 'lsitopic' or classifier_name == 'rptopic':
            return load_gensimtopicmodel(filename, preprocessor=preprocessor, compact=True)
        if classifier_name == 'kerasautoencoder':
            return load_autoencoder_topicmodel(filename, preprocessor=preprocessor, compact=True)
        if classifier_name == 'topic_sklearn':
            topicmodel = cio.get_model_config_field(filename, 'topicmodel')
            try:
                if topicmodel == 'ldatopic' or topicmodel == 'lsitopic' or topicmodel == 'rptopic':
                    return load_gensim_topicvec_sklearnclassifier(filename, preprocessor=preprocessor, compact=True)
                if topicmodel == 'kerasautoencoder':
                    return load_autoencoder_topic_sklearnclassifier(filename, preprocessor=preprocessor, compact=True)
            except e.AlgorithmNotExistException:
                raise e.AlgorithmNotExistException(topicmodel)
        if classifier_name == 'nnlibvec':
            return load_varnnlibvec_classifier(wvmodel, filename, compact=True, vecsize=vecsize)
        if classifier_name == 'sumvec':
            return load_sumword2vec_classifier(wvmodel, filename, compact=True, vecsize=vecsize)
        if classifier_name == 'maxent':
            return load_maxent_classifier(filename, compact=True)
        if classifier_name == 'dtm':
            return load_DocumentTermMatrix(filename, compact=True)
        if classifier_name == 'kerasseq2seq':
            return loadSeq2SeqWithKeras(filename, compact=True)
        if classifier_name == 'charbases2s':
            return loadCharBasedSeq2SeqGenerator(filename, compact=True)
        if classifier_name == 'scrnn_spell':
            return loadSCRNNSpellCorrector(filename, compact=True)
    except e.AlgorithmNotExistException:
        raise e.AlgorithmNotExistException(classifier_name)
    raise e.AlgorithmNotExistException(classifier_name)