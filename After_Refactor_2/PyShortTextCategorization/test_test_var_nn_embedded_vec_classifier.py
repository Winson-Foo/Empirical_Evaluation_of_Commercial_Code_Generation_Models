import os
import unittest
import urllib.request

import shorttext
import logging

logging.basicConfig(level=logging.INFO)


class TestVarNNEmbeddedVecClassifier(unittest.TestCase):
    w2v_model_file = "test_w2v_model.bin"
    w2v_model_link = "https://shorttext-data-northernvirginia.s3.amazonaws.com/trainingdata/test_w2v_model.bin"
    test_classes = {'mathematics', 'physics', 'theology'}
    trainclass_dict = shorttext.data.subjectkeywords()
    nb_labels = len(trainclass_dict.keys())

    def setUp(self):
        # Download word-embedding model
        logging.info("Downloading word-embedding model....")
        if not os.path.isfile(self.w2v_model_file):
            urllib.request.urlretrieve(self.w2v_model_link, self.w2v_model_file)

        self.w2v_model = shorttext.utils.load_word2vec_model(self.w2v_model_file, binary=True)

    def tearDown(self):
        # Clean up
        logging.info("Removing word-embedding model")
        if os.path.isfile(self.w2v_model_file):
            os.remove(self.w2v_model_file)

    def assertScoreSum(self, scores):
        total_score = sum(scores.values())
        self.assertAlmostEqual(total_score, 1.0, 1)

    def test_CNNWordEmbed_without_Gensim(self):
        # Test CNN
        logging.info("Testing CNN...")
        # create keras model using `CNNWordEmbed` class
        logging.info("Keras model")
        keras_model = shorttext.classifiers.frameworks.CNNWordEmbed(wvmodel=self.w2v_model, nb_labels=self.nb_labels)

        # create and train classifier using keras model constructed above
        logging.info("Training")
        main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model)
        main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)

        # compute classification score
        logging.info("Testing")
        scores = main_classifier.score('artificial intelligence')
        self.assertScoreSum(scores)

    def test_DoubleCNNWordEmbed_without_Gensim(self):
        # Test DoubleCNN
        logging.info("Testing DoubleCNN...")
        # create keras model using `DoubleCNNWordEmbed` class
        logging.info("Keras model")
        keras_model = shorttext.classifiers.frameworks.DoubleCNNWordEmbed(wvmodel=self.w2v_model,
                                                                          nb_labels=self.nb_labels)

        # create and train classifier using keras model constructed above
        logging.info("Training")
        main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model)
        main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)

        # compute classification score
        logging.info("Testing")
        scores = main_classifier.score('artificial intelligence')
        self.assertScoreSum(scores)

    def test_CLSTMWordEmbed_without_Gensim(self):
        # Test CLSTM
        logging.info("Testing CLSTM...")
        # create keras model using `CLSTMWordEmbed` class
        logging.info("Keras model")
        keras_model = shorttext.classifiers.frameworks.CLSTMWordEmbed(wvmodel=self.w2v_model, nb_labels=self.nb_labels)

        # create and train classifier using keras model constructed above
        logging.info("Training")
        main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model)
        main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)

        # compute classification score
        logging.info("Testing")
        scores = main_classifier.score('artificial intelligence')
        self.assertScoreSum(scores)

    def test_SumEmbed(self):
        # Test SumEmbed
        logging.info("Testing SumEmbed")
        classifier = shorttext.classifiers.SumEmbeddedVecClassifier(self.w2v_model)
        class_dict = shorttext.data.subjectkeywords()
        classifier.train(class_dict)

        # compute
        self.assertEqual(classifier.score('linear algebra').keys(), self.test_classes)
        self.assertEqual(classifier.score('big data').keys(), self.test_classes)


if __name__ == '__main__':
    unittest.main()