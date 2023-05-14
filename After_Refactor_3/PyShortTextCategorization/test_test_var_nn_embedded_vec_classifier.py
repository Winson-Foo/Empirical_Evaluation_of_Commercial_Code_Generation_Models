import os
import unittest
import urllib

import shorttext


class TestVarNNEmbeddedVecClassifier(unittest.TestCase):
    W2V_MODEL_FILE = "test_w2v_model.bin"

    def setUp(self):
        self.download_word_embedding_model()
        self.load_word2vec_model()
        self.load_training_data()

    def tearDown(self):
        self.remove_word_embedding_model()

    def download_word_embedding_model(self):
        # Download word-embedding model from s3 bucket
        print("Downloading word-embedding model....")
        link = "https://shorttext-data-northernvirginia.s3.amazonaws.com/trainingdata/test_w2v_model.bin"
        if not os.path.isfile(TestVarNNEmbeddedVecClassifier.W2V_MODEL_FILE):
            urllib.request.urlretrieve(link, TestVarNNEmbeddedVecClassifier.W2V_MODEL_FILE)

    def load_word2vec_model(self):
        # Load word2vec model
        self.w2v_model = shorttext.utils.load_word2vec_model(TestVarNNEmbeddedVecClassifier.W2V_MODEL_FILE, binary=True)

    def load_training_data(self):
        # Load training data
        self.trainclass_dict = shorttext.data.subjectkeywords()

    def remove_word_embedding_model(self):
        # Remove word-embedding model file
        print("Removing word-embedding model")
        if os.path.isfile(TestVarNNEmbeddedVecClassifier.W2V_MODEL_FILE):
            os.remove(TestVarNNEmbeddedVecClassifier.W2V_MODEL_FILE)

    def assert_equal_dict(self, dict1, dict2):
        """Assert that two dictionaries are equal"""
        self.assertTrue(len(dict1)==len(dict2))
        for classlabel in dict1:
            self.assertTrue(classlabel in dict2)
            self.assertAlmostEqual(dict1[classlabel], dict2[classlabel], places=4)

    def test_cnn_word_embed_without_gensim(self):
        # Test CNN
        print("Testing CNN...")
        # Create keras model using `CNNWordEmbed` class
        keras_model = shorttext.classifiers.frameworks.CNNWordEmbed(wvmodel=self.w2v_model,
                                                                    nb_labels=len(self.trainclass_dict.keys()))
        # Create and train classifier using keras model constructed above
        main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model)
        main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)
        # Compute classification score
        score_vals = main_classifier.score('artificial intelligence')
        self.assertAlmostEqual(score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'], 1.0, 1)

    def test_double_cnn_word_embed_without_gensim(self):
        # Test DoubleCNN
        print("Testing DoubleCNN...")
        # Create keras model using `DoubleCNNWordEmbed` class
        keras_model = shorttext.classifiers.frameworks.DoubleCNNWordEmbed(wvmodel=self.w2v_model,
                                                                          nb_labels=len(self.trainclass_dict.keys()))
        # Create and train classifier using keras model constructed above
        main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model)
        main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)
        # Compute classification score
        score_vals = main_classifier.score('artificial intelligence')
        self.assertAlmostEqual(score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'], 1.0, 1)

    def test_clstm_word_embed_without_gensim(self):
        # Test CLSTM
        print("Testing CLSTM...")
        # Create keras model using `CLSTMWordEmbed` class
        keras_model = shorttext.classifiers.frameworks.CLSTMWordEmbed(wvmodel=self.w2v_model,
                                                                      nb_labels=len(self.trainclass_dict.keys()))
        # Create and train classifier using keras model constructed above
        main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model)
        main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)
        # Compute classification score
        score_vals = main_classifier.score('artificial intelligence')
        self.assertAlmostEqual(score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'], 1.0, 1)

    def test_sum_embed(self):
        # Test SumEmbed
        print("Testing SumEmbed")
        classifier = shorttext.classifiers.SumEmbeddedVecClassifier(self.w2v_model)
        classdict = shorttext.data.subjectkeywords()
        classifier.train(classdict)

        # Compute
        self.assert_equal_dict(classifier.score('linear algebra'),
                               {'mathematics': 0.9986082046096036,
                                'physics': 0.9976047404871671,
                                'theology': 0.9923434326310248})
        self.assert_equal_dict(classifier.score('learning'),
                               {'mathematics': 0.998968177605999,
                                'physics': 0.9995439648885027,
                                'theology': 0.9965552994894663})


if __name__ == '__main__':
    unittest.main()