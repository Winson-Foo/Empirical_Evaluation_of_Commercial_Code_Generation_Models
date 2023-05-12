import os
import unittest
import urllib

import shorttext


W2V_MODEL_LINK = "https://shorttext-data-northernvirginia.s3.amazonaws.com/trainingdata/test_w2v_model.bin"
W2V_MODEL_FILENAME = "test_w2v_model.bin"


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        print("Downloading word-embedding model....")
        if not os.path.isfile(W2V_MODEL_FILENAME):
            urllib.request.urlretrieve(W2V_MODEL_LINK, W2V_MODEL_FILENAME)
        self.w2v_model = shorttext.utils.load_word2vec_model(W2V_MODEL_FILENAME, binary=True)
        self.trainclass_dict = shorttext.data.subjectkeywords()

    def tearDown(self):
        print("Removing word-embedding model")
        if os.path.isfile(W2V_MODEL_FILENAME):
            os.remove(W2V_MODEL_FILENAME)

    def comparedict(self, dict1, dict2):
        self.assertTrue(len(dict1)==len(dict2))
        print(dict1, dict2)
        for classlabel in dict1:
            self.assertTrue(classlabel in dict2)
            self.assertAlmostEqual(dict1[classlabel], dict2[classlabel], places=4)

    def test_cnn_wordembed(self):
        print("Testing CNN...")
        keras_model = shorttext.classifiers.frameworks.CNNWordEmbed(
            wvmodel=self.w2v_model,
            nb_labels=len(self.trainclass_dict.keys())
        )
        main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model)
        main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)
        score_vals = main_classifier.score('artificial intelligence')
        self.assertAlmostEqual(score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'], 1.0, 1)

    def test_double_cnn_wordembed(self):
        print("Testing DoubleCNN...")
        keras_model = shorttext.classifiers.frameworks.DoubleCNNWordEmbed(
            wvmodel=self.w2v_model,
            nb_labels=len(self.trainclass_dict.keys())
        )
        main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model)
        main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)
        score_vals = main_classifier.score('artificial intelligence')
        self.assertAlmostEqual(score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'], 1.0, 1)

    def test_clstm_wordembed(self):
        print("Testing CLSTM...")
        keras_model = shorttext.classifiers.frameworks.CLSTMWordEmbed(
            wvmodel=self.w2v_model,
            nb_labels=len(self.trainclass_dict.keys())
        )
        main_classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.w2v_model)
        main_classifier.train(self.trainclass_dict, keras_model, nb_epoch=2)
        score_vals = main_classifier.score('artificial intelligence')
        self.assertAlmostEqual(score_vals['mathematics'] + score_vals['physics'] + score_vals['theology'], 1.0, 1)

    def test_sum_embed(self):
        print("Testing SumEmbed")
        classifier = shorttext.classifiers.SumEmbeddedVecClassifier(self.w2v_model)
        classdict = shorttext.data.subjectkeywords()
        classifier.train(classdict)

        # compute
        self.comparedict(classifier.score('linear algebra'),
                         {'mathematics': 0.9986082046096036,
                          'physics': 0.9976047404871671,
                          'theology': 0.9923434326310248})
        self.comparedict(classifier.score('learning'),
                         {'mathematics': 0.998968177605999,
                          'physics': 0.9995439648885027,
                          'theology': 0.9965552994894663})


if __name__ == '__main__':
    unittest.main()