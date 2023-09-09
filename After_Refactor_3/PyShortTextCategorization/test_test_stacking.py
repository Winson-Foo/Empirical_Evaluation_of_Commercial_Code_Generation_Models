import unittest
import os

import shorttext
from shorttext.stack import LogisticStackedGeneralization
from shorttext.smartload import smartload_compact_model
from sklearn.svm import SVC

class ClassifierTrainer:
    def __init__(self, nihdict):
        self.nihdict = nihdict

    def train_maxent_classifier(self):
        maxent_classifier = shorttext.classifiers.MaxEntClassifier()
        maxent_classifier.train(self.nihdict, nb_epochs=100)
        return maxent_classifier

    def train_topic_modeler(self):
        topicmodeler = shorttext.generators.LDAModeler()
        topicmodeler.train(self.nihdict, 8)
        return topicmodeler

    def train_svm_classifier(self, topicmodeler):
        svm_classifier = shorttext.classifiers.TopicVectorSkLearnClassifier(topicmodeler, SVC())
        svm_classifier.train(self.nihdict)
        return svm_classifier

    def train_stacked_generalization(self, maxent_classifier, svm_classifier, topicdisclassifier):
        stacked_classifier = LogisticStackedGeneralization({
            'maxent': maxent_classifier,
            'svm': svm_classifier,
            'topiccosine': topicdisclassifier
        })
        stacked_classifier.train(self.nihdict)
        return stacked_classifier

class TestStacking(unittest.TestCase):
    def setUp(self):
        self.nihdict = shorttext.data.nihreports(sample_size=None)
        self.trainer = ClassifierTrainer(self.nihdict)

    def tearDown(self):
        for filepath in os.listdir('.'):
            if filepath.endswith('.bin'):
                os.remove(os.path.join('.', filepath))

    def test_stacking(self):
        maxent_classifier = self.trainer.train_maxent_classifier()
        maxent_classifier.save_compact_model('./bio_maxent.bin')

        topicmodeler = self.trainer.train_topic_modeler()
        topicdisclassifier = shorttext.classifiers.TopicVectorCosineDistanceClassifier(topicmodeler)
        topicmodeler.save_compact_model('./bio_lda.bin')

        svm_classifier = self.trainer.train_svm_classifier(topicmodeler)
        svm_classifier.save_compact_model('./bio_svm.bin')

        stacked_classifier = self.trainer.train_stacked_generalization(maxent_classifier, svm_classifier, topicdisclassifier)
        stacked_classifier.save_compact_model('./bio_logistics.bin')

        self.assertIn('maxent', stacked_classifier.classifiers)
        self.assertIn('svm', stacked_classifier.classifiers)
        self.assertIn('topiccosine', stacked_classifier.classifiers)

    def comparedict(self, dict1, dict2):
        self.assertTrue(len(dict1)==len(dict2))
        for classlabel in dict1:
            self.assertTrue(classlabel in dict2)
            self.assertAlmostEqual(dict1[classlabel], dict2[classlabel], places=4)

    def test_stacked_classifiers(self):
        self.test_stacking()

        maxent_classifier2 = smartload_compact_model('./bio_maxent.bin', None)
        topicmodeler2 = smartload_compact_model('./bio_lda.bin', None)
        topicdisclassifier2 = shorttext.classifiers.TopicVectorCosineDistanceClassifier(topicmodeler2)
        svm_classifier2 = smartload_compact_model('./bio_svm.bin', None)
        stacked_classifier2 = LogisticStackedGeneralization({
            'maxent': maxent_classifier2,
            'svm': svm_classifier2,
            'topiccosine': topicdisclassifier2
        })
        stacked_classifier2.load_compact_model('./bio_logistics.bin')

        terms = ['stem cell', 'grant', 'system biology']
        for term in terms:
            self.comparedict(maxent_classifier2.score(term), maxent_classifier.score(term))
            self.comparedict(topicdisclassifier2.score(term), topicdisclassifier.score(term))
            self.comparedict(svm_classifier2.score(term), svm_classifier.score(term))
            self.comparedict(stacked_classifier2.score(term), stacked_classifier.score(term))

    def test_svm_classifier(self):
        nihdict = {'NCCAM': self.nihdict['NCCAM'], 'NCATS': self.nihdict['NCATS']}
        topicmodeler = shorttext.generators.LDAModeler()
        topicmodeler.train(nihdict, 16)
        svm_classifier = shorttext.classifiers.TopicVectorSkLearnClassifier(topicmodeler, SVC())
        svm_classifier.train(nihdict)
        svm_classifier.save_compact_model('./bio_svm2.bin')
        svm_classifier2 = smartload_compact_model('./bio_svm2.bin', None)

        terms = ['stem cell', 'grant', 'system biology']
        for term in terms:
            topicvec = svm_classifier.getvector(term)
            topicvec2 = svm_classifier2.getvector(term)

            for idx, classlabel in enumerate(svm_classifier.classlabels):
                self.assertAlmostEqual(svm_classifier.classifier.score([topicvec], [idx]), svm_classifier2.classifier.score([topicvec2], [idx]), places=4)

if __name__ == '__main__':
    unittest.main()