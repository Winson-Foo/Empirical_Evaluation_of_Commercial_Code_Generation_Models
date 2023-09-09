import unittest
import os
import shorttext
from shorttext.stack import LogisticStackedGeneralization
from shorttext.smartload import smartload_compact_model
from sklearn.svm import SVC

DATA_PATH = '.' # set the data path as a constant

class TestStacking(unittest.TestCase):
    
    def setUp(self):
        self.nih_dict = shorttext.data.nihreports(sample_size=None)
    
    def tearDown(self):
        # Remove all binary files created during testing
        for filepath in os.listdir(DATA_PATH):
            if filepath.endswith('.bin'):
                os.remove(os.path.join(DATA_PATH, filepath))

    def test_training_stacking(self):
        # loading NIH Reports
        nih_dict = {'NCCAM': self.nih_dict['NCCAM'], 'NCATS': self.nih_dict['NCATS']}

        # training classifiers
        maxent_classifier = self.train_maxent_classifier(nih_dict) # maxent classifier
        topicmodeler, topicdisclassifier = self.train_lda_classifier(nih_dict) # LDA classifier
        svm_classifier = self.train_svm_classifier(nih_dict, topicmodeler) # SVM classifier
        stacked_classifier = self.train_stacked_classifier(maxent_classifier, svm_classifier, topicdisclassifier, nih_dict) # stacking classifier

        # save the classifiers
        self.save_classifier(maxent_classifier, 'bio_maxent.bin')
        self.save_classifier(svm_classifier, 'bio_svm.bin')
        self.save_classifier(topicmodeler, 'bio_lda.bin')
        self.save_classifier(stacked_classifier, 'bio_logistics.bin')

    def test_smartload(self):
        # loading NIH Reports
        nih_dict = {'NCCAM': self.nih_dict['NCCAM'], 'NCATS': self.nih_dict['NCATS']}

        # training classifiers
        maxent_classifier, topicmodeler, svm_classifier, stacked_classifier = self.get_saved_classifiers() # get saved classifiers

        # smartload the classifiers
        maxent_classifier2 = smartload_compact_model(os.path.join(DATA_PATH, 'bio_maxent.bin'), None)
        topicmodeler2 = smartload_compact_model(os.path.join(DATA_PATH, 'bio_lda.bin'), None)
        topicdisclassifier2 = shorttext.classifiers.TopicVectorCosineDistanceClassifier(topicmodeler2)
        svm_classifier2 = smartload_compact_model(os.path.join(DATA_PATH, 'bio_svm.bin'), None)
        stacked_classifier2 = LogisticStackedGeneralization({'maxent': maxent_classifier2,
                                                             'svm': svm_classifier2,
                                                             'topiccosine': topicdisclassifier2})
        stacked_classifier2.load_compact_model(os.path.join(DATA_PATH, 'bio_logistics.bin'))

        # compare the classifiers
        terms = ['stem cell', 'grant', 'system biology']
        for term in terms:
            print(term)
            print('maximum entropy')
            self.compare_scores(maxent_classifier.score(term), maxent_classifier2.score(term))
            print('LDA')
            self.compare_scores(topicdisclassifier.score(term), topicdisclassifier2.score(term))
            print('SVM')
            self.compare_scores(svm_classifier.score(term), svm_classifier2.score(term))
            print('combined')
            self.compare_scores(stacked_classifier.score(term), stacked_classifier2.score(term))

    def test_svm(self):
        # loading NIH Reports
        nih_dict = {'NCCAM': self.nih_dict['NCCAM'], 'NCATS': self.nih_dict['NCATS']}

        # training SVM classifier
        topicmodeler, svm_classifier = self.train_svm_classifier(nih_dict, n_topics=16)

        # save the classifier
        self.save_classifier(svm_classifier, 'bio_svm2.bin')

        # smartload the classifier
        svm_classifier2 = smartload_compact_model(os.path.join(DATA_PATH, 'bio_svm2.bin'), None)

        # compare the classifiers
        terms = ['stem cell', 'grant', 'system biology']
        for term in terms:
            print(term)
            topicvec = svm_classifier.getvector(term)
            topicvec2 = svm_classifier2.getvector(term)
            print(topicvec)
            print(topicvec2)
            for idx, classlabel in enumerate(svm_classifier.classlabels):
                print(str(idx)+' '+classlabel)
                print(svm_classifier.classifier.score([topicvec], [idx]))
            for idx, classlabel in enumerate(svm_classifier2.classlabels):
                print(str(idx)+' '+classlabel)
                print(svm_classifier2.classifier.score([topicvec2], [idx]))
            print({classlabel: svm_classifier.classifier.score([topicvec], [idx])
                   for idx, classlabel in enumerate(svm_classifier.classlabels)})
            print({classlabel: svm_classifier2.classifier.score([topicvec], [idx])
                   for idx, classlabel in enumerate(svm_classifier2.classlabels)})

        for term in terms:
            print(term)
            self.compare_scores(svm_classifier.score(term), svm_classifier2.score(term))

    def train_maxent_classifier(self, nih_dict):
        # maxent classifier
        maxent_classifier = shorttext.classifiers.MaxEntClassifier()
        maxent_classifier.train(nih_dict, nb_epochs=100)
        return maxent_classifier

    def train_lda_classifier(self, nih_dict, n_topics=8):
        # LDA classifier
        topicmodeler = shorttext.generators.LDAModeler()
        topicmodeler.train(nih_dict, n_topics)
        topicdisclassifier = shorttext.classifiers.TopicVectorCosineDistanceClassifier(topicmodeler)
        return topicmodeler, topicdisclassifier

    def train_svm_classifier(self, nih_dict, topicmodeler, n_topics=16):
        # SVM classifier
        svm_classifier = shorttext.classifiers.TopicVectorSkLearnClassifier(topicmodeler, SVC())
        svm_classifier.train(nih_dict)
        return topicmodeler, svm_classifier

    def train_stacked_classifier(self, maxent_classifier, svm_classifier, topicdisclassifier, nih_dict):
        # stacking classifier
        stacked_classifier = LogisticStackedGeneralization({'maxent': maxent_classifier,
                                                            'svm': svm_classifier,
                                                            'topiccosine': topicdisclassifier})
        stacked_classifier.train(nih_dict)
        return stacked_classifier

    def save_classifier(self, classifier, filename):
        classifier.save_compact_model(os.path.join(DATA_PATH, filename))

    def get_saved_classifiers(self):
        maxent_classifier = smartload_compact_model(os.path.join(DATA_PATH, 'bio_maxent.bin'), None)
        topicmodeler = smartload_compact_model(os.path.join(DATA_PATH, 'bio_lda.bin'), None)
        topicdisclassifier = shorttext.classifiers.TopicVectorCosineDistanceClassifier(topicmodeler)
        svm_classifier = smartload_compact_model(os.path.join(DATA_PATH, 'bio_svm.bin'), None)
        stacked_classifier = LogisticStackedGeneralization({'maxent': maxent_classifier,
                                                             'svm': svm_classifier,
                                                             'topiccosine': topicdisclassifier})
        stacked_classifier.load_compact_model(os.path.join(DATA_PATH, 'bio_logistics.bin'))

        return maxent_classifier, topicmodeler, svm_classifier, stacked_classifier

    def compare_scores(self, dict1, dict2):
        # compare the scores of two classifiers
        self.assertTrue(len(dict1) == len(dict2))
        for class_label in dict1:
            self.assertTrue(class_label in dict2)
            self.assertAlmostEqual(dict1[class_label], dict2[class_label], places=4)

if __name__ == '__main__':
    unittest.main()