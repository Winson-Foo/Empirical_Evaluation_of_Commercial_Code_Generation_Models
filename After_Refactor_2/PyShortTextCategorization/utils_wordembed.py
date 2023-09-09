import numpy as np
import requests
from gensim.models.keyedvectors import KeyedVectors


class RESTfulKeyedVectors(KeyedVectors):
    """ RESTfulKeyedVectors, for connecting to the API of the preloaded word-embedding vectors loaded
        by `WordEmbedAPI`.

        This class inherits from :class:`gensim.models.keyedvectors.KeyedVectors`.

    """
    def __init__(self, url, port='5000'):
        """ Initialize the class.

        :param url: URL of the API, usually `http://localhost`
        :param port: Port number
        :type url: str
        :type port: str
        """
        self.url = url
        self.port = port

    def closer_than(self, entity1, entity2):
        """

        :param entity1: word 1
        :param entity2: word 2
        :type entity1: str
        :type entity2: str
        :return: list of words
        :rtype: list
        """
        r = requests.post(self.url + ':' + self.port + '/closerthan',
                          json={'entity1': entity1, 'entity2': entity2})
        return r.json()

    def distance(self, entity1, entity2):
        """

        :param entity1: word 1
        :param entity2: word 2
        :type entity1: str
        :type entity2: str
        :return: distance between two words
        :rtype: float
        """
        r = requests.post(self.url + ':' + self.port + '/distance',
                          json={'entity1': entity1, 'entity2': entity2})
        return r.json()['distance']

    def distances(self, entity1, other_entities=()):
        """

        :param entity1: word
        :param other_entities: list of words
        :type entity1: str
        :type other_entities: list
        :return: list of distances between `entity1` and each word in `other_entities`
        :rtype: list
        """
        r = requests.post(self.url + ':' + self.port + '/distances',
                          json={'entity1': entity1, 'other_entities': other_entities})
        return np.array(r.json()['distances'], dtype=np.float32)

    def get_vector(self, entity):
        """

        :param entity: word
        :type: str
        :return: word vectors of the given word
        :rtype: numpy.ndarray
        """
        r = requests.post(self.url + ':' + self.port + '/get_vector', json={'token': entity})
        returned_dict = r.json()
        if 'vector' in returned_dict:
            return np.array(returned_dict['vector'])
        else:
            raise KeyError('The token {} does not exist in the model.'.format(entity))

    def most_similar(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        r = requests.post(self.url + ':' + self.port + '/most_similar', json=kwargs)
        return [tuple(pair) for pair in r.json()]

    def most_similar_to_given(self, entity1, entities_list):
        """

        :param entity1: word
        :param entities_list: list of words
        :type entity1: str
        :type entities_list: list
        :return: list of similarities between the given word and each word in `entities_list`
        :rtype: list
        """
        r = requests.post(self.url + ':' + self.port + '/most_similar_to_given',
                          json={'entity1': entity1, 'entities_list': entities_list})
        return r.json()['token']

    def rank(self, entity1, entity2):
        """

        :param entity1: word 1
        :param entity2: word 2
        :type entity1: str
        :type entity2: str
        :return: rank
        :rtype: int
        """
        r = requests.post(self.url + ':' + self.port + '/rank',
                          json={'entity1': entity1, 'entity2': entity2})
        return r.json()['rank']

    def save(self, fname_or_handle, **kwargs):
        """

        :param fname_or_handle:
        :param kwargs:
        :return:
        """
        raise IOError('The class RESTfulKeyedVectors do not persist models to a file.')

    def similarity(self, entity1, entity2):
        """

        :param entity1: word 1
        :param entity2: word 2
        :return: similarity between two words
        :type entity1: str
        :type entity2: str
        :rtype: float
        """
        r = requests.post(self.url + ':' + self.port + '/similarity',
                          json={'entity1': entity1, 'entity2': entity2})
        return r.json()['similarity']