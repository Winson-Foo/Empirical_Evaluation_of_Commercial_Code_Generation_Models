class ModelNotTrainedException(Exception):
    """Raised when a model is called before training"""
    def __init__(self):
        self.message = 'Model not trained.'


class AlgorithmNotExistException(Exception):
    """Raised when a non-existent algorithm is called"""
    def __init__(self, algoname):
        self.message = f'Algorithm {algoname} not exist.'


class WordEmbeddingModelNotExistException(Exception):
    """Raised when a non-existent path is given for the word-embedding model"""
    def __init__(self, path):
        self.message = f'Given path of the word-embedding model does not exist: {path}.'


class UnequalArrayLengthsException(Exception):
    """Raised when two arrays have unequal lengths"""
    def __init__(self, arr1, arr2):
        self.message = f'Unequal lengths: {len(arr1)} and {len(arr2)}.'


class NotImplementedException(Exception):
    """Raised when a method is not yet implemented"""
    def __init__(self):
        self.message = 'Method not implemented.'


class IncorrectClassificationModelFileException(Exception):
    """Raised when a model file does not match the expected name"""
    def __init__(self, expectedname, actualname):
        self.message = f'Incorrect model file (expected: {expectedname}; actual: {actualname}).'


class OperationNotDefinedException(Exception):
    """Raised when an operation is not defined"""
    def __init__(self, opname):
        self.message = f'Operation {opname} not defined.'