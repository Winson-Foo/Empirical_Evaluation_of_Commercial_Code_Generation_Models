class CustomException(Exception):
    def __init__(self, message):
        self.message = message

class ModelNotTrainedException(CustomException):
    def __init__(self):
        super().__init__('Model not trained.')

class AlgorithmNotExistException(CustomException):
    def __init__(self, algoname):
        super().__init__(f'Algorithm {algoname} not exist.')

class WordEmbeddingModelNotExistException(CustomException):
    def __init__(self, path):
        super().__init__(f'Given path of the word-embedding model not exist: {path}')

class UnequalArrayLengthsException(CustomException):
    def __init__(self, arr1, arr2):
        super().__init__(f'Unequal lengths: {len(arr1)} and {len(arr2)}')

class NotImplementedException(CustomException):
    def __init__(self):
        super().__init__('Method not implemented.')

class IncorrectClassificationModelFileException(CustomException):
    def __init__(self, expectedname, actualname):
        super().__init__(f'Incorrect model (expected: {expectedname}; actual: {actualname})')

class OperationNotDefinedException(CustomException):
    def __init__(self, opname):
        super().__init__(f'Operation {opname} not defined')