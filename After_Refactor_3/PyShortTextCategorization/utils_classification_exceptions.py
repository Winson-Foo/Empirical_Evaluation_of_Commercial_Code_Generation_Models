# models/exceptions.py

class ModelNotTrainedError(Exception):
    """Raised when a model is used before being trained."""
    

class AlgorithmNotFoundError(Exception):
    """Raised when an algorithm name is not recognized."""
    
    def __init__(self, algorithm_name):
        super().__init__(f"Algorithm '{algorithm_name}' not found.")
        

class WordEmbeddingModelError(Exception):
    """Raised when a word embedding model file is not found."""
    
    def __init__(self, path):
        super().__init__(f"Word embedding model file not found: {path}")
        

class UnequalArrayLengthsError(Exception):
    """Raised when two arrays have different lengths."""
    
    def __init__(self, arr1, arr2):
        super().__init__(f"Arrays have unequal lengths: {len(arr1)} and {len(arr2)}")
        
        
class NotImplementedError(Exception):
    """Raised when a method or function is not implemented."""
    
    def __init__(self, method_name=None):
        message = "Method not implemented." if method_name is None else f"Method '{method_name}' not implemented."
        super().__init__(message)
        

class IncorrectClassificationModelFileError(Exception):
    """Raised when a classification model file name does not match the expected format."""
    
    def __init__(self, expected_name, actual_name):
        super().__init__(f"Incorrect model file name (expected: '{expected_name}', actual: '{actual_name}').")
        
        
class OperationNotDefinedError(Exception):
    """Raised when an operation is not defined for a particular data type."""
    
    def __init__(self, operation_name, data_type):
        super().__init__(f"Operation '{operation_name}' not defined for data type '{data_type}'.")