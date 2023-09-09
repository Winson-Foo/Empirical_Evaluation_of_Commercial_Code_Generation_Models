import warnings
from functools import wraps


def deprecated(func):
    """This is a decorator which can be used to mark functions as deprecated.
    It will result in a warning being emitted when the function is used."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        return func(*args, **kwargs)

    return wrapper

# usage example:

@deprecated
def some_old_function(x):
    return x + 1