import itertools


class HyperParameter:
    def __init__(self, id, param):
        """
        Represents a hyperparameter value.

        :param id: Identifier for this value.
        :param param: Dictionary of param name to value mapping.
        """
        self.id = id
        self.param = dict()
        for key, item in param:
            self.param[key] = item

    def __str__(self):
        """
        Gets a string representation of this value.

        :return: String representation.
        """
        return str(self.id)

    def dict(self):
        """
        Gets a dictionary representation of this value.

        :return: Dictionary representation.
        """
        return self.param


class HyperParameters:
    def __init__(self, ordered_params):
        """
        Represents a set of hyperparameters.

        :param ordered_params: Ordered dictionary of parameter name to values.
        """
        if not isinstance(ordered_params, OrderedDict):
            raise NotImplementedError
        params = []
        for key in ordered_params.keys():
            param = [[key, iterm] for iterm in ordered_params[key]]
            params.append(param)
        self.params = list(itertools.product(*params))

    def __getitem__(self, index):
        """
        Gets a hyperparameter value at the specified index.

        :param index: Index of the value to retrieve.
        :return: Hyperparameter value.
        """
        return HyperParameter(index, self.params[index])

    def __len__(self):
        """
        Gets the number of hyperparameter values.

        :return: Number of hyperparameter values.
        """
        return len(self.params)