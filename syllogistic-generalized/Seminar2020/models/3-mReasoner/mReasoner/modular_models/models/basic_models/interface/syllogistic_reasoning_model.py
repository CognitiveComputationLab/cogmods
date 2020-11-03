import abc
import itertools


class SyllogisticReasoningModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        # Dict representing current parameter values, e.g. {"param1": 0.4, "param2": "x"}
        self.params = {}

        # Dict representing possible parameter values, e.g. {"param1": [0.0, 0.5, 1.0], "param2": ["x", "y"]}
        self.param_grid = {}

        # Whether or not the model's prediction is stochastic (not deterministic)
        self.is_stochastic = True

    def generate_param_configurations(self):
        return list((dict(zip(self.param_grid, x)) for x in itertools.product(*self.param_grid.values())))

    def set_params(self, params):
        self.params = params

    def get_params(self):
        return self.params

    @abc.abstractmethod
    def predict(self, syllogism, additional_premises=None):
        """ Predict a list of conclusions (like ["Aac", "Iac"]) for a syllogism (like "AA1") """
        raise NotImplementedError()
