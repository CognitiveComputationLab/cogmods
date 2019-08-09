import copy
import json
import operator
import os
import sys

import ccobra
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..")))
from modular_models.util import sylutil

# For basic model predictions
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prediction_cache/"))


class CCobraWrapper:
    """ Overwrites the methods of ccobra.CCobraModel and makes them available as generic functionality to basic model
    classes. """

    def __init__(self, model):
        # Internal reasoning model (basic model)
        self.model = model()

        # Initial parameter values
        self.default_params = self.model.get_params()

        # Store parameter values and errors obtained in pre_train to reset to for each new subject
        self.pre_train_params = None
        self.pre_train_errors = []

        # List of the models parameter configurations
        self.configurations = self.model.generate_param_configurations()

        # Averaging for stochastic models
        self.prediction_runs = 100 if self.model.is_stochastic else 1

        # "Cached" predictions of the internal model for each syllogism and each parameter configuration for speedup.
        self.predictions = self.generate_predictions()

        # Data structure for storing a prediction error for each parameter configuration
        self.errors = []

        # Number of subjects the model is pre-trained on
        self.n_pre_subjects = 0

        # Factor by which the error from *one* response of the subject the model is adapting to is weighted more than
        # the average error from all pre-training subjects (-> used to determine parameters in adapt step)
        # 1: 1/1 in 1st adaption step, 2/1 in 2nd step, 3/1 in 3rd step, ..., n/1 in nth step (n = # items)
        # 0.5: 1/2 -> 2/2 -> 3/2 -> 4/2 -> 5/2 -> ... -> n/2
        self.extra_weight = 5

    def generate_predictions(self):
        """ Generate all predictions for all configurations of model parameters for storage and quick retrieval """

        # Name of cache file is built from name of internal model
        cache_file = os.path.join(CACHE_DIR + self.model.__class__.__name__ + "_PREDICITON_CACHE.json")

        # Try loading predictions from cache
        try:
            predictions = json.load(open(cache_file, 'r'))
        except (IOError, ValueError):
            predictions = {}

        # No cache available - generate predictions regularly
        if predictions == {}:
            parameters_before = copy.deepcopy(self.model.params)
            predictions = {syllogism: [] for syllogism in ccobra.syllogistic.SYLLOGISMS}
            for syllogism in ccobra.syllogistic.SYLLOGISMS:
                for param_configuration in self.configurations:
                    self.model.set_params(param_configuration)
                    y = [0.0] * 9
                    for _ in range(self.prediction_runs):
                        conclusions = self.model.predict(syllogism)
                        y = list(
                            map(operator.add, y, [1 if c in conclusions else 0 for c in ccobra.syllogistic.RESPONSES]))
                    y = [el / sum(y) for el in y]
                    predictions[syllogism].append(y)

            # Reset parameters
            self.model.set_params(parameters_before)

            # Write cache
            json.dump(predictions, open(cache_file, 'w'))

        return predictions

    @staticmethod
    def compute_error(l1, l2):
        """
        >>> CCobraWrapper.compute_error([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
        0.0
        >>> CCobraWrapper.compute_error([0.1, 0.0, 0.9], [0.0, 0.0, 1.0])
        0.1
        >>> CCobraWrapper.compute_error([0.5, 0.5], [0.5, 0.5])
        0.5
        """
        return sum([x1*x2 for i, x1 in enumerate(l1) for j, x2 in enumerate(l2) if i != j])

    def best_param_configuration(self):
        min_error = min(self.errors)
        return self.configurations[self.errors.index(min_error)]

    def start_participant(self, **kwargs):
        pass

    def pre_train(self, dataset):
        # Only pre-train if the model has parameters in the first place
        if self.configurations == [{}]:
            return

        # Store number of subjects for later as weight for adapt
        self.n_pre_subjects = len(dataset)

        self.errors = []
        data = sylutil.aggregate_data(dataset)
        for params in self.configurations:
            self.model.set_params(params)
            error = 0
            for syllogism in ccobra.syllogistic.SYLLOGISMS:
                y = self.cached_prediction(syllogism)
                data_flat = [data[syllogism][c] for c in ccobra.syllogistic.RESPONSES]
                error += self.compute_error(y, data_flat)
            self.errors.append(error)

        best_configuration = self.best_param_configuration()
        self.model.set_params(best_configuration)
        self.pre_train_params = best_configuration
        self.pre_train_errors = self.errors

    def adapt(self, item, target, **kwargs):
        # Only adapt if the model has parameters in the first place
        if self.configurations == [{}]:
            return

        target_enc = ccobra.syllogistic.encode_response(target, item.task)
        syl_enc = ccobra.syllogistic.Syllogism(item).encoded_task
        for i, params in enumerate(self.configurations):
            y = self.predictions[syl_enc][i]
            y_target = [1.0 if c == target_enc else 0.0 for c in ccobra.syllogistic.RESPONSES]
            error = self.compute_error(y, y_target) * self.n_pre_subjects * self.extra_weight
            self.errors[i] += error
        best_configuration = self.best_param_configuration()
        self.model.set_params(best_configuration)

    def cached_prediction(self, syllogism):
        """ retrieve prediction from cache """

        actual_params = self.model.get_params()
        for i, params in enumerate(self.configurations):
            if params == actual_params:
                return self.predictions[syllogism][i]
        raise Exception

    def predict(self, item, **kwargs):
        syl_enc = ccobra.syllogistic.Syllogism(item).encoded_task
        y = self.cached_prediction(syl_enc)
        conclusion = np.random.choice(ccobra.syllogistic.RESPONSES, 1, p=y)[0]
        return ccobra.syllogistic.decode_response(conclusion, item.task)
