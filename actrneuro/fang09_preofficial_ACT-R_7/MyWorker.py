import numpy
import time

import ConfigSpace as CS
from hpbandster.core.worker import Worker
import os
from evaluate import evaluate

from set_up_run_folder import set_up_run_folder
class MyWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, config, budget, **kwargs):
        """
        Simple example for a compute function
        The loss is just a the config + some noise (that decreases with the budget)

        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """


        runs = int(budget)
        model = "pmm"
        ans = 0.22
        rt = -0.65
        lf = 0.4

        # bold_scale = 0.88
        bold_scale = config["bold-scale"]

        # neg_bold_scale = 1
        neg_bold_scale = config["neg-bold-scale"]

        # bold_exp = 4
        bold_exp = config["bold-exp"]


        # neg_bold_exp = 15
        neg_bold_exp = config["neg-bold-exp"]


        # bold_positive = 1
        bold_positive = config["bold-positive"]

        # bold_negative = 0
        bold_negative = config["bold-negative"]








        # clean run env
        set_up_run_folder()


        # run the model n times
        os.system("ccl64 -n -l -b run_direct.lisp  -- " + str(runs) + " " + str(model) + " " + str(ans) + " " + str(rt) + " " + str(lf) + " " + str(bold_scale) + " " + str(neg_bold_scale) + " " + str(bold_exp) + " " + str(neg_bold_exp) + " " + str(bold_positive) + " " + str(bold_negative))

        print("model run done")
        corr = 1 - evaluate(plot=False)

        os.system("echo \"Loss: " + str(corr) + " with " + str(runs) + " " + str(model) + " " + str(ans) + " " + str(rt) + " " + str(lf) + " " + str(bold_scale) + " " + str(neg_bold_scale) + " " + str(bold_exp) + " " + str(neg_bold_exp) + " " + str(bold_positive) + " " + str(bold_negative) + "\" >> optimizer.log")




        return({
                    'loss': float(corr),  # this is the a mandatory field to run hyperband
                    'info': 1 - corr  # can be used for any user-defined information - also mandatory
                })

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('bold-scale', lower=0, upper=5))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('neg-bold-scale', lower=0, upper=5))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('bold-exp', lower=1, upper=30))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('neg-bold-exp', lower=1, upper=30))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('bold-positive', lower=0, upper=5))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('bold-negative', lower=0, upper=5))
        return(config_space)
