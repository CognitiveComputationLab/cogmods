""" Implementation of the most-frequent answer (MFA) model which predicts responses based on the
most-frequently selected choice from the available background (training) data.

"""

import collections

import ccobra
import numpy as np


class MFAModel(ccobra.CCobraModel):
    def __init__(self, name='MFA'):
        super(MFAModel, self).__init__(name, ['syllogistic-generalized'], ['single-choice'])

        # Initialize member variables
        self.mfa = {}

    def pre_train(self, dataset, **kwargs):
        """ Determine most-frequent answers from the training data.

        """

        mfa = {}

        # Iterate over subjects
        for subject_data in dataset:
            # Iterate over tasks
            for task_data in subject_data:
                # Encode the task
                syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(task_data['item'])
                enc_task = syl.encoded_task
                enc_resp = syl.encode_response(task_data['response'])

                if enc_task not in mfa:
                    mfa[enc_task] = {}

                if enc_resp not in mfa[enc_task]:
                    mfa[enc_task][enc_resp] = 0

                mfa[enc_task][enc_resp] += 1

        self.mfa = mfa

    def pre_train_person(self, dataset, **kwargs):
        """ The MFA model is not supposed to be person-trained.

        """

        pass

    def predict(self, item, **kwargs):
        """ Generate prediction based on the most-frequent answer.

        """

        # Encode the task information
        syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(item)
        task_enc = syl.encoded_task
        enc_choices = [syl.encode_response(x) for x in item.choices]

        # Obtain counts for choices
        best_cnt = -1
        best_resps = []
        for resp, cnt in self.mfa[task_enc].items():
            if cnt < best_cnt:
                continue
            elif resp not in enc_choices:
                continue

            if cnt > best_cnt:
                best_cnt = cnt
                best_resps = []

            best_resps.append(resp)

        # Handle empty resps
        if not best_resps:
            best_resps = enc_choices

        # Extract and return prediction
        pred = np.random.choice(best_resps)
        return syl.decode_response(pred)

    def adapt(self, item, truth, **kwargs):
        """ Continuously adapt the MFA by incrementing responses throughout the experiment.

        """

        syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(item)
        task_enc = syl.encoded_task
        resp_enc = syl.encode_response(truth)

        if task_enc not in self.mfa:
            self.mfa[task_enc] = {}

        if resp_enc not in self.mfa[task_enc]:
            self.mfa[task_enc][resp_enc] = 0

        self.mfa[task_enc][resp_enc] += 1
