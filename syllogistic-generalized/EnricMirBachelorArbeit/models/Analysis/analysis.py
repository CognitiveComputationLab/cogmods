""" Implementation of the most-frequent answer (MFA) model which predicts responses based on the
most-frequently selected choice from the available background (training) data.

"""

import collections

import ccobra
import numpy as np


class AnalysisModel(ccobra.CCobraModel):
    def __init__(self, name='Analysis'):
        super(AnalysisModel, self).__init__(name, ['syllogistic-generalized'], ['single-choice'])

    def check_key_exists(self, dict, key):
        try:
            value = dict[key]
            return value
        except KeyError:
            return ''

    def encode_task(self, task):
        map = {
            'All' : 'A',
            'Most': 'T',
            'Most not': 'D',
            'Some': 'I',
            'Some not': 'O',
            'Few': 'B',
            'Few not': 'T',
            'No': 'E'
        }

        return map[task[0]] + task[1].lower() + task[2].lower()

    def pre_train(self, dataset, **kwargs):
        """ Here we analyse the dataset

        """

        answers = {}
        total_answers = 0
        total_nvc = 0
        for subject_data in dataset:
            for task_data in subject_data:
                syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(task_data['item'])
                enc_task = syl.encoded_task
                enc_resp = syl.encode_response(task_data['response'])
                if enc_task not in answers:
                     p1 = syl.p1
                     p2 = syl.p2
                     p1[1] = 'A' if syl.A == p1[1] else 'B' if syl.B == p1[1] else 'C'
                     p1[2] = 'A' if syl.A == p1[2] else 'B' if syl.B == p1[2] else 'C'
                     p2[1] = 'A' if syl.A == p2[1] else 'B' if syl.B == p2[1] else 'C'
                     p2[2] = 'A' if syl.A == p2[2] else 'B' if syl.B == p2[2] else 'C'
                     answers[enc_task] = {'dec_task': self.encode_task(syl.p1) + ',' + self.encode_task(syl.p2)}
                if enc_resp not in answers[enc_task]:
                     answers[enc_task][enc_resp] = 0
                answers[enc_task][enc_resp] += 1
                total_answers += 1
                if (enc_resp == 'NVC'): total_nvc += 1

        print('Total:' + str(total_answers) + ' nvc: ' + str(total_nvc))
        my_string = "{}       & {:<2}  & {:<2}  & {:<2}  & {:<2}  & {:<2}  & {:<2}  & {:<2}  & {:<2}  & {:<2}  & {:<2}  & {:<2}  & {:<2}  & {:<2}  & {:<2} \\\\"
        for answer in sorted(answers):
            task = answer
            dec_task  = answers[answer]['dec_task']
            aac = self.check_key_exists(answers[answer], 'Aac')
            tac = self.check_key_exists(answers[answer], 'Tac')
            iac = self.check_key_exists(answers[answer], 'Iac')
            oac = self.check_key_exists(answers[answer], 'Oac')
            dac = self.check_key_exists(answers[answer], 'Dac')
            eac = self.check_key_exists(answers[answer], 'Eac')
            aca = self.check_key_exists(answers[answer], 'Aca')
            tca = self.check_key_exists(answers[answer], 'Tca')
            ica = self.check_key_exists(answers[answer], 'Ica')
            oca = self.check_key_exists(answers[answer], 'Oca')
            dca = self.check_key_exists(answers[answer], 'Dca')
            eca = self.check_key_exists(answers[answer], 'Eca')
            nvc = self.check_key_exists(answers[answer], 'NVC')

            #print(my_string.format(task, dec_task, aac, tac, iac, oac, dac, eac, aca, tca, ica, oca, dca, eca, nvc))
        return

    def pre_train_person(self, dataset, **kwargs):
        """

        """

        pass

    def predict(self, item, **kwargs):
        """ Generate prediction based on the most-frequent answer.

        """

        return 'NVC'

    def adapt(self, item, truth, **kwargs):
        """ Analysis adapt

        """
