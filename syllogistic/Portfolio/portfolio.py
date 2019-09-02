import ccobra

import os
import collections

import pandas as pd

class Portfolio(ccobra.CCobraModel):
    def __init__(self, name='Portfolio', modellist=None):
        super(Portfolio, self).__init__(name, ['syllogistic'], ['single-choice'])

        if not modellist:
            modellist = os.listdir('models')

        # Load the model predictions
        self.predictions = {}
        self.scores = {}
        for modelfile in modellist:
            df = pd.read_csv('models' + os.sep + modelfile)
            pred_dict = dict(zip(df['Syllogism'], df['Prediction'].apply(lambda x: x.split(';'))))

            self.predictions[modelfile] = pred_dict
            self.scores[modelfile] = 0


    def pre_train(self, dataset, **kwargs):
        for subj_data in dataset:
            for task_data in subj_data:
                self.adapt(task_data['item'], task_data['response'], weight=(1 / (len(subj_data) * 5)))

    def person_train(self, data, **kwargs):
        pass

    def predict(self, item, **kwargs):
        # Obtain the best predictor
        best_models = [list(self.scores.keys())[0]]
        for model, score in self.scores.items():
            top_score = self.scores[best_models[0]]
            if score > top_score:
                best_models = [model]
            elif score == top_score:
                best_models.append(model)

        # Process syllogism
        syllogism = ccobra.syllogistic.Syllogism(item)

        # Generate voting of responses
        votes = []
        for model in best_models:
            votes.extend(self.predictions[model][syllogism.encoded_task])

        counts = dict(collections.Counter(votes))
        ranking = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        return syllogism.decode_response(ranking[0][0])

    def adapt(self, item, target, weight=1, **kwargs):
        # Process syllogism
        syllogism = ccobra.syllogistic.Syllogism(item)
        enc_target = syllogism.encode_response(target)

        for model, predictions in self.predictions.items():
            modelpred = predictions[syllogism.encoded_task]
            if enc_target in modelpred:
                self.scores[model] += (weight / len(modelpred))
