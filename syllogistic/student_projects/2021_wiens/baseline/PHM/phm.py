import pandas as pd
import numpy as np

import ccobra

class PHM(ccobra.CCobraModel):
    def __init__(self, name='PHM'):
        super(PHM, self).__init__(
            name, ['syllogistic'], ['verify'])

        pred_df = pd.read_csv('PHM.csv')
        self.predictions = dict(
            zip(
                pred_df['Syllogism'].tolist(),
                [x.split(';') for x in pred_df['Prediction']]))

    def predict(self, item, **kwargs):
        enc_task = ccobra.syllogistic.encode_task(item.task)
        enc_resp = self.predictions[enc_task]
        enc_conclusion = ccobra.syllogistic.encode_response(item.choices[0], item.task)
        enc_resp = enc_resp[np.random.randint(0, len(enc_resp))]
        if enc_conclusion == enc_resp:
            return True
        return False
