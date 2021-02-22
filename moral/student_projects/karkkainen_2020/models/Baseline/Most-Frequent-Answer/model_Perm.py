import numpy as np

import ccobra

mostFrequentAnswer = {
    'AccidentCONTACT' : 'PERMISSIBLE',
    'AccidentNOCONTACT' : 'PERMISSIBLE',
    'FiremanCONTACT' : 'IMPERMISSIBLE',
    'FiremanNOCONTACT' : 'IMPERMISSIBLE' ,
    'MotorboatCONTACT' : 'IMPERMISSIBLE' ,
    'MotorboatNOCONTACT' : 'IMPERMISSIBLE' ,
    'PregnancyCONTACT' : 'PERMISSIBLE' ,
    'PregnancyNOCONTACT' : 'IMPERMISSIBLE',
    'RailroadCONTACT' : 'IMPERMISSIBLE' ,
    'RailroadNOCONTACT' : 'IMPERMISSIBLE' ,
    'ShipCONTACT' : 'PERMISSIBLE' ,
    'ShipNOCONTACT' : 'IMPERMISSIBLE',
    }
    
class MFAModel(ccobra.CCobraModel):
    def __init__(self, name='MFA'):
        super(MFAModel, self).__init__(
            name, ['moral'], ['single-choice'])



    def predict(self, item, **kwargs):
        prediction = mostFrequentAnswer[item.task[0][0] + item.task[0][1]]

        return prediction
