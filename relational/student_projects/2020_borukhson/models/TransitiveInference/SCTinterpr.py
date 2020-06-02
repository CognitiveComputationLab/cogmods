""" Transitive-Inference model implementation.
"""
import ccobra
import random
import numpy as np
from modelfunctions import *


class SCTinterpr(ccobra.CCobraModel):
    """ TransitivityInt CCOBRA implementation.
    """
    def __init__(self, name='SCTinterpr-McIlvane2003'): #Stimulus Control Topography
        """ Initializes the TransitivityInt model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.valueSCT = {} #[n.Node('referencePoint')]
        self.lastChoiceSCT = {} #interpretation for tendency
        super().__init__(name, ['spatial-relational'], ['single-choice'])

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given syllogism.
        """ 
        # Generate and return the current prediction
        first, second = item.choices[0][0][0], item.choices[1][0][0]
        for elem in [first, second]:
            if elem not in self.valueSCT.keys():
                self.valueSCT[elem] = 0
        if self.valueSCT[first] > self.valueSCT[second]:
            return int(first)
        if self.valueSCT[first] < self.valueSCT[second]:
            return int(second)
        return int([first, second][np.random.randint(0, len([first, second]))])

    def predictS(self, pair):
        """ Predicts weighted responses to a given syllogism.
        """ 
        # Generate and return the current prediction
        first, second = pair
        for elem in [first, second]:
            if elem not in self.valueSCT.keys():
                self.valueSCT[elem] = 0
        if self.valueSCT[first] > self.valueSCT[second]:
            return 1
        if self.valueSCT[first] < self.valueSCT[second]:
            return 0
        return 0.5

    def adapt(self, item, target, **kwargs):
        pair = item.choices[0][0][0], item.choices[1][0][0]
        first, second = sortedPair(pair)
        for elem in [first, second]:
            if elem not in self.valueSCT.keys():
                self.valueSCT[elem] = 0
        modes = []
        if (first, second) in self.lastChoiceSCT.keys():
            modes.append(self.lastChoiceSCT[(first, second)])
        if first in self.lastChoiceSCT.keys():
            if self.lastChoiceSCT[first] == 'onlyRejectOther':
                modes.append('onlyReject')
            if self.lastChoiceSCT[first] == 'onlySelectMe':
                modes.append('onlySelect')
        if second in self.lastChoiceSCT.keys():
            if self.lastChoiceSCT[second] == 'onlyRejectMe':
                modes.append('onlyReject')
            if self.lastChoiceSCT[second] == 'onlySelectOther':
                modes.append('onlySelect')
        if len(modes) == 0:
            modes = ['onlySelect','onlyReject','bothSelectAndReject']
        case = random.choice(modes)
        if case == 'onlySelect':
            self.valueSCT[first] += 1
            self.lastChoiceSCT[first] = 'onlySelectMe'
            self.lastChoiceSCT[second] = 'onlySelectOther'
        if case == 'onlyReject':
            self.valueSCT[second] -= 1
            self.lastChoiceSCT[second] = 'onlyRejectMe'
            self.lastChoiceSCT[first] = 'onlyRejectOther'
        if case == 'bothSelectAndReject':
            self.lastChoiceSCT[(first, second)] = 'bothSelectAndReject'
            self.valueSCT[first] += 1
            self.valueSCT[second] -= 1
    
    def adaptS(self, pair):
        first, second = sortedPair(pair)
        for elem in [first, second]:
            if elem not in self.valueSCT.keys():
                self.valueSCT[elem] = 0
        modes = []
        if (first, second) in self.lastChoiceSCT.keys():
            modes.append(self.lastChoiceSCT[(first, second)])
        if first in self.lastChoiceSCT.keys():
            if self.lastChoiceSCT[first] == 'onlyRejectOther':
                modes.append('onlyReject')
            if self.lastChoiceSCT[first] == 'onlySelectMe':
                modes.append('onlySelect')
        if second in self.lastChoiceSCT.keys():
            if self.lastChoiceSCT[second] == 'onlyRejectMe':
                modes.append('onlyReject')
            if self.lastChoiceSCT[second] == 'onlySelectOther':
                modes.append('onlySelect')
        if len(modes) == 0:
            modes = ['onlySelect','onlyReject','bothSelectAndReject']
        case = random.choice(modes)
        if case == 'onlySelect':
            self.valueSCT[first] += 1
            self.lastChoiceSCT[first] = 'onlySelectMe'
            self.lastChoiceSCT[second] = 'onlySelectOther'
        if case == 'onlyReject':
            self.valueSCT[second] -= 1
            self.lastChoiceSCT[second] = 'onlyRejectMe'
            self.lastChoiceSCT[first] = 'onlyRejectOther'
        if case == 'bothSelectAndReject':
            self.lastChoiceSCT[(first, second)] = 'bothSelectAndReject'
            self.valueSCT[first] += 1
            self.valueSCT[second] -= 1
