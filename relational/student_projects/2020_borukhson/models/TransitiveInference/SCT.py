""" Transitive-Inference model implementation.
"""
import ccobra
import random
import numpy as np

class SCT(ccobra.CCobraModel):
    """ TransitivityInt CCOBRA implementation.
    """
    def __init__(self, name='SCT-McIlvane2003'): #Stimulus Control Topography
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

    def predictS(self,pair):
        """ Predicts weighted responses to a given syllogism.
        """ 
        # Generate and return the current prediction
        (first, second) = pair
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
        (first, second) = self.sortedPair(pair)
        for elem in [first, second]:
            if elem not in self.valueSCT.keys():
                self.valueSCT[elem] = 0
        case = random.choice(['onlySelect','onlyReject','bothSelectAndReject'])
        if case is 'onlySelect':
            self.valueSCT[first] += 1
        if case is 'onlyReject':
            self.valueSCT[second] -= 1
        if case is 'bothSelectAndReject':
            self.valueSCT[first] += 1
            self.valueSCT[second] -= 1

    
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

    def adaptS(self, pair):
        first, second = self.sortedPair(pair)
        for elem in [first, second]:
            if elem not in self.valueSCT.keys():
                self.valueSCT[elem] = 0
        case = random.choice(['onlySelect','onlyReject','bothSelectAndReject'])
        #print(self.valueSCT)
        if case == 'onlySelect':
            self.valueSCT[first] += 1
        if case == 'onlyReject':
            self.valueSCT[second] -= 1
        if case == 'bothSelectAndReject':
            self.valueSCT[first] += 1
            self.valueSCT[second] -= 1
