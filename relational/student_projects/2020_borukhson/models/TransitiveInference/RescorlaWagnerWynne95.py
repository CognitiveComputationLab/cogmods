
""" Transitive-Inference model implementation.
"""
import ccobra
import random
import math
from modelfunctions import *


class RescorlaWagnerWynne95(ccobra.CCobraModel):
    """ TransitivityInt CCOBRA implementation.
    """
    def __init__(self, name='RescorlaWagner-Wynne95'):
        """ Initializes the TransitivityInt model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        #Done except for Vz; Vz set to 0 as in Lazareva
        #self.vInit = 'random'   #init associative values
        self.vInit = 0.0001
        self.B = 0.04423            #CONSTANTREWARD
        self.a = 30.6357           #SCALINGPARAMETER
        self.assocV = {}
        self.Vz = 0 #Vz LEFT OUT
        super().__init__(name, ['spatial-relational'], ['single-choice'])


    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given syllogism.
        """ 
        # Generate and return the current prediction
        pair = item.choices[0][0][0], item.choices[1][0][0]
        first, second = pair
        probabilityOfFirst = self.rw_probability(first, pair)
        if random.random() < probabilityOfFirst:
            return int(first)
        return int(second)

    def predictS(self, itemPair):
        first, second = itemPair[0], itemPair[1]
        pair = (str(first), str(second))
        return self.rw_probability(first, pair)
        probabilityOfFirst = self.rw_probability(first, pair)
        if random.random() < probabilityOfFirst:
            return int(first)
    def adapt(self, item, target, **kwargs):
        pair = item.choices[0][0][0], item.choices[1][0][0]
        first, second = sortedPair(pair)
        self.assocVinit(pair)
        self.assocV[first] += self.B*(1-(self.assocV[first] + self.Vz))
        self.assocV[second] -= self.B*(self.assocV[second] + self.Vz)
    def adaptS(self, itemPair):
        left, right = itemPair[0], itemPair[1]
        pair = (left, right)
        first, second = sortedPair(pair)
        self.assocVinit(pair)
        self.assocV[first] += self.B*(1-(self.assocV[first] + self.Vz))
        self.assocV[second] -= self.B*(self.assocV[second] + self.Vz)

    def rw_probability(self, elem, pair):
        first, second = pair
        self.assocVinit(pair)
        first, second = elem, [a for a in pair if a != elem][0]
        r = (self.assocV[first] + self.Vz) / max(self.assocV[first] + self.assocV[second] + 2*self.Vz, 0.00001)
        divisor =math.exp( max(-10,min((self.a*((-2)*r+1)),10)))
        return 1/(1+divisor)

    def assocVinit(self, first, second = None):
        if isinstance(first, tuple):
            first, second = first
        for each in [first, second]:
            if not str(each) in self.assocV.keys() and each != None:
                self.assocV[str(each)] = float(self.vInit)
