
""" Transitive-Inference model implementation.
"""
import ccobra
import random
import math
from modelfunctions import *


class VTTInt(ccobra.CCobraModel):
    
    def rewardedStimulus(self, pair, stim = None):
        if stim:
            return min([int(a) for a in pair]) == int(stim)
        return min([int(a) for a in pair])

    def orderedPair(self, pair):
        return min([int(a) for a in pair]), max([int(a) for a in pair])

    """ News reasoning CCOself.BRA implementation.
    """
    def __init__(self, name='VTT'):
        """ Initializes the news reasoning model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.a = 0    #WEIGHTINGFACTOR
        self.V = {}
        self.R = {}
        self.itemsLength = 7
        self.items = [1,2,3,4,5,6,7]
        self.make_R()
        super(VTTInt, self).__init__(name, ['spatial-relational'], ['single-choice'])

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given syllogism.
        """ 
        # Generate and return the current prediction
        pair = int(item.choices[0][0][0]), int(item.choices[1][0][0])
        prediction = self.vtt_predict(pair)
        return prediction

    def pre_train(self, dataset):
        pass

    def make_R(self):
        for itemIndex in range(0, self.itemsLength):
            if itemIndex == 0:
                self.R[self.items[itemIndex]] = 2
            elif itemIndex == self.itemsLength - 1:
                self.R[self.items[itemIndex]] = 0
            else:
                self.R[self.items[itemIndex]] = 1

    def getV(self, item):
        if isinstance(item, str):
            print('tried string: ', item, 'correction: ', int(item) - 1 )
            self.getV(int(item) - 1)
        if not item >= 0 and item < self.itemsLength:
            return 0
        if not int(item) in self.V.keys():
            self.V[int(item)] = self.R[int(item)]
        return self.V[int(item)]

    def vtt_predict(self, pair):
        first, second = pair
        probabilityOfFirst = self.vtt_probability(first, pair)
        if random.random() < probabilityOfFirst:
            return int(first)
        return int(second)
    
    def vtt_probability(self, item, pair):
        first, second = pair
        if item != first:
            first, second = second, first
        return abs(self.getV(first) - self.getV(second))

    def rw_probability(self, elem, pair):
        first, second = pair
        self.assocVinit(pair)
        first, second = elem, [a for a in pair if a != elem][0]
        Vz = 0 #Vz LEFT OUT
        r = (self.assocV[first] + Vz) / (self.assocV[first] + self.assocV[second] + Vz)
        return 1 / math.exp((-1*self.a*(2*r-1)))
