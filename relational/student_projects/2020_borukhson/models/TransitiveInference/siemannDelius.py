
""" Transitive-Inference model implementation.
"""
import ccobra
import random
from modelfunctions import *

def rewardedStimulus(stim, pair):
    return min([int(a) for a in pair]) == int(stim)

class SiemannDelius(ccobra.CCobraModel):
    """ TransitivityInt CCORA implementation.
    """
    def __init__(self, name='SiemannDelius-Guez2013'):
        """ Initializes the TransitivityInt model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        #Completed, parameters not optimized
        self.bPlus = 0.5            #CONSTANTBETAREWARD
        self.bMinus =0.5            #CONSTANTBETA-NON-REWARD
        self.e = 0.5            #ELEMENTALWEIGHT
        self.elemVinit = 0.0001
        self.confVinit = 0.0001
        self.K = 1 - self.e     #CONFIGURALWEIGHT  
        self.elemV = {}
        self.confV = {}
        super().__init__(name, ['spatial-relational'], ['single-choice'])

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given syllogism.
        """ 
        # Generate and return the current prediction
        pair = item.choices[0][0][0], item.choices[1][0][0]
        prediction = self.siemann_delius_predict(pair)
        return prediction

    def predictS(self, itemPair):
        left, right = itemPair[0], itemPair[1]
        pair = (left, right)
        return self.sd_probability(left, pair)

        prediction = self.siemann_delius_predict(pair)
        return prediction
    def adaptS(self, itemPair):
        left, right = itemPair[0], itemPair[1]
        pair = (left, right)
        first, second = pair
        for element in pair:
            if not element in self.elemV.keys():
                self.elemV[element] = self.elemVinit
        for item in pair:
            if not (item, pair) in self.confV.keys():
                self.confV[item, pair] = self.confVinit
                self.confV[item, (second, first)] = self.confVinit
        if rewardedStimulus(first, pair):
            self.elemV[first] += (self.bPlus* self.elemV[first]*self.sd_probability(first, pair)) * self.e
            self.confV[first, pair] += (self.bPlus * self.confV[first, pair] * self.sd_probability(first, pair) * self.K)
            self.elemV[second] -= (self.bMinus* self.elemV[second]*(1 - self.sd_probability(first, pair))) * self.e
            self.confV[second, pair] = self.confV[first, pair] - (self.bMinus * self.confV[first, pair] * (1 - self.sd_probability(first, pair)) * self.K)
        elif rewardedStimulus(second, pair):
            self.elemV[second] += (self.bPlus* self.elemV[second]*self.sd_probability(second, pair)) * self.e
            self.confV[second, pair] += (self.bPlus * self.confV[second, pair] * self.sd_probability(second, pair) * self.K) 
            self.elemV[first] -= (self.bMinus* self.elemV[first]*(1 - self.sd_probability(second, pair))) * self.e
            self.confV[first, pair] = self.confV[second, pair] - (self.bMinus * self.confV[second, pair] * (1 - self.sd_probability(second, pair)) * self.K)
        
    def adapt(self, item, target, **kwargs):
        pair = item.choices[0][0][0], item.choices[1][0][0]
        first, second = pair
        for element in pair:
            if not element in self.elemV.keys():
                self.elemV[element] = self.elemVinit
        for item in pair:
            if not (item, pair) in self.confV.keys():
                self.confV[item, pair] = self.confVinit
                self.confV[item, (second, first)] = self.confVinit
        if rewardedStimulus(first, pair):
            self.elemV[first] += (self.bPlus* self.elemV[first]*self.sd_probability(first, pair)) * self.e
            self.confV[first, pair] += (self.bPlus * self.confV[first, pair] * self.sd_probability(first, pair) * self.K)
            self.elemV[second] -= (self.bMinus* self.elemV[second]*(1 - self.sd_probability(first, pair))) * self.e
            self.confV[second, pair] = self.confV[first, pair] - (self.bMinus * self.confV[first, pair] * (1 - self.sd_probability(first, pair)) * self.K)
        elif rewardedStimulus(second, pair):
            self.elemV[second] += (self.bPlus* self.elemV[second]*self.sd_probability(second, pair)) * self.e
            self.confV[second, pair] += (self.bPlus * self.confV[second, pair] * self.sd_probability(second, pair) * self.K) 
            self.elemV[first] -= (self.bMinus* self.elemV[first]*(1 - self.sd_probability(second, pair))) * self.e
            self.confV[first, pair] = self.confV[second, pair] - (self.bMinus * self.confV[second, pair] * (1 - self.sd_probability(second, pair)) * self.K)
        
    def siemann_delius_predict(self, pair):
        first, second = pair
        probabilityOfFirst = self.sd_probability(first, pair)
        if random.random() < probabilityOfFirst:
            return int(first)
        return int(second)

    def sd_predict_firstTest(self, item, pair):
        first, second = pair 
        return self.elemV[first]/(self.elemV[first] + self.elemV[second]) if (self.elemV[first] + self.elemV[second]) > self.elemV[first] else int(self.elemV[first] != 0)
    
    def sd_probability(self, item, pair):
        first, second = pair
        for element in pair:
            if not element in self.elemV.keys():
                self.elemV[element] = self.elemVinit
        if not (item, pair) in self.confV.keys():
            return self.sd_predict_firstTest(item, pair)
        return (self.elemV[first] * self.confV[first, pair])/min(0.0001,self.elemV[first] * self.confV[first, pair] + self.elemV[second] * self.confV[second, pair])
