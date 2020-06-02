""" Transitive-Inference model implementation.
"""
import ccobra
import random
import math
import numpy as np
from modelfunctions import *


class RLELO_F(ccobra.CCobraModel):
    """ TransitivityInt CCOself.BRA implementation.
    """
    def __init__(self, name='RL_ELO_F-Kumaran2016'):
        """ Initializes the TransitivityInt model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.b = 0.5           #TEMPERATURE
        self.a = 0.525           #LEARNING RATE
        self.s = 0.5           #GAUSSIAN NOISE STANDARD DEVIATION
        self.vInit = 0.001
        self.V = {}             #ranks
        self.lastChosen = None
        super().__init__(name, ['spatial-relational'], ['single-choice'])

    def predict(self, item, **kwargs):
        left, right = int(item.choices[0][0][0]), int(item.choices[1][0][0])
        if random.random() < self.p(0, (left, right)):
            chosen = int(left)
        else:
            chosen= int(right)
        self.lastChosen = chosen
        return chosen
    def predictS(self, itemPair):
        left, right = int(itemPair[0]), int(itemPair[1])
        pair = (left, right)
        return self.p(0, (left, right))
        if random.random() < self.p(0, (left, right)):
            chosen = int(left)
        else:
            chosen= int(right)
        self.lastChosen = chosen
        return chosen
    def adaptS(self, itemPair):
        left, right = int(itemPair[0]), int(itemPair[1])
        if correctReply((left, right)) == str(left):
            self.V[left] = self.a*(1-self.p(0, (left, right))) + self.v(left)
            self.V[right] = (-1)*self.a*(1-self.p(0, (left, right))) + self.v(right)
        elif correctReply((left, right)) == str(right):
            self.V[left] = (-1)*self.a*(1-self.p(1, (left, right))) + self.v(left) 
            self.V[right] = self.a*(1-self.p(1, (left, right)))  + self.v(right)
        else:
            print('error')
        self.V[left] += np.random.normal(0,abs(self.s),1)[0]
        self.V[right] += np.random.normal(0,abs(self.s),1)[0]
        
    def adapt(self, item, target, **kwargs):
        left, right = int(item.choices[0][0][0]), int(item.choices[1][0][0])
        if correctReply((left, right)) == str(left):
            self.V[left] = self.a*(1-self.p(0, (left, right))) + self.v(left)
            self.V[right] = (-1)*self.a*(1-self.p(0, (left, right))) + self.v(right)
        elif correctReply((left, right)) == str(right):
            self.V[left] = (-1)*self.a*(1-self.p(1, (left, right))) + self.v(left) 
            self.V[right] = self.a*(1-self.p(1, (left, right)))  + self.v(right)
        else:
            print('error')
        self.V[left] += np.random.normal(0,abs(self.s),1)[0]
        self.V[right] += np.random.normal(0,abs(self.s),1)[0]

    def p(self, leftOrRight, pair):
        left, right = pair
        if leftOrRight == 0: #left
            exponent = float(max(-10, min(10, -1*self.b*(self.v(left)-self.v(right)))))
            return 1/(1+math.exp(exponent))
        if leftOrRight == 1: #right
            return 1-self.p(0, pair)
        
    def v(self, item):
        if item not in self.V.keys():
            self.V[item] = self.vInit
        return self.V[item]
    
