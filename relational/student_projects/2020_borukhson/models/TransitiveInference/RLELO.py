""" Transitive-Inference model implementation.
"""
import ccobra
import random
import math
from modelfunctions import *


class RLELO(ccobra.CCobraModel):
    """ TransitivityInt CCOBRA implementation.
    """
    def __init__(self, name='RL_ELO-Kumaran2016'):
        """ Initializes the TransitivityInt model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.b = 6.80454            #TEMPERATURE
        self.a = 3.71694            #LEARNING RATE
        self.V = {}             #ranks
        self.vInit = 0.001
        self.lastChosen = None
        super().__init__(name, ['spatial-relational'], ['single-choice'])

    def predict(self, item, **kwargs):
        left, right = int(item.choices[0][0][0]), int(item.choices[1][0][0])
        if random.random() < self.p('left', (left, right)):
            chosen = int(left)
        else:
            chosen= int(right)
        self.lastChosen = chosen
        return chosen
    def predictS(self, itemPair):
        left, right = int(itemPair[0]), int(itemPair[1])
        pair = (left, right)
        return self.p('left', (left, right))
        if random.random() < self.p('left', (left, right)):
            chosen = int(left)
        else:
            chosen= int(right)
        self.lastChosen = chosen
        return chosen
    def adapt(self, item, target, **kwargs):
        left, right = int(item.choices[0][0][0]), int(item.choices[1][0][0])
        if correctReply((left, right)) == str(left):
            self.V[left] = self.a*(1-self.p('left', (left, right))) + self.v(left)
            self.V[right] = (-1)*self.a*(1-self.p('left', (left, right))) + self.v(right)
        elif correctReply((left, right)) == str(right):
            self.V[left] = (-1)*self.a*(1-self.p('right', (left, right))) + self.v(left) 
            self.V[right] = self.a*(1-self.p('right', (left, right)))  + self.v(right)
        else:
            print('error')
    def adaptS(self, itemPair):
        left, right = int(itemPair[0]), int(itemPair[1])
        if correctReply((left, right)) == str(left):
            self.V[left] = self.a*(1-self.p('left', (left, right))) + self.v(left)
            self.V[right] = (-1)*self.a*(1-self.p('left', (left, right))) + self.v(right)
        elif correctReply((left, right)) == str(right):
            self.V[left] = (-1)*self.a*(1-self.p('right', (left, right))) + self.v(left) 
            self.V[right] = self.a*(1-self.p('right', (left, right)))  + self.v(right)
        else:
            print('error')        
    def p(self, leftOrRight, pair):
        left, right = pair
        if leftOrRight == 'left' : #left
            return 1/(1+math.exp(max(-10,min(10,-1*self.b*(self. v(left)-self.v(right))))))
        if leftOrRight == 'right' : #right
            return 1-self.p('left', pair)
        
    def v(self, item):
        #print(self.V)
        if item not in self.V.keys():
            self.V[item] = self.vInit
        return self.V[item]
    
