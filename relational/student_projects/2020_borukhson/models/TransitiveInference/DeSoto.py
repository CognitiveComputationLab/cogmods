""" Transitive-Inference model implementation.
"""
import ccobra
import random
import numpy as np
from modelfunctions import *



class DeSoto(ccobra.CCobraModel):
    """ TransitivityInt CCOBRA implementation.
    """
    def __init__(self, name='DeSoto-1965'):
        """ Initializes the TransitivityInt model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.spacialPos = {}
        super().__init__(name, ['spatial-relational'], ['single-choice'])

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given syllogism.
        """ 
        #MathematicalImpl; Done, yet other interpretations of spacial representation and insertion actons in this represetation are conceivable.
        first, second = item.choices[0][0][0], item.choices[1][0][0]
        self.putInSpace((first, second))
        if self.spacialPos[first] > self.spacialPos[second]:
            return int(first)
        if self.spacialPos[first] == self.spacialPos[second]:
            return int([first, second][np.random.randint(0, len([first, second]))])
        if self.spacialPos[first] < self.spacialPos[second]:
            return int(second)
    def predictS(self, pair):
        first, second = pair
        self.putInSpace(pair)
        if self.spacialPos[first] > self.spacialPos[second]:
            return int(first)
        if self.spacialPos[first] == self.spacialPos[second]:
            return int([first, second][np.random.randint(0, len([first, second]))])
        if self.spacialPos[first] < self.spacialPos[second]:
            return int(second)
    def adaptS(self, pair):
        self.putInSpace(pair)
    def putInSpace(self, pair):
        first, second = sortedPair(pair)
        if first in self.spacialPos.keys() and second in self.spacialPos.keys():
            if self.spacialPos[first] > self.spacialPos[second]:
                return #all ok
            if self.spacialPos[first] == self.spacialPos[second]:
                self.moveToBothSides((first, second))
            if self.spacialPos[first] < self.spacialPos[second]:
                newPosInMiddle = float(self.spacialPos[first] + self.spacialPos[second])/2
                self.spacialPos[first], self.spacialPos[second] = newPosInMiddle,newPosInMiddle
        if first not in self.spacialPos.keys() or second not in self.spacialPos.keys():
            minValue = 0 if 0 == len([a for a in self.spacialPos.keys()]) else min([self.spacialPos[a] for a in self.spacialPos.keys()])
            maxValue = 0 if 0 == len([a for a in self.spacialPos.keys()]) else max([self.spacialPos[a] for a in self.spacialPos.keys()])
            newPosInMiddle = float(maxValue - minValue)/2 + minValue 
            self.spacialPos[first], self.spacialPos[second] = newPosInMiddle,newPosInMiddle
            self.moveToBothSides((first, second))

    def moveToBothSides(self, pair): #pair on same number
        first, second = pair
        newPositions = self.spacialPos.copy()
        for elem in self.spacialPos.keys():
            if self.spacialPos[elem] < self.spacialPos[first]:
                newPositions[elem] = self.spacialPos[elem] + 0.5
            elif elem == first:
                newPositions[elem] = self.spacialPos[elem] + 0.5
            elif self.spacialPos[elem] > self.spacialPos[second]:
                newPositions[elem] = self.spacialPos[elem] - 0.5
            elif elem == second:
                newPositions[elem] = self.spacialPos[elem] - 0.5
            else:
                newPositions[elem] = self.spacialPos[elem]
        self.spacialPos = newPositions
