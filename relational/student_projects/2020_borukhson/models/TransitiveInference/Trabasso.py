""" Transitive-Inference model implementation.
"""
import ccobra
import random
import numpy as np

def setEq(a, b):
    for aI in a:
        if aI not in b:
            return False 
    for bI in b:
        if bI not in a:
            return False
    return True

class Trabasso(ccobra.CCobraModel):
    """ TransitivityInt CCOBRA implementation.
    """
    def __init__(self, name='Trabasso-Riley74'):
        """ Initializes the TransitivityInt model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        #Done; various other interpretations conceivable 
        self.intArrs = [] 
        self.h = 0.5
        super().__init__(name, ['spatial-relational'], ['single-choice'])

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given syllogism.
        """ 
        # Generate and return the current prediction
        first, second = item.choices[0][0][0], item.choices[1][0][0]
        for array in self.intArrs:
            if first in array and second in array:
                if array.index(first) < array.index(second):
                    chance = random.random()
                    if chance < array.index(first)/max(0.0001,len(array)*self.h):
                        return int(first)
                    if chance < (len(array) - array.index(second))/max(0.0001,len(array)*self.h):
                        return int(first)
                elif array.index(second) < array.index(first):
                    chance = random.random()
                    if chance < array.index(second)/max(0.0001,len(array)*self.h):
                        return int(second)
                    if chance < (len(array) - array.index(first))/max(0.0001,len(array)*self.h):
                        return int(second)
        return int([first, second][np.random.randint(0, len([first, second]))])
    def predictS(self, pair):
        first, second = pair
        for array in self.intArrs:
            if first in array and second in array:
                if array.index(first) < array.index(second):
                    return max(array.index(first)/max(0.0001,len(array)*self.h), (len(array) - array.index(second))/max(0.0001,len(array)*self.h))
                elif array.index(second) < array.index(first):
                    return max(1 - array.index(second)/max(0.0001,len(array)*self.h), 1 - array.index(first)/max(0.0001,len(array)*self.h))
        return 0.5
    def adaptS(self, pair):
        first, second = self.sortedPair(pair)
        first, second = str(first), str(second)
        pairsFound = 0
        for array in self.intArrs:
            if first in array and second in array:
                pairsFound += 1
                if array.index(first) < array.index(second):
                    return
                if array.index(first) > array.index(second):
                    array[array.index(first)] = second
                    array[array.index(second)] = first
            elif first in array:
                pairsFound += 1
                newArrs = [array[:array.index(first)+1] + [second] + array[array.index(first)+1:]]
                for eachArray in self.intArrs:
                    if not setEq(eachArray, array):
                        newArrs.append(eachArray)
                self.intArrs = newArrs
            elif second in array:
                pairsFound += 1
                newArrs = [array[:array.index(second)] + [first] + array[array.index(second):]]
                for eachArray in self.intArrs:
                    if not setEq(eachArray, array):
                        newArrs.append(eachArray)
                self.intArrs = newArrs
            for otherArray in self.intArrs:
                if array[0] == otherArray[-1]:
                    newArrs = [otherArray[:-1] + array]
                    for eachArray in self.intArrs:
                        if not setEq(eachArray, otherArray):
                            newArrs.append(eachArray)
                    self.intArrs = newArrs
                if array[-1] == otherArray[0]:
                    newArrs = [array[:-1] + otherArray]
                    for eachArray in self.intArrs:
                        if not setEq(eachArray, otherArray):
                            newArrs.append(eachArray)
                    self.intArrs = newArrs
        if pairsFound == 0:
            self.intArrs.append([first, second])

    def adapt(self, item, target, **kwargs):
        first, second = self.sortedPair((item.choices[0][0][0], item.choices[1][0][0]))
        pairsFound = 0
        for array in self.intArrs:
            if first in array and second in array:
                pairsFound += 1
                if array.index(first) < array.index(second):
                    return
                if array.index(first) > array.index(second):
                    array[array.index(first)] = second
                    array[array.index(second)] = first
            elif first in array:
                pairsFound += 1
                newArrs = [array[:array.index(first)+1] + [second] + array[array.index(first)+1:]]
                for eachArray in self.intArrs:
                    if not setEq(eachArray, array):
                        newArrs.append(eachArray)
                self.intArrs = newArrs
            elif second in array:
                pairsFound += 1
                newArrs = [array[:array.index(second)] + [first] + array[array.index(second):]]
                for eachArray in self.intArrs:
                    if not setEq(eachArray, array):
                        newArrs.append(eachArray)
                self.intArrs = newArrs
            for otherArray in self.intArrs:
                if array[0] == otherArray[-1]:
                    newArrs = [otherArray[:-1] + array]
                    for eachArray in self.intArrs:
                        if not setEq(eachArray, otherArray):
                            newArrs.append(eachArray)
                    self.intArrs = newArrs
                if array[-1] == otherArray[0]:
                    newArrs = [array[:-1] + otherArray]
                    for eachArray in self.intArrs:
                        if not setEq(eachArray, otherArray):
                            newArrs.append(eachArray)
                    self.intArrs = newArrs
        if pairsFound == 0:
            self.intArrs.append([first, second])
