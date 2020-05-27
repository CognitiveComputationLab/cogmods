""" Transitive-Inference model implementation.
"""
import ccobra
import random
import math

class BushMosteller(ccobra.CCobraModel):
    """ TransitivityInt CCOBRA implementation.
    """
    def __init__(self, name='Bush-Mosteller-Wynne95', commands = []):
        """ Initializes the TransitivityInt model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.Db = 0.125             #rate parameter for the effect of nonreward.
        self.V = {}             #ranks
        self.vInit = 0.001
        self.lastChosen = None
        super().__init__(name, ['spatial-relational'], ['single-choice'], commands)

    def predict(self, item, **kwargs):
        left, right = int(item.choices[0][0][0]), int(item.choices[1][0][0])
        X, Y = left, right
        r = self.v(X)/max(self.v(X) + self.v(Y), 0.00001)
        if r >= 0.5:
            if random.random() < 0.5 + 0.883*pow(2*r-1,0.75):
                chosen = int(left)
            else:
                chosen= int(right)
        else:
            if random.random() < 0.5 - 0.883*pow(1-2*r,0.75):
                chosen = int(left)
            else:
                chosen = int(right)
        return chosen

    def predictS(self, itemPair):
        left, right = int(itemPair[0]), int(itemPair[1])
        X, Y = left, right
        r = float(self.v(X))/max(self.v(X) + self.v(Y),0.00001)# if (self.v(X) + self.v(Y)) > self.v(X) else int(self.v(X)!=0)
        if r >= 0.5:
            return 0.5 + 0.883*pow(2*r-1,0.75)
        return 0.5 - 0.883*pow(1-2*r,0.75)
    def adaptS(self, itemPair):
        left, right = int(itemPair[0]), int(itemPair[1])
        if self.correctReply((left, right)) == str(left):
            self.V[left] = self.Db*(1-self.v(left)) + self.v(left)
            self.V[right] = (-1)*self.Db*self.v(right) + self.v(right)
        elif self.correctReply((left, right)) == str(right):
            self.V[left] = (-1)*self.Db*self.v(left) + self.v(left)
            self.V[right] = self.Db*(1-self.v(right)) + self.v(right)
        else:
            print('error')
    def adapt(self, item, target, **kwargs):
        left, right = int(item.choices[0][0][0]), int(item.choices[1][0][0])
        if self.correctReply((left, right)) == str(left):
            self.V[left] = self.Db*(1-self.v(left)) + self.v(left)
            self.V[right] = (-1)*self.Db*self.v(right) + self.v(right)
        elif self.correctReply((left, right)) == str(right):
            self.V[left] = (-1)*self.Db*self.v(left) + self.v(left)
            self.V[right] = self.Db*(1-self.v(right)) + self.v(right)
        else:
            print('error')
        
    def v(self, item):
        if item not in self.V.keys():
            self.V[item] = self.vInit
        return self.V[item]
    