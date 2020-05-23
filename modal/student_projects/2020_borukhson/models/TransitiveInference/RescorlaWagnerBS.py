""" Transitive-Inference model implementation.
"""
import ccobra
import random
import math

class RescorlaWagnerBS(ccobra.CCobraModel):
    """ TransitivityInt CCOself.BRA implementation.
    """
    def __init__(self, name='RescorlaWagner-Kumaran2016', commands = []):
        """ Initializes the TransitivityInt model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.b = 21.77117            #TEMPERATURE
        self.a = 0.08958            #LEARNING RATE
        self.V = {}             #ranks
        self.vInit = 0.001
        self.lastChosen = None
        super().__init__(name, ['spatial-relational'], ['single-choice'], commands)

    def predict(self, item, **kwargs):
        left, right = int(item.choices[0][0][0]), int(item.choices[1][0][0])
        if random.random() < self.p('left', (left, right)):
            chosen = int(left)
        else:
            chosen= int(right)
        return chosen
    def predictS(self, itemPair):
        left, right = int(itemPair[0]), int(itemPair[1])
        return self.p('left', (left, right))
        if random.random() < self.p('left', (left, right)):
            chosen = int(left)
        else:
            chosen= int(right)
        return chosen
    def adaptS(self, itemPair):
        left, right = int(itemPair[0]), int(itemPair[1])
        if self.correctReply((left, right)) == str(left):
            Ol = 1
            Or = -1
        elif self.correctReply((left, right)) == str(right):
            Ol = -1
            Or = 1
        else:
            print('error')
        self.V[left] = (Ol- self.v(left)) * self.a
        self.V[right] = (Or- self.v(right)) * self.a


    def adapt(self, item, target, **kwargs):
        left, right = int(item.choices[0][0][0]), int(item.choices[1][0][0])
        if self.correctReply((left, right)) == str(left):
            Ol = 1
            Or = -1
        elif self.correctReply((left, right)) == str(right):
            Ol = -1
            Or = 1
        else:
            print('error')
        self.V[left] = (Ol- self.v(left)) * self.a
        self.V[right] = (Or- self.v(right)) * self.a

    def p(self, leftOrRight, pair):
        left, right = pair
        if leftOrRight == 'left': #left
            exponent = float(max(-10, min(10, -1*self.b*(self.v(left)-self.v(right)))))
            return 1/(1+math.exp(exponent))
        if leftOrRight == 'right': #right
            return 1-self.p('left', pair)
        
    def v(self, item):
        if item not in self.V.keys():
            self.V[item] = self.vInit
        return self.V[item]
    