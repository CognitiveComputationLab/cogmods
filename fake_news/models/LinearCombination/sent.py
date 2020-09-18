""" News Item Processing model implementation.
"""
import ccobra
from random import random 
import math
from LinearCombination.sentimentanalyzer import SentimentAnalyzer

from scipy.optimize._basinhopping import basinhopping
from numpy import mean


class LP(ccobra.CCobraModel):
    """ TransitivityInt CCOBRA implementation.
    """
    def __init__(self, name='SentimentAnalysis', commands = []):
        """ Initializes the TransitivityInt model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.thresh = 1
        self.parameter = {}
        self.relevant = ['negative_emotion', 'health', 'dispute', 'government', 'leisure', 'healing', 'military', 'fight', 'meeting', 'shape_and_size', 'power', 'terrorism', 'competing', 'optimism', 'sexual', 'zest', 'love', 'joy', 'lust', 'office', 'money', 'aggression', 'wealthy', 'banking', 'kill', 'business', 'fabric', 'speaking', 'work', 'valuable', 'economics', 'clothing', 'payment', 'feminine', 'worship', 'affection', 'friends', 'positive_emotion', 'giving', 'help', 'school', 'college', 'real_estate', 'reading', 'gain', 'science', 'negotiate', 'law', 'crime', 'stealing', 'white_collar_job', 'weapon', 'night', 'strength']
        self.relevant = ['negative_emotion', 'fight', 'optimism', 'sexual', 'money', 'aggression', 'affection', 'positive_emotion', 'science', 'law', 'crime']
        for a in self.relevant:
            self.parameter[a] = 1

        #dictionary for testing with value from rough optimization on Experiment 1
        optdict = {'negative_emotion': 3.488183752051738, 'fight': 4.795255469272864, 'optimism': 5.4718777782354735, 'sexual': 3.583167795093339, 'money': 5.249519675409447, 'aggression': 5.464386990594476, 'affection': -2.3986185486873572, 'positive_emotion': -1.4074963226019577, 'science': 1.5522568483198222, 'law': 12.874721726587898, 'crime': -2.9929120337902457}
        for a in optdict.keys():
            self.parameter[a] = optdict[a]
        super().__init__(name, ['misinformation'], ['single-choice'])

    def predictS(self, item, **kwargs):
        if len(kwargs.keys()) == 1:
            kwargs = kwargs['kwargs']
        analysis = SentimentAnalyzer.analysis(item)
        p = 0
        for a in self.relevant:
            p += analysis[a]*  self.parameter[a]
        #print(p)
        return 1 if self.thresh < p else 0

    def adapt(self, item, target, **kwargs):
        pass

    def adaptS(self, itemPair):
        pass
    
    def predict(self, item, **kwargs):
        return 'Accept' if random() < self.predictS(item, **kwargs) else 'Reject'

    def toCommandList(self,pars):
        optCommands = []
        i = 0
        parKeys = sorted(self.parameter.keys())
        for a in parKeys:
            if len(pars)<=i: 
                print('keys length error', self.name)
                break
            optCommands.append('self.parameter[\'' + a + '\'] = ' + str(pars[i]))
            i += 1
        return optCommands
    
    def executeCommands(self, commands):
        for command in commands:
            exec(command)

    def pre_person_background(self, dataset):
        trialList = []
        for pers in dataset:
            trialList.extend([pers])
        if len(self.parameter.keys()) > 0:
            personOptimum = basinhopping(self.itemsOnePersonThisModelPeformance, [1]*len(self.parameter.keys()), niter=200, stepsize=3, T=4,  minimizer_kwargs={"args" : (trialList)})
            optpars = personOptimum.x
        else: 
            optpars = [] 
        self.executeCommands(self.toCommandList(optpars))

    def itemsOnePersonThisModelPeformance(self, pars, items):
        #input: list of items
        items = [a for a in items]
        performanceOfPerson = []
        self.executeCommands(self.toCommandList(pars))
        for item in items:
            #print(item['item'], item['aux'])
            pred = min(1.0,max(self.predictS(item=item['item'], kwargs= item['aux']),0.0)) 
            if item['aux']['binaryResponse']:
                predictionPerf = min(1.0,max(self.predictS(item=item['item'], kwargs=item['aux']),0.0)) 
            elif not item['aux']['binaryResponse']:
                predictionPerf = 1.0 - pred
            else:
                print('Error')
            performanceOfPerson.append(predictionPerf)
        return -1*mean(performanceOfPerson) 




