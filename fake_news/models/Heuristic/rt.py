""" News Item Processing model implementation.
"""
import ccobra
from random import random 
import math
from scipy.optimize._basinhopping import basinhopping
from numpy import mean
from scipy import mean

class RT(ccobra.CCobraModel):
    """ TransitivityInt CCOBRA implementation.
    """
    def __init__(self, name='CR&time', commands = []):
        """ Initializes the TransitivityInt model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.parameter = {}
        self.Cparameter = {}
        for a in ['alpha']:
            self.parameter[a] = 0
        #dictionary for testing with value from rough optimization on Experiment 1
        optdict = {'alpha': -0.015201555811670677}
        for a in optdict.keys():
            self.parameter[a] = optdict[a]
        super().__init__(name, ['misinformation'], ['single-choice'])

    def predictS(self, item, **kwargs):
        if len(kwargs.keys()) == 1:
            kwargs = kwargs['kwargs']
        self.Cparameter['Cr'] = 0.65 
        self.Cparameter['Cf'] = 0.2 
        self.Cparameter['Mr'] = 0.13                              
        self.Cparameter['Mf'] = - 0.12 
        if kwargs['truthful']:
            threshold = self.Cparameter['Cr'] + self.Cparameter['Mr'] * kwargs['crt']
        if not kwargs['truthful']:
            threshold = self.Cparameter['Cf'] + self.Cparameter['Mf'] * kwargs['crt']
        return threshold + kwargs['reaction_time']*self.parameter['alpha']

    def adapt(self, item, target, **kwargs):
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


