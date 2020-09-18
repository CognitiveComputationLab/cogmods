""" News Item Processing model implementation.
"""
import ccobra
from random import random
import math
from scipy.optimize._basinhopping import basinhopping
from numpy import mean


class S2MR(ccobra.CCobraModel):
    """ TransitivityInt CCOBRA implementation.
    """
    def __init__(self, name='S2MR'):
        """ Initializes the TransitivityInt model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.parameter = {}
        #self.parameter['Dcons'] = 3
        #self.parameter['Dlib'] = 1.5

        self.parameter['Kc'] = 0.55
        self.parameter['Kl'] = 0.3
        
        self.parameter['Mc'] = 0.2
        self.parameter['Ml'] = - 0.2

        #dictionary for testing with value from rough optimization on Experiment 1
        optdict = {'Kc': -0.056536155629514195, 'Kl': -0.09074348945552314, 'Mc': -0.027349753930657073, 'Ml': -0.036805428943731885}
        for a in optdict.keys():
            self.parameter[a] = optdict[a]
        super().__init__(name, ['misinformation'], ['single-choice'])

    def predictS(self, item, **kwargs):
        if len(kwargs.keys()) == 1:
            kwargs = kwargs['kwargs']
        conservativePerson = kwargs['conservatism'] >= 3.5
        liberalPerson = kwargs['conservatism'] <= 3.5
        if kwargs['Partisanship_All_Combined']>3.8 and conservativePerson:
            threshold = self.parameter['Kc'] + kwargs['crt'] * self.parameter['Mc']
        elif kwargs['Partisanship_All_Combined']<2.2 and liberalPerson:
            threshold = self.parameter['Kc'] + kwargs['crt'] * self.parameter['Mc']
        elif kwargs['Partisanship_All_Combined']<2.2 and conservativePerson:
            threshold = self.parameter['Kl'] + kwargs['crt'] * self.parameter['Ml']
        elif kwargs['Partisanship_All_Combined']>3.8 and liberalPerson:
            threshold = self.parameter['Kl'] + kwargs['crt'] * self.parameter['Ml']
        else:
            threshold = 0.5
        return threshold

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

        
