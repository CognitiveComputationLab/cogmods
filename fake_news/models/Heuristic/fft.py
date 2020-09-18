""" News Item Processing model implementation.
"""
import ccobra
from random import random 
import math
import numpy as np
import pandas as pd
from LinearCombination.sentimentanalyzer import SentimentAnalyzer
from Heuristic.fasttrees.fasttrees import FastFrugalTreeClassifier
from sklearn.model_selection import train_test_split
from Heuristic.fftTool import FFTtool
from scipy.optimize import * 


class FFTifan(ccobra.CCobraModel):
    """ TransitivityInt CCOBRA implementation.
    """
    
    def __init__(self, name='Fast-Frugal-Tree-ifan', commands = []):
        """ Initializes the TransitivityInt model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        SentimentAnalyzer.initialize()
        self.parameter = {}
        #self.parameter['thresh'] = 10
        self.componentKeys = ['crt','ct','conservatism','panasPos','panasNeg','education', 'reaction_time','accimp','age','gender','Exciting_Democrats_Combined', 'Exciting_Republicans_Combined', 'Familiarity_Democrats_Combined', 'Familiarity_Republicans_Combined', 'Importance_Democrats_Combined', 'Importance_Republicans_Combined', 'Likelihood_Democrats_Combined', 'Likelihood_Republicans_Combined', 'Partisanship_All_Combined', 'Partisanship_All_Partisan', 'Partisanship_Democrats_Combined', 'Partisanship_Republicans_Combined','Sharing_Democrats_Combined', 'Sharing_Republicans_Combined', 'Worrying_Democrats_Combined','Worrying_Republicans_Combined', 'Sent: negative_emotion', 'Sent: health', 'Sent: dispute', 'Sent: government', 'Sent: healing', 'Sent: military', 'Sent: fight', 'Sent: meeting', 'Sent: shape_and_size', 'Sent: power', 'Sent: terrorism', 'Sent: competing', 'Sent: office', 'Sent: money', 'Sent: aggression', 'Sent: wealthy', 'Sent: banking', 'Sent: kill', 'Sent: business', 'Sent: speaking', 'Sent: work', 'Sent: valuable', 'Sent: economics', 'Sent: payment', 'Sent: friends', 'Sent: giving', 'Sent: help', 'Sent: school', 'Sent: college', 'Sent: real_estate', 'Sent: reading', 'Sent: gain', 'Sent: science', 'Sent: negotiate', 'Sent: law', 'Sent: crime', 'Sent: stealing', 'Sent: strength']#Keys.person + Keys.task 
        super().__init__(name, ['misinformation'], ['single-choice'])

    def pre_train(self, dataset):
        #print('Pretrain started')
        trialList = []
        for pers in dataset:
            perslist = []
            for a in pers:
                persdict = {}
                persdict = a['aux']
                persdict['item'] = a['item']
                perslist.append(persdict)
            trialList.extend(perslist)
        return self.fitTreeOnTrials(trialList)

    def fitTreeOnTrials(self, trialList, maxLength=-1, person='global'):
        for item in trialList:
            for a in self.componentKeys:
                if a not in item.keys():
                    continue
                if item['conservatism'] >= 3.5:
                    if 'Republicans' in a:
                        item[a.replace('Republicans', 'Party')] = item[a]
                        item.pop(a,None)
                        item.pop(a.replace('Republicans','Democrats'))
                elif item['conservatism'] <= 3.5:
                    if 'Democrats' in a:
                        item[a.replace('Democrats', 'Party')] = item[a]
                        item.pop(a,None)
                        item.pop(a.replace('Democrats','Republicans'))
                if 'Sent' in a:
                    item[a] = SentimentAnalyzer.analysis(item['item'])[a.split(' ')[1]]
        #print(len(trialList))
        data = self.toDFFormatTruthful(trialList)
        FFTtool.fc[person] = FastFrugalTreeClassifier(max_levels=5)
        #print('Started Fitting Fast-Frugal-Tree')
        FFTtool.fc[person].fit(data.drop(columns='response'), data['response'])
        #print('   Done Fitting Fast-Frugal-Tree')
        #print(FFTtool.fc[person].get_tree(top10=True,decision_view=True))
        #print(FFTtool.fc[person].get_tree(decision_view=False))

    def toDFFormat(self, trialList):
        featList = []
        for item in trialList:
            newPars = [item['binaryResponse']] + [item[a] for a in self.componentKeys if a in item.keys()]
            featList.append(newPars)
        cat_columns = [] 
        nr_columns = [a for a in self.componentKeys
            if a not in cat_columns and a in item.keys()]
        data_columns = ['response'] +cat_columns + nr_columns
        data = pd.DataFrame(data=featList, columns=data_columns)
        for col in cat_columns:
            data[col] = data[col].astype('category')
            for col in nr_columns:
                if data[col].dtype != 'float' and data[col].dtype != 'int':
                    print('type error: ' + data[col])
                    data.loc[data[col] == '?', col] = np.nan
                    data[col] = data[col].astype('float')
        data['response'] = data['response'].apply(lambda x: True if x==1 else False).astype(bool)
        return data
        
    def toDFFormatTruthful(self, trialList):
        featList = []
        errorcount = 0
        for item in trialList:
            newPars = [item['truthful']] + [item[a] for a in self.componentKeys if a in item.keys()]
            if any(a for a in newPars[1:] if type(a) == type('hi')):
                for a in newPars[1:]:
                    if type(a) == type('hi'):
                        errorcount += 1
                        print('FOUND ERROR:', a, type(a),newPars, errorcount)
                        print(self.componentKeys)
                        continue
            else:
                featList.append(newPars)
        #print('number of FFT error trials:', errorcount)
        cat_columns = [] 
        nr_columns = [a for a in self.componentKeys
            if a not in cat_columns and a in item.keys()]
        data_columns = ['response'] +cat_columns + nr_columns
        data = pd.DataFrame(data=featList, columns=data_columns)
        #print(len(featList), len(data))
        for col in cat_columns:
            data[col] = data[col].astype('category')
            for col in nr_columns:
                if data[col].dtype != 'float' and data[col].dtype != 'int':
                    print('type error: ' + data[col])
                    data.loc[data[col] == '?', col] = np.nan
                    data[col] = data[col].astype('float')
        data['response'] = data['response'].apply(lambda x: True if x==1 else False).astype(bool)
        return data

    def predictS(self, item, person='global', **kwargs):
        if 'item' not in kwargs.keys():
            kwargs['item'] = item
        for a in self.componentKeys:
            if kwargs['conservatism'] >= 3.5:
                if 'Republicans' in a and a in kwargs.keys():
                    kwargs[a.replace('Republicans', 'Party')] = kwargs[a]
            elif kwargs['conservatism'] <= 3.5:
                if 'Democrats' in a and a in kwargs.keys():
                    kwargs[a.replace('Democrats', 'Party')] = kwargs[a]
            if 'Sent' in a:
                kwargs[a] = SentimentAnalyzer.analysis(kwargs['item'])[a.split(' ')[1]]
        pred = FFTtool.fc[person].predict(self.toDFFormat([kwargs]))
        return int(pred[0])

    def predict(self, item, **kwargs):
        return 'Accept' if random() < self.predictS(item, **kwargs) else 'Reject'

    def adapt(self, item, target, **kwargs):
        pass

    def adaptS(self, itemPair):
        pass