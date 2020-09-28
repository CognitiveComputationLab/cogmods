#adjust import structure if started as script
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


""" 
News Item Processing model implementation.
"""
import ccobra
from random import random 
import math
import numpy as np
import pandas as pd
from LinearCombination.sentimentanalyzer import SentimentAnalyzer
from fasttrees.fasttrees import FastFrugalTreeClassifier
from sklearn.model_selection import train_test_split
from Heuristic.fftTool import FFTtool
from scipy.optimize import * 

class FFTzigzag(ccobra.CCobraModel):
    """ FFTzigzag CCOBRA implementation.
    """
    
    def __init__(self, name='Fast-Frugal-Tree-ZigZag(Z+)', commands = []):
        """ Initializes the model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        SentimentAnalyzer.initialize()
        self.parameter = {}
        self.fft = None
        self.lastnode = None
        self.componentKeys = ['crt','ct','conservatism','panasPos','panasNeg','education', 'reaction_time','accimp','age','gender','Exciting_Democrats_Combined', 'Exciting_Republicans_Combined', 'Familiarity_Democrats_Combined', 'Familiarity_Republicans_Combined', 'Importance_Democrats_Combined', 'Importance_Republicans_Combined', 'Likelihood_Democrats_Combined', 'Likelihood_Republicans_Combined', 'Partisanship_All_Combined', 'Partisanship_All_Partisan', 'Partisanship_Democrats_Combined', 'Partisanship_Republicans_Combined','Sharing_Democrats_Combined', 'Sharing_Republicans_Combined', 'Worrying_Democrats_Combined','Worrying_Republicans_Combined','Sent: negative_emotion', 'Sent: health', 'Sent: dispute', 'Sent: government', 'Sent: healing', 'Sent: military', 'Sent: fight', 'Sent: meeting', 'Sent: shape_and_size', 'Sent: power', 'Sent: terrorism', 'Sent: competing', 'Sent: office', 'Sent: money', 'Sent: aggression', 'Sent: wealthy', 'Sent: banking', 'Sent: kill', 'Sent: business', 'Sent: speaking', 'Sent: work', 'Sent: valuable', 'Sent: economics', 'Sent: payment', 'Sent: friends', 'Sent: giving', 'Sent: help', 'Sent: school', 'Sent: college', 'Sent: real_estate', 'Sent: reading', 'Sent: gain', 'Sent: science', 'Sent: negotiate', 'Sent: law', 'Sent: crime', 'Sent: stealing', 'Sent: strength']#Keys.person + Keys.task 
        super().__init__(name, ['misinformation'], ['single-choice'])

    def pre_train(self, dataset):
        #Globally trains zigzag FFT on data for all persons
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
        if FFTtool.ZigZag != None:
            return
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
        maxLength = -1
        predictionQuality = {}
        predictionMargin = {}
        #precalculated predictive quality of individual cues
        predictionQuality, predictionMargin = {'>crt': -0.499969808586438, '<crt': -0.499969808586438, '>conservatism': -0.499969808586438, '<conservatism': -0.499969808586438, '>ct': -0.499969808586438, '<ct': -0.499969808586438, '>education': -0.499969808586438, '<education': -0.499969808586438, '>accimp': -0.499969808586438, '<accimp': -0.499969808586438, '>panasPos': -0.499969808586438, '<panasPos': -0.499969808586438, '>panasNeg': -0.499969808586438, '<panasNeg': -0.499969808586438, '>Exciting_Party_Combined': -0.499969808586438, '<Exciting_Party_Combined': -0.9967320261437909, '>Familiarity_Party_Combined': -0.9996378123868164, '<Familiarity_Party_Combined': -0.499969808586438, '>Importance_Party_Combined': -0.9978308026030369, '<Importance_Party_Combined': -0.7963446475195822, '>Partisanship_All_Combined': -0.9978308026030369, '<Partisanship_All_Combined': -0.777589954117363, '>Partisanship_All_Partisan': -0.9978308026030369, '<Partisanship_All_Partisan': -0.9997585124366095, '>Partisanship_Party_Combined': -0.9967845659163987, '<Partisanship_Party_Combined': -0.7990093847758082, '>Worrying_Party_Combined': -0.499969808586438, '<Worrying_Party_Combined': -0.9935897435897436}, {'>crt': 1.972872196401318, '<crt': 0.0, '>conservatism': 0.0, '<conservatism': 0.0, '>ct': 0.0, '<ct': 0.0, '>education': 0.0, '<education': 0.0, '>accimp': 0.0, '<accimp': 0.0, '>panasPos': 0.0, '<panasPos': 0.0, '>panasNeg': 0.0, '<panasNeg': 0.0, '>Exciting_Party_Combined': 0.0, '<Exciting_Party_Combined': 3.574226008657847, '>Familiarity_Party_Combined': 2.6042025215277933, '<Familiarity_Party_Combined': 0.0, '>Importance_Party_Combined': 2.2036394876853738, '<Importance_Party_Combined': 4.255574086579121, '>Partisanship_All_Combined': 1.965200689848336, '<Partisanship_All_Combined': 3.8353730503393244, '>Partisanship_All_Partisan': 1.2298451980672356, '<Partisanship_All_Partisan': 0.7176463353940941, '>Partisanship_Party_Combined': 4.223723273224007, '<Partisanship_Party_Combined': 3.885513240139143, '>Worrying_Party_Combined': 0.0, '<Worrying_Party_Combined': 1.6378138233962547}
        orderedConditionsPos = []
        orderedConditionsNeg = []
        #calculate order and direction of cues for both Accept (Pos) and Reject (Neg) exits
        for a in sorted(predictionQuality.items(), key=lambda x: x[1], reverse=False):
            if a[0][1:] not in item.keys():
                continue
            b = a[0][1:]
            s = a[0][0]
            cond = 'item[\'aux\'][\'' + b + '\'] ' + s + ' ' + str(predictionMargin[a[0]])
            newnode = Node(cond,True,False)
            rep0preds, rep1preds, length0, length1 = predictiveQuality(newnode, trialList)
            #determine exit direction
            if rep1preds/length1 >= rep0preds/length0:
                if a[0][1:] not in [i[1:] for i in orderedConditionsPos + orderedConditionsNeg] and a[0][1:] in self.componentKeys:
                    orderedConditionsPos.append(a[0])
            else:
                if a[0][1:] not in [i[1:] for i in orderedConditionsNeg +orderedConditionsPos] and a[0][1:] in self.componentKeys:
                    orderedConditionsNeg.append(a[0])
        orderedConditions = []
        for i in range(max(len(orderedConditionsNeg), len(orderedConditionsPos))):
            if len(orderedConditionsNeg) > i:
                orderedConditions.append(orderedConditionsNeg[i])
            if len(orderedConditionsPos) > i:
                orderedConditions.append(orderedConditionsPos[i])
        exitLeft = True #as Z+ version implemented
        #assemble tree
        for sa in orderedConditions[:maxLength] if maxLength > 0 else orderedConditions:
            b = sa[1:]
            s = sa[0]
            cond = 'item[\'aux\'][\'' + b + '\'] ' + s + ' ' + str(predictionMargin[sa])
            newnode = Node(cond,True,False)
            if self.fft == None:
                self.fft = newnode
                self.lastnode = self.fft
            else:
                if not exitLeft:
                    self.lastnode.left = newnode
                    self.lastnode = self.lastnode.left
                else:
                    self.lastnode.right = newnode
                    self.lastnode = self.lastnode.right
            exitLeft = not exitLeft
        FFTtool.ZigZag = self.fft

    def predictS(self, item, **kwargs):
        if len(kwargs.keys()) == 1:
            kwargs = kwargs['kwargs']
        return FFTtool.ZigZag.run(item, **kwargs)

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

def predictiveQuality(node, trialList):
    rep0preds = 0
    rep1preds = 0
    length0 = 1
    length1 = 1
    for item in trialList:
        if 1 == node.run(item):
            rep1preds += int(bool(item['aux']['truthful'] == 1))
            length1 += 1
        else:
            rep0preds += int(bool(item['aux']['truthful']  == 0))
            length0 += 1
    return rep0preds, rep1preds, length0, length1

class Node:
    def __init__(self, conditionstr, left, right):
        self.condition = conditionstr
        self.left = left
        self.right = right
    
    def run(self, item, **kwargs):
        #get prediction of tree
        try:
            if 'aux' not in item.keys():
                item['aux'] = item
        except:
            tempitem = item
            item = {}
            item['item'] = tempitem
            item['aux'] = kwargs
        if eval(self.condition):
            if isinstance(self.left,bool):
                return self.left
            return self.left.run(item)
        else:
            if isinstance(self.right,bool):
                return self.right
            return self.right.run(item)

    def getstring(self):
        #visualize tree
        a = ''
        if isinstance(self.left,bool):
            a = 'If ' + self.condition.split('\'')[3] + self.condition.split(']')[1] + ' then return ' + str(self.left) + ', else: ' 
            a += 'Return ' + str(self.right) if isinstance(self.right,bool) else self.right.getstring()
        else:
            a = 'If ' + self.condition.split('\'')[3] + self.condition.split(']')[1] + ' then return ' + str(self.right) + ', else: '
            a += 'Return ' + str(self.left) if isinstance(self.left,bool) else self.left.getstring()
        return a 
