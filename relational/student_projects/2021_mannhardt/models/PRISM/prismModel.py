""" Interface for .

"""

import sys, os, importlib
from collections import OrderedDict
from pprint import pprint
import numpy as np
import ccobra
import pandas as pd
from spatialreasoner.prism import Prism

from difflib import SequenceMatcher

UP = np.array([0,1,0])
DOWN = UP * (-1)
RIGHT = np.array([1,0,0])
LEFT = RIGHT * (-1)


interpretations = OrderedDict([("north", UP), ("south", DOWN), ("west", LEFT), ("east", RIGHT),
                               ("north-west", UP + LEFT), ("north-east", UP + RIGHT), 
                               ("south-west", DOWN + LEFT), ("south-east", DOWN + RIGHT)])

correspondence = {"left":"west", "right":"east", "above":"north", "below":"south",
                  "west":"left", "east":"right", "north":"above", "south":"below"}

VP = 0


def interpretation(relation, cd):
    if not cd:
        relation = correspondence[relation.lower()]
    return interpretations[relation.lower()]


def build_model(model, cd):
    model = [[interpretation(prem[0], cd)] + prem[1:] for prem in model]
    rep = {}
    objects = []
    for prem in model:
        items = prem[1], prem[2]
        for i in items:
            if i not in objects:
                objects.append(i)

    base = np.zeros(3, dtype=int)

    for obj in objects:
        rep[tuple(base)] = obj
        base += RIGHT

    return rep

def build_model_from_string(model, cd=False):
    mod = dict()

    posV = 0
    for char in model:
        mod[(posV, 0, 0)] = char
        posV += 1
        
    return mod

def model_similarity(mod1, mod2, pref):
    '''
    calcs similarity between mod1 and mod2 based on individual similarity preferences
    mod1 is initial model and mod2 one of the two choices.
    '''
    if pref[0] > pref[1]:
        if mod1[(0,0,0)] == mod2[(2,0,0)]:
            return 1
        elif mod1[(2,0,0)] == mod2[(0,0,0)]:
            return 2
    else:
        if mod1[(0,0,0)] == mod2[(2,0,0)]:
            return 2
        elif mod1[(2,0,0)] == mod2[(0,0,0)]:
            return 1
        
    return 1

class PrismModel(ccobra.CCobraModel):
    """ Model producing randomly generated responses.

    """

    def __init__(self, name='PRISM'):
        """ Initializes the random model.

        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the CCOBRA
            framework as a means for identifying the model.

        """

        super(PrismModel, self).__init__(
            name, ["spatial-relational"], ["verify", "single-choice", "model_constr"])

        self.prism = Prism(False)
        self.last_prediction = None
        self.counter = 0
        self.model = None

        # stores preferred way of similarity
        # [<LeftTwoObjects>, <RightTwoObjects>]
        self.pref_similarity = [0, 0]

    def start_participant(self, **kwargs):
        global VP
        VP += 1

    def target_positions_in_model(self, targets, model):
        '''
        Apparently expects to only get one model
        '''
        target_positions = {}
        for position, obj in model.items():
            if obj in targets:
                target_positions[obj] = np.array(position)
        return target_positions

    def predict_model_constr(self, premises, choices, cd):
        model = self.prism.build_model(premises)[-1]
        self.model = model
        return model
    
    def include_counterfact(self, mod, choices, cd=False):
        model = build_model_from_string(mod[0][0])
        targets = [build_model_from_string(choices[0][0][0]), build_model_from_string(choices[1][0][0])]
        sim1 = model_similarity(model, targets[0], self.pref_similarity)
        sim2 = model_similarity(model, targets[1], self.pref_similarity)
        return choices[0][0][0] if sim1 > sim2 else choices[1][0][0]

    def predict_verification(self, mod, choices, cd=False):
        choice = choices[0][0]
        choice[0] = interpretation(choice[0], cd)

        validity = self.prism.decide(self.prism.build_premise(*choice), [build_model_from_string(mod[0][0])])
        return validity != [None]

    def predict_single_choice(self, premises, choices, cd):
        model = self.prism.build_model(premises)[-1]
        self.model = model
        
        for mod in choices:
            if build_model_from_string(mod[0][0]) == self.model:
                return mod[0][0]

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given syllogism.

        Parameters
        ----------
        task : str
            task to produce a response for.

        """
        self.counter += 1

        self.prism = self.prism.reset()
        cd = item.task[0][0].lower() not in ["left", "right","above", "below"]
        premises = []
        try:
            premises = [[interpretation(prem[0], cd)] + prem[1:] for prem in item.task]
        except:
            pass

        if item.response_type == 'verify':
            cd = item.choices[0][0][0].lower() not in ["left", "right","above", "below"]
            self.last_prediction = self.predict_verification(item.task, item.choices, cd)
        
        elif item.response_type == 'single-choice':
            if item.sequence_number == 3:
                self.last_prediction = self.include_counterfact(item.task, item.choices, cd)
            else:
                self.last_prediction = self.predict_single_choice(premises, item.choices, cd)
        return self.last_prediction

        return False # default
    
    def adapt(self, item, response, **kwargs):
        if item.sequence_number != 3:
            return  
        response = response[0][0]
        initialModel = item.task[0][0]
        # Left two objects relation stays the same
        if initialModel[2] == response[0]:
            self.pref_similarity[0] += 1
        # Right two objects relation stays the same
        elif initialModel[0] == response[2]:
            self.pref_similarity[1] += 1
            
    def pre_train(self, dataset):
        """ Pretrains the model for a given dataset.

        Parameters
        ----------
        dataset : list
            Pairs of items and responses that the model should be fitted
            against.

        """
        for person_data in dataset:
            for item_data in person_data:
                if item_data["item"].sequence_number != 3:
                    continue
                task = item_data["item"].task[0][0]
                response = item_data["response"][0][0]
                # Left two objects relation stays the same
                if task[2] == response[0]:
                    self.pref_similarity[1] += 1
                # Right two objects relation stays the same
                elif task[0] == response[2]:
                    self.pref_similarity[0] += 1
        self.pref_similarity[0] = self.pref_similarity[0] / len(dataset)
        self.pref_similarity[1] = self.pref_similarity[1] / len(dataset)
