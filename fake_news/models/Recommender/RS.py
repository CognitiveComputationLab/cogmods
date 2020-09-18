""" News Item Processing model implementation.
"""
import ccobra
from random import random 
import math




class RS:
    recommendation = {}
    featuresOfAllPeople = {}
    repliesOfAllPeople = {}
    trained = False