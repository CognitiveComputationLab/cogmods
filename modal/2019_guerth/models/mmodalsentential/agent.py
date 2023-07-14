"""This module implements a reasoning agent.
The agent models a reasoner by providing parameters that determine how the
agent reasons. The parameters are thresholds that determine whether a random
process causes a different reasoning strategy.
"""
import reasoner as r
from random import random

class Agent():
    def __init__(self, sigma=0.9, gamma=0.5):
        self.sigma = sigma # if random number is higher then sigma calls system 2
        self.gamma = gamma # if random number is higher then gamma uses weak validity.

    def model(self, premisses):
        # system = 1 if random() <= self.sigma else 2
        return r.model(premisses)

    def what_follows(self, premisses):
        system = 1 if random() <= self.sigma else 2
        return r.what_follows(premisses, system)

    def necessary(self, premisses, conclusion):
        system = 1 if random() <= self.sigma else 2
        weak = False if random() <= self.gamma else True
        return r.necessary(premisses, conclusion, system, weak)

    def possible(self, premisses, conclusion):
        system = 1 if random() <= self.sigma else 2
        return r.possible(premisses, conclusion, system)

    def probability(self, premisses, conclusion):
        system = 1 if random() <= self.sigma else 2
        return r.probability(premisses, conclusion, system)

    def verify(self, premisses, evidence):
        system = 1 if random() <= self.sigma else 2
        return r.verify(premisses, evidence, system)

    def defeasance(self, premisses, fact):
        system = 1 if random() <= self.sigma else 2
        return r.defeasance(premisses, fact, system)



if __name__ == "__main__":
    a = Agent()
    print(a.model('a | b'))
    print(a.necessary(['a|b'], 'a^b'))
    a.gamma = 0.9
    a.sigma = 1
    print(a.necessary(['a|b'], 'a^b'))
    print(a.necessary(['a|b'], 'a^b'))
    print(a.necessary(['a|b'], 'a^b'))