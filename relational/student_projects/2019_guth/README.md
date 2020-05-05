Genetic Algorithms for Spatial Reasoning 
=======

Spatial Reasoning Models implemented by Manuel Guth as part of the bachelor thesis in early 2019.

Two different models based on genetic algorithms.
The rule approach is based on logic rules for reasoning.
The mental model instruction approach (MMI) is based on the Mental Model Theory.

## Prerequisites

Works with Python 3.6
Used libraries: ccobra, random, pickle

## Models

Two models have been implemented for the four benchmarks (figural, premiseorder, singlechoice, verification):

- Rule Approach: This approach uses rules of propositional logic for spatial reasoning. Different sets of rules were created and improved with the help of a genetic algorithm. In the prediction phase, each participant is predicted with the set of rules most fitting to him/her.

- Mental Model Instruction Approach: This approach is based on the Mental Model Theory. It lets each participant build its own mental model which were pre-treined with a genetic algorithm.

## Quickstart

1) To run the model in the CCOBRA framework and compare it with other models:

   - Create a json-file for the benchmark (see CCOBRA documentation, already an .json added to repo for all four benchmarks)
   - Insert this model and all other models to be tested in the benchmark-file
   - Run the benchmark (ccobra file_name.json)
