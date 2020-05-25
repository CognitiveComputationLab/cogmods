Prediction models for propositinal logical tasks
=======

propositional reasoning models implemented as part of the Bachelor project by Julian Giessl in 2019.

## Prerequisites

Python 3.5
Libraries: ccobra, (pandas, numpy)

## Models

Several models have been implemented:
- Logic model:
 choses responces for a given task according to normal propositional logic.
- MFA model:
 choses responces for a task according to the most frequent response to that task.
- Recommender model:
 choses responses for a task based upon previous responses (to similar tasks) of the participants.

## Files

data_structures.py contains a tree data structure for the premises, with which the truth value
of a premise can be calculated according to the truth values of the variables.
Furthermore it contains the classification tags for the model.

task_processor.py contains two components.
First is a task processor class, with which the input data is parsed.
I.e. the premises get parsed into a tree-data-structure and feature vectors, describing the premises,
are constructed. Furthermore the possible answers are classified.
Secondly it contains predictor classes, with which the predictions are calculated based upon the logic
of the corresponding model.

In mfa_model.py the most frequent answers to the the tasks are calculated by pre-training the model
on a data-set. Model chooses the most frequent answers given as a response.

In logic_model.py the TaskProcessor the LogicPredictor classes are initialized for a strict logical
model. The model chooses the answer according to strict formal logic.
All the calculations are done in task_processor.py.

In model.py the TaskProcessor and the Predictor classes are initialized for the model. The model chooses
the responses based upon previous answers of the participants (basicly a recommender system). The calculations are done in
task_processor.py.
