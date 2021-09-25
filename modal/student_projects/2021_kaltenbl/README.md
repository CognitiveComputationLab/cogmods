# Optimization of Predictive models for individual human reasoning:
Bachelor's Thesis of Leon Kaltenbrunn (2021). Investigates modal reasoning using mModalSentential and baselines for
comparison.

## Prerequisites:
Python 3.8 Libraries: ccobra, pandas, numpy

## Models:
### Baseline:
- Random Model: Predicts a response for a given task at random.
- Logic Model: Predicts a response for a given task with a pure logical approach. Many thanks and credit to its author [Joey Thaidigsman](https://github.com/joeytman/Modal_Logic_Tableaux_Solver "Logic Solver").
- MFA Model: Predicts a response for a given task by choosing the **M**ost **F**requent **A**nswer
- UBCF Model: Predicts a response by computing a similarity of the top k participants to the current participant and task.
- MBCF Model: Modified variant of the UBCF Model to predict a response by computing a similarity of the top k strategies of mModalSentential to the current participant and task.



#### mModalSentential:
- mModalSentential: As a baseline as implemented by [Guerth 2019](https://github.com/CognitiveComputationLab/cogmods/blob/master/modal/student_projects/2019_guerth/models/mModalSentential_model.py "mModalSentential")
- mModalSentential pre trained: As a baseline as implemented by [Guerth 2019](https://github.com/CognitiveComputationLab/cogmods/blob/master/modal/student_projects/2019_guerth/models/mModalSentential_pretrained_model.py "mModalSentential pre trained")
- mModalSentential optimized: Predicts a response by using the cognitive model mModalSentential with some basic optimizations from the Thesis.
- mModalSentential optimized **+** : Predicts a response by using the cognitive model mModalSentential with basic and further optimizations from the Thesis.
