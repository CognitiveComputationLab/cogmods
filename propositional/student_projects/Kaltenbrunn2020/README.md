# Predictive models for individual human reasoning:
Bachelor's project of Leon Kaltenbrunn (2020). Investigates propositional reasoning using mSentential and baselines for
comparison.

## Prerequisites:
Python 3.8 Libraries: ccobra, pandas, numpy

## Models:
### Baseline:
- Random Model: Predicts a response for a given task at random.
- Logic Model: Predicts a response for a given task with a pure logical approach.
- MFA Model: Predicts a response for a given task by choosing the **M**ost **F**requent **A**nswer
- UBCF: Predicts a response by computing a similarity of the top k participants to the current participant and task.

#### mSentential:
- mSentential: Predicts a response by using the Cognitive Model mSentential
- mSentential tuned: Predicts a response by using the Cognitive Model mSentential but with additional 
options to improve performance


