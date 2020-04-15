OCF and Reiter Models for the Suppression Task
=======

nonmonotonic reasoning models implemented as part of the Master project by Francine Wagner in 2020.

We implemented several based on two approaches: OCF and Reiter Default Logic. Several improvements for the Reiter model are implemented such as adding a default rule to model modus tollens and affirmation of the consequent.

## Prerequisites

Works with Python 3.7
Used libraries: ccobra, random, numpy

## Quickstart

1) To run the model in the CCOBRA framework and compare it with other models:

   - Create a json-file for the benchmark (see CCOBRA documentation, already an evaluation.json added to repo)
   - Insert this model and all other models to be tested in the benchmark-file
   - Run the benchmark (./ccobra -s results.csv evaluation.json)