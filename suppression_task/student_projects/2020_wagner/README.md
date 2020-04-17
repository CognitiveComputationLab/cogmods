OCF and Reiter Models for the Suppression Task
=======

nonmonotonic reasoning models implemented as part of the Master project by Francine Wagner in 2020.

We implemented several based on two approaches: OCF and Reiter Default Logic. Several improvements for the Reiter model are implemented such as adding a default rule to model modus tollens and affirmation of the consequent.

## Prerequisites

Works with Python 3.7
Used libraries: ccobra, random, numpy

## Models

Several models have been implemented:

- Reiter Model:
  constructs defaults based on the conditional and the statement and executes them. 
- Reiter Model probability
  The keywords "Rarely" and "Mostly" are interpreted differently than in the previous model. We introduce a probability factor. Based on this factor the fact is added to the knowledge base.
- Reiter Model improved
  Some participants tend to use modus tollens or the affirmation of the consequence inference rule. This model adds new defaults in order to model these inference rules.
- Reiter Model improved pretrain
  This models adds the new default rules based on a threshold which is computed in the pretrain method. The threshold is based on the experience of other participants. It is the probability that we mispredict the response.
- OCF Model
  uses the idea of conditional objects with an nonmonotonic behaviour as well as the possible world semantic. The main idea is that the worlds are ranked by their plausibility. The most promising world, that matches our choices will be returned accordingly. We use a belief revision algorithm to compute the ranks.

## Quickstart

1) To run the model in the CCOBRA framework and compare it with other models:

   - Create a json-file for the benchmark (see CCOBRA documentation, already an evaluation.json added to repo)
   - Insert this model and all other models to be tested in the benchmark-file
   - Run the benchmark (./ccobra -s results.csv evaluation.json)