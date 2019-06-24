# Weak Completion Semantics Model for the Wason Selection Task

The WCSWason model can solve instances of the Wason Selection Task using the Weak Completion Semantics. The Wason Selection Task is a problem demonstrating the systematic deviations of human reasoning from classical logic and was formulated in a variety of different cases - the abstract, social and deontic case. In every problem formulation, subjects have to turn cards (in the abstract case: "D", "K", "3", "7") to verify a given rule (implication, in the abstract case: 3 <-- D). Common patterns of cards which are turned, comprise the following: "D", "D, 3", "D, 3, 7", "D, 7". Depending on the case, some patterns are more often selected than others - in the abstract case, most people tend to only turn the cards "D" or "D" and "3", in the deontic case however, many people are able to derive the logically correct solution.

The WCSWason model attempts to model these different cases with two suggested principles - abduction and contraposition, by assigning them different probabilities to be activated for the three case types. For more detailed information on the model and the background, see the paper [Breu, Ind, Mertesdorf, Ragni (2019) The Weak Completion Semantics Can Model Inferences of Individual Human Reasoners](https://link.springer.com/chapter/10.1007/978-3-030-19570-0_33).


## Prerequisites

The code works with python 3.7
Used libraries: numpy, random, scipy.optimize


## Quickstart

1) To try out one run with manually specified probabilities for the additional principles:
    - Create a model and set the principle_probabilities (f.i. model.principle_probabilities = \[0.5, 0.5\])
    - Simulate one run by calling the function "compute_one_trial()"

2) To see how every pattern of the four canonical cases is derived:
    - model.compute_every_variation_once()

3) To compute a certain case-type for a variabel number of times (result is in percentage):
    - model.compute_case_xtimes("abstract", 10000)

4) To optimize the probabilities for the additional principles for certain target-data (returns the optimal probability-values for the given target-data):
    - Select a list of target values for the four possible card patterns (f.i. target = \[36, 39, 5, 19\])
    - Call the function "optimize" with the specified target values: "model.optimize(target)"

5) To calculate the average result for a specific case over many runs (result in percentage):
    - model.average_results("abstract", 50)
  
The problem solving process can be printed by enabling the model parameter "print_output" (model.print_output = True)

## Example calls

1) Simulate one run of the wason selection task with the selected probabilities of 0.7 (abduction) and 0.4 (additional rule - contraposition) and print the problem solving process.
```
>>> model = WCSWason()
>>> model.print_output = True
>>> model.principle_probabilities = [0.7, 0.4]
>>> model.compute_one_trial()
```

3) Compute the deontic case 5000 times 
```
>>> model = WCSWason()
>>> model.compute_case_xtimes("deontic", 5000)
```
