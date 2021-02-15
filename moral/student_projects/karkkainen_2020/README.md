Author: Saku Kärkkäinen


This folder contains the data and models related to my master thesis "Predictive modelling of human moral reasoning".


## To run the benchmark:
* First, make sure you have python and required libraries installed.
* Required libraries: ccobra, NumPy, Pytorch, sklearn
	 
* You can run benchmarks individually with the command: ccobra _benchmarkname_.json

## Models included:

- Baseline models:
    - Random guess
    - Most frequent answer

- Ethical theories:
    - Utilitarianism
    - Deontologies (Mean principle, 'thou shalt not kill')

- Machine learning:
    - Logistic regression
    - Ridge regression
    - Neural network
    - Random forests

- Other models:
    - Archetypes
        - Custom predictions based on participant 'archetypes'.

## Benchmarks:

- Judgement:
    - Participants are asked to give their judgements on tree variations of the trolley problem.

- Compare:
    - Participants were asked to compare two moral dilemmas, and tell which of them is harder.

- Permissibility:
    - Participants were presented with moral dilemmas, an action related to it, and the question: "Is the action permissible?"

Sources for each benchmark can be found in the 'data' folder.