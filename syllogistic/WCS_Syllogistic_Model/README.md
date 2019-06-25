# Weak Completion Semantics Model for Syllogistic Reasoning

This model implements the Weak Completion Semantics approach to solve syllogistic reasoning tasks. The background and concepts of the implementation are based on the following papers:
- [E.A. Dietz (2017). From Logic Programming to Human Reasoning: How to be Artificially Human (Dissertation, Chapter 7)](https://run.unl.pt/bitstream/10362/31248/1/Saldanha_2017.pdf)
- [Costa, Dietz, Hölldobler, Ragni (2016). Syllogistic Reasoning under the Weak Completion Semantics](https://www.researchgate.net/profile/Steffen_Hoelldobler/publication/305800319_Two-Valued_Logic_is_Not_Sufficient_to_Model_Human_Reasoning_but_Three-Valued_Logic_is_A_Formal_Analysis/links/57d297e808ae601b39a3fa3a/Two-Valued-Logic-is-Not-Sufficient-to-Model-Human-Reasoning-but-Three-Valued-Logic-is-A-Formal-Analysis.pdf)
- [Costa, Dietz, Hoelldobler (2017). Monadic Reasoning using Weak Completion Semantics](http://ysip2.computational-logic.org/ceur/YSIP2/paper9.pdf)
- [Costa, Dietz, Hölldobler, Ragni (2017). A Computational Logic Approach to Human Syllogistic Reasoning](https://pdfs.semanticscholar.org/b51c/c0b9d6e37ee78d8bd3587bcc1bc5412df51f.pdf)
- [Dietz, Hoelldobler, Moerbitz (2017). The Syllogistic Reasoning Task: Reasoning Principles and Heuristic Strategies in Modeling Human Clusters](https://link.springer.com/chapter/10.1007/978-3-030-00801-7_10)

Altogether, the model uses 6 basic and 4 advanced reasoning princples to encode and solve syllogistic reasoning tasks (for further information, the princples are explained in detail in the according papers). The 6 basic principles are used for every subject and task, whereas the advanced principles are only activated if they accurately predict the subject´s answers (usage of the adapt-function of CCOBRA-models). For every subject, an array of counters keeps track of the most succesfull principle combinations to successfully predict the answers of the current subject. For the next task, the principle combination which has been the most succesfull so far, is selected, in order to adapt the model to the current subject.


## Prerequisites

The code works with python 3.7.
Used libraries: ccobra, random, operator, pandas


## Quickstart

1) To run the model in the CCOBRA framework and compare it with other models:
- Create a json-file for the benchmark (see CCOBRA documentation)
- Insert this model and all other models to be tested in the benchmark-file
- Run the benchmark (python runner.py ..\path\to\benchmark\name_of_benchmark.json). 

2) To calculate and save the solutions to the 64 syllogisms for one specified principle combination (f. i. \[0, 0, 0, 0, 1\]), run the following function. The results of the computation are saved as a csv file in the folder, in which this python file is located.
- compute_one_trial(\[0, 0, 0, 0, 1\], "csv_table_name")

3) To calculate the solutions to the 64 syllogisms for all possible principle combinations (32 in total), run the following function (this needs to be done before using the WCS-model in a CCOBRA benchmark, so that the answers can be looked up in the csv-file).
- compute_all_variations()

4) To see what the WCS model is calculating in detail for one given problem and principle combination, enable the model parameter "print_output" and call the following function:
- compute_problem_from_scratch(syllogistic_problem, principle_combination)


## Example calls

4) Running the model stand-alone and computing a syllogistic problem (here: "AA1") with a manually selected principle combination (here: \[1, 1, 1, 1, 0\]) and printed problem solving process
```
>>> model = WCSSyllogistic()
>>> model.print_output = True
>>> model.compute_problem_from_scratch("AA1", [1, 1, 1, 1, 0])
```


## Statistics for the prediction-accuracy of the principle combinations

The two statistics-csv-files contain some information on the potential prediction accuracy of the wcs model with regards to the principle combinations
(one file for the training-set (Ragni2016) and one for the test-set (Veser2018)). 

The collected statistics comprise the following:

- id: id of the subject
- theoretical_best: the number of syllogisms that can be potentially predicted by the wcs-model for the according subject
- max: the number of syllogisms that can be predicted by the best principle combination for the according sujbect (one fix combination). 
- min: the number of syllogisms that can be predicted by the worst principle combination for the according subject
- span_max_min: max - min
- span_best_max: theoretical_best - max

The columns starting with "mean_" show the mean values for the five statistics over all subjects of the training/test-set.
