# Spatial and Temporal Reasoning Model

This module contains a spatial and a temporal reasoning model, which were originally implemented by Philip Johnson-Laird in Lisp
(the original Lisp-Code can be found [here](http://mentalmodels.princeton.edu/models/)). This Python translation was implemented by
Christian Breu and Julia Mertesdorf in order to compare the two models with other cognitive models implemented in Python.

Both models implement a cognitive domain of the Mental Model Theory invented by P. Johnson-Laird and can
solve temporal and spatial deduction problems. A detailed description of the functionality of the two
models can be found at the [website of the Cognitive Computation chair](https://www.cognitive-computation.uni-freiburg.de/student-projects-and-theses/documents/2018-project-bachelor-breu-mertesdorf.pdf).

## Prerequisites

The code works with python 3.7.  
Used Libraries: copy, unittest, numpy, matplotlib 


## Quickstart

1) To run a Spatial Problem with the Spatial Model:
   - process_problem(problem-number, name-of-spatial-problem-set, "spatial")
2) To run a Spatial Problem with the Temporal Model:
   - process_problem(problem-number, name-of-spatial-problem-set, "temporal", True)
3) To run a Temporal Problem with the Temporal Model:
   - process_problem(problem-number, name-of-temporal-problem-set, "temporal")
4) To run a Temporal Problem with the Spatial Model:
   - process_problem(problem-number, name-of-temporal-problem-set, "spatial", False)
5) Instead of processing only one problem of a problem-set, all problems of a problem-set can be processed by calling  "process_all_problems" without a problem-number.


## Example calls

1)  One spatial problem solved with the spatial model
```
>>> spatial_model = MainModule()
>>> spatial_model.process_problem(1, INDETERMINATE_PROBLEMS, "spatial")
```

5) All temporal problems of the set "COMBINATION_PROBLEMS" solved with the temporal model
```
>>> temporal_model = MainModule() 
>>> temporal_model.process_all_problems(COMBINATION_PROBLEMS, "temporal")
```
