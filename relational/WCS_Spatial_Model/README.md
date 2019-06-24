# Weak Completion Semantics Model for Spatial Reasoning

This model implements the Weak Completion Semantics approach to solve spatial reasoning tasks in python. The problem solving process is divided into three phases: 
  1. model construction phase (the first and preferred mental model is constructed)
  2. model inspection phase (searching for the query-relation in the preferred model and verifying whether it holds)
  3. model variation phase (construction of alternative models of the problem premises)

The model was origially implemented in Prolog and comprised only the first two phases. The rules 1-11 for the construction of the preferred mental model are taken from the according paper [Dietz, Hölldobler, Höps: A Computational Logic Approach to Human Spatial Reasoning (2015).](http://www.wv.inf.tu-dresden.de/Publications/2015/report-15-02.pdf) Their model is here extended to the third phase (variation phase).

Note: This model is restricted to left- and right-relations. To be able to solve two-dimensional spatial reasoning tasks, the model needs to be extended to all four cardinal directions.

## Prerequisites

The code works with python 3.7.  
Used Libraries: ccobra, itertools


## Quickstart

1) To run the model in the CCOBRA framework and compare it with other models:
   - Create a json-file for the benchmark (see CCOBRA documentation)
   - Insert this model and all other models to be tested in the benchmark-file
   - Run the benchmark (python runner.py ..\path\to\benchmark\name_of_benchmark.json). 

2) To run the model stand-alone:
   - Create an instance of the model
   - To print the problem solving process, set the model parameter "print_output" to True
   - To enable or disable the variation phase, set the model parameter "perform_variation" accordingly to True or False.
   - Create a problem consistiting of multiple premises and a query, encoded als lists of strings
   - Call the function:"compute_problem(problem_premises, query)" to start the problem computation.


## Example calls 

2) Running the model stand-alone and computing a simple spatial problem
```
>>> model = WCSSpatial()
>>> model.print_output = True
>>> model.perform_variation = True

>>> problem_premises = [["left", "A", "B"], ["left", "M", "B"]]
>>> query = ["left", "M", "A"]
>>> model.compute_problem(problem_premises, query)
```
