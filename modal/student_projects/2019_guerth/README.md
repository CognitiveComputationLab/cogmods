# mModalSentential

The goals of this project are:
1. Writing a modular Python program, that has the same functionality as the existing Lisp program 'mSentential.lisp' \
(Which can be found here: http://mentalmodels.princeton.edu/models/)
2. Extending this program to support modal operators (both in premises and conclusions).

Models based on this program can then be compared to models based on formal modal logic. The modal logic solver used is a slightly modified version of this project: https://github.com/joeytman/Modal_Logic_Tableaux_Solver . Many thanks and credit to its author Joey Thaidigsman.

## Models
The idea for the CCOBRA models is to use the reasoning tasks of the developed
mModalSentential program to predict answers of the subjects in modal reasoning
verification tasks. In other words, the model gets as input premises and a conclusion
and has to decide if the subject answers ’Yes’ or ’No’ to the question wheter or not the conclusion follows from the premises. A specific individual may reason on a more intuitive basis (system 1) or on a more deliberate basis (system 2). He or she may use weak necessity. He or she may understand the task as to choose a reasoning strategy that more resembles that of the programs possible function rather than the necessary funtion. And so on.

The model **mModalSentential_model.py** uses the adapt function to change the programs strategy to
better fit the subjects strategy by keeping a statistic of what strategy in the past
would have predicted an individual best. This strategy is then chosen for the next
prediction. This approach is biased for using the system 1 necessary function as
default start strategy.

The model **mModalSentential_pretrained_model.py** uses the pre_train function to make an initial order of the best strategies and tries to make out the best default strategy
which gets used as an initial guess for a new subject and the best alternative to the
strategy which gets applied if it is better than the default strategy. \
NOTE: The current implementation of this model is too slow and therefore not included in the benchmarks.

The models **modal_logic_K.py**, **modal_logic_S4.py**, **modal_logic_B.py** and **modal_logic_T.py** are used for comparison. They are based on a satisfiability solver for modal logic that allows for setting up frames to be reflexive, transitive or symmetric. (K = none. S4 = reflexive, transitive. B = reflexive, symmetric. T = reflexive). In order to apply the program to a verification task ’Does conclusion B follows from premise A?’, you have to test if the solver decides that ¬(A → B) is not satisfiable. If that is the case then A → B is valid in modal logic and the answer to the verification task is ’Yes’ otherwise it is ’No’.



## Dependencies
- numpy
- networkx \
(only necessary for the modal logic models, not for the mModalSentential models)