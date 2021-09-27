## mModalSentential created by [Guerth 2019](https://github.com/CognitiveComputationLab/cogmods/tree/master/modal/student_projects/2019_guerth/models/mmodalsentential "mModalSentential")

There are three main modules: Parser, Model Builder and Reasoner. Furthermore
there are two minor modules: Agent and Logger.
The Reasoner module is the central module which provides the reasoning task
functions. It calls the Parser module to parse input strings and the Model builder
module to build models. The Agent module simulate a reasoning agent, for example
a human individual. The Agent module is basically a wrapper around the Reasoner
module and stochastic parameters which control whether system 1 or 2 is used and
whether or not weak necessity is used. Lastly, the Logger module implements a
simple logging/printing mechanism so that the other modules can output additional
information other than the results of the reasoning tasks.

### Added two  function: count_operators, count_models 