# SpatialCCobra

This folder contains relational reasoning models implemented as part of the Master project by Saku Kärkkäinen in 2020.

Aim of the project was to study the predictive performance of different computational models for spatial reasoning and create a comprehensive benchmark.

Benchmarking is done with the help of the CCOBRA framework.

## To run the benchmark:

* First, make sure you have python and required libraries installed.
    * Required libraries: NumPy, Pytorch, auto-sklearn, tqdm, pandas

* NOTE: A customized version of the CCOBRA framework is used on some of the models, if you're already working with CCOBRA, I recommend setting up a virtual environment (https://github.com/pyenv/pyenv) to prevent clashing. Install the customized CCOBRA framework with the following command: pip install /PATH/TO/ccobra_modified
    * However, the customized version is only needed by the Most-Frequent-Answer and Dynamic field theory -models. All the other models will work on the normal version of ccobra.

* To use the PRISM-model, you also have to install the spatial reasoner.
    * This is done by traversing to the prism folder under: benchmarking/models/cognitive/prism and running the command: python setup.py install
	   

* To run the benchmarks:
    * On linux or ubuntu for windows you can use the shell script: runBenchmarks.sh
    * Othwerwise, you can run benchmarks individually with the command: ccobra benchmarkname.json
    * NOTE: Running the benchmark in its entirety takes several hours.




## Current status
The benchmark is ran on 4 different datasets, which all contain subject data from different spatial relational reasoning tasks.

The benchmark currently consists of the following models:

* Baseline models:
    * Random guess (all benchmarks)
    * Transitive-closure (all benchmarks)
    * Most Frequent Answer (all benchmarks)

* Cognitive models:
    * Dynamic Field Theory -model (only on 'singlechoice' and 'verification'.) 
    * Preferred Mental Models (all benchmarks) 
    * Verbal reasoning (all benchmarks)


* Machine learning models:
    * Spatial artificial neural network (ANN) (only on 'singlechoice)
    * Multilayer perceptron (MLP) (all benchmarks)
    * Recurrent neural network (RNN) (all benchmarks)
    * Recurrent neural network with LSTM cells (LSTM) (all benchmarks)
    * AutoML (all benchmarks)


