# SpatialCCobra

This folder contains relational reasoning models benchmarked as part of the Master project by Saku Kärkkäinen in 2020.

Aim of the project was to study the predictive performance of different computational models for spatial reasoning and create a comprehensive benchmark.

Benchmarking is done with the help of the CCOBRA framework.

## To run the benchmark:

* First, make sure you have python and required libraries installed.
    * Required libraries: ccobra, NumPy, Pytorch, auto-sklearn, tqdm, pandas


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
    * Random guess (all benchmarks) (*)
    * Transitive-closure (all benchmarks) (*)
    * Most Frequent Answer (all benchmarks)

* Cognitive models:
    * Dynamic Field Theory -model (only on 'singlechoice' and 'verification'.) (*)  (Kounatidou,  Richter & Schöner, 2018)
    * Preferred Mental Models (all benchmarks) (*) (Ragni & Knauff, 2013)
    * Verbal reasoning (all benchmarks) (*) (Krumnack, Bucher, Nejasmic, & Knauff, 2010)


* Machine learning models:
    * Spatial artificial neural network (ANN) (only on 'singlechoice) (*) (Ragni & Klein, 2012)
    * Multilayer perceptron (MLP) (all benchmarks)
    * Recurrent neural network (RNN) (all benchmarks)
    * Recurrent neural network with LSTM cells (LSTM) (all benchmarks)
    * AutoML (all benchmarks)

(Models marked with a star (*) were originally implemented by other people working at the lab.)

References to cognitive models:

    Kounatidou, P., Richter, M., & Schöner, G.  (2018).  A neural  dynamic  architecture  that  autonomously  builds  mental  models.In  C.  Kalish,  M.  A.  Rau,  X.  J.  Zhu,  &T. T. Rogers (Eds.),Proceedings of the 40th annual meeting of the cognitive science society, cogsci 2018, madison,wi, usa, july 25-28, 2018

    Krumnack, A., Bucher, L., Nejasmic, J., & Knauff, M. (2010). Spatial reasoning as verbal reasoning. In Proceedings of the Annual Meeting of the Cognitive Science Society (Vol. 32, No. 32).

    Ragni, M., & Klein, A. (2012). Deductive Reasoning-Using Artificial Neural Networks to Simulate Preferential Reasoning. In IJCCI (pp. 635-638).

    Ragni,  M., & Knauff,  M.   (2013).   A theory and a compu-tational model of spatial reasoning with preferred mental models.Psychological review,120(3), 561



