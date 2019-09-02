# Using Dynamic Field Theory to Autonomously Build Mental Models

This project is aimed at using the architecture presented in Kounatidou et al. (2018) to create mental models for relational spatial reasoning that is compatible with [CCOBRA](https://github.com/CognitiveComputationLab/ccobra).

Here we propose using the architecture presented in Kounatidou et al. (2018) and the C++ libraries presented in Lomp et al. (2013) to create a CCOBRA compatible model. The CEDAR framework is run in the background and invoked using a python a script which represents the interface between CCOBRA and CEDAR.

Our program is able to translate the tasks from CCOBRA into an experiment file that the DFT model implemented using the CEDAR framework is able to understand.
This experiment file is then manually fed into CEDAR and the result manually returned through a simple GUI into our program.
These results are then processed by CCOBRA to create the statistics and accuracy evaluation of the model.

Unfortunately, we were not able to find a reasonable method to automatically run the DFT architecture, nor were we able to find a reasonable method to connect through the backend of the CEDAR framework to run experiments.
The DFT architecture always returns the same results, given the same input.
As a result, after we were able to run every variation of the tasks given by CCOBRA manually with our program, we were able to create a lookup table.
This lookup table and its interface to the CCOBRA framework is what was used to gather the final data and statistics that we presented in our paper.

## Directory Structure
The original DFT qrchitecture can be found under `final-model/mental_imagery_extended.json` and our extended DFT architecture can be found under `final-model/mental_imagery_extended`.

Our final lookup table based program can be found under the `final-model` directory while the original program can be found within `src`.