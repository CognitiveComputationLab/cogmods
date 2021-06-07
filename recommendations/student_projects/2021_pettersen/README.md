Author: Peter Pettersen

This folder contains the data and models related to my master thesis "Recommendation Algorithms with Cognitive Analysis".

## To run the benchmark:
* First, make sure you have python and required libraries installed.
* Required libraries: ccobra, numpy, pandas, surprise
* You can run benchmarks individually with the command: ccobra _benchmarkname_.json

## Models included:

- Baseline:
    - Mean value of training data

- Content-based:
    - Using similar genres

- Collaborative filtering
    - Item-based
    - User-based
    - Matrix Factorization, SVD

## Task:

- Recommend movie ratings (0.5 - 5.0) for individual users based on their rating history, similar users, and movie similarities.
- Supplemental cognitive analysis includes the study of influential users measured by nearest neighbors (for the user-based CF)
