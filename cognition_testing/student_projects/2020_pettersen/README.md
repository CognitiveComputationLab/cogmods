# Cognitive Reflection Test (CRT) Predictive Model

The goals of this project are:
1. Determine the most descriptive CRT questions
2. Develop a model to predict CRT responses

## Models
The models can be run and compared side-by-side in CCOBRA with the file "CRT_bench.json". The CRT data tested is located in the "data" folder and is split for training/testing as "CRT_trainset.csv" and "CRT_testset.csv". There are 7 CRT questions with responses regarded as TRUE or FALSE. There are 138 participants who fully answered all 7 questions. The models are graded on predictive ability of reponses to 3 of the CRT questions. Some of the models may use the other remaining 4 CRT questions as prior knowledge. The accuracy results are displayed on an overall level, based on participants and the 3 CRT questions.

### Random
Generates are random TRUE/FALSE response.

### Always False
Returns the FALSE response.

### MFA
The most frequent answer (MFA) returns the most frequently reported response of all the particpants so far, per CRT question.

### Conditional
Trains and uses a naive Bayes classifier to generate a condtional reponse, based on other CRT responses.

### Combined
Uses a weighting for the MFA and Conditional models, combining the results to varying degrees. The current weight is w = 0.5, which is an equal share of each model.

## Dependencies
-numpy
-sklearn.naive_bayes