# Models for modal reasoning
___

### 1. Model: Random Model

**Filename:** random_model.py

**Description:** 
The Random Model assumes a uniform distribution between the possible conclusions and then randomly samples 
one of the two to create a baseline that should be outperformed by every other model.
___
### 2. Model: MFA Model

**Filename:** mfa_model.py

**Description:** 
Most Frequent Answer Model for modal reasoning task.

___
### 3. Model: Logic Model

**Filename:** modal_logic.py

**Description:** 
The Logic Model, because of the nature of the task, should give us useful insights on how accurate our 
reasoners compare against a logic-based approach. Many thanks and credit to its author [Joey Thaidigsman](https://github.com/joeytman/Modal_Logic_Tableaux_Solver "Logic Solver").

**Parameter:** 
The parameter "t" can be used to use different axioms.
1. K = none 
2. S4 = reflexive, transitive 
3. B = reflexive, symmetric 
4. T = reflexive

**Example:** ```{"filename": "../../models/modal_logic.py", "args": {"t": "K"}}```
___

### 4. Model: UBCF Model

**Filename:** modal_ubcf.py

**Description:** 
User Based Collaborative Filtering for modal reasoning.

**Parameter:**
1. The parameter "k" can be used to set the number of top k similar participants to the current one.
2. The parameter "exp" can be used to reduce the impact of the less similar top k participants.

**Example:** ```{"filename": "../../models/modal_ubcf.py", "args": { "k": 5, "exp": 0 }}```
___
### 5. Model: MBCF Model

**Filename:** mModalSentential_mbcf.py

**Description:** 
MBCF stands for Model Based Collaborative Filtering and instead of creating a database out of the 
answering behaviour of all other participants in pre-training, it creates a database of predictions 
on the tasks by all the six strategies for a modal reasoning task possible in mModalSentential.

**Parameter:**
1. The parameter "k" can be used to set the number of top k similar strategies to the current participant.
2. The parameter "exp" can be used to reduce the impact of the less similar top k strategies.

**Example:** ```{"filename": "../../models/mModalSentential_mbcf.py", "args": { "k": 1, "exp": 0 }}```
___
### 6. Model: mModalSentential

**Filename:** mModalSentential_model.py

**Description:** 
Baseline implementation by [Guerth 2019](https://github.com/CognitiveComputationLab/cogmods/blob/master/modal/student_projects/2019_guerth/models/mModalSentential_model.py "mModalSentential")

___
### 7. Model: mModalSentential pre trained

**Filename:** mModalSentential_pretrained_model.py

**Description:** 
Pretrained baseline implementation by [Guerth 2019](https://github.com/CognitiveComputationLab/cogmods/blob/master/modal/student_projects/2019_guerth/models/mModalSentential_pretrained_model.py "mModalSentential pre trained")

___

### 8. Model: mModalSentential optimized (+)

**Filename:** mModalSentential_optimized.py

**Description:** 
For a full description check the bachelors thesis *Optimization of Predictive Models for Individual Human Reasoning*.
Because of permutation testing, prediction times are very long. Enable debug to see what's going on.

**Parameter:**
1. Enable/disable further optimizations (mModalSentential optimized +) with the "ex" parameter.
2. Enable/disable debug prints for each fold to be displayed with the "debug" parameter.

**Example:** ```{"filename": "../../models/mModalSentential_optimized.py", "args": {"ex": true, "debug": true}}```
___


