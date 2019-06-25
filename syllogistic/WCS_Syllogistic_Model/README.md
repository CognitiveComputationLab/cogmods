# Weak Completion Semantics Model for Syllogistic Reasoning

This model implements the Weak Completion Semantics approach to solve syllogistic reasoning tasks. The background and concepts of the implementation
are based on the following papers:
- E.A. Dietz (2017). From Logic Programming to Human Reasoning: How to be artificially Human (Dissertation, Chapter 7)
- Costa, Dietz, Hölldobler, Ragni (2016). Syllogistic Reasoning under the Weak Completion Semantics
- Costa, Dietz, Hoelldobler (2017). Monadic Reasoning using Weak Completion Semantics
- Costa, Dietz, Hölldobler, Ragni (2017). A Computational Logic Approach to Human Syllogistic Reasoning
- Dietz, Hoelldobler, Moerbitz (2017). The Syllogistic Reasoning Task: Reasoning Principles and Heuristic Strategies in Modeling Human
  Clusters

Altogether, the model uses 6 basic and 4 advanced reasoning princples to encode and solve syllogistic reasoning tasks
(for further information, the princples are explained in detail in the according papers). The 6 basic principles are used for every
subject and task, whereas the advanced principles are only activated if they accurately predict the subject´s answers (usage of the adapt-function of CCOBRA-models).
For every subject, an array of counters keeps track of the most succesfull principle combinations to successfully predict the answers of the current
subject. For the next task, the principle combination which has been the most succesfull so far, is selected, in order to adapt the model to the current
subject.
