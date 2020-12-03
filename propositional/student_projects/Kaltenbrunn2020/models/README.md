# mSentential for propositional reasoning

### 1. Model: mSentential

**Filename:** mSentential_model.py

**Description:** mSentential for propositional reasoning (based on the mModalSentential by Guerth) and Paper: 
"Facts and Possibilities: A Model-Based Theory of Sentential Reasoning"

**Parameter:** Sigma - Is the probability that system 2 is engaged in making inferences Defaults to 0.2 as stated in paper

### 2. Model: mSentential tuned

**Filename:** mSentential_model_tuned.py

**Description:** This model is tuned for better performance with the CCobra framework:
Generally system 2 performs much better then system 1, so system 2 always
provides the prediction, for this model. Further, a dict of answered questions
and responses is kept. With that said, a bool of answered 'nothing' is saved
as well. This works as an indicator if a participant is not possible to
comprehended a given task. And with that, might not be able to comprehended
similar tasks.

**Options:**
1. Consistence: If System 2 predicts multiple possible answers, this checks if
                system 1 has predicted an answer and chooses the answer that is
                predicted by both systems. This provides prediction consistence
                between both systems.

2. Necessary:   mSentential provides both necessary and possible predictions
                for a given premise. Some participants may only provide an
                answer if it follows necessarily. So with that Option it is
                possible to return 'nothing' if nothing follows necessarily.

3. Size_limit:  Possible working memory size limit. This option, when enabled
                checks the task for more then 3 sentential connectives because
                this may lead to working memory overload of the participant and
                nothing is predicted.

4. Memory:      If this option is enabled the dict for answered questions and
                responses is used to predict answers if question was already
                answered.