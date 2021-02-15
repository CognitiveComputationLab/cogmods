"""
CCobra Model - UBCF (User based collaborative filtering) for propositional reasoning

Parameter:

1. k:   Is the number of similar participants to be considered in computing the collaborative prediction vector.
        Defaults to 12 as that is the recommended Value.
2. exp: With this parameter the similarities of the top k participants are modified so that the impact of the not
        so similar ones is reduced. Increasing this parameter reduces the impact of the not so similar participants.
        Defaults to 0.1 as that is the recommended Value.
"""


import copy
import ccobra
import numpy as np


class PropositionalUBCF(ccobra.CCobraModel):
    def __init__(self, name='Propositional UBCF', k=12, exp=0.1):
        name = f"k = {k}"
        super(PropositionalUBCF, self).__init__(name, ["propositional"], ["single-choice"])

        # Lists of all propositional tasks and responses
        self.PROP_TASK = {
            'Not;B/or;A;B/not;and;A;B': 0,
            'Not;A/or;A;B/not;and;A;B': 1,
            'Not;B/or;or;A;B;and;A;B': 2,
            'Not;A/or;or;A;B;and;A;B': 3,
            'B/or;or;A;B;and;A;B': 4,
            'A/or;or;A;B;and;A;B': 5,
            'Not;A/if;A;B': 6,
            'Not;B/if;A;B': 7,
            'B/if;A;B': 8,
            'A/if;A;B': 9,
            'B/iff;A;B': 10,
            'A/iff;A;B': 11,
            'Not;B/iff;A;B': 12,
            'Not;A/iff;A;B': 13,
            'B/or;A;B/not;and;A;B': 14,
            'A/or;A;B/not;and;A;B': 15,
        }
        self.PROP_RESP = {
            'A': 0,
            'not;A': 1,
            'B': 2,
            'not;B': 3,
            'nothing': 4
        }

        # Vector of the participants answers
        self.user_pred = np.zeros((16,), dtype='int')
        self.database = []

        # Propositional Similarity Parameters
        self.p_props = np.random.uniform(size=16)
        self.p_props = np.ones((16,))

        # Recommender Parameters
        self.k = k  # 12
        self.exp = exp  # 0.1

    def pre_train(self, dataset):
        # Create a Database with all the necessary information about the other participents
        for subj_data in dataset:
            self.database.append((
                self.extract_pred_info(subj_data)
            ))

    def predict(self, item, **kwargs):
        task_idx = self.PROP_TASK[self.tuple_to_string(item.task)]
        sims = []

        # Find similarities between the current participant and the others
        for other in self.database:
            sim = self.similarity(self.user_pred, other)
            sims.append((sim, other[task_idx] - 1))

        # Compute the collaborative prediction vector
        top_sim = sorted(sims, reverse=True)[:self.k]
        prediction = np.zeros((5,))
        for sim, resp in top_sim:
            prediction[resp] += sim ** self.exp

        # Slice the prediction vector and return the prediction
        pred_idx = np.argmax(prediction)
        for key, value in self.PROP_RESP.items():
            if pred_idx == value:
                return key

    def adapt(self, item, response, **kwargs):
        # Create User_pred with all necessary information about the current participant
        task_idx = self.PROP_TASK[self.tuple_to_string(item.task)]
        resp_idx = self.PROP_RESP[self.tuple_to_string(response)]
        self.user_pred[task_idx] = resp_idx + 1

    def similarity(self, a, b):
        # Compute similarity based on responses
        pred_sim = np.sum((a == b) * self.p_props) / max(1, 16 - np.sum(a == 0))
        return pred_sim + 0.000001  # Add a const to avoid problems with 0

    def extract_pred_info(self, user_data):
        # Take the user_data and extract the data for a specific user
        pred_info = np.zeros((16,), dtype='int')
        for task_data in user_data:
            task_idx = self.PROP_TASK[self.tuple_to_string(task_data['item'].task)]
            resp_idx = self.PROP_RESP[self.tuple_to_string(task_data['response'])]
            pred_info[task_idx] = resp_idx + 1

        # Return a encode prediction vector
        return pred_info

    @staticmethod
    def tuple_to_string(tuptup):
        def join_deepest(tup, sep=';'):
            if not isinstance(tup, list):
                return tup
            if not isinstance(tup[0], list):
                return sep.join(tup)
            else:
                for idx in range(len(tup)):
                    tup[idx] = join_deepest(tup[idx], sep)
                return tup

        tup = copy.deepcopy(tuptup)
        tup = join_deepest(tup, ';')
        tup = join_deepest(tup, '/')

        # Sort the tuples
        tup = sorted(tup) if isinstance(tup, list) else tup

        # tup = join_deepest(tup, '|')
        return tup
