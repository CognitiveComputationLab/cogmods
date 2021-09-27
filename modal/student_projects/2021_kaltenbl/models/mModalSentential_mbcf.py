import ccobra
import numpy as np
from mmodalsentential.assertion_parser import ccobra_to_assertion
from mmodalsentential.reasoner import necessary, possible


class mModalSentential_MBCF(ccobra.CCobraModel):
    def __init__(self, name='mModalSentential_MBCF', k=3, exp=0):
        name = f"k = {k}, exp = {exp}"
        super(mModalSentential_MBCF, self).__init__(name, ["modal"], ["verify"])

        # Lists of all modal tasks and responses
        self.PROP_TASK = {}
        self.PROP_RESP = {
            False: 0,
            True: 1,
        }

        # List of all mModalSentential models
        self.MODEL_LIST = {
            "system 1": {"system": 1, "weak": False, "poss": False},
            "system 2": {"system": 2, "weak": False, "poss": False},
            "system 1 weak": {"system": 1, "weak": True, "poss": False},
            "system 2 weak": {"system": 2, "weak": True, "poss": False},
            "system 1 poss": {"system": 1, "weak": False, "poss": True},
            "system 2 poss": {"system": 2, "weak": False, "poss": True}
        }

        # Number of tasks in current dataset
        self.c_task_ids = 0

        # Vector of the current participants answers
        self.user_pred = np.zeros((self.c_task_ids,), dtype='int')

        # Database for all model prediction vectors
        self.database = []

        # Modal similarity parameters
        self.m_props = np.random.uniform(size=self.c_task_ids)
        self.m_props = np.ones((self.c_task_ids,))

        # Recommender Parameters
        self.k = k
        self.exp = exp

    def pre_train(self, dataset):
        task_id = 0

        # Fill task list
        for subj_train_data in dataset:
            for seq_train_data in subj_train_data:
                item = seq_train_data['item']
                task = ccobra_to_assertion(item.task[0])
                choices = ccobra_to_assertion(item.choices[0][0])
                if task not in self.PROP_TASK:
                    self.PROP_TASK[task] = {}
                if choices not in self.PROP_TASK[task]:
                    self.PROP_TASK[task][choices] = task_id
                    task_id += 1

        # Count all tasks in dataset to correctly choose size of np arrays
        self.c_task_ids = task_id
        self.user_pred = np.zeros((self.c_task_ids,), dtype='int')
        self.m_props = np.random.uniform(size=self.c_task_ids)
        self.m_props = np.ones((self.c_task_ids,))

        # Create prediction vectors for every model
        for model in self.MODEL_LIST:
            pred_info = np.zeros((self.c_task_ids,), dtype='int')
            for task in self.PROP_TASK:
                for choice in self.PROP_TASK[task]:
                    task_idx = self.PROP_TASK[task][choice]
                    if self.MODEL_LIST[model]["poss"]:
                        resp_idx = self.PROP_RESP[possible(task, choice, self.MODEL_LIST[model]["system"])]
                    else:
                        resp_idx = self.PROP_RESP[
                            necessary(task, choice, self.MODEL_LIST[model]["system"],
                                      self.MODEL_LIST[model]["weak"])]
                    pred_info[task_idx] = resp_idx + 1
            self.database.append(pred_info)

    def similarity(self, a, b):
        # Compute similarity based on responses
        pred_sim = np.sum((a == b) * self.m_props) / max(1, self.c_task_ids - np.sum(a == 0))
        return pred_sim + 0.000001  # Add a const to avoid problems with 0

    def predict(self, item, **kwargs):
        task = ccobra_to_assertion(item.task[0])
        choices = ccobra_to_assertion(item.choices[0][0])
        task_idx = self.PROP_TASK[task][choices]
        sims = []

        # Find similarities between the current participant and the others
        for other in self.database:
            sim = self.similarity(self.user_pred, other)
            sims.append((sim, other[task_idx] - 1))

        # Compute the collaborative prediction vector
        top_sim = sorted(sims, reverse=True)[:self.k]
        prediction = np.zeros((2,))
        for sim, resp in top_sim:
            prediction[resp] += sim ** self.exp

        # Slice the prediction vector and return the prediction
        pred_idx = np.argmax(prediction)
        for key, value in self.PROP_RESP.items():
            if pred_idx == value:
                return key

    def adapt(self, item, response, **kwargs):
        task = ccobra_to_assertion(item.task[0])
        choices = ccobra_to_assertion(item.choices[0][0])

        # Create User_pred with all necessary information about the current participant
        task_idx = self.PROP_TASK[task][choices]
        resp_idx = self.PROP_RESP[response]
        self.user_pred[task_idx] = resp_idx + 1
