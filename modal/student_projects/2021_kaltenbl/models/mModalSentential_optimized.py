import itertools
import ccobra
from collections import Counter
import operator
import numpy as np
from mmodalsentential.assertion_parser import ccobra_to_assertion, count_operators
from mmodalsentential.reasoner import necessary, possible, count_models

curr_subj_count = 1


class MentalModel(ccobra.CCobraModel):
    def __init__(self, name='mModalSentential optimized', ex=True, debug=False):
        if ex:
            name = f'mModalSentential optimized +'
        super(MentalModel, self).__init__(name, ['modal'], ['verify'])

        self.task_profile = {}  # Task knowledgebase
        self.user_profile = {}  # User Profile
        self.last_prediction = False  # Last prediction of Cobra model
        self.last_model_prediction = ""  # Model to make the last prediction
        self.model_profile = {"system 1": 0,
                              "system 2": 0,
                              "system 1 poss": 0,
                              "system 2 poss": 0,
                              "system 1 weak": 0,
                              "system 2 weak": 0}

        self.external = ex  # Enable/disable external optimizations
        self.debug = debug  # Enable/disable debug prints
        self.pcs_threshold = -0.5  # Init Value for PCS lower threshold range(-0.5 -> -1.0)
        self.max_subj_count = 1  # Get Max Subject Count for Epoch Print
        self.best_param = []  # Save best parameters learned in permutation pre training
        self.r_type_delta = [1, 2, 3, 4, 5]  # Init Value for Reasoner Type delta (1 -> 5)
        self.m_type_delta = [1, 2, 3, 4, 5]  # Init Value for Modality Type delta (1 -> 5)
        self.delta = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Init Value for delta between s_pre model and p model (1 -> 10)
        self.s_limit = [0, 1]  # Init Value for s_connect threshold s_limit range(5+)
        self.memory = [True, False]  # Enable/disable predictions by Memory
        self.s_connect = [True, False]  # Enable/disable predictions by max sentential connectives (Size Limit)
        self.beta = [True, False]  # Enable/disable prediction modifications by beta

    def pre_train(self, dataset):
        # Creating knowledgebase
        for subj_train_data in dataset:
            self.max_subj_count += 1
            for seq_train_data in subj_train_data:
                init = {"dist": {True: 0,
                                 False: 0},
                        "mPred": {},
                        "pcs": {"dRatio": 0,
                                "dMajority": False,
                                "mRatio": 0,
                                "mMajority": "",
                                "match": True,
                                "score": 0
                                },
                        "indicator": {"r_type": 0,
                                      "m_type": 0,
                                      "n_model1": 0,
                                      "n_model2": 0,
                                      "s_connect": 0}
                        }
                item = seq_train_data['item']
                response = seq_train_data['response']
                task = ccobra_to_assertion(item.task[0])
                choices = ccobra_to_assertion(item.choices[0][0])

                # Add Task & Choice to knowledgebase
                if task not in self.task_profile:
                    self.task_profile[task] = {}
                if choices not in self.task_profile[task]:
                    self.task_profile[task][choices] = init

                # Create response distribution
                if response not in self.task_profile[task][choices]["dist"]:
                    self.task_profile[task][choices]["dist"][response] = 1
                else:
                    self.task_profile[task][choices]["dist"][response] += 1

                # Get predictions of the model
                mpred = self.task_profile[task][choices]["mPred"]
                if "system 1" not in mpred:
                    mpred["system 1"] = necessary(task, choices)
                if "system 2" not in mpred:
                    mpred["system 2"] = necessary(task, choices, 2)
                if "system 1 weak" not in mpred:
                    mpred["system 1 weak"] = necessary(task, choices, 1, True)
                if "system 2 weak" not in mpred:
                    mpred["system 2 weak"] = necessary(task, choices, 2, True)
                if "system 1 poss" not in mpred:
                    mpred["system 1 poss"] = possible(task, choices, 1)
                if "system 2 poss" not in mpred:
                    mpred["system 2 poss"] = possible(task, choices, 2)

                # Setup Strategy Array in pre Training
                for system, value in mpred.items():
                    if value == response:
                        self.model_profile[system] += 1

                indicator = self.task_profile[task][choices]["indicator"]

                # Count sentential connectives for Task & Choice
                # Note <>, [] are not counted!
                if indicator["s_connect"] == 0:
                    count_task = count_operators(item.task[0])
                    count_choice = count_operators(item.choices[0][0])
                    indicator["s_connect"] = count_task + count_choice

                    # Find the max for s_limit
                    if max(self.s_limit) < indicator["s_connect"]:
                        self.s_limit.clear()
                        self.s_limit.append(indicator["s_connect"])

                # Count number of mental models for the Task & Choice
                if indicator["n_model1"] == 0 or indicator["n_model2"] == 0:
                    n_model1 = count_models(task, 1) + count_models(choices, 1)
                    n_model2 = count_models(task, 2) + count_models(choices, 2)
                    indicator["n_model1"] = n_model1
                    indicator["n_model2"] = n_model2

        # Modify Knowledgebase
        for task in self.task_profile:
            for choices in self.task_profile[task]:
                pcs = self.task_profile[task][choices]["pcs"]
                indicator = self.task_profile[task][choices]["indicator"]
                mpred = self.task_profile[task][choices]["mPred"]

                # Dist Calculations
                c_true = self.task_profile[task][choices]["dist"][0]
                c_false = self.task_profile[task][choices]["dist"][1]
                if c_true > c_false:
                    pcs["dRatio"] = c_true / (c_true + c_false)
                    pcs["dMajority"] = False
                else:
                    pcs["dRatio"] = c_false / (c_true + c_false)
                    pcs["dMajority"] = True

                # mPred Calculations
                mratioarray = Counter(mpred.values()).most_common()
                if len(mratioarray) == 1:
                    pcs["mRatio"] = 1
                    pcs["mMajority"] = mratioarray[0][0]
                else:
                    c_true = sorted(mratioarray, reverse=True)[0]
                    c_false = sorted(mratioarray, reverse=True)[1]
                    if c_true[1] > c_false[1]:
                        pcs["mRatio"] = c_true[1] / (c_true[1] + c_false[1])
                        pcs["mMajority"] = True
                    elif c_true[1] == c_false[1]:
                        pcs["mRatio"] = 0.5
                        pcs["mMajority"] = pcs["dMajority"]
                    else:
                        pcs["mRatio"] = c_false[1] / (c_true[1] + c_false[1])
                        pcs["mMajority"] = False

                # Check for matching majorities
                if pcs["mMajority"] == pcs["dMajority"]:
                    pcs["match"] = True
                elif pcs["mMajority"] == "equal":
                    pcs["match"] = True
                else:
                    pcs["match"] = False

                # Create PCS Score
                if pcs["match"] or (pcs["mMajority"] == "equal"):
                    pcs["score"] = (pcs["mRatio"] + pcs["dRatio"]) / 2
                else:
                    pcs["score"] = -((pcs["mRatio"] + pcs["dRatio"]) / 2)

                # Indicator (Task) Modifications
                s1n = mpred["system 1"]
                s2n = mpred["system 2"]
                s1nw = mpred["system 1 weak"]
                s2nw = mpred["system 2 weak"]
                s1p = mpred["system 1 poss"]
                s2p = mpred["system 2 poss"]

                # System 1 has other output then System 2 => Reasoner Type (Ignore necessity and weak)
                if s1n != s2n:
                    indicator["r_type"] = 1
                # System 1 has other output then System 2 => Reasoner Type (Ignore necessity)
                if s1p == s1nw and s2p == s2nw and s1p != s2p:
                    indicator["r_type"] = 2
                # System 1 has other output then System 2 => Reasoner Type (Ignore possibility)
                if s1n == s1nw and s2n == s2nw and s1n != s2n:
                    indicator["r_type"] = 3
                # System 1 has other output then System 2 => Reasoner Type
                if s1n == s1nw == s1p and s2n == s2nw == s2p and s1n != s2n:
                    indicator["r_type"] = 4

                # Possibility has other output then necessity => Modality Type
                if s1n == s2n and s1p == s2p and s1n != s1p:
                    indicator["m_type"] = 1
                # Possibility has other output then necessity => Modality Type
                # But weak necessity has other output then necessity as well
                if s1n == s2n and s1p == s2p and s1n != s1p and s1n != s1nw and s2n != s2nw:
                    indicator["m_type"] = 3

        if self.external:
            # Build Permutation Array
            args = {"rtd": self.r_type_delta,
                    "mtd": self.m_type_delta,
                    "d": self.delta,
                    "m": self.memory,
                    "s": self.s_connect,
                    "sl": self.s_limit,
                    "b": self.beta}
        else:
            # No external optimizations
            args = {"rtd": self.r_type_delta,
                    "mtd": self.m_type_delta,
                    "d": self.delta,
                    "m": [False],
                    "s": [False],
                    "sl": self.s_limit,
                    "b": [False]}

        values_arr = []
        arg_arr = []

        for arg, values in args.items():
            values_arr.append(values)
            arg_arr.append(arg)

        # Create all Permutations of all args
        tmp_arr = list(itertools.product(*values_arr))

        pred_permute = {}
        c_epoch = 0

        # Permutation testing for best parameters
        for permute in tmp_arr:
            c_epoch += 1
            correct = 0
            total = 0
            for subj_train_data in dataset:
                for seq_train_data in subj_train_data:
                    item = seq_train_data['item']
                    p_id = item.identifier
                    response = seq_train_data['response']
                    task = ccobra_to_assertion(item.task[0])
                    choices = ccobra_to_assertion(item.choices[0][0])
                    prediction = self.make_prediction(p_id, task, choices, permute)
                    self.make_adapt(p_id, task, choices, response, permute)

                    total += 1
                    if response == prediction:
                        correct += 1
            # print(f"Epoch {c_epoch}/{len(tmp_arr)} - Parameters: {permute}")
            pred_p = correct/total
            pred_permute[permute] = pred_p
            self.user_profile.clear()

        # Set best parameters for actually prediction
        self.best_param = max(pred_permute.items(), key=operator.itemgetter(1))[0]

        global curr_subj_count
        if self.debug:
            print(f"Fold {curr_subj_count}/{self.max_subj_count} - Parameters: {self.best_param}")
        curr_subj_count += 1

    def make_prediction(self, p_id, c_task, c_choices, parameters):
        # Set current parameters:
        delta = parameters[2]
        memory = parameters[3]
        s_connect = parameters[4]
        s_limit = parameters[5]
        beta = parameters[6]

        # Get knowledgebase for current task
        c_task_profile = self.task_profile[c_task][c_choices]
        pcs = c_task_profile["pcs"]
        mpred = c_task_profile["mPred"]
        indicator = c_task_profile["indicator"]

        # Add User to User_profile if it doesnt exist already
        init = {"amount": 0,
                "correct": 0,
                "beta": [],
                "beta_value": 0,
                "p_pred": {"system 1": 0,
                           "system 2": 0,
                           "system 1 poss": 0,
                           "system 2 poss": 0,
                           "system 1 weak": 0,
                           "system 2 weak": 0},
                "memory": {}}

        if p_id not in self.user_profile:
            self.user_profile[p_id] = init
        p = self.user_profile[p_id]

        # Choose correct strategy for current participant
        self.last_model_prediction = self.choose_model_strategy(p, delta)
        prediction = mpred[self.last_model_prediction]

        # Access Memory
        if memory:
            if c_task in p["memory"]:
                if c_choices in p["memory"][c_task]:
                    if p["memory"][c_task][c_choices]["resp"] != prediction:
                        self.last_model_prediction = "memory"
                        prediction = p["memory"][c_task][c_choices]["resp"]

        # Sentential Connectives
        if s_connect:
            if indicator["s_connect"] >= s_limit and pcs["score"] < self.pcs_threshold:
                self.last_model_prediction = "s_connect"
                prediction = pcs["dMajority"]

        # Beta
        if beta:
            if p["beta_value"] < 0 and np.array(p["beta"][-2:]).sum() < 1 and pcs["score"] < self.pcs_threshold:
                self.last_model_prediction = "beta - < 0"
                prediction = not pcs["dMajority"]

        # Make prediction
        self.last_prediction = prediction
        return prediction

    def make_adapt(self, p_id, c_task, c_choices, response, parameters):
        # Set Parameters
        r_type_delta = parameters[0]
        m_type_delta = parameters[1]

        c_task_profile = self.task_profile[c_task][c_choices]
        indicator = c_task_profile["indicator"]
        mpred = c_task_profile["mPred"]
        pcs = c_task_profile["pcs"]
        lp = self.last_prediction
        cp = self.user_profile[p_id]

        # Add current task into memory
        if c_task not in cp["memory"]:
            cp["memory"][c_task] = {}
        if c_choices not in cp["memory"][c_task]:
            cp["memory"][c_task][c_choices] = {"resp": response,
                                               "mPred": lp}

        # Modify Beta parameter
        if response != pcs["dMajority"]:
            cp["beta"].append(-1)
        if response == pcs["dMajority"]:
            cp["beta"].append(1)
        cp["beta_value"] = np.array(cp["beta"]).sum()

        # Modify p_pred
        for system, value in cp["p_pred"].items():
            if mpred[system] == response:
                cp["p_pred"][system] += 1
                # Give boost if indicator task was answered matching response
                if indicator["r_type"] > 0:
                    cp["p_pred"][system] += r_type_delta
                if indicator["m_type"] > 0:
                    cp["p_pred"][system] += m_type_delta

    def choose_model_strategy(self, p, delta):
        # Get best strategy from pre training
        s_pre = max(self.model_profile.items(), key=operator.itemgetter(1))[0]
        # Get best strategy from current participant p
        s_p = max(p["p_pred"].items(), key=operator.itemgetter(1))[0]

        # If the strategies are not the same check if the count of the pre training strategy +
        # a delta is smaller then the count of the participants strategy (avoids choosing suboptimal strategy)
        if s_pre != s_p:
            if p["p_pred"][s_pre] + delta < p["p_pred"][s_p]:
                return s_p
        return s_pre

    def predict(self, item, **kwargs):
        c_task = ccobra_to_assertion(item.task[0])
        c_choices = ccobra_to_assertion(item.choices[0][0])
        p_id = item.identifier
        return self.make_prediction(p_id, c_task, c_choices, self.best_param)

    def adapt(self, item, response, **kwargs):
        c_task = ccobra_to_assertion(item.task[0])
        c_choices = ccobra_to_assertion(item.choices[0][0])
        p_id = item.identifier
        return self.make_adapt(p_id, c_task, c_choices, response, self.best_param)
