import ccobra
from mmodalsentential.assertion_parser import ccobra_to_assertion
from mmodalsentential.reasoner import necessary, possible

global_statistic = {"system 1":0,
                         "system 2":0,
                         "system 1 weak":0,
                         "system 2 weak":0,
                         "system 1 poss": 0,
                         "system 2 poss": 0,
                         }

class MentalModel(ccobra.CCobraModel):
    def __init__(self, name='mModalSentential pretrained'):
        super(MentalModel, self).__init__(
            name, ['modal'], ['verify'])

        self.profiles = {1:0, 2:0}
        self.reasoner_type = 1
        self.last_predictions = {1:None, 2:None}
        self.predict_1 = None
        self.predict_2 = None

    def predict(self, item, **kwargs):
        # print("predict")
        task = ccobra_to_assertion(item.task[0])
        choices = ccobra_to_assertion(item.choices[0][0])

        prediction_1 = self.select_prediction(task, choices, 1)
        prediction_2 = self.select_prediction(task, choices, 2)

        self.last_predictions[1] = prediction_1
        self.last_predictions[2] = prediction_2

        if self.reasoner_type == 1:
            prediction = prediction_1
        elif self.reasoner_type == 2:
            prediction = prediction_2
        else:
            raise Exception
        return prediction

    def pre_train(self, dataset):
        #TODO: maybe do a statistic for each type of question?
        # print("pretrain")
        # print(len(dataset))

        profiles = {"system 1": [0, {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}],
                    "system 2": [0, {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}],
                    "system 2 weak": [0, {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}],
                    "system 1 weak": [0, {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}],
                    "system 1 poss": [0, {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}],
                    "system 2 poss": [0, {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}]
                    }
        predictions = {"system 1":None,
                       "system 2":None,
                       "system 2 weak":None,
                       "system 1 weak":None,
                       "system 1 poss": None,
                       "system 2 poss": None
                       }

        for subj_train_data in dataset:
            # print(subj_train_data)
            for seq_train_data in subj_train_data:
                # print(i)
        #         print(seq_train_data['item'])
        #         # print(seq_train_data['item'].identifier, seq_train_data['response'])
                item = seq_train_data['item']
                response = seq_train_data['response']
                task = ccobra_to_assertion(item.task[0])
                choices = ccobra_to_assertion(item.choices[0][0])

                prediction_1 = necessary(task, choices)
                prediction_2 = necessary(task, choices, 2)
                prediction_2_weak = necessary(task, choices, 2, True) #2
                prediction_1_weak = necessary(task, choices, 1, True) #2
                prediction_1_poss = possible(task, choices, 1)
                prediction_2_poss = possible(task, choices, 2)


                predictions["system 1"] = prediction_1
                predictions["system 2"] = prediction_2
                predictions["system 2 weak"] = prediction_2_weak
                predictions["system 1 weak"] = prediction_1_weak
                predictions["system 1 poss"] = prediction_1_poss
                predictions["system 2 poss"] = prediction_2_poss
                for i, p in predictions.items():
                    if p == response:
                        profiles[i][0] += 1

                        for j, (ii, pp) in enumerate(predictions.items()):
                            if pp == response:
                                profiles[i][1][j+1] += 1

        best_val = -1
        best = None
        for key, val in profiles.items():
            if val[0] > best_val:
                best_val = val[0]
                best = (key, val)
        # print("Best: ", best)

        for i, (key, val) in enumerate(profiles.items()):
            if key == best[0]:
                best_index = i+1
                break
        # print("best index = ", best_index)


        second_best_val = -1
        second_best = None
        for key, val in profiles.items():
            if key == best[0]:
                continue
            if val[0] - val[1][best_index] > second_best_val:
                second_best_val = val[0] - val[1][best_index]
                second_best = (key, val, val[0] - val[1][best_index])
        # print("Second best: ", second_best)


        self.predict_1 = best[0]
        self.predict_2 = second_best[0]
        # if best == "system 1":
        #     self.predict_1 = necessary
        # elif best == "system 2":
        #     self.predict_1 = necessary
        # elif best == "system 2 weak":
        #     self.predict_1 = necessary
        # elif best == "system 1 poss":
        #     self.predict_1 = possible
        # elif best == "system 2 poss":
        #     self.predict_1 = possible

        # if second_best == "system 1":
        #     self.predict_2 = necessary(task, choices)
        # elif second_best == "system 2":
        #     self.predict_2 = necessary(task, choices, 2)
        # elif second_best == "system 2 weak":
        #     self.predict_2 = necessary(task, choices, 2, True)
        # elif second_best == "system 1 poss":
        #     self.predict_2 = possible(task, choices, 1)
        # elif second_best == "system 2 poss":
        #     self.predict_2 = possible(task, choices, 2)


        # prediction_1 = necessary(task, choices)
        # prediction_2 = necessary(task, choices, 2)
        # prediction_2_weak = necessary(task, choices, 2, True)
        # prediction_1_poss = possible(task, choices, 1)
        # prediction_2_poss = possible(task, choices, 2)


        # print("Here is the statistic:")
        # print(profiles)


    def adapt(self, item, response, **kwargs):
        # print("adapt")
        # print(item.task)
        # print(response)
        # print()

        for i, p in self.last_predictions.items():
            if p == response:
                self.profiles[i] += 1
        # print(self.profiles)

        best = -1
        for key, val in self.profiles.items():
            if val > best:
                best = val
                if self.reasoner_type != key:
                    self.reasoner_type = key
        # print(self.reasoner_type)

        # if self.reasoner_type == 1:
        #     strategy = self.predict_1
        # elif self.reasoner_type == 2:
        #     strategy = self.predict_2
        # global_statistic[strategy] += 1
        # s = 0
        # for k,v in global_statistic.items():
        #     s += v
        # for k,v in global_statistic.items():
        #     print(k + " & " + str(v) + " & " + "{:.2f}".format(round(v/s, 2)) + "\\\\")
        # print()

    def select_prediction(self, task, choices, select):
        if select == 1:
            strategy = self.predict_1
        elif select == 2:
            strategy = self.predict_2



        if strategy == "system 1":
            return necessary(task, choices)
        elif strategy == "system 2":
            return necessary(task, choices, 2)
        elif strategy == "system 2 weak":
            return necessary(task, choices, 2, True)
        elif strategy == "system 1 weak":
            return necessary(task, choices, 1, True)
        elif strategy == "system 1 poss":
            return possible(task, choices, 1)
        elif strategy == "system 2 poss":
            return possible(task, choices, 2)