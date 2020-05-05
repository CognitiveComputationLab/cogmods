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
    def __init__(self, name='mModalSentential'):
        super(MentalModel, self).__init__(
            name, ['modal'], ['verify'])

        self.profiles = {"system 1":0,
                         "system 2":0,
                         "system 2 weak":0,
                         "system 1 weak":0,
                         "system 1 poss": 0,
                         "system 2 poss": 0,
                         }
        self.reasoner_type = "system 1"
        self.last_predictions = {"system 1":None,
                                 "system 2":None,
                                 "system 2 weak":None,
                                 "system 1 weak":None,
                                 "system 1 poss": None,
                                 "system 2 poss": None,
                                }

    def predict(self, item, **kwargs):
        # print("predict")
        task = ccobra_to_assertion(item.task[0])
        choices = ccobra_to_assertion(item.choices[0][0])

        prediction_1 = necessary(task, choices)
        prediction_2 = necessary(task, choices, 2)
        prediction_2_weak = necessary(task, choices, 2, True) #2
        prediction_1_weak = necessary(task, choices, 1, True) #2
        prediction_1_poss = possible(task, choices, 1)
        prediction_2_poss = possible(task, choices, 2)


        self.last_predictions["system 1"] = prediction_1
        self.last_predictions["system 2"] = prediction_2
        self.last_predictions["system 2 weak"] = prediction_2_weak
        self.last_predictions["system 1 weak"] = prediction_1_weak
        self.last_predictions["system 1 poss"] = prediction_1_poss
        self.last_predictions["system 2 poss"] = prediction_2_poss

        if self.reasoner_type == "system 1":
            prediction = prediction_1
        elif self.reasoner_type == "system 2":
            prediction = prediction_2
        elif self.reasoner_type == "system 2 weak":
            prediction = prediction_2_weak
        elif self.reasoner_type == "system 1 weak":
            prediction = prediction_1_weak
        elif self.reasoner_type == "system 1 poss":
            prediction = prediction_1_poss
        elif self.reasoner_type == "system 2 poss":
            prediction = prediction_2_poss
        else:
            raise Exception
        global_statistic[self.reasoner_type] += 1
        # print(global_statistic)
        # s = 0
        # for k,v in global_statistic.items():
        #     s += v
        # # print(s)
        # for k,v in global_statistic.items():
        #     # print(k + " & " + str(v) + " & " + str(round(v/s, 2)) + "\\\\")
        #     print(k + " & " + str(v) + " & " + "{:.2f}".format(round(v/s, 2)) + "\\\\")
        #     # print(k + " & " + str(v) + " & " + str((int(round(v/s, 2)*100))) + "\\\\")
        # print()

        # print("reasoner_type: ", self.reasoner_type)
        # print(self.i)

        # print(task)
        # print(choices)
        # print(prediction)
        # print()
        return prediction

    def pre_train(self, dataset):
        pass


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

    def person_train(self, data):
        """ Trains the model based on background data of the individual to
        be tested on.
        Parameters
        ----------
        data : list(dict(str, object))
            Training data for the model. List of tasks containing the items
            and corresponding responses.
        """
        print("PERSON TRAIN")
        print(data)
        print("END PERSON TRAIN")