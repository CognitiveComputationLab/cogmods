import ccobra
import random
import base_models.mReasoner as mReasoner


class ccobra_mReasoner(ccobra.CCobraModel):
    """
    This model uses ccobras adapt-predict cycle to determine when individuals
    search for alternative models with mReasoner.
    """
    def __init__(self, name='mReasoner adaptiv'):
        super(ccobra_mReasoner, self).__init__(
            name, ['syllogistic'], ['verify'])

        self.previous_answers = []
        self.sigma = 0
        self.omega = 0
        self.predictions = {}

    def pre_train_person(self, dataset):
        self.size = len(dataset)
        for task_data in dataset:
            self.adapt(
                task_data['item'],
                task_data['response'],
                **task_data['aux']
            )
        self.predictions = {}

    def fit_parameter(self, dataset):
        best = 0
        best_parameter = []
        for omega in [0, 1]:
            for sigma in [0, 1]:
                score = 0
                for task in dataset:
                    self.sigma = sigma
                    self.omega = omega
                    prediction = self.predict(task["item"], **task["kwargs"])
                    if prediction == task["response"]:
                        score += 1

                    if score == best:
                        best_parameter.append([sigma, omega])

                    elif score > best:
                        best = score
                        best_parameter = [[sigma, omega]]

        # don't change if default setting is already the best setting
        if [self.sigma, self.omega] in best_parameter:
            return
        # randomly choose best parameter of best_parameter list
        best_parameter = random.choice(best_parameter)
        self.sigma = best_parameter[0]
        self.omega = best_parameter[1]

    def predict(self, item, **kwargs):
        mr = mReasoner.mReasoner()
        enc_task = ccobra.syllogistic.encode_task(item.task)
        enc_conclusion = ccobra.syllogistic.encode_response(item.choices[0], item.task)
        conclusions = mr.predict(enc_task, system2=self.sigma, weaken=self.omega)

        if enc_conclusion in conclusions:
            prediction = True
        else:
            prediction = False
        self.predictions[item.sequence_number] = prediction
        return prediction

    def adapt(self, item, target, **kwargs):
        setting = {"item": item, "response": target, "kwargs": kwargs}
        self.previous_answers.append(setting)
        self.fit_parameter(self.previous_answers)

    def end_participant(self, identifier, model_log, **kwargs):
        # calculate accuracy
        hits = 0
        for element in self.previous_answers:
            if element["response"] == self.predictions[element["item"].sequence_number]:
                hits += 1

        accuracy = round(hits / len(self.predictions), 2)
        with open("prediction\\prediction.txt", "a") as f:
            f.write(self.name + ";" + str(identifier) + ";" + str(accuracy) + ";" + str(self.predictions) + "\n")
