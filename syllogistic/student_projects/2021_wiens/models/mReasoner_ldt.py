import ccobra
import random
import base_models.mReasoner as mReasoner


class mReasoner_ldt(ccobra.CCobraModel):
    """
    This model uses the results of the ldt and the response time of the participants
    in order to decide when to search for alternative models
    """

    def __init__(self, name='mReasoner adaptiv + selective processing'):
        super(mReasoner_ldt, self).__init__(
            name, ['syllogistic'], ['verify'])

        self.previous_answers = []
        self.sigma = 1
        self.omega = 1
        self.omega2 = 1
        self.predictions = {}

    def pre_train_person(self, dataset):
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
        for sigma in [1, 0]:
            for omega in [1, 0]:
                self.sigma = sigma
                self.omega = omega
                score = 0
                for task in dataset:
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
        sigma = self.sigma
        omega = self.omega
        enc_task = ccobra.syllogistic.encode_task(item.task)
        enc_conclusion = ccobra.syllogistic.encode_response(item.choices[0], item.task)

        if kwargs["ldt_uw"] - kwargs["ldt_rw"] < 15 and kwargs["response_time"] > 9000:
            sigma = 1

        conclusions = mr.predict(enc_task, system2=sigma, weaken=omega, verify_target=enc_conclusion)

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
