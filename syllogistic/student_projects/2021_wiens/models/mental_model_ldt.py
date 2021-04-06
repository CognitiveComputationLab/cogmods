import ccobra
import random
import base_models.mental_model as mental_model


class Mental_Model_ldt(ccobra.CCobraModel):
    """ This model uses the results of the ldt and the response time of the participants
    in order to decide when to search for alternative models
    """
    def __init__(self, name='MMT adaptiv + selective processing'):
        super(Mental_Model_ldt, self).__init__(
            name, ['syllogistic'], ['verify'])

        # default parameters
        self.countermodelsearch = 1
        self.previous_answers = []
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
        # grid search for best parameter
        for countermodelsearch in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            self.countermodelsearch = countermodelsearch
            score = 0
            for task in dataset:
                prediction = self.predict(task["item"], **task["kwargs"])
                if prediction == task["response"]:
                    score += 1

                if score == best:
                    best_parameter.append([countermodelsearch])

                elif score > best:
                    best = score
                    best_parameter = [[countermodelsearch]]

        if countermodelsearch in best_parameter:
            return
        # randomly choose best parameter of best_parameter list
        best_parameter = random.choice(best_parameter)
        self.countermodelsearch = best_parameter[0]

    def predict(self, item, **kwargs):
        enc_task = ccobra.syllogistic.encode_task(item.task)
        enc_conclusion = ccobra.syllogistic.encode_response(item.choices[0], item.task)
        mm = mental_model.Mental_Model()
        amp = self.countermodelsearch

        # semantic inhibition or high response time -> always search for alternative models
        if kwargs["ldt_uw"] - kwargs["ldt_rw"] < 15 and kwargs["response_time"] > 9000:
            amp = 1
        conclusions = mm.predict(enc_task, amp=amp, verify_target=enc_conclusion)

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
