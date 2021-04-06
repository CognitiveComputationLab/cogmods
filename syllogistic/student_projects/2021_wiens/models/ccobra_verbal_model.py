import ccobra
import random
import base_models.verbal_model as verbal_model


class ccobra_VerbalReasoning(ccobra.CCobraModel):
    """ This model uses ccobras adapt-predict cycle to determine what
    additional information reasoners infer.
    """

    def __init__(self, name='Verbal Reasoning adaptiv'):
        super(ccobra_VerbalReasoning, self).__init__(
            name, ['syllogistic'], ['verify'])

        self.previous_answers = []
        self.parameter = ["a"] * 14

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
        vm = verbal_model.Verbal_Model()
        best = 0
        best_parameter = []
        possibilites = vm.get_individual_diffrence_parameter()
        for parameterization in possibilites:
            score = 0
            for task in dataset:
                self.parameter = parameterization
                prediction = self.predict(task["item"], **task["kwargs"])
                if prediction == task["response"]:
                    score += 1

                if score == best:
                    best_parameter.append(parameterization)

                elif score > best:
                    best = score
                    best_parameter = [parameterization]

        # best parameter already used
        if self.parameter in best_parameter:
            return
        # randomly choose best parameter of best_parameter list
        best_parameter = random.choice(best_parameter)
        self.parameter = best_parameter

    def predict(self, item, **kwargs):
        vm = verbal_model.Verbal_Model()
        vm.set_individual_diffrence_parameter(self.parameter)
        enc_task = ccobra.syllogistic.encode_task(item.task)
        enc_conclusion = ccobra.syllogistic.encode_response(item.choices[0], item.task)
        conclusions = vm.predict(enc_task)
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
