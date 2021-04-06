import ccobra
import random
import base_models.mReasoner as mReasoner


class conflict_resolution(ccobra.CCobraModel):

    def __init__(self, name='additives Wahrscheinlichkeitsmodel'):
        super(conflict_resolution, self).__init__(
            name, ['syllogistic'], ['verify'])

        # default parameters
        self.previous_answers = []
        self.sigma = 1
        self.omega_valid = 1
        self.omega_invalid = 1
        self.h = 0
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
        for sigma in [1, 0.5, 0]:
            for omega_valid in [1, 0.5, 0]:
                for omega_invalid in [1, 0.5, 0]:
                    for h in [0, 0.5, 1]:
                        self.h = h
                        self.sigma = sigma
                        self.omega_valid = omega_valid
                        self.omega_invalid = omega_invalid
                        score = 0
                        for task in dataset:
                            prediction = self.predict(task["item"], **task["kwargs"])
                            if prediction == task["response"]:
                                score += 1
                            if score == best:
                                best_parameter.append([h, sigma, omega_valid, omega_invalid])

                            elif score > best:
                                best = score
                                best_parameter = [[h, sigma, omega_valid, omega_invalid]]

        # don't change if default setting is already the best setting
        if [self.h, self.sigma, self.omega_valid, self.omega_invalid] in best_parameter:
            return
        # randomly choose best parameter of best_parameter list
        best_parameter = random.choice(best_parameter)
        self.h = best_parameter[0]
        self.sigma = best_parameter[1]
        self.omega_valid = best_parameter[2]
        self.omega_invalid = best_parameter[3]

    def predict(self, item, **kwargs):
        enc_task = ccobra.syllogistic.encode_task(item.task)
        enc_conclusion = ccobra.syllogistic.encode_response(item.choices[0], item.task)
        mr = mReasoner.mReasoner()

        omega = self.omega_valid if kwargs["type_of_syllogism"].endswith("valid") else self.omega_invalid

        # analytical output
        if kwargs["ldt_uw"] - kwargs["ldt_rw"] < 15 and kwargs["response_time"] > 9000:
            output = mr.predict(enc_task, system2=self.sigma, weaken=omega)
            output = enc_conclusion in output
        # heuristic output
        else:
            if kwargs["type_of_syllogism"].startswith("mismatch"):
                if random.random() < self.h:
                    output = True
                else:
                    output = enc_conclusion in ["Eac", "Eca"]
            else:
                output = enc_conclusion in ["Eac", "Eca"]

        self.predictions[item.sequence_number] = output
        return output

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
