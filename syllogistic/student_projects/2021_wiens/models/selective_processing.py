import ccobra
import base_models.mental_model as mental_model


class Selective_processing(ccobra.CCobraModel):
    """
    This model models the reasoning process based on the selective processing of belief bias (Evans 2000)
    Returns True if there is a mental model (including alternative models) in which the conclusion holds
    Returns False otherwise
    """
    def __init__(self, name='Selective-Processing'):
        super(Selective_processing, self).__init__(
            name, ['syllogistic'], ['verify'])

        self.predictions = {}
        self.previous_answers = []

    def predict(self, item, **kwargs):
        enc_task = ccobra.syllogistic.encode_task(item.task)
        enc_conclusion = ccobra.syllogistic.encode_response(item.choices[0], item.task)
        mm = mental_model.Mental_Model()
        conclusions = mm.predict(enc_task, verify_target=enc_conclusion)
        if enc_conclusion in conclusions:
            prediction = True
        else:
            prediction = False
        self.predictions[item.sequence_number] = prediction
        return prediction

    def adapt(self, item, target, **kwargs):
        setting = {"item": item, "response": target, "kwargs": kwargs}
        self.previous_answers.append(setting)

    def end_participant(self, identifier, model_log, **kwargs):
        # calculate accuracy
        hits = 0
        for element in self.previous_answers:
            if element["response"] == self.predictions[element["item"].sequence_number]:
                hits += 1

        accuracy = round(hits / len(self.predictions), 2)
        with open("prediction\\prediction.txt", "a") as f:
            f.write(self.name + ";" + str(identifier) + ";" + str(accuracy) + ";" + str(self.predictions) + "\n")