import ccobra


class composite_model(ccobra.CCobraModel):

    def __init__(self, name='Ensemble-Model'):
        super(composite_model, self).__init__(
            name, ['syllogistic'], ['verify'])

        self.predictions = {}

    def get_predictions(self, identifier):
        if len(self.predictions) > 0:
            return
        predictions = []
        with open("prediction\\prediction.txt", "r") as f:
            for line in f:
                line = line.split(";")
                if int(line[1]) == identifier:
                    predictions.append((line[2], line[3].strip("\n")))
        # sort by accuracy
        predictions.sort()
        return eval(predictions[-1][1])

    def predict(self, item, **kwargs):
        predictions = self.get_predictions(item.identifier)
        return predictions[item.sequence_number]
