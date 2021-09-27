import ccobra
import random


class RandomModel(ccobra.CCobraModel):
    def __init__(self, name='RandomModel'):
        super(RandomModel, self).__init__(
            name, ['modal'], ['verify'])

    def predict(self, item, **kwargs):
        return random.choice([True, False])
