import ccobra
from Modal_Logic.ccobra_adapter import ccobra_to_assertion
from Modal_Logic.solver import does_a_follow_from_b


class MentalModel(ccobra.CCobraModel):
    def __init__(self, name='Modal Logic System T'):
        super(MentalModel, self).__init__(
            name, ['modal'], ['verify'])

    def predict(self, item, **kwargs):
        task = ccobra_to_assertion(item.task[0])
        choices = ccobra_to_assertion(item.choices[0][0])
        return does_a_follow_from_b(task, choices, ['reflexive'])

    def pre_train(self, dataset):
        pass
        # print("pretrain")
        # print(len(dataset))

        # for subj_train_data in dataset:
        #     for seq_train_data in subj_train_data:
        #         print(seq_train_data['item'].identifier, seq_train_data['response'])

    def adapt(self, item, response, **kwargs):
        # print(item.task)
        # print(response)
        # print()
        pass

