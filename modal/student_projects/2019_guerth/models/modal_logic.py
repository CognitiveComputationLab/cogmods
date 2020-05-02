import ccobra
from Modal_Logic.ccobra_adapter import ccobra_to_assertion
from Modal_Logic.solver import does_a_follow_from_b


class MentalModel(ccobra.CCobraModel):
    def __init__(self, name='Modal Logic System K,T,B,S4'):
        super(MentalModel, self).__init__(
            name, ['modal'], ['verify'])
        self.last_response = None

    def predict(self, item, **kwargs):
        task = ccobra_to_assertion(item.task[0])
        choices = ccobra_to_assertion(item.choices[0][0])
        r_K = does_a_follow_from_b(task, choices)
        if does_a_follow_from_b(task, choices, ['reflexive']) != r_K:
            raise Exception
        if does_a_follow_from_b(task, choices, ['reflexive', 'symmetric']) != r_K:
            raise Exception
        if does_a_follow_from_b(task, choices, ['reflexive', 'transitive']) != r_K:
            raise Exception
        return r_K

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
        # if self.last_response != response:
        #     print(item.task[0])
        #     print(item.choices[0][0])
        #     print("my_response: ", self.last_response)
        #     print("their_response: ", response)
        #     print()
        pass

