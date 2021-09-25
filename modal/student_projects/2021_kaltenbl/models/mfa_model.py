import ccobra
from mmodalsentential.assertion_parser import ccobra_to_assertion


class MentalModel(ccobra.CCobraModel):
    def __init__(self, name='MFA Model'):
        super(MentalModel, self).__init__(
            name, ['modal'], ['verify'])
        self.task_profile = {}

    def predict(self, item, **kwargs):
        task = ccobra_to_assertion(item.task[0])
        choices = ccobra_to_assertion(item.choices[0][0])
        task_profile = self.task_profile[task]
        choices_profile = task_profile[choices]
        sorted_choice = sorted(choices_profile.items(), key=lambda v: v[1], reverse=True)
        return sorted_choice[0][0]

    def pre_train(self, dataset):
        for subj_train_data in dataset:
            for seq_train_data in subj_train_data:
                item = seq_train_data['item']
                response = seq_train_data['response']
                task = ccobra_to_assertion(item.task[0])
                choices = ccobra_to_assertion(item.choices[0][0])
                if task not in self.task_profile:
                    self.task_profile[task] = {}
                if choices not in self.task_profile[task]:
                    self.task_profile[task][choices] = {}
                if response not in self.task_profile[task][choices]:
                    self.task_profile[task][choices][response] = 1
                else:
                    self.task_profile[task][choices][response] += 1
