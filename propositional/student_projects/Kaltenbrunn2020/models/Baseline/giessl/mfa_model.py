import copy
import ccobra

counter_dict = {}


class MFAModel(ccobra.CCobraModel):
    def __init__(self, name='MFAModel'):
        super(MFAModel, self).__init__(
            name, ['propositional'], ['single-choice'])

    def pre_train(self, dataset):
        global counter_dict
        for subj_train_data in dataset:
            for seq_train_data in subj_train_data:
                rep = self.tuple_to_string(seq_train_data['response'])
                task = self.tuple_to_string(seq_train_data['item'].task).lower()

                if task not in counter_dict:
                    counter_dict[task] = {}
                if rep not in counter_dict[task]:
                    counter_dict[task][rep] = 1
                else:
                    counter_dict[task][rep] += 1

    def predict(self, item, **kwargs):
        global counter_dict
        key = self.tuple_to_string(item.task).lower()
        tmp = counter_dict[key]
        temp = sorted(tmp.items(), key=lambda v: v[1], reverse=True)
        for y in item.choices:
            z = self.tuple_to_string(y)
            if z.lower() == temp[0][0].lower():
                return y
        
        

    def tuple_to_string(self, tuptup):
        def join_deepest(tup, sep=';'):
            if not isinstance(tup, list):
                return tup
            if not isinstance(tup[0], list):
                return sep.join(tup)
            else:
                for idx in range(len(tup)):
                    tup[idx] = join_deepest(tup[idx], sep)
                return tup

        tup = copy.deepcopy(tuptup)
        tup = join_deepest(tup, ';')
        tup = join_deepest(tup, '/')

        # Sort the tuples
        tup = sorted(tup) if isinstance(tup, list) else tup

        # tup = join_deepest(tup, '|')
        return tup
