
import copy
import ccobra
from task_processor import TaskProcessor
from task_processor import LogicPredictor


class LogicModel(ccobra.CCobraModel):
    def __init__(self, name='LogicModel'):
        super(LogicModel, self).__init__(
            name, ['propositional'], ['single-choice'])
        self.task_processor = TaskProcessor()
        self.pred = LogicPredictor(self.task_processor)

    def predict(self, item, **kwargs):
        key = self.tuple_to_string(item.task)
        self.task_processor.add_task_to_task_map(key)
        s = self.pred.calculate_predictions(key)
        for y in item.choices:
            z = self.tuple_to_string(y)
            if z.lower() == s:
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
