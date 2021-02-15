
import copy
import ccobra
from task_processor import TaskProcessor
from task_processor import LogicPredictor
from task_processor import VectorPredictor


choices_given = {}


class MyModel(ccobra.CCobraModel):
    def __init__(self, name='Recommendation-Model'):
        super(MyModel, self).__init__(
            name, ['propositional'], ['single-choice'])
        self.task_processor = TaskProcessor()
        self.pred = LogicPredictor(self.task_processor)
        self.participant_finished_tasks = []
        self.vec_pred = VectorPredictor(self.task_processor, self.participant_finished_tasks)

    def start_participant(self, **kwargs):
        global choices_given
        self.participant_finished_tasks.clear()
        self.vec_pred.choices_given = choices_given

    def predict(self, item, **kwargs):
        key = self.tuple_to_string(item.task)
        self.task_processor.add_task_to_task_map(key)
        s = self.pred.calculate_predictions(key)
        c = self.tuple_to_string(item.choices)
        self.task_processor.add_feature_vector(key)
        self.task_processor.assign_choice_classes(key, c, s)
        p = self.vec_pred.calculate_prediction_from_top(key)
        for y in item.choices:
            z = self.tuple_to_string(y)
            if z.lower() == p.lower():
                return y

    def adapt(self, item, target, **kwargs):
        global choices_given
        task = self.tuple_to_string(item.task)
        truth = self.tuple_to_string(target)
        s = self.pred.calculate_predictions(task)
        self.participant_finished_tasks.append((task, truth))
        if task not in choices_given:
            choices_given[task] = {}
        if truth not in choices_given[task]:
            choices_given[task][truth] = 1
        else:
            choices_given[task][truth] += 1


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
