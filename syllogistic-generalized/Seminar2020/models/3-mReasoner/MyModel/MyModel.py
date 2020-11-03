from collections import namedtuple
from random      import sample, randint, choice
from math        import floor, ceil
import ccobra
from copy import deepcopy
import logging
import sys


rank_order_chater_oaksford = \
    ['Some not', 'No', 'Some', 'Few', 'Most', 'All']

rank_order_default = rank_order_chater_oaksford

rank_order_remove_some_not_flip_few_some = \
    ['No', 'Few', 'Some', 'Most', 'All']

# Based on implications from actual logic.
weakening_strict = {
    'All' : 'Most', 'Most' : 'Some',     'Some'     : 'NVC',
    'No'  : 'Few',  'Few'  : 'Some not', 'Some not' : 'NVC'
}

# negative: Whether we refer to the intersection or the difference,
#     in the latter case the class membership will be negated.
# initial_size: The numbers of individuals created for the first premise.
# min_size The minimum number of individuals of the subject class of the second premise
#     that we need to satisfy it.
# lower_bound, upper_bound: Minimal resp. maximal ratio between the number of individuals
#     in the referred to set and the number of individuals in the subject set.
# existential: Whether an individual outside the reffered to set should be added.
Intension = namedtuple( \
    'Parameters', 'negative initial_size min_size lower_bound upper_bound existential')

# Based on the table in Khemlani & Johnson with min_size for Most adjusted.
# The values might be too strict in practice.
intensions_kj12 = {
    'All'      : Intension(False, (4, 4), 1, 1.0, 1.0, False),
    'Some'     : Intension(False, (4, 2), 1, 0.1, 1.0, True),
    'Most'     : Intension(False, (4, 3), 3, 0.6, 0.9, True),
    'No'       : Intension(True,  (4, 4), 1, 1.0, 1.0, False),
    'Some not' : Intension(True,  (4, 2), 1, 0.1, 1.0, True),
    'Few'      : Intension(True,  (4, 3), 3, 0.6, 0.9, True),
}

intensions_default = {
    'All'      : Intension(False, (4, 4), 1, 1.0, 1.0, False),
    'Some'     : Intension(False, (4, 2), 1, 0.001, 1.0, True),
    'Most'     : Intension(False, (4, 3), 3, 0.501, 1.0, True),
    'No'       : Intension(True,  (4, 4), 1, 1.0, 1.0, False),
    'Some not' : Intension(True,  (4, 2), 1, 0.001, 1.0, True),
    'Few'      : Intension(True,  (4, 3), 3, 0.501, 1.0, True),
}


def pp_statement(statement):
    if statement[0] == 'NVC':
        return "NVC"
    if statement[0] == 'Some not':
        return " ".join(["Some", statement[1], "are not", statement[2]])
    return " ".join([statement[0], statement[1], "are", statement[2]])


def pp_syllogism(logger: logging.Logger, task, response=None):
    logger.debug(pp_statement(task[0]) + '\n' + pp_statement(task[1]))
    if response:
        logger.debug("-----------------")
        for r in response:
            logger.debug(pp_statement(r))
    return None


def pp_model(model, classes=['A', 'B', 'C']):
    res = ""
    for ind in model:
        for c in classes:
            if ind.get(c):
                res += " " + c
            elif ind.get(c) == False:
                res += "-" + c
            else:
                res += " " * (1 + len(c))
            res += " "
        res += "\n"
    return res


Parameters = namedtuple('Parameters', 'disallow_nvc max_weakenings max_tries max_size order_from_choices strategic most_not', defaults=(False, 3, 500, 21, False, True, True))


class MyModel(ccobra.CCobraModel):

    params_2010_base = Parameters(disallow_nvc=True, order_from_choices=True, strategic=False, most_not=False)
    params_2010_improved = Parameters(disallow_nvc=True, max_weakenings=0, order_from_choices=True, strategic=False, most_not=False)

    params_2020_base = Parameters(strategic=False)
    params_2020_improved = Parameters()

    params = params_2020_improved # modify for ccobra run

    def __init__(self, log_level=logging.DEBUG):

        super(MyModel, self).__init__(
            'Improved Model', ['syllogistic-generalized'], ['single-choice'])

        self.rank_order  = rank_order_default
        self.weakening   = weakening_strict
        self.intensions  = intensions_default
        self.some_no_sym = False # Some / No A B = Some / No B A, requires response to be given

        params = MyModel.params
        self.operations1    = ['add', 'move', 'break']
        self.operations2    = ['move']
        if params.strategic:
            self.operations1.append('strategic')
            self.operations2.append('strategic')
        self.disallow_nvc   = params.disallow_nvc
        self.max_weakenings = params.max_weakenings
        self.max_tries      = params.max_tries
        self.max_size       = params.max_size
        self.order_from_choices = params.order_from_choices
        self.most_not       = params.most_not

        self.log_level      = log_level
        handler = logging.StreamHandler(sys.stdout)
        self.logger = logging.Logger('model', self.log_level)
        self.logger.addHandler(handler)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['logger']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = logging.Logger('model', self.log_level)

    def holds(self, model, statement):
        if statement[0] == 'NVC':
            return True # Allow ourselves to infer an empty conclusion

        params = self.intensions.get(statement[0])

        s = 0 # Individuals in subject class
        p = 0 # Number of individuals referred to

        for ind in model:
            if not ind.get(statement[1]):
                continue
            s += 1
            if not params.negative and ind.get(statement[2]) \
            or params.negative     and not ind.get(statement[2]):
                p += 1

        if s == 0:
            return not params.existential

        ratio = p / s

        return params.lower_bound <= ratio <= params.upper_bound

    def weaken(self, model, conclusion):

        while not self.holds(model, conclusion) and self.weakenings > 0:
            c = self.weakening.get(conclusion[0])
            if self.disallow_nvc and c == 'NVC':
                self.weakenings = 0
                break
            conclusion[0] = c
            self.weakenings -= 1
        return conclusion

    def initial_conclusion(self, item, task):

        dominant_quantifier = next(x for x in self.rank_order if x in { task[0][0], task[1][0] })

        if self.order_from_choices:
            conclusion_classes = [item.choices[0][0][1], item.choices[0][0][2]]
        else:
            dominant_premise = next(x for x in task if x[0] == dominant_quantifier)
            other_premise    = next(x for x in task if x != dominant_premise)

            conclusion_classes = [dominant_premise[1], dominant_premise[2]]
            if conclusion_classes[0] == other_premise[1]:
                conclusion_classes[0] = other_premise[2]
            elif conclusion_classes[1] == other_premise[1]:
                conclusion_classes[1] = other_premise[2]
            elif conclusion_classes[0] == other_premise[2]:
                conclusion_classes[0] = other_premise[1]
            elif conclusion_classes[1] == other_premise[2]:
                conclusion_classes[1] = other_premise[1]

        return [dominant_quantifier, conclusion_classes[0], conclusion_classes[1]]

    def initial_model(self, task):

        params0 = self.intensions.get(task[0][0])
        params1 = self.intensions.get(task[1][0])

        # First premise

        model = [{ task[0][1] : True, task[0][2] : not params0.negative } \
                    for _ in range(params0.initial_size[1])] \
              + [{ task[0][1] : True } for _ in range( \
                    (params0.initial_size[0] - params0.initial_size[1]))]

        if params0.existential:
            model.append({ task[0][2] : True })

        # Second premise

        matches = lambda i: \
            i.get(task[1][1]) or i.get(task[1][2]) == (not params1.negative) and \
                task[1][2] == task[0][1]

        s = sum([1 for ind in model if matches(ind)])

        model += [{ task[1][1] : True } for _ in range(params1.min_size - s)]

        candidates = [i for i, ind in enumerate(model) if matches(ind)]

        amount = randint( \
            ceil(params1.lower_bound * len(candidates)), \
            floor(params1.upper_bound * len(candidates)))

        for i in sample(candidates, amount):
            model[i].update({ task[1][1] : True })
            model[i].update({ task[1][2] : not params1.negative })

        if params1.existential and task[1][2] != task[0][1]:
            model.append({ task[1][2] : True })

        return model

    def op_add(self, model, classes):
        key = choice(classes)
        model.append({key: True})
        return model

    def op_move(self, model):
        row_idx = randint(0, len(model) - 1)
        items = list(model[row_idx].items())
        if len(items) == 0:
            return model
        element = choice(items)
        dest_targets = []
        for j, row in enumerate(model):
            if element[0] in row:
                continue
            dest_targets.append(j)
        if len(dest_targets) == 0:
            return model
        target_idx = choice(dest_targets)
        model[target_idx][element[0]] = element[1]
        del model[row_idx][element[0]] # Might (deliberately) leave 'empty' individuals
        return model

    def op_break(self, model, classes):
        candidates = []
        for j, row in enumerate(model):
            if len(row.keys()) == 3:
                candidates.append(j)
        if len(candidates) == 0:
            return model
        c = choice(candidates)
        c3 = choice(classes) # classes = C1, C2, C3
        classes1 = classes.copy()
        classes1.remove(c3)
        c1 = choice(classes1)
        classes1.remove(c1)
        classes1.append(c3) # classes1 = C2, C3
        row = model[c]
        addtl_row = dict()
        for x in classes1:
            addtl_row[x] = row[x]
        del row[c3] # row = C1, C2
        model.append(addtl_row)
        return model

    def op_strategic(self, model, conclusion):
        params = self.intensions.get(conclusion[0])
        candidates = []
        for ind in model:
            if ind.get(conclusion[1]):
                if not params.negative and ind.get(conclusion[2]) \
                or params.negative     and not ind.get(conclusion[2]):
                    candidates.append(ind)
        if len(candidates) == 0:
            model.append({conclusion[1] : True, conclusion[2] : params.negative})
        else:
            c = choice(candidates)
            if choice([True, False]):
                del c[conclusion[1]]
            else:
                c[conclusion[2]] = params.negative

        return model

    def system2(self, model, stmt1, stmt2, conclusion,
        classes=['A', 'B', 'C']):
        """
        System 2 expects a model and tries to find
        a counterexample which doesn't break the premise.
        It then returns the weakened model
        """

        for _ in range(self.max_tries):
            if conclusion[0] == 'NVC':
                return model, conclusion

            model_copy = deepcopy(model)

            op = choice(
                self.operations1
            if len(model_copy) < self.max_size
            else
                self.operations2
            )

            if op == 'add':
                model_copy = self.op_add(model_copy, classes)
            elif op == 'move':
                model_copy = self.op_move(model_copy)
            elif op == 'break':
                model_copy = self.op_break(model_copy, classes)
            elif op == 'strategic':
                model_copy = self.op_strategic(model_copy, conclusion)

            if self.holds(model_copy, stmt1) and self.holds(model_copy, stmt2):
                model = model_copy
                conclusion = self.weaken(model, conclusion)

        return model, conclusion

    def predict(self, item, response=None, **kwargs):

        # Init state
        self.weakenings = self.max_weakenings

        task = ccobra.syllogistic_generalized.GeneralizedSyllogism(item).task

        if self.most_not:
            for x in task:
                if x[0] == 'Most not':
                    x[0] = 'Few'
            if response is not None and response[0] == 'Most not':
                response[0] = 'Few'

        self.logger.debug("#####" + str(item.identifier) + str(item.sequence_number) + "#####")
        self.logger.debug('')
        pp_syllogism(self.logger, task, response)

        classes = list({task[0][1], task[0][2], task[1][1], task[1][2]})

        conclusion = self.initial_conclusion(item, task)

        # See whether we can predict correctly with the right amount of weakening
        if response is None:
            self.logger.warning("Response not given. Cannot assess heuristic performance")
        else:
            intersection = [x for x in response if x in self.orbit(conclusion)]
            if len(intersection) == 0:
                self.logger.debug("The heuristic fails!")

        model = self.initial_model(task)

        self.logger.debug(pp_model(model, classes))
        self.logger.debug(pp_statement(conclusion) + "(heuristic)")

        assert self.holds(model, task[0])
        assert self.holds(model, task[1])

        conclusion = self.weaken(model, conclusion)
        self.logger.debug(pp_statement(conclusion) + "(system 1)")

        model, conclusion = self.system2(model, task[0], task[1], conclusion, classes)

        self.logger.debug(pp_statement(conclusion) + '(system 2)')
        self.logger.debug('')
        self.logger.debug(pp_model(model, classes))

        assert self.holds(model, task[0])
        assert self.holds(model, task[1])

        if self.some_no_sym:
            if response is None:
                logger.error("Please provide subject response for this configuration (some_no_sym)")
            elif conclusion[0] in ['Some', 'No'] and conclusion[0] == response[0][0]:
                return [response[0]]

        if conclusion[0] == 'NVC':
            conclusion = ['NVC']
        if self.most_not and conclusion[0] == 'Few':
            conclusion[0] = 'Most not'
        return [conclusion]

    # All possible weakenings
    def orbit(self, conclusion):
        res = [['NVC'], conclusion]
        while not conclusion[0] == 'NVC':
            conclusion = conclusion.copy()
            if self.some_no_sym:
                if conclusion[0] in ['Some', 'No']:
                    c_sym = conclusion.copy()
                    c_sym[1] = conclusion[2]
                    c_sym[2] = conclusion[1]
                    res.append(c_sym)
            quantifier = self.weakening.get(conclusion[0])
            conclusion[0] = quantifier
            res.append(conclusion)
        return res
