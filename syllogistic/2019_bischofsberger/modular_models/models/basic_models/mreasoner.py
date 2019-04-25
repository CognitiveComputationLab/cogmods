import ccobra
import random
import copy
import sys
import os

from .interface import SyllogisticReasoningModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.util import sylutil


class MReasoner(SyllogisticReasoningModel):
    """ Funktion passt. Doku fehlt. If "NVC" in conclusions unbefriedigend.
    """
    def __init__(self):
        SyllogisticReasoningModel.__init__(self)
#        self.params["falsify"] = 0.6
#        self.param_grid["falsify"] = [0.0, 0.2, 0.4, 0.6, 0.8]

    def encode(self, syllogism):
        """ Returns initial MM representation of a syllogism + list of exhausted terms

        >>> mm = MentalModels()
        >>> mm.encode("AA1")
        ([['a', 'b', 'c'], ['a', 'b', 'c']], ['a', 'b'])
        >>> mm.encode("AA2")
        ([['a', 'b', 'c'], ['a', 'b', 'c']], ['b', 'c'])
        >>> mm.encode("AA3")
        ([['a', 'b', 'c'], ['a', 'b', 'c']], ['a', 'c'])
        >>> mm.encode("AA4")
        ([['a', 'b', 'c'], ['a', 'b', 'c']], ['b'])
        >>> mm.encode("AE1")
        ([['a', 'b', '-c'], ['a', 'b', '-c'], ['c'], ['c']], ['a', 'b', 'c'])
        >>> mm.encode("AE2")
        ([['-b', 'c'], ['-b', 'c'], ['a', 'b'], ['a', 'b']], ['b', 'c'])
        >>> mm.encode("AE3")
        ([['a', 'b'], ['a', 'b'], ['-b', 'c'], ['-b', 'c']], ['a', 'b', 'c'])
        >>> mm.encode("AE4")
        ([['a', 'b', '-c'], ['a', 'b', '-c'], ['c'], ['c']], ['b', 'c'])
        >>> mm.encode("AI1")
        ([['a', 'b', 'c'], ['a', 'b'], ['c']], ['a'])
        >>> mm.encode("AI2")
        ([['a', 'b', 'c'], ['c'], ['a', 'b']], ['b'])
        >>> mm.encode("AI3")
        ([['a', 'b', 'c'], ['a', 'b'], ['c']], ['a'])
        >>> mm.encode("AI4")
        ([['a', 'b', 'c'], ['a', 'b'], ['c']], ['b'])
        >>> mm.encode("AO1")
        ([['a', 'b', '-c'], ['a', 'b', '-c'], ['c'], ['c']], ['a'])
        >>> mm.encode("AO2")
        ([['-b', 'c'], ['-b', 'c'], ['a', 'b'], ['a', 'b']], ['b'])
        >>> mm.encode("AO3")
        ([['a', 'b'], ['a', 'b'], ['-b', 'c'], ['-b', 'c']], ['a'])
        >>> mm.encode("AO4")
        ([['a', 'b', '-c'], ['a', 'b', '-c'], ['c'], ['c']], ['b'])
        >>> mm.encode("EA1")
        ([['a', '-b'], ['a', '-b'], ['b', 'c'], ['b', 'c']], ['a', 'b'])
        >>> mm.encode("EA2")
        ([['-b', 'c'], ['-b', 'c'], ['-a', 'b'], ['-a', 'b'], ['a'], ['a']], ['a', 'b'])
        >>> mm.encode("EA3")
        ([['a', '-b'], ['a', '-b'], ['b', 'c'], ['b', 'c']], ['a', 'b', 'c'])
        >>> mm.encode("EI1")
        ([['a', '-b'], ['a', '-b'], ['b', 'c'], ['b'], ['c']], ['a', 'b'])
        >>> mm.encode("EI3")
        ([['a', '-b'], ['a', '-b'], ['b', 'c'], ['b'], ['c']], ['a', 'b'])
        >>> mm.encode("IA1")
        ([['a', 'b', 'c'], ['a'], ['b', 'c']], ['b'])
        >>> mm.encode("IA3")
        ([['a', 'b', 'c'], ['a'], ['b', 'c']], ['c'])
        >>> mm.encode("IA4")
        ([['a', 'b', 'c'], ['a'], ['b', 'c']], ['b'])
        >>> mm.encode("IE1")
        ([['a', 'b', '-c'], ['a'], ['b', '-c'], ['c'], ['c']], ['b', 'c'])
        >>> mm.encode("II1")
        ([['a', 'b', 'c'], ['a'], ['b'], ['c']], [])
        >>> mm.encode("IO1")
        ([['a', 'b', '-c'], ['a'], ['b', '-c'], ['c'], ['c']], [])
        >>> mm.encode("OA4")
        ([['-a', 'b', 'c'], ['-a', 'b', 'c'], ['a'], ['a']], ['b'])
        >>> mm.encode("OE4")
        ([['-a', 'b', '-c'], ['-a', 'b', '-c'], ['a'], ['a'], ['c'], ['c']], ['b', 'c'])
        >>> mm.encode("OI4")
        ([['-a', 'b', 'c'], ['-a', 'b'], ['a'], ['a'], ['c']], [])
        """

        mm = []
        exhausted = set()
        to = sylutil.term_order(syllogism[2])
        subj_0, pred_0 = to[0]
        subj_1, pred_1 = to[1]
        if syllogism[0] == "A":
            mm.extend((["a", "b"], ["a", "b"]))
            exhausted.add(subj_0)
        if syllogism[0] == "I":
            mm.extend((["a", "b"], ["a"], ["b"]))
        if syllogism[0] == "E":
            mm.extend(([subj_0, "-" + pred_0], [subj_0, "-" + pred_0], [pred_0], [pred_0]))
            exhausted.update((subj_0, pred_0))
        if syllogism[0] == "O":
            mm.extend(([subj_0, "-" + pred_0], [subj_0, "-" + pred_0], [pred_0], [pred_0]))

        i_b = [i for i, row in enumerate(mm) if "b" in row]
        if syllogism[1] == "A":
            for i in i_b:
                mm[i].append("c")
            exhausted.add(subj_1)
        if syllogism[1] == "I":
            mm[i_b[0]].append("c")
            mm.append(["c"])
        if syllogism[1] == "E" or syllogism[1] == "O":
            if subj_1 == "b":
                for i in i_b:
                    mm[i].append("-c")
                mm.extend((["c"], ["c"]))
            else:
                mm.extend((["-b", "c"], ["-b", "c"]))
            if syllogism[1] == "E":
                exhausted.update(("b", "c"))

        return [sorted(row, key=lambda e: e[-1]) for row in mm], sorted(list(exhausted))

    def rule_break(self, mental_model, exhausted):
        """ Apply Add Negative Token rule to MM, attempting to refute a particular conclusion. """
        if "b" in exhausted:
            return mental_model
        new_model = []
        for i, row in enumerate(mental_model):
            if len(row) == 3:
                new_model.extend(mental_model[:i])
                ia = row.index("a") if "a" in row else row.index("-a")
                ib = row.index("b") if "b" in row else row.index("-b")
                ic = row.index("c") if "c" in row else row.index("-c")
                new_model.extend([[row[ia], row[ib]], [row[ib], row[ic]]])
                new_model.extend(mental_model[i + 1:])
                return new_model
        return mental_model

    def rule_add_pos(self, mental_model, exhausted, conclusion):
        """ Apply Add Positive Token rule to MM, attempting to refute a particular conclusion. """
        if conclusion[0] != "A":
            return mental_model
        new_mental_model = copy.deepcopy(mental_model)
        for x in ["a", "c"]:
            if [x] not in mental_model and x not in exhausted:
                new_mental_model.append([x])
                return new_mental_model
        return mental_model

    def rule_add_neg(self, mental_model, exhausted, conclusion):
        """ Apply Add Negative Token rule to MM, attempting to refute a particular conclusion. """
        new_model = copy.deepcopy(mental_model)
        for x in ["a", "c"]:
            if x in exhausted:
                continue
            x_other = "a" if x == "c" else "c"
            i_targets = [i for i, row in enumerate(new_model) if
                         (x not in row and "-" + x not in row) and (
                                     x_other in row or "-" + x_other in row)]
            if len(i_targets) == 0:
                continue
            if conclusion[0] == "O":
                for i in i_targets:
                    new_model[i].append(x)
            else:
                new_model[i_targets[0]].append(x)
        #            return new_model
        return new_model

    def rule_move(self, mental_model, exhausted, conclusion):
        """ Apply Move Token rule to MM, attempting to refute a particular conclusion. """
        ia = next((i for i, row in enumerate(mental_model) if
                   "a" in row and "c" not in row and "-c" not in row), None)
        ic = next((i for i, row in enumerate(mental_model) if
                   "c" in row and "a" not in row and "-a" not in row), None)

        if ia is not None and ic is not None and self.consistent_b(mental_model[ia],
                                                                   mental_model[ic]):
            if "a" in exhausted and "c" in exhausted or conclusion[0] == "O":
                new_model = self.do_move(mental_model, ia, ic)
                return self.rule_move(new_model, exhausted, conclusion)
            else:
                return self.do_move(mental_model, ia, ic)
        return mental_model

    def do_move(self, mental_model, i_from, i_to):
        """ Generate a MM from another MM by moving tokens from one row to another """
        new_model = copy.deepcopy(mental_model)
        new_model[i_to] = sorted(list(set(new_model[i_from] + new_model[i_to])))
        del new_model[i_from]
        return new_model

    def consistent_b(self, row1, row2):
        """ Check if the sign of the middle token ("b") is the same in two rows """
        if "b" in row1 and "-b" in row2 or "-b" in row1 and "b" in row2:
            return False
        return True

    def model_includes_negative(self, mental_model):
        """ Check if MM contains a negative token """
        for row in mental_model:
            for symbol in row:
                if symbol[0] == "-":
                    return True
        return False

    def models_equal(self, m1, m2):
        """ Checks if two MMs have equal content. """
        s1 = sorted([sorted(row) for row in m1])
        s2 = sorted([sorted(row) for row in m2])
        if s1 == s2:
            return True
        return False

    def conclude(self, mental_model, exhausted, exclude_weaker=True):
        """
        Generate a list of conclusions from a MM.

        Incomplete number of old cases obtained from Public-Syllog.lisp
        >>> mm = MentalModels()
        >>> mm.conclude([['a', 'b', 'c'], ['a', 'b', 'c']], ['a', 'b']) # AA1
        ['Aac', 'Aca']
        >>> mm.conclude([['a', 'b', 'c'], ['a', 'b', 'c'], ['c']], ['a', 'b'])
        ['Aac', 'Ica']
        >>> mm.conclude([['a', 'b', 'c'], ['a', 'b'], ['c']], ['a']) # AI1
        ['Iac', 'Ica']
        >>> mm.conclude([['a', 'b'], ['b', 'c'], ['a', 'b'], ['c']], ['a'])
        ['NVC', 'NVC']
        >>> mm.conclude([['a', 'b', '-c'], ['a', 'b', '-c'], ['c'], ['c']], ['a', 'b', 'c']) # AE1
        ['Eac', 'Eca']
        >>> mm.conclude([['a', 'b', '-c'], ['a', 'b', '-c'], ['c'], ['c']], ['a']) # AO1
        ['Oac', 'Oca']
        >>> mm.conclude([['a', 'b', 'c'], ['a', 'b', 'c'], ['b', '-c'], ['b', '-c']], ['a'])
        ['NVC', 'NVC']

        >>> mm.conclude([['a', 'b', 'c'], ['a'], ['b', 'c']], ['b']) # IA1
        ['Iac', 'Ica']
        >>> mm.conclude([['a', 'b', 'c'], ['a'], ['b'], ['c']], []) # II1
        ['Iac', 'Ica']
        >>> mm.conclude([['a', 'b'], ['b', 'c'], ['a'], ['b'], ['c']], [])
        ['NVC', 'NVC']
        >>> mm.conclude([['a', 'b', '-c'], ['a'], ['b', '-c'], ['c'], ['c']], ['b', 'c']) # IE1
        ['Eac', 'Eca']
        >>> mm.conclude([['a', 'c'], ['a', 'b', '-c'], ['b', '-c'], ['c']], ['b', 'c'])
        ['Oac', 'Oca']
        >>> mm.conclude([['a', 'c'], ['a', 'b', '-c'], ['b', '-c'], ['a', 'c']], ['b', 'c'])
        ['Oac', 'NVC']
        >>> mm.conclude([['a', 'b', '-c'], ['a'], ['b', '-c'], ['c'], ['c']], []) # IO1
        ['Oac', 'Oca']
        >>> mm.conclude([['a', 'c'], ['a', 'b', 'c'], ['b', '-c'], ['b', '-c']], [])
        ['NVC', 'NVC']

        >>> mm.conclude([['a', '-b'], ['a', '-b'], ['b', 'c'], ['b', 'c']], ['a', 'b']) # EA1
        ['Eac', 'Eca']
        >>> mm.conclude([['a', '-b', 'c'], ['a', '-b'], ['b', 'c'], ['b', 'c']], ['a', 'b'])
        ['Oac', 'Oca']
        >>> mm.conclude([['a', '-b', 'c'], ['a', '-b', 'c'], ['b', 'c'], ['b', 'c']], ['a', 'b'])
        ['NVC', 'Oca']
        >>> mm.conclude([['a', '-b'], ['a', '-b'], ['b', 'c'], ['b'], ['c']], ['a', 'b']) # EI1
        ['Eac', 'Eca']
        >>> mm.conclude([['a', '-b', 'c'], ['a', '-b'], ['b', 'c'], ['b'], ['c']], ['a', 'b'])
        ['Oac', 'Oca']
        >>> mm.conclude([['a', '-b', 'c'], ['a', '-b', 'c'], ['b', 'c'], ['b'], ['c']], ['a', 'b'])
        ['NVC', 'Oca']
        >>> mm.conclude([['a', '-b'], ['a', '-b'], ['b', '-c'], ['b', '-c'], ['c'], ['c']], ['a', 'b', 'c']) # EE1
        ['Eac', 'Eca']
        >>> mm.conclude([['a', '-b', 'c'], ['a', '-b', 'c'], ['b', '-c'], ['b', '-c']], ['a', 'b', 'c'])
        ['NVC', 'NVC']
        >>> mm.conclude([['a', '-b'], ['a', '-b'], ['b', '-c'], ['b', '-c'], ['c'], ['c']], ['a', 'b']) # EO1
        ['Eac', 'Eca']
        >>> mm.conclude([['a', '-b', 'c'], ['a', '-b'], ['b', '-c'], ['b', '-c'], ['c']], ['a', 'b'])
        ['Oac', 'Oca']
        >>> mm.conclude([['a', '-b', 'c'], ['a', '-b', 'c'], ['b', '-c'], ['b', '-c']], ['a', 'b'])
        ['NVC', 'NVC']
        """
        res_pos = {"Aac": True, "Aca": True, "Iac": False, "Ica": False}
        res_neg = {"Eac": True, "Eca": True, "Oac": False, "Oca": False}
        res = res_neg if self.model_includes_negative(mental_model) else res_pos
        conclusions = []

        for subj in ["a", "c"]:
            obj = "a" if subj == "c" else "c"
            so = subj + obj
            for row in [row for row in mental_model if subj in row]:
                if obj in row:
                    res_pos["I" + so] = True
                if obj not in row:
                    res_pos["A" + so] = False
                if obj not in row:
                    res_neg["O" + so] = True
                if obj in row:
                    res_neg["E" + so] = False

            if len(exhausted) < 2:
                res_neg["E" + so] = False

            if exclude_weaker:
                if res_pos["A" + so]:
                    res_pos["I" + so] = False
                if res_neg["E" + so]:
                    res_neg["O" + so] = False

            new_concl = [c for c in res if res[c] is True and so in c]
            if not new_concl:
                conclusions.append("NVC")
            else:
                if len(new_concl) != 1:
                    raise Exception
                conclusions.append(new_concl[0])

        return conclusions

    def falsify(self, mental_model, exhausted, conclusion):
        """ Returns a new mental model generated from an existing one as a counterexample for a
        conclusion
        """
        if conclusion == "NVC":
            return mental_model
        negative = self.model_includes_negative(mental_model)
        if not negative:
            new_model = self.rule_break(mental_model, exhausted)
        else:
            intermediate_model = copy.deepcopy(mental_model)
            while True:
                intermediate_model_old = copy.deepcopy(intermediate_model)
                intermediate_model = self.rule_break(intermediate_model, exhausted)
                if self.models_equal(intermediate_model_old, intermediate_model):
                    break
            new_model = self.rule_move(intermediate_model, exhausted, conclusion)
        if self.models_equal(new_model, mental_model):
            if not negative:
                new_model = self.rule_add_pos(mental_model, exhausted, conclusion)
            else:
                new_model = self.rule_add_neg(mental_model, exhausted, conclusion)
                if self.models_equal(new_model, mental_model):
                    return new_model
            return new_model
        else:
            return new_model

    def predict(self, syllogism):
        initial_model, exhausted = self.encode(syllogism)
        conclusions = self.conclude(initial_model, exhausted)
        current_model = initial_model
        while random.random() < self.params["falsify"]:
            conclusions = self.conclude(current_model, exhausted)
            new_model = self.falsify(current_model, exhausted, conclusions[0])
            if self.models_equal(new_model, current_model):
                new_model = self.falsify(current_model, exhausted, conclusions[1])
                if self.models_equal(new_model, current_model):
                    break
            current_model = new_model

        return conclusions
