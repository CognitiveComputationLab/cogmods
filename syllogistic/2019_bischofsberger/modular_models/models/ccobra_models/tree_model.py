# -*- coding: utf-8 -*-

# vm_reencode konditioniert, Operationen rausgenommen, ...
# conclusions == ["NVC"] bei vm_reencode als precondition

import ccobra
import copy
import numpy as np
import os
import pprint
import random
import sys
from anytree import AnyNode, RenderTree, Resolver, search, PreOrderIter
from enum import Enum
from functools import lru_cache
from os.path import dirname

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.util import sylutil
from modular_models.models.basic_models import GeneralizedMatching, PHM
from modular_models.models.ccobra_models import CCobraPSYCOP, CCobraVerbalModels, CCobraMentalModels


psycop = CCobraPSYCOP()
mm = CCobraMentalModels()
vm = CCobraVerbalModels()
gm = GeneralizedMatching()
phm = PHM()


def resp_prefer_stronger(mr):
    conclusions = mr[1]
    prefered_conclusions = []
    for c in conclusions:
        if c[0] in ["I", "O"] and {"I": "A", "O": "E"}[c[0]] + c[1:] in conclusions:
            continue
        prefered_conclusions.append(c)
    return random.choice(prefered_conclusions)


def resp_prefer(mr, stronger_weaker=None, ac_ca=None):
    conclusions = mr[1]
    prefered_conclusions = []
    for c in conclusions:
        if stronger_weaker == "stronger":
            if c[0] in ["I", "O"] and {"I": "A", "O": "E"}[c[0]] + c[1:] in conclusions:
                continue
        elif stronger_weaker == "weaker":
            if c[0] in ["A", "E"] and {"A": "I", "E": "O"}[c[0]] + c[1:] in conclusions:
                continue
        if ac_ca is not None:
            if c[1:] != ac_ca and (c[0] + ac_ca[0] + ac_ca[1]) in conclusions:
                continue
        prefered_conclusions.append(c)
    return random.choice(prefered_conclusions)


def phm_heuristic_min(item):
    syl = ccobra.syllogistic.Syllogism(item).encoded_task
    return [phm.f_min_heuristic[syl[:2]] + ac for ac in ["ac", "ca"]]


def phm_p_entailment(mr):
    conclusions = mr[1]
    new_conclusions = []
    for c in conclusions:
        if c == "NVC":
            new_conclusions.append("NVC")
        else:
            new_conclusions.append(phm.f_p_entailment[c[0]] + c[1:])
    return new_conclusions


def heuristic_gm(item, par_total_order, co):
    syl = ccobra.syllogistic.Syllogism(item).encoded_task

    before_params = gm.get_params()
    gm.params["total_order"] = par_total_order
    moods = gm.heuristic_generalized_matching(syl)
    gm.set_params(before_params)

    return [m + co for m in moods]


def vm_encode_new(item):
    syllogism = ccobra.syllogistic.Syllogism(item).encoded_task
    exhausted = mm_vmstyle_encode(item)[1]

    return vm.model.encode(syllogism), exhausted


def IC_reverse_premises(item, first=True, second=True):
    new_item = copy.deepcopy(item)

    # reverse first premise
    if first:
        new_item.task[0][-1], new_item.task[0][-2] = item.task[0][-2], item.task[0][-1]

    # reverse second premise
    if second:
        new_item.task[1][-1], new_item.task[1][-2] = item.task[1][-2], item.task[1][-1]

    return new_item


def ic_reverse_premise(item, first=True, second=True):
    new_premises = []
    p1, p2 = (encode_proposition(x, item) for x in item.task)
    new_premises.extend([p1, p2])
    if first:
        new_premises.append(p1[0] + p1[2] + p1[1])
    if second:
        new_premises.append(p2[0] + p2[2] + p2[1])
    return new_premises


def heuristic_atmosphere(item):
    f = {"AA": ["Aac", "Aca"],
         "AE": ["Eac", "Eca"], "EA": ["Eac", "Eca"],
         "AI": ["Iac", "Ica"], "IA": ["Iac", "Ica"],
         "AO": ["Oac", "Oca"], "OA": ["Oac", "Oca"],
         "EE": ["Eac", "Eca"],
         "EI": ["Oac", "Oca"], "IE": ["Oac", "Oca"],
         "EO": ["Oac", "Oca"], "OE": ["Oac", "Oca"],
         "II": ["Iac", "Ica"],
         "IO": ["Oac", "Oca"], "OI": ["Oac", "Oca"],
         "OO": ["Oac", "Oca"]}
    syl = ccobra.syllogistic.Syllogism(item)
    return f[syl.encoded_task[:2]]

def heuristic_matching(item):
    f2 = {"AA": ["Aac", "Aca"],
          "AE": ["Eac", "Eca"], "EA": ["Eac", "Eca"],
          "AI": ["Iac", "Ica"], "IA": ["Iac", "Ica"],
          "AO": ["Oac", "Oca"], "OA": ["Oac", "Oca"],
          "EE": ["Eac", "Eca"],
          "EI": ["Eac", "Eca", "Iac", "Ica"], "IE": ["Eac", "Eca", "Iac", "Ica"],
          "EO": ["Eac", "Eca", "Oac", "Oca"], "OE": ["Eac", "Eca", "Oac", "Oca"],
          "II": ["Iac", "Ica"],
          "IO": ["Iac", "Ica", "Oac", "Oca"], "OI": ["Iac", "Ica", "Oac", "Oca"],
          "OO": ["Oac", "Oca"]}
    syl = ccobra.syllogistic.Syllogism(item)
    return f2[syl.encoded_task[:2]]

t_mm = -1
def get_time():
    global t_mm
    t_mm += 1
    return t_mm


def vm_to_mm(verbal_model):
    mental_model = []
    for i, ind in enumerate(verbal_model):
        mental_model.append([])
        for p in ind.props:
            p_str = "-"+p.name if p.neg else p.name
            mental_model[i].append(p_str)
    return mental_model


def mm_to_vm(mental_model):
    verbal_model = []
    for row in mental_model:
        props = []
        for p in row:
            neg = True if p[0] == "-" else False
            props.append(VerbalModels.Prop(name=p[-1], neg=neg, identifying=False, t=-1))
        verbal_model.append(VerbalModels.Individual(props=props, t=-1))
    return verbal_model


def mm_vmstyle_encode_premises(premises):
#    syllogism = ccobra.syllogistic.Syllogism(item).encoded_task
    mm = []
    exhausted = []
#    to = sylutil.term_order(syllogism[2])
#    subj_0, pred_0 = to[0]
    subj_0, pred_0 = premises[0][1], premises[0][2]
#    subj_1, pred_1 = to[1]

    t0 = get_time()
    t1 = get_time()

    s0 = VerbalModels.Prop(name=subj_0, neg=False, identifying=False, t=t0)
    p0 = VerbalModels.Prop(name=pred_0, neg=False, identifying=False, t=t0)
    non_p0 = VerbalModels.Prop(name=pred_0, neg=True, identifying=False, t=t0)
#    s1 = VerbalModels.Prop(name=subj_1, neg=False, identifying=False, t=t0)
#    p1 = VerbalModels.Prop(name=pred_1, neg=False, identifying=False, t=t0)

    ### First premise ###
    if premises[0][0] == "A":
        mm.extend([VerbalModels.Individual(props=[s0, p0], t=t0),
                   VerbalModels.Individual(props=[s0, p0], t=t0)])
        if s0 not in exhausted:
           exhausted.append(s0)
    if premises[0][0] == "I":
        mm.extend((VerbalModels.Individual(props=[s0, p0], t=t0),
                   VerbalModels.Individual(props=[s0], t=t0),
                   VerbalModels.Individual(props=[p0], t=t0)))
    if premises[0][0] == "E":
        mm.extend([VerbalModels.Individual(props=[s0, non_p0], t=t0),
                   VerbalModels.Individual(props=[s0, non_p0], t=t0),
                   VerbalModels.Individual(props=[p0], t=t0),
                   VerbalModels.Individual(props=[p0], t=t0)])
        if s0 not in exhausted:
            exhausted.append(s0)
        if p0 not in exhausted:
            exhausted.append(p0)
    if premises[0][0] == "O":
        mm.extend([VerbalModels.Individual(props=[s0, non_p0], t=t0),
                   VerbalModels.Individual(props=[s0, non_p0], t=t0),
                   VerbalModels.Individual(props=[p0], t=t0),
                   VerbalModels.Individual(props=[p0], t=t0)])

    for n, prem in enumerate(premises[1:]):
        tn = get_time()
        subj_n, pred_n = prem[1], prem[2]
        end_term_n = subj_n if subj_n != "b" else pred_n
        sn = VerbalModels.Prop(name=subj_n, neg=False, identifying=False, t=t0)
        pn = VerbalModels.Prop(name=pred_n, neg=False, identifying=False, t=t0)
#        non_pn = VerbalModels.Prop(name=pred_0, neg=True, identifying=False, t=t0)

        ### Second premise ###
        prop_b = VerbalModels.Prop(name="b", neg=False, identifying=False, t=t0)
        i_b = [i for i, row in enumerate(mm) if row.contains(prop_b)]

        prop_et = VerbalModels.Prop(name=end_term_n, neg=False, identifying=False, t=tn)
        prop_non_et = VerbalModels.Prop(name=end_term_n, neg=True, identifying=False, t=tn)
        ind_et = VerbalModels.Individual(props=[prop_et], t=tn)

#        prop_c = VerbalModels.Prop(name="c", neg=False, identifying=False, t=t1)
#        prop_non_c = VerbalModels.Prop(name="c", neg=True, identifying=False, t=t1)
#        ind_c = VerbalModels.Individual(props=[prop_c], t=t1)
        prop_non_b = VerbalModels.Prop(name="b", neg=False, identifying=False, t=tn)

        if prem[0] == "A":
            for i in i_b:
                mm[i].props.append(prop_et)
            if sn not in exhausted:
                exhausted.append(sn)
        if prem[0] == "I":
            mm[i_b[0]].props.append(prop_et)
            mm.append(ind_et)
        if prem[0] == "E" or prem[0] == "O":
            if sn == prop_b:
                for i in i_b:
                    mm[i].props.append(prop_non_et)
                mm.extend((ind_et, ind_et))
            else:
                mm.extend([VerbalModels.Individual(props=[prop_non_b, prop_et], t=tn),
                           VerbalModels.Individual(props=[prop_non_b, prop_et], t=tn)])
            if prem[0] == "E":
                if prop_b not in exhausted:
                    exhausted.append(prop_b)
                if prop_et not in exhausted:
                    exhausted.append(prop_et)

    [ind.props.sort(key=lambda p: p.name) for ind in mm]
    return mm, sorted(exhausted, key=lambda p: p.name)

def mm_vmstyle_encode(item):
    syllogism = ccobra.syllogistic.Syllogism(item).encoded_task
    mm = []
    exhausted = []
    to = sylutil.term_order(syllogism[2])
    subj_0, pred_0 = to[0]
    subj_1, pred_1 = to[1]

    t0 = get_time()
    t1 = get_time()

    s0 = VerbalModels.Prop(name=subj_0, neg=False, identifying=False, t=t0)
    p0 = VerbalModels.Prop(name=pred_0, neg=False, identifying=False, t=t0)
    non_p0 = VerbalModels.Prop(name=pred_0, neg=True, identifying=False, t=t0)
    s1 = VerbalModels.Prop(name=subj_1, neg=False, identifying=False, t=t0)
    p1 = VerbalModels.Prop(name=pred_1, neg=False, identifying=False, t=t0)

    ### First premise ###
    if syllogism[0] == "A":
        mm.extend([VerbalModels.Individual(props=[s0, p0], t=t0),
                   VerbalModels.Individual(props=[s0, p0], t=t0)])
        if s0 not in exhausted:
           exhausted.append(s0)
    if syllogism[0] == "I":
        mm.extend((VerbalModels.Individual(props=[s0, p0], t=t0),
                   VerbalModels.Individual(props=[s0], t=t0),
                   VerbalModels.Individual(props=[p0], t=t0)))
    if syllogism[0] == "E":
        mm.extend([VerbalModels.Individual(props=[s0, non_p0], t=t0),
                   VerbalModels.Individual(props=[s0, non_p0], t=t0),
                   VerbalModels.Individual(props=[p0], t=t0),
                   VerbalModels.Individual(props=[p0], t=t0)])
        if s0 not in exhausted:
            exhausted.append(s0)
        if p0 not in exhausted:
            exhausted.append(p0)
    if syllogism[0] == "O":
        mm.extend([VerbalModels.Individual(props=[s0, non_p0], t=t0),
                   VerbalModels.Individual(props=[s0, non_p0], t=t0),
                   VerbalModels.Individual(props=[p0], t=t0),
                   VerbalModels.Individual(props=[p0], t=t0)])

    ### Second premise ###
    prop_b = VerbalModels.Prop(name="b", neg=False, identifying=False, t=t0)
    i_b = [i for i, row in enumerate(mm) if row.contains(prop_b)]

    prop_c = VerbalModels.Prop(name="c", neg=False, identifying=False, t=t1)
    prop_non_c = VerbalModels.Prop(name="c", neg=True, identifying=False, t=t1)
    ind_c = VerbalModels.Individual(props=[prop_c], t=t1)
    prop_non_b = VerbalModels.Prop(name="b", neg=False, identifying=False, t=t1)

    if syllogism[1] == "A":
        for i in i_b:
            mm[i].props.append(prop_c)
        if s1 not in exhausted:
            exhausted.append(s1)
    if syllogism[1] == "I":
        mm[i_b[0]].props.append(prop_c)
        mm.append(ind_c)
    if syllogism[1] == "E" or syllogism[1] == "O":
        if s1 == prop_b:
            for i in i_b:
                mm[i].props.append(prop_non_c)
            mm.extend((ind_c, ind_c))
        else:
            mm.extend([VerbalModels.Individual(props=[prop_non_b, prop_c], t=t1),
                       VerbalModels.Individual(props=[prop_non_b, prop_c], t=t1)])
        if syllogism[1] == "E":
            if prop_b not in exhausted:
                exhausted.append(prop_b)
            if prop_c not in exhausted:
                exhausted.append(prop_c)

    [ind.props.sort(key=lambda p: p.name) for ind in mm]
    return mm, sorted(exhausted, key=lambda p: p.name)

def mm_conclude(mr, exclude_weaker=True):
    verbal_model, exhausted = mr[0]
    return mm.model.conclude(vm_to_mm(verbal_model), exhausted, exclude_weaker)

def vm_conclude(mr):
    verbal_model, exhausted = mr[0]
    return vm.model.conclude(verbal_model)

def mm_falsify(mr, i=0):
    verbal_model, exhausted = mr[0]
    conclusions = mr[1]

    new_mental_model = mm.model.falsify(vm_to_mm(verbal_model), [e.name for e in exhausted], conclusions[i])
    new_verbal_model = mm_to_vm(new_mental_model)

    return new_verbal_model, exhausted

def vm_encode(item, p1=None, p2=None, p3=None, p4=None, p5=None, p6=None):
    syllogism = ccobra.syllogistic.Syllogism(item).encoded_task
    params_before = vm.model.get_params()

    exhausted = mm_vmstyle_encode(item)[1]

    if p1 is not None:
        vm.model.params["p1"] = p1
        vm.model.params["p2"] = p2
        vm.model.params["p3"] = p3
        vm.model.params["p4"] = p4
        vm.model.params["p5"] = p5
        vm.model.params["p6"] = p6
    new_model = vm.model.encode(syllogism)

    vm.model.set_params(params_before)
    return new_model, exhausted


def encode_proposition(prop, item):
    # ["Some", "x", "y"] -> Iac (or Ica, dependent on task)
    quantor = {"All": "A", "Some": "I", "No": "E", "Some not": "O"}[prop[0]]
    middle_term = list(set(item.task[0][1:]).intersection(item.task[1][1:]))[0]
    a_term = list(set(item.task[0][1:]) - set(middle_term))[0]
    c_term = list(set(item.task[1][1:]) - set(middle_term))[0]

#    print("encode_proposition")

#    print(prop)
#    print(item)
#    print(quantor)
#    print(a_term)
#    print(c_term)
#    print(middle_term)

    i = prop[1:].index(middle_term)
    end_term = "a" if a_term in prop else "c"
#    print(end_term)
    pr_enc = quantor + "b" + end_term if i == 0 else quantor + end_term + "b"

#    print(pr_enc)
    return pr_enc



def vm_encode_premises(premises):
    verbal_model = []
#    premises = [syllogism[i] + sylutil.term_order(syllogism[2])[i] for i in [0, 1]]
    for premise in premises:
#        p_enc = encode_proposition(premise, )
        verbal_model = vm.model.extend_vm(verbal_model, premise, reencoding=False)

    exhausted = mm_vmstyle_encode_premises(premises)[1]
    return verbal_model, exhausted


def vm_reencode(mr, p10=None, p11=None, p12=None, p13=None):
    verbal_model, exhausted = mr[0]
    conclusions = mr[1]
    syllogism = current_syllogism

    params_before = vm.model.get_params()

    if p10 is not None:
        vm.model.params["p10"] = p10
        vm.model.params["p11"] = p11
        vm.model.params["p12"] = p12
        vm.model.params["p13"] = p13

    new_conclusions = vm.model.reencode(syllogism, verbal_model, conclusions)

    vm.model.set_params(params_before)
    return new_conclusions


def response_by_order(mr, order="ac"):
    for conclusion in mr[1]:
        if order in conclusion:
            return conclusion
    # prefered order not present -> return first conclusion in list
    return mr[1][0]

#def psycop_forward_rules(item):
#    syllogism = ccobra.syllogistic.Syllogism(item).encoded_task
#    premises = self.encode_premises(syllogism,
#                                    ex_implicatures=self.params["premise_implicatures_existential"],
#                                    grice_implicatures=self.params["premise_implicatures_grice"])
#    fw_propositions = self.run_forward_rules(premises)


def psycop_check(mr):
    conclusions = mr[1]
    syllogism = current_syllogism
    psycop_predictions = psycop.cached_prediction(syllogism)
    confirmed_conclusions = []
    for c in conclusions:
        if c == "NVC" or c in psycop_predictions:
            confirmed_conclusions.append(c)
    if len(confirmed_conclusions) == 0:
        return None
    return confirmed_conclusions


OpType = Enum("OpType", "PREENCODE, HEURISTIC, ENCODE, CONCLUDE, REENCODE, RESPOND, FALSIFY, ENTAIL")
State = Enum("State", "Item, Premises, MR, Response")
optype_to_states = {OpType.PREENCODE: (State.Item, State.Premises),
                    OpType.HEURISTIC: (State.Item, State.MR),
                    OpType.ENCODE: (State.Premises, State.MR),
                    OpType.RESPOND: (State.MR, State.Response),
                    OpType.CONCLUDE: (State.MR, State.MR),
                    OpType.REENCODE: (State.MR, State.MR),
                    OpType.ENTAIL: (State.MR, State.MR),
                    OpType.FALSIFY: (State.MR, State.MR),
                    }

current_syllogism = ""

class AbstractModel(ccobra.CCobraModel):
    class Operation:
        def __init__(self, optype, fnc, args):
            self.optype = optype
            self.fnc = fnc
            self.args = args
            self.pre_state, self.post_state = optype_to_states[optype]


        def eval_preconditions(self, current_node):
            if self.fnc == IC_reverse_premises:
                if current_node.vars["num_preencodes"] == 0:
                   return True
                return False
            if self.optype == OpType.RESPOND:
                if current_node.content[1] is None:
                    return False
            if self.optype == OpType.CONCLUDE:
                if current_node.vars["num_concludes"] > 0 or current_node.content[0] is None:
                    return False
            if self.optype == OpType.REENCODE:
                if current_node.vars["num_reencodes"] > 0 or current_node.content[0] is None or current_node.content[1] is None:
                    return False
                if self.fnc == vm_reencode:
                    if current_node.content[1] != ["NVC"]:
                        return False
            if self.optype == OpType.ENTAIL:
                if current_node.vars["num_entails"] > 0 or current_node.content[1] is None:
                    return False
            if self.optype == OpType.FALSIFY:
                if current_node.vars["num_falsifies"] > 1 or current_node.content[0] is None or current_node.content[1] is None:
                    return False
                if self.fnc == mm_falsify:
#                    if self.args[0] == -1 and (not current_node.vars["mm_falsify_0"] or current_node.vars["mm_falsify_n1"]): # first conclusion not yet falsified or last (second, hopefully) conclusion already falsified
#                        return False
                    if self.args[0] >= len(current_node.content[1]):
                        return False
                    if current_node.vars["mm_last_falsify"] is None:
                        if self.args[0] > 0:
                            return False
                    elif current_node.vars["mm_last_falsify"] != self.args[0] - 1:
                        return False
            return True

#    inits = 0
    def __init__(self, name="\"Tree\" Model"):
#        print("inits", self.inits)
#        self.inits += 1

        super(AbstractModel, self).__init__(name, ["syllogistic"], ["single-choice"])
        self.state = State.Item

        self.extra_weight = 5

        self.operations = [
                           # Preencode
                           self.Operation(OpType.PREENCODE, ic_reverse_premise, (False, False)),
                           self.Operation(OpType.PREENCODE, ic_reverse_premise, (False, True)),
                           self.Operation(OpType.PREENCODE, ic_reverse_premise, (True, False)),
                           self.Operation(OpType.PREENCODE, ic_reverse_premise, (True, True)),

                           # Heuristic
                           self.Operation(OpType.HEURISTIC, heuristic_atmosphere, ()),
                           self.Operation(OpType.HEURISTIC, heuristic_matching, ()),
                           self.Operation(OpType.HEURISTIC, phm_heuristic_min, ()),

                           # Encode
 #                          self.Operation(OpType.ENCODE, mm_vmstyle_encode_premises, ()),
#                           self.Operation(OpType.ENCODE, vm_encode, ()),
                           self.Operation(OpType.ENCODE, vm_encode_premises, ()),

            # Conclude
                           self.Operation(OpType.CONCLUDE, mm_conclude, ()),
                           self.Operation(OpType.CONCLUDE, vm_conclude, ()),

                           # Reencode
                           self.Operation(OpType.REENCODE, vm_reencode, ()),
                           self.Operation(OpType.REENCODE, psycop_check, ()),

                           # Falsify
                           self.Operation(OpType.FALSIFY, mm_falsify, [0]),
                           self.Operation(OpType.FALSIFY, mm_falsify, [1]),
                           self.Operation(OpType.FALSIFY, mm_falsify, [2]),
                           self.Operation(OpType.FALSIFY, mm_falsify, [3]),

                           # Entail
                           self.Operation(OpType.ENTAIL, phm_p_entailment, ()),

                           # Respond
###                           self.Operation(OpType.RESPOND, response_by_order, ["ac"]),
###                           self.Operation(OpType.RESPOND, response_by_order, ["ca"]),
                           self.Operation(OpType.RESPOND, lambda mr: random.choice(mr[1]), ()),
###                           self.Operation(OpType.RESPOND, resp_prefer_stronger, ()),
                           self.Operation(OpType.RESPOND, resp_prefer, ("stronger", "ac")),
                           self.Operation(OpType.RESPOND, resp_prefer, ("weaker", "ac")),
                           self.Operation(OpType.RESPOND, resp_prefer, ("stronger", "ca")),
                           self.Operation(OpType.RESPOND, resp_prefer, ("weaker", "ca")),

                           self.Operation(OpType.RESPOND, lambda mr: mr[1][0], ()),
        ]

#        x = [self.Operation(OpType.ENCODE, vm_encode, ("a", "a", "a", "a", "a", "a"))]
#        y = [self.Operation(OpType.REENCODE, vm_reencode, ("b", "b", "b", "c"))]

#        x = [self.Operation(OpType.ENCODE, vm_encode, (p1, p2, p3, p4, p5, p6))
#             for p1 in vm.model.param_grid["p1"] for p2 in vm.model.param_grid["p2"] for p3 in vm.model.param_grid["p3"]
#             for p4 in vm.model.param_grid["p4"] for p5 in vm.model.param_grid["p5"] for p6 in vm.model.param_grid["p6"]]

#        y = [self.Operation(OpType.REENCODE, vm_reencode, (p10, p11, p12, p13))
#             for p10 in vm.model.param_grid["p10"] for p11 in vm.model.param_grid["p11"]
#             for p12 in vm.model.param_grid["p12"] for p13 in vm.model.param_grid["p13"]]

        #        x = [self.Operation(OpType.HEURISTIC, heuristic_gm, [par_to, co]) for par_to in gm.param_grid["total_order"] for co in ["ac", "ca"]]
#        self.operations.extend(x)
#        self.operations.extend(y)

        self.best_plan = ""

        # cache tree for every syllogism
        self.trees = self.span_all_trees()

        # cache path names from Item to Response
        self.targets = self.fill_plans()

#    nsp = 0
    def start_participant(self, **kwargs):
#        print("START_PARTICIPANT")
#        print("nsp", self.nsp)
#        self.nsp += 1
        if mm.pre_train_params is not None:
            # pre_training has taken place
            mm.model.set_params(mm.pre_train_params)
        if vm.pre_train_params is not None:
            # pre_training has taken place
            vm.model.set_params(vm.pre_train_params)
        self.all_plans = self.pre_train_all_plans
        self.best_plan = self.pre_train_best_plan

    def applicable(self, operation, current_node):
        if operation.pre_state == current_node.state and operation.eval_preconditions(current_node):
            return True
        return False

    def update_variables(self, operation, current_node, post_content):
        new_vars = copy.deepcopy(current_node.vars)
        if operation.optype == OpType.PREENCODE:
            new_vars["num_preencodes"] += 1
        elif operation.optype == OpType.CONCLUDE:
            new_vars["num_concludes"] += 1
        elif operation.optype == OpType.REENCODE:
            new_vars["num_reencodes"] += 1
        elif operation.optype == OpType.ENTAIL:
            new_vars["num_entails"] += 1
        elif operation.optype == OpType.FALSIFY:
            new_vars["num_falsifies"] += 1
            if operation.fnc == mm_falsify:
                new_vars["mm_last_falsify"] = operation.args[0]
#                if operation.args[0] == -1:
#                    new_vars["mm_falsify_n1"] = True
#                elif operation.args[0] == 0:
#                    new_vars["mm_falsify_0"] = True
        if operation.post_state == State.MR and post_content[1] is not None:
            new_vars["n_conclusions"] = len(post_content[1])
        return new_vars

    def new_child(self, operation, current_node):
        """ Return a new child node by applying a function with arguments to the content of
        current_node.
        """

        # Variables for operation preconditions
        if operation.optype == OpType.PREENCODE:
            post_content = operation.fnc(current_node.content, *operation.args)

        elif operation.optype == OpType.HEURISTIC:
            post_content = (None, operation.fnc(current_node.content, *operation.args))
        elif operation.optype == OpType.ENCODE:
            post_content = (operation.fnc(current_node.content, *operation.args), None)
        elif operation.optype == OpType.CONCLUDE or operation.optype == OpType.REENCODE or operation.optype == OpType.ENTAIL:
            post_content = (current_node.content[0], operation.fnc(current_node.content, *operation.args))
        elif operation.optype == OpType.FALSIFY:
            post_content = (operation.fnc(current_node.content, *operation.args), current_node.content[1])
        elif operation.optype == OpType.RESPOND:
            post_content = operation.fnc(current_node.content, *operation.args)


        # new child node
        return AnyNode(name="{0}/{1}{2}".format(current_node.name, operation.fnc.__name__, operation.args),
                       content=post_content,
                       state=operation.post_state,
                       vars=self.update_variables(operation, current_node, post_content),
                       parent=current_node)

    def follow_plan(self, root, plan):
        plan_segments = plan.split("/")
        current_node = root
        for i, strop in enumerate(plan_segments):
            part_plan = "/".join(plan_segments[:i+1])
            current_node = search.findall_by_attr(current_node, value=part_plan, name="name", maxlevel=2)[0]
        return current_node

#        return search.findall_by_attr(root, value=plan, name="name")[0]

#    npt = 0
    def pre_train(self, dataset):
#        print("PRE_TRAIN")
#        print("npt", self.npt)
#        self.npt += 1

        mm.pre_train(dataset)
        vm.pre_train(dataset)
        psycop.pre_train(dataset)

        # Store number of subjects for later as weight for adapt
        self.n_pre_subjects = len(dataset)

        self.all_plans = {}
        n = 0

        for generic_task in sylutil.GENERIC_TASKS:
            generic_item = ccobra.Item(0, "syllogistic", generic_task, "single-choice", generic_choices)
            root = self.span_tree(generic_item)
            for leaf in root.leaves:
                self.all_plans[leaf.name] = 0
#            for node in PreOrderIter(root):
#                if node.content in ["Aac", "Aca", "Iac", "Ica", "Eac", "Eca", "Oac", "Oca", "NVC"]:
#                    self.all_plans[node.name] = 0

            #        for syl in ccobra.syllogistic.SYLLOGISMS:

#            generic_item = ccobra.Item(0, "syllogistic", "Some not;y;x/Some;z;y", "single-choice", generic_choices)

#            if syl == "AI2":
#                print(syl)
#                for pre, _, node in RenderTree(self.span_tree_by_syllogism(syl)):
#                    if "root/mm_vmstyle_encode()/mm_conclude()/vm_reencode('a', 'a', 'a', 'a')/mm_falsify[0]" in node.name:
#                        print("%s%s\t%s%s" % (pre, node.name, "CONTENT: ", node.content))

#            root = self.span_tree_by_syllogism(syl)
#            for node in PreOrderIter(root):
#                if node.content in ["Aac", "Aca", "Iac", "Ica", "Eac", "Eca", "Oac", "Oca", "NVC"]:
#                    self.all_plans[node.name] = 0
 #       print("hello", "root/mm_vmstyle_encode()/mm_conclude()/vm_reencode('a', 'a', 'a', 'a')/mm_falsify[0]/phm_p_entailment()/mm_falsify[1]/resp_prefer_stronger()" in self.all_plans)

        for participant in dataset:
            for task in participant:
                n += 1
                item = task["item"]

                target = ccobra.syllogistic.encode_response(task["response"], task)
                target_nodes = self.find_plans(item, target)
                for tn in target_nodes:
                    plan = tn.name
#                    if plan not in self.all_plans:
#                        self.all_plans[plan] = 0
#                    val = self.all_plans[plan].get
                    self.all_plans[plan] += 1

        self.best_plan = max(self.all_plans, key=self.all_plans.get)
#        for i in range(10):
#            print("#" + str(i) + "")
        self.pre_train_best_plan = copy.deepcopy(self.best_plan)
        self.pre_train_all_plans = copy.deepcopy(self.all_plans)
        print("Best plan:", self.best_plan, "\napplied # of times:", self.all_plans[self.best_plan], "from total:", n, "ratio:", self.all_plans[self.best_plan] / float(n))

    def plan_vote(self, item):
#        root = self.span_tree(item)
        conclusions = ccobra.syllogistic.RESPONSES
        scores = []

        for concl in conclusions:
            target_nodes = self.find_plans(item, concl)
            scores.append(sum([self.all_plans[tn.name] for tn in target_nodes]))

#        for plan in self.all_plans:
#            prediction_node = self.follow_plan(root, plan)
#            score = self.all_plans[plan]
#            scores[conclusions.index(prediction_node.content)] += score
        n = sum(scores)
        scores = [s/n for s in scores]
        return conclusions[scores.index(max(scores))]
#        return np.random.choice(conclusions, p=scores)

#    nadapt = 0
    def adapt(self, item, target, **kwargs):
#        print("ADAPT")
#        self.nadapt += 1
#        print("nadapt", self.nadapt)
#        return 0
        mm.adapt(item, target)
        vm.adapt(item, target)
        psycop.adapt(item, target)
        for target_node in self.find_plans(item, ccobra.syllogistic.encode_response(target, item.task)):
            plan = target_node.name
#            try:
            self.all_plans[plan] += self.n_pre_subjects * self.extra_weight
#            except: # TODO: KeyException - aber wieso?
#                print("Is the plan in?", plan in self.all_plans)
#                print("target_node", target_node)

#                print(item)
#                print(ccobra.syllogistic.Syllogism(item).encoded_task)

#                for pre, _, node in RenderTree(self.span_tree(item)):
#                    if "root/mm_vmstyle_encode()/mm_conclude()/vm_reencode('a', 'a', 'a', 'a')/mm_falsify[0]" in node.name:
#                        print("%s%s\t%s%s" % (pre, node.name, "CONTENT: ", node.content))

#                self.all_plans[plan] = self.n_pre_subjects * self.extra_weight
#                raise
        self.best_plan = max(self.all_plans, key=self.all_plans.get)

    @lru_cache(maxsize=1)
    def fill_plans(self):
        targets = {}
        for generic_item in sylutil.GENERIC_ITEMS:
#            generic_item = ccobra.Item(0, "syllogistic", generic_task, "single-choice", generic_choices)
            syl = ccobra.syllogistic.Syllogism(generic_item).encoded_task
            root = self.span_tree(generic_item)
            for c in ccobra.syllogistic.RESPONSES:
                targets[(syl, c)] = search.findall_by_attr(root, c, name="content")
        return targets

    def find_plans(self, item, target):
#        root = self.span_tree(item)
        syl = ccobra.syllogistic.Syllogism(item).encoded_task
#        if ((syl, target)) not in self.targets:
#        self.targets[(syl, target)] = search.findall_by_attr(root, target, name="content")
        return self.targets[(syl, target)]

#    npred = 0
    def predict(self, item, **kwargs):
#        print("PREDICT")
#        print("npred", self.npred)
#        self.npred += 1
#        prediction_node = self.follow_plan(self.span_tree(item), self.best_plan)
        c = self.plan_vote(item)
#        return ccobra.syllogistic.decode_response(prediction_node.content, item.task)
        return ccobra.syllogistic.decode_response(c, item.task)

    def value_space(self, root, var):
        vals = set()
        for node in PreOrderIter(root):
            vals.add(node.vars[var])
        return vals

#    trees = {}
#    targets = {}

    def span(self, item):
        global current_syllogism
        current_syllogism = ccobra.syllogistic.encode_task(item.task)  # ccobra.syllogistic.Syllogism(item).encoded_task

        root = AnyNode(name="root", content=item, state=State.Item, vars={"num_preencodes": 0,
                                                                          "num_concludes": 0,
                                                                          "num_falsifies": 0,
                                                                          "num_entails": 0,
                                                                          "num_reencodes": 0,
                                                                          "mm_last_falsify": None,
                                                                          "n_conclusions": 0,
                                                                          })
        fringe = [root]

        while True:
            if len(fringe) == 0:
                break
            current_node = fringe.pop()
            for op in self.operations:
                if self.applicable(op, current_node):
                    fringe.append(self.new_child(op, current_node))

        #        for pre, _, node in RenderTree(root):
        #            if "vm_encode('b', 'b', 'a', 'c', 'a', 'c')" in node.name:
        #                print("%s%s" % (pre, node.content))
        #            print("%s%s\t%s" % (pre, node.name, node.content))

#        self.trees[current_syllogism] = root
        return root


    def span_all_trees(self):
        trees = {}
        for item in sylutil.GENERIC_ITEMS:
            syl = ccobra.syllogistic.encode_task(item.task)
            trees[syl] = self.span(item)
        return trees

    """
    def span_tree_by_syllogism(self, item):
#        if current_syllogism in self.trees:
#            return self.trees[current_syllogism]

        root = AnyNode(name="root", content=item, state=State.Item, vars={"num_preencodes": 0,
                                                                          "num_concludes": 0,
                                                                          "num_falsifies": 0,
                                                                          "num_entails": 0,
                                                                          "num_reencodes": 0,
                                                                          "mm_last_falsify": None,
                                                                          "n_conclusions": 0,
                                                                          })
        fringe = [root]

        while True:
            if len(fringe) == 0:
                break
            current_node = fringe.pop()
            for op in self.operations:
                if self.applicable(op, current_node):
                    fringe.append(self.new_child(op, current_node))

        #        for pre, _, node in RenderTree(root):
        #            if "vm_encode('b', 'b', 'a', 'c', 'a', 'c')" in node.name:
        #                print("%s%s" % (pre, node.content))
        #            print("%s%s\t%s" % (pre, node.name, node.content))

        self.trees[current_syllogism] = root
        return root

    """

    def span_tree(self, item):
#        global current_syllogism
        syl = ccobra.syllogistic.encode_task(item.task)#ccobra.syllogistic.Syllogism(item).encoded_task
#        return self.span_tree_by_syllogism(item)
        return self.trees[syl]



#generic_premises = [(encode_proposition(i.task[0], i), encode_proposition(i.task[1], i)) for i in sylutil.GENERIC_ITEMS]

#for i, ps in enumerate(generic_premises):
#    print(i, ccobra.syllogistic.encode_task(sylutil.GENERIC_ITEMS[i].task))

#    modl = mm_vmstyle_encode_premises(ps)


#print("mm_orig", mm_encode(item))
#print("mm_new", mm_vmsytle_encode(item))
#print("vm", vm_encode(item))
#print("reverted", vm_to_mm(mm_vmsytle_encode(item)[0]))

#m = AbstractModel()
#root = m.span_tree(sylutil.GENERIC_ITEMS[0])

#print("print TREE")
#for pre, _, node in RenderTree(root):
#    if "vm_encode('b', 'b', 'a', 'c', 'a', 'c')" in node.name:
 #       print("%s%s" % (pre, node.content))
#    print("%s%s\t%s" % (pre, node.name, node.content))


#m.pre_train([])
#m.all_plans = []
#m.plan_vote(item)
#plan = "root/vm_encode('b', 'b', 'a', 'c', 'a', 'c')/vm_conclude()/vm_reencode('b', 'b', 'b', 'c')/phm_p_entailment()/mm_falsify[0]/resp_prefer_stronger()"
#m.follow_plan(root, plan)
