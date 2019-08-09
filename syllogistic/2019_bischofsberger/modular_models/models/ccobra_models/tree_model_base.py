# -*- coding: utf-8 -*-

import copy
import os
import random
import re
import sys
from enum import Enum

import ccobra
import numpy as np
from anytree import AnyNode, search, RenderTree

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.util import sylutil
from modular_models.models.basic_models import GeneralizedMatching, PHM, VerbalModels, MentalModels, PSYCOP, MReasoner, Atmosphere, Matching, IllicitConversion
from modular_models.models.ccobra_models.interface import CCobraWrapper

# State machine from which to build trees/plans.
OpType = Enum("OpType", "PREENCODE, HEURISTIC, ENCODE, ENCODE_PREMISES, CONCLUDE, REENCODE, RESPOND, FALSIFY, ENTAIL, NONE, XYZ")
State = Enum("State", "Item, Premises, MR, Response")
OPTYPE_TO_STATES = {OpType.PREENCODE: (State.Item, State.Premises),
                    OpType.HEURISTIC: (State.Item, State.MR),
                    OpType.ENCODE: (State.Item, State.MR),
                    OpType.ENCODE_PREMISES: (State.Premises, State.MR),
                    OpType.RESPOND: (State.MR, State.Response),
                    OpType.CONCLUDE: (State.MR, State.MR),
                    OpType.REENCODE: (State.MR, State.MR),
                    OpType.ENTAIL: (State.MR, State.MR),
                    OpType.FALSIFY: (State.MR, State.MR),
                    OpType.NONE: (State.MR, State.MR),
                    OpType.XYZ: (State.Item, State.MR),
                    }


class Operation:
    """ Class for operation objects """
    def __init__(self, optype, fnc, args):
        # Type of operations
        self.optype = optype

        # Function
        self.fnc = fnc

        # Function arguments
        self.args = args

        # State transition
        self.pre_state, self.post_state = OPTYPE_TO_STATES[optype]

    def __str__(self):
        return "Operation(" + self.optype.name + ", " + self.fnc.__name__ + str(self.args) + ")"

    def eval_preconditions(self, current_node):
        """ Evaluate preconditions on current_node. Contains conditions of particular operations. """

        if "flex" in self.fnc.__name__:
            if len(current_node.content) != 2 or current_node.content[0] is not None:
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

        if self.fnc.__name__ == "psycop_check":
            if current_node.vars["num_psycop_checks"] > 0 or current_node.content[1] is None:
                return False

        if self.fnc.__name__ == "phm_p_entailment":
            if current_node.vars["num_entails"] > 0 or current_node.content[1] is None:
                return False
        if self.fnc.__name__ == "phm_reply":
            if current_node.vars["num_phm_replies"] > 0 or current_node.content[1] is None:
                return False

        if self.fnc.__name__ == "mreasoner_falsify" or self.fnc.__name__ == "mreasoner_falsify_with_model":
            if self.fnc.__name__ == "mreasoner_falsify_with_model" and current_node.content[0] is None:
                return False
            if current_node.content[1] is None:# or self.args[0] >= len(current_node.content[1]):
                return False
            if current_node.vars["mreasoner_last_falsify"] is None:
                if self.args[0] > 0:
                    return False
            elif current_node.vars["mreasoner_last_falsify"] != self.args[0] - 1:
                return False
        if self.fnc.__name__ == "weaken_conclusion":
            if current_node.content[1] is None or current_node.vars["num_weakens"] > 0:# or self.args[0] >= len(current_node.content[1]) or current_node.content[1][self.args[0]][0] not in ["A", "E"]:
                return False

        if self.optype == OpType.FALSIFY:
            if current_node.vars["num_falsifies"] > 1 or current_node.content[0] is None or current_node.content[1] is None:
                return False
            if self.fnc.__name__ == "mm_falsify":
                if current_node.vars["mm_last_falsify"] is None:
                    if self.args[0] > 0:
                        return False
                elif current_node.vars["mm_last_falsify"] != self.args[0] - 1:
                    return False
        return True


class Operations:
    """ Encapsulates individual models and their operations """

    def __init__(self):
        # Individual models
        self.psycop = CCobraWrapper(model=PSYCOP)
        self.ic = IllicitConversion()
        self.mm = CCobraWrapper(model=MentalModels)
        self.vm = CCobraWrapper(model=VerbalModels)
        self.mreasoner = CCobraWrapper(model=MReasoner)
        self.gm = GeneralizedMatching()
        self.phm = CCobraWrapper(model=PHM)

        # Store mapping operation -> model
        self.op2model = {"heuristic_atmosphere": "Atmosphere",
                         "psycop_predict": "PSYCOP",
                         "ic_predict": "IC",
                         "heuristic_matching": "Matching",
                         "ic_reverse_premise": "IC",
                         "phm_heuristic_min": "PHM",
                         "phm_p_entailment": "PHM",
                         "phm_reply": "PHM",
                         "psycop_check": "PSYCOP",
                         "mm_encode_premises_vmstyle": "MM",
                         "mm_encode_vmstyle": "MM",
                         "mm_conclude": "MM",
                         "mm_falsify": "MM",
                         "mm_encode_flex": "MM",
                         "vm_conclude": "VM",
                         "vm_encode": "VM",
                         "vm_encode_flex": "VM",
                         "vm_encode_premises": "VM",
                         "vm_reencode": "VM",
                         "mreasoner_encode_flex": "MReasoner",
                         "mreasoner_falsify": "MReasoner",
                         "mreasoner_heuristic": "MReasoner",
                         "mreasoner_encode_vmstyle": "MReasoner",
                         "mreasoner_encode": "MReasoner",
                         "mreasoner_falsify_with_model": "MReasoner",
                         "weaken_conclusion": "MReasoner",
                         }

        # Store current syllogism to acccess from within operations
        self.current_syllogism = ""

    ### Special operations only for pureness test ###
    def psycop_predict(self, item):
        return self.psycop.model.predict(ccobra.syllogistic.encode_task(item.task))

    def ic_predict(self, item):
        return self.ic.predict(ccobra.syllogistic.encode_task(item.task))

    ### 1. Atmosphere ###
    def heuristic_atmosphere(self, item):
        """
        >>> m = Atmosphere()
        >>> ops = Operations()
        >>> for item in sylutil.GENERIC_ITEMS:
        ...     syl = ccobra.syllogistic.encode_task(item.task)
        ...     print(syl, ops.heuristic_atmosphere(item) == m.heuristic_atmosphere(syl))
        AA1 True
        AA2 True
        AA3 True
        AA4 True
        AI1 True
        AI2 True
        AI3 True
        AI4 True
        AE1 True
        AE2 True
        AE3 True
        AE4 True
        AO1 True
        AO2 True
        AO3 True
        AO4 True
        IA1 True
        IA2 True
        IA3 True
        IA4 True
        II1 True
        II2 True
        II3 True
        II4 True
        IE1 True
        IE2 True
        IE3 True
        IE4 True
        IO1 True
        IO2 True
        IO3 True
        IO4 True
        EA1 True
        EA2 True
        EA3 True
        EA4 True
        EI1 True
        EI2 True
        EI3 True
        EI4 True
        EE1 True
        EE2 True
        EE3 True
        EE4 True
        EO1 True
        EO2 True
        EO3 True
        EO4 True
        OA1 True
        OA2 True
        OA3 True
        OA4 True
        OI1 True
        OI2 True
        OI3 True
        OI4 True
        OE1 True
        OE2 True
        OE3 True
        OE4 True
        OO1 True
        OO2 True
        OO3 True
        OO4 True
        """
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

    ### 2. Matching ###
    def heuristic_matching(self, item):
        """
        >>> m = Matching()
        >>> ops = Operations()
        >>> for item in sylutil.GENERIC_ITEMS:
        ...     syl = ccobra.syllogistic.encode_task(item.task)
        ...     print(syl, ops.heuristic_matching(item) == m.heuristic_matching(syl))
        AA1 True
        AA2 True
        AA3 True
        AA4 True
        AI1 True
        AI2 True
        AI3 True
        AI4 True
        AE1 True
        AE2 True
        AE3 True
        AE4 True
        AO1 True
        AO2 True
        AO3 True
        AO4 True
        IA1 True
        IA2 True
        IA3 True
        IA4 True
        II1 True
        II2 True
        II3 True
        II4 True
        IE1 True
        IE2 True
        IE3 True
        IE4 True
        IO1 True
        IO2 True
        IO3 True
        IO4 True
        EA1 True
        EA2 True
        EA3 True
        EA4 True
        EI1 True
        EI2 True
        EI3 True
        EI4 True
        EE1 True
        EE2 True
        EE3 True
        EE4 True
        EO1 True
        EO2 True
        EO3 True
        EO4 True
        OA1 True
        OA2 True
        OA3 True
        OA4 True
        OI1 True
        OI2 True
        OI3 True
        OI4 True
        OE1 True
        OE2 True
        OE3 True
        OE4 True
        OO1 True
        OO2 True
        OO3 True
        OO4 True
        """
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

    ### 3. Illicit Conversion ###
    def ic_reverse_premise(self, item, first=True, second=True):
        """
        >>> ops = Operations()
        >>> aa1_item = sylutil.GENERIC_ITEMS[0]
        >>> ops.ic_reverse_premise(aa1_item, first=True, second=True)
        ['Aab', 'Abc', 'Aba', 'Acb']
        >>> ops.ic_reverse_premise(aa1_item, first=False, second=False)
        ['Aab', 'Abc']
        >>> ops.ic_reverse_premise(aa1_item, first=True, second=False)
        ['Aab', 'Abc', 'Aba']
        """
        new_premises = []
        p1, p2 = (sylutil.encode_proposition(x, item) for x in item.task)
        new_premises.extend([p1, p2])
        if first:
            new_premises.append(p1[0] + p1[2] + p1[1])
        if second:
            new_premises.append(p2[0] + p2[2] + p2[1])
        return new_premises

    ### 4. PHM ###
    def phm_heuristic_min(self, item):
        syl = ccobra.syllogistic.Syllogism(item).encoded_task
        concl_mood = self.phm.model.f_min_heuristic[syl[:2]]
        return self.phm.model.attachment(concl_mood, syl)

    def phm_p_entailment(self, mr):
        """
        >>> ops = Operations()
        >>> ops.phm_p_entailment((None, ["Aac", "Iac", "Oac", "Oca"]))
        ['Iac', 'Oac', 'Ica']
        """
        conclusions = mr[1]
        new_conclusions = []
        for c in conclusions:
            if c == "NVC":
                new_conclusions.append("NVC")
            else:
                new_conclusions.append(self.phm.model.f_p_entailment[c[0]] + c[1:])
        new_conclusions = sylutil.uniquify_keep_order(new_conclusions)
        if len(new_conclusions) == 2 and new_conclusions[0][0] == new_conclusions[1][0]:
            new_conclusions = self.phm.model.attachment(new_conclusions[0][0], self.current_syllogism)
        return new_conclusions

    def phm_reply(self, mr):
        if random.random() >= self.phm.model.params["confidence" + self.phm.model.max_premise(self.current_syllogism)[0]]:
            return ["NVC"]
        else:
            return mr[1]

    ### 5. PSYCOP ###
    def psycop_check(self, mr):
        conclusions = mr[1]
        return [c for c in conclusions if c in self.psycop.model.conclusions_positive_checks(self.current_syllogism)]

    ### 6. Mental Models ###
    def mm_encode_vmstyle(self, item):
        """
        >>> mm = MentalModels()
        >>> ops = Operations()
        >>> for syl in ccobra.syllogistic.SYLLOGISMS:
        ...     item = sylutil.syllogism_to_item(syl)
        ...     model, exh = ops.mm_encode_vmstyle(item)
        ...     model = sylutil.vm_to_mm(model)
        ...     exh = [e.name for e in exh]
        ...     print(syl, (model, exh) == mm.encode(syl))
        AA1 True
        AA2 True
        AA3 True
        AA4 True
        AI1 True
        AI2 True
        AI3 True
        AI4 True
        AE1 True
        AE2 True
        AE3 True
        AE4 True
        AO1 True
        AO2 True
        AO3 True
        AO4 True
        IA1 True
        IA2 True
        IA3 True
        IA4 True
        II1 True
        II2 True
        II3 True
        II4 True
        IE1 True
        IE2 True
        IE3 True
        IE4 True
        IO1 True
        IO2 True
        IO3 True
        IO4 True
        EA1 True
        EA2 True
        EA3 True
        EA4 True
        EI1 True
        EI2 True
        EI3 True
        EI4 True
        EE1 True
        EE2 True
        EE3 True
        EE4 True
        EO1 True
        EO2 True
        EO3 True
        EO4 True
        OA1 True
        OA2 True
        OA3 True
        OA4 True
        OI1 True
        OI2 True
        OI3 True
        OI4 True
        OE1 True
        OE2 True
        OE3 True
        OE4 True
        OO1 True
        OO2 True
        OO3 True
        OO4 True
        """

        syllogism = ccobra.syllogistic.Syllogism(item).encoded_task
        mm = []
        exhausted = []
        to = sylutil.term_order(syllogism[2])
        subj_0, pred_0 = to[0]
        subj_1, pred_1 = to[1]

        t0 = sylutil.get_time()
        t1 = sylutil.get_time()

        s0 = VerbalModels.Prop(name=subj_0, neg=False, identifying=True, t=t0)
        p0 = VerbalModels.Prop(name=pred_0, neg=False, identifying=False, t=t0)
        non_p0 = VerbalModels.Prop(name=pred_0, neg=True, identifying=False, t=t0)
        s1 = VerbalModels.Prop(name=subj_1, neg=False, identifying=True, t=t0)
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
        b_is_subject = True if syllogism[2] in ["1", "4"] else False
        c_ident = True if not b_is_subject else False

        prop_c = VerbalModels.Prop(name="c", neg=False, identifying=c_ident, t=t1)
        prop_non_c = VerbalModels.Prop(name="c", neg=True, identifying=False, t=t1)
        ind_c = VerbalModels.Individual(props=[prop_c], t=t1)
        prop_non_b = VerbalModels.Prop(name="b", neg=True, identifying=False, t=t1)

        if syllogism[1] == "A":
            for i in i_b:
                mm[i].props.append(prop_c)
                mm[i].t = t1
                if b_is_subject:
                    for prop in mm[i].props:
                        if prop.name == "b":
                            prop.identifying = True
            if s1 not in exhausted:
                exhausted.append(s1)
        if syllogism[1] == "I":
            mm[i_b[0]].props.append(prop_c)
            mm[i_b[0]].t = t1
            if b_is_subject:
                for prop in mm[i_b[0]].props:
                    if prop.name == "b":
                        prop.identifying = True
            mm.append(ind_c)
        if syllogism[1] == "E" or syllogism[1] == "O":
            if s1 == prop_b:
                for i in i_b:
                    mm[i].props.append(prop_non_c)
                    mm[i].t = t1
                    if b_is_subject:
                        for prop in mm[i].props:
                            if prop.name == "b":
                                prop.identifying = True
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

    def mm_encode_premises_vmstyle(self, premises):
        """
        >>> mm = MentalModels()
        >>> ops = Operations()
        >>> for syl in ccobra.syllogistic.SYLLOGISMS:
        ...     prems = sylutil.syllogism_to_premises(syl)
        ...     model, exh = ops.mm_encode_premises_vmstyle(prems)
        ...     model = sylutil.vm_to_mm(model)
        ...     exh = [e.name for e in exh]
        ...     print(syl, (model, exh) == mm.encode(syl))
        AA1 True
        AA2 True
        AA3 True
        AA4 True
        AI1 True
        AI2 True
        AI3 True
        AI4 True
        AE1 True
        AE2 True
        AE3 True
        AE4 True
        AO1 True
        AO2 True
        AO3 True
        AO4 True
        IA1 True
        IA2 True
        IA3 True
        IA4 True
        II1 True
        II2 True
        II3 True
        II4 True
        IE1 True
        IE2 True
        IE3 True
        IE4 True
        IO1 True
        IO2 True
        IO3 True
        IO4 True
        EA1 True
        EA2 True
        EA3 True
        EA4 True
        EI1 True
        EI2 True
        EI3 True
        EI4 True
        EE1 True
        EE2 True
        EE3 True
        EE4 True
        EO1 True
        EO2 True
        EO3 True
        EO4 True
        OA1 True
        OA2 True
        OA3 True
        OA4 True
        OI1 True
        OI2 True
        OI3 True
        OI4 True
        OE1 True
        OE2 True
        OE3 True
        OE4 True
        OO1 True
        OO2 True
        OO3 True
        OO4 True
        """
        premises = premises[:2]

    #    syllogism = ccobra.syllogistic.Syllogism(item).encoded_task
        mm = []
        exhausted = []
    #    to = sylutil.term_order(syllogism[2])
    #    subj_0, pred_0 = to[0]
        subj_0, pred_0 = premises[0][1], premises[0][2]
    #    subj_1, pred_1 = to[1]

        t0 = sylutil.get_time()
        t1 = sylutil.get_time()

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
            tn = sylutil.get_time()
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
            prop_non_b = VerbalModels.Prop(name="b", neg=True, identifying=False, t=tn)

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

    def mm_encode_flex(self, mr):
        item = sylutil.syllogism_to_item(self.current_syllogism)
        return self.mm_encode_vmstyle(item)

    def mm_conclude(self, mr, exclude_weaker=True):
        verbal_model, exhausted = mr[0]
        return sorted(list(set(self.mm.model.conclude(sylutil.vm_to_mm(verbal_model), exhausted, exclude_weaker))))

    def mm_falsify(self, mr, i=0):
        verbal_model, exhausted = mr[0]
        conclusions = mr[1]

        if i >= len(conclusions):
            return mr[0]

        new_mental_model = self.mm.model.falsify(sylutil.vm_to_mm(verbal_model), [e.name for e in exhausted], conclusions[i])
        new_verbal_model = sylutil.mm_to_vm(new_mental_model)

        return new_verbal_model, exhausted

    ### 7. Verbal Models ###
    def vm_encode(self, item, p1=None, p2=None, p3=None, p4=None, p5=None, p6=None):
        syllogism = ccobra.syllogistic.Syllogism(item).encoded_task
        params_before = copy.deepcopy(self.vm.model.get_params())

        exhausted = self.mm_encode_vmstyle(item)[1]

        if p1 is not None:
            self.vm.model.params["p1"] = p1
            self.vm.model.params["p2"] = p2
            self.vm.model.params["p3"] = p3
            self.vm.model.params["p4"] = p4
            self.vm.model.params["p5"] = p5
            self.vm.model.params["p6"] = p6
        new_model = self.vm.model.encode(syllogism)
        self.vm.model.set_params(params_before)

        return new_model, exhausted

    def vm_encode_premises(self, premises):
        verbal_model = []
        for premise in premises:
            verbal_model = self.vm.model.extend_vm(verbal_model, premise, reencoding=False)

        exhausted = self.mm_encode_premises_vmstyle(premises)[1]
        return verbal_model, exhausted

    def vm_encode_flex(self, mr):
        item = sylutil.syllogism_to_item(self.current_syllogism)
        return self.vm_encode(item)

    def vm_conclude(self, mr):
        verbal_model, exhausted = mr[0]
        return self.vm.model.conclude(verbal_model)

    def vm_reencode(self, mr, p10=None, p11=None, p12=None, p13=None):
        verbal_model, exhausted = mr[0]
        conclusions = mr[1]
        syllogism = self.current_syllogism

        if conclusions != ["NVC"]:
            return (verbal_model, exhausted), conclusions

        params_before = copy.deepcopy(self.vm.model.get_params())

        if p10 is not None:
            self.vm.model.params["p10"] = p10 #b
            self.vm.model.params["p11"] = p11 #b
            self.vm.model.params["p12"] = p12 #b
            self.vm.model.params["p13"] = p13 #c

        new_model, new_conclusions = self.vm.model.reencode(syllogism, verbal_model, conclusions)

        new_model = verbal_model if any([row.contains(VerbalModels.Prop(name="a", neg=False)) and row.contains(VerbalModels.Prop(name="a", neg=True))
                                         or row.contains(VerbalModels.Prop(name="b", neg=False)) and row.contains(VerbalModels.Prop(name="b", neg=True))
                                         or row.contains(VerbalModels.Prop(name="c", neg=False)) and row.contains(VerbalModels.Prop(name="c", neg=True))
                                         for row in new_model]) else new_model

        self.vm.model.set_params(params_before)
        return (verbal_model, exhausted), new_conclusions#(new_model, exhausted), new_conclusions

    ### 8. MReasoner ###
    def mreasoner_heuristic(self, item):
        syl = ccobra.syllogistic.Syllogism(item).encoded_task
        return self.mreasoner.model.heuristic(syl)

    def mreasoner_encode(self, item, size, deviation):
        exhausted = []
        syllogism = ccobra.syllogistic.encode_task(item.task)

        t0 = sylutil.get_time()
        t1 = sylutil.get_time()

        mm = []
        (subj0, pred0), (subj1, pred1) = sylutil.term_order(syllogism[2])

        s0 = VerbalModels.Prop(name=subj0, neg=False, identifying=False, t=t0)
        non_s0 = VerbalModels.Prop(name=subj0, neg=True, identifying=False, t=t0)
        p0 = VerbalModels.Prop(name=pred0, neg=False, identifying=False, t=t0)
        non_p0 = VerbalModels.Prop(name=pred0, neg=True, identifying=False, t=t0)
        s1 = VerbalModels.Prop(name=subj1, neg=False, identifying=False, t=t0)
        prop_b = VerbalModels.Prop(name="b", neg=False, identifying=False, t=t0)

        xy = VerbalModels.Individual(props=[s0, p0], t=t0)
        xnony = VerbalModels.Individual(props=[s0, non_p0], t=t0)
        nonxy = VerbalModels.Individual(props=[non_s0, p0], t=t0)
        nonxnony = VerbalModels.Individual(props=[non_s0, non_p0], t=t0)

        ### Encode first premise ###
        # individuals that must be present in the model of the 1st premise
        required_inds = {"A": [xy], "I": [xy], "E": [xnony, nonxy], "O": [xnony]}[syllogism[0]]

        appendix = []

        def draw_individual(mood, subj, pred, complete):
            s0 = VerbalModels.Prop(name=subj, neg=False, identifying=False, t=t0)
            non_s0 = VerbalModels.Prop(name=subj, neg=True, identifying=False, t=t0)
            p0 = VerbalModels.Prop(name=pred, neg=False, identifying=False, t=t0)
            non_p0 = VerbalModels.Prop(name=pred, neg=True, identifying=False, t=t0)
            s1 = VerbalModels.Prop(name=subj, neg=False, identifying=False, t=t0)

            x = VerbalModels.Individual(props=[s0], t=t0)
            y = VerbalModels.Individual(props=[p0], t=t0)
            xy = VerbalModels.Individual(props=[s0, p0], t=t0)
            xnony = VerbalModels.Individual(props=[s0, non_p0], t=t0)
            nonxy = VerbalModels.Individual(props=[non_s0, p0], t=t0)
            nonxnony = VerbalModels.Individual(props=[non_s0, non_p0], t=t0)

            if mood == "A":
                if complete:
                    # All A are B -> possible inds: [a b], [-a b], [-a -b], impossible ind: [a -b]
                    return random.choice([xy, nonxy, nonxnony])
                return xy
            elif mood == "I":
                if complete:
                    # Some A are B -> possible inds: [a b], [-a b], [a, -b], [-a -b]
                    return random.choice([xy, nonxy, xnony, nonxnony])
                return random.choice([xy, x])
            elif mood == "E":
                if complete:
                    # No A are B -> possible inds: [-a b], [a, -b], [-a -b], impossible ind: [a b]
                    return random.choice([nonxy, xnony, nonxnony])
                return random.choice([nonxy, xnony])
            elif mood == "O":
                if complete:
                    # Some A are not B -> possible inds: [a b], [-a b], [a, -b], [-a -b]
                    return random.choice([xy, nonxy, xnony, nonxnony])
                return random.choice([xy, xnony, y])

        # O needs an additional requirement to make sure "b" is present (otherwise e.g. [a -b] [-a -b] would be allowed)
        while not all([any([ind.props == i.props for i in appendix]) for ind in required_inds]) or \
                syllogism[0] == "O" and not any([ind.contains(prop_b) for ind in appendix]):
            appendix = []
            for i in range(size):
                yyy = draw_individual(syllogism[0], subj0, pred0, random.random() < deviation)
                appendix.append(yyy)
        mm.extend(appendix)

        ### Encode second premise ###
        prop_non_b = VerbalModels.Prop(name="b", neg=True, identifying=False, t=t1)
        i_b = [i for i, ind in enumerate(mm) if ind.contains(prop_b)]

        prop_c = VerbalModels.Prop(name="c", neg=False, identifying=False, t=t1)
        prop_non_c = VerbalModels.Prop(name="c", neg=True, identifying=False, t=t1)
        ind_c = VerbalModels.Individual(props=[prop_c], t=t1)

        if syllogism[1] == "A":
            for i in i_b:
                mm[i].props.append(prop_c)
            if s1 not in exhausted:
                exhausted.append(s1)
        elif syllogism[1] == "I":
            if subj1 == "b":
                for n, i in enumerate(i_b):
                    mm[i].props.append(prop_c)
                    if n == 1:
                        break
            elif subj1 == "c":
                mm[i_b[0]].props.append(prop_c)
                mm.append(ind_c)
        elif syllogism[1] == "E":
            if subj1 == "b":
                for i in i_b:
                    mm[i].props.append(prop_non_c)
                mm.append(ind_c)
            elif subj1 == "c":
                mm.extend([VerbalModels.Individual(props=[prop_non_b, prop_c], t=t1),
                           VerbalModels.Individual(props=[prop_non_b, prop_c], t=t1),
                           VerbalModels.Individual(props=[prop_non_b, prop_c], t=t1),
                           VerbalModels.Individual(props=[prop_non_b, prop_c], t=t1)])
            if prop_b not in exhausted:
                exhausted.append(prop_b)
            if prop_c not in exhausted:
                exhausted.append(prop_c)
        elif syllogism[1] == "O":
            if subj1 == "b":
                for n, i in enumerate(i_b):
                    mm[i].props.append(prop_non_c)
                    # augment max 2 individuals
                    if n == 1:
                        break
                mm.append(ind_c)
            elif subj1 == "c":
                mm.extend([VerbalModels.Individual(props=[prop_non_b, prop_c], t=t1),
                           VerbalModels.Individual(props=[prop_c], t=t1)])

        return mm, sorted(exhausted, key=lambda p: p.name)

    def mreasoner_encode_flex(self, mr, size, epsilon):
        item = sylutil.syllogism_to_item(self.current_syllogism)
        return self.mreasoner_encode(item, size, epsilon)

    def mreasoner_falsify(self, mr, i=0):
        conclusions = copy.deepcopy(mr[1])
        if i >= len(conclusions):
            return conclusions
        if conclusions[i] != "NVC":
            conclusions[i] = self.mreasoner.model.verify_conclusion(conclusions[i], self.current_syllogism)
        return sylutil.uniquify_keep_order(conclusions)

    def mreasoner_falsify_with_model(self, mr, weaken=False, i=0):
        conclusions = copy.deepcopy(mr[1])
        if i >= len(conclusions):
            return conclusions
        if self.mreasoner.model.check_if_holds(sylutil.vm_to_mm(mr[0][0]), conclusions[i]):
            if weaken and conclusions[i][0] in ["A", "E"]:
                conclusions[i] = {"A": "I", "E": "O"}[conclusions[i][0]] + conclusions[i][1:]
            else:
                conclusions[i] = "NVC"
        return sylutil.uniquify_keep_order(conclusions)

    def weaken_conclusion(self, mr):
        conclusions = copy.deepcopy(mr[1])
        new_conclusions = []
        for c in conclusions:
            if c[0] in ["A", "E"]:
                nc = {"A": "I", "E": "O"}[c[0]] + c[1:]
            else:
                nc = c
            new_conclusions.append(nc)
        return sylutil.uniquify_keep_order(new_conclusions)

    ### 9. Response selection ###
    def resp_prefer(self, mr, stronger_weaker=None, ac_ca=None, mood_first=None, posneg=None):
        """
        >>> ops = Operations()
        >>> ops.resp_prefer((None, ["Aac", "Aca", "Iac", "Ica", "NVC"]), "stronger", "ac")
        'Aac'
        >>> ops.resp_prefer((None, ["Aac", "Aca", "Iac", "Ica", "NVC"]), "weaker", "ac")
        'Iac'
        >>> ops.resp_prefer((None, ["Aac", "Aca", "Iac", "Ica", "NVC"]), "stronger", "ca")
        'Aca'
        >>> ops.resp_prefer((None, ["Aac", "Aca", "Iac", "Ica", "NVC"]), "weaker", "ca")
        'Ica'
        >>> ops.resp_prefer((None, ["Aac", "Aca"]), "weaker", "ca")
        'Aca'

        >>> ops.resp_prefer((None, ["NVC"]), "weaker", "ca")
        'NVC'
        >>> ops.resp_prefer((None, ["NVC", "NVC", "NVC", "NVC", "NVC", "NVC", "NVC"]), "weaker", "ca")
        'NVC'
        >>> ops.resp_prefer((None, ["NVC", "NVC", "NVC", "NVC", "NVC", "NVC", "Aac"]), "weaker", "ca")
        'Aac'

        >>> ops.resp_prefer((None, ["Aac", "Ica"]), "stronger", "ca", True, "pos")
        'Aac'
        >>> ops.resp_prefer((None, ["Aac", "Ica"]), "weaker", "ca", False, "pos")
        'Ica'
        >>> ops.resp_prefer((None, ['Oac', 'Eca', 'Iac', 'Ica']), "stronger", "ca", True, "pos")
        'Ica'
        >>> ops.resp_prefer((None, ['Oac', 'Eca', 'Iac', 'Ica']), "weaker", "ac", True, "neg")
        'Oac'
        >>> ops.resp_prefer((None, ['Iac', 'Ica', 'Oac', 'Oca']), "weaker", "ac", True, "pos")
        'Iac'
        """

        conclusions = [c for c in mr[1] if c != "NVC"]
        if len(conclusions) == 0:
            return "NVC"

        prefered_conclusions = []
        for c in conclusions:
            if stronger_weaker == "stronger":
                if c[0] in ["I", "O"] and ({"I": "A", "O": "E"}[c[0]] + c[1:] in conclusions or {"I": "A", "O": "E"}[c[0]] + c[2] + c[1] in conclusions):
                    continue
            elif stronger_weaker == "weaker":
                if c[0] in ["A", "E"] and ({"A": "I", "E": "O"}[c[0]] + c[1:] in conclusions or {"A": "I", "E": "O"}[c[0]] + c[2] + c[1] in conclusions):
                    continue
            if ac_ca is not None:
                if c[1:] != ac_ca and (c[0] + ac_ca[0] + ac_ca[1]) in conclusions:
                    continue
            prefered_conclusions.append(c)

        if len(prefered_conclusions) > 1:
            if posneg == "pos":
                prefered_conclusions = [c for c in prefered_conclusions if c[0] in ["A", "I"]]
            elif posneg == "neg":
                prefered_conclusions = [c for c in prefered_conclusions if c[0] in ["E", "O"]]

        if len(prefered_conclusions) > 1 or len(prefered_conclusions) == 0:
            raise Exception

        return prefered_conclusions[0]

    def resp_random(self, mr):
        return random.choice(mr[1])


# All implemented basic operations
BASIC_OPERATIONS = [
    Operation(OpType.NONE, Operations.vm_encode_flex, ()),
    Operation(OpType.NONE, Operations.mm_encode_flex, ()),
    Operation(OpType.NONE, Operations.mreasoner_encode_flex, (2, 0.0)),
    Operation(OpType.NONE, Operations.mreasoner_encode_flex, (3, 0.0)),

    # Preencode
    Operation(OpType.PREENCODE, Operations.ic_reverse_premise, (False, False)),
    Operation(OpType.PREENCODE, Operations.ic_reverse_premise, (False, True)),
    Operation(OpType.PREENCODE, Operations.ic_reverse_premise, (True, False)),
    Operation(OpType.PREENCODE, Operations.ic_reverse_premise, (True, True)),

    # Heuristic
    Operation(OpType.HEURISTIC, Operations.heuristic_atmosphere, ()),
    Operation(OpType.HEURISTIC, Operations.heuristic_matching, ()),
    Operation(OpType.HEURISTIC, Operations.phm_heuristic_min, ()),
    Operation(OpType.HEURISTIC, Operations.mreasoner_heuristic, ()),

    # Encode
    Operation(OpType.ENCODE_PREMISES, Operations.mm_encode_premises_vmstyle, ()),
    Operation(OpType.ENCODE, Operations.mreasoner_encode, (2, 0.0)),
    Operation(OpType.ENCODE, Operations.mreasoner_encode, (3, 0.0)),
    Operation(OpType.ENCODE, Operations.mm_encode_vmstyle, ()),
    Operation(OpType.ENCODE, Operations.vm_encode, ()),
    Operation(OpType.ENCODE_PREMISES, Operations.vm_encode_premises, ()),

    # Conclude
    Operation(OpType.CONCLUDE, Operations.mm_conclude, ()),
    Operation(OpType.CONCLUDE, Operations.vm_conclude, ()),

    # Reencode
    Operation(OpType.REENCODE, Operations.vm_reencode, ()),
    Operation(OpType.ENTAIL, Operations.psycop_check, ()),

    # Falsify
    Operation(OpType.FALSIFY, Operations.mm_falsify, [0]),
    Operation(OpType.FALSIFY, Operations.mm_falsify, [1]),
    Operation(OpType.FALSIFY, Operations.mm_falsify, [2]),
    Operation(OpType.FALSIFY, Operations.mm_falsify, [3]),
    Operation(OpType.ENTAIL, Operations.mreasoner_falsify, [0]),
    Operation(OpType.ENTAIL, Operations.mreasoner_falsify, [1]),
    Operation(OpType.ENTAIL, Operations.mreasoner_falsify, [2]),
    Operation(OpType.ENTAIL, Operations.mreasoner_falsify, [3]),
    Operation(OpType.ENTAIL, Operations.mreasoner_falsify_with_model, [0]),
    Operation(OpType.ENTAIL, Operations.mreasoner_falsify_with_model, [1]),
    Operation(OpType.ENTAIL, Operations.mreasoner_falsify_with_model, [2]),
    Operation(OpType.ENTAIL, Operations.mreasoner_falsify_with_model, [3]),

    # Entail
    Operation(OpType.ENTAIL, Operations.phm_p_entailment, ()),
    Operation(OpType.ENTAIL, Operations.weaken_conclusion, ()),
    Operation(OpType.ENTAIL, Operations.phm_reply, ()),

    # Respond
    Operation(OpType.RESPOND, Operations.resp_prefer, ("stronger", "ac", True, "pos")),  # wichtig
    Operation(OpType.RESPOND, Operations.resp_prefer, ("weaker", "ac", True, "pos")),  # wichtig
    Operation(OpType.RESPOND, Operations.resp_prefer, ("stronger", "ca", True, "pos")),  # wichtig
    Operation(OpType.RESPOND, Operations.resp_prefer, ("weaker", "ca", True, "pos")),  # wichtig
#    Operation(OpType.RESPOND, Operations.resp_random, ()),  # wichtig
]

# Additional operations to create complete 1-model-plans of every model
EXTRA_OPERATIONS = [
    Operation(OpType.XYZ, Operations.psycop_predict, ()),  # TEMPORÄR FÜR PSYCOP
    Operation(OpType.XYZ, Operations.ic_predict, ()),  # TEMPORÄR FÜR IC
]


class AbstractModelBase:
    def __init__(self, operations, only_pure=False, extra_weight=0.8, max_depth=6, name="Tree_Model (base)", init_ops=True):
        self.name = name

        # Initialize container with all implemented operations
        if init_ops:
            self.ops = Operations()

        # Actually used set of operations
        self.operations = operations

        # Maximum depth of search trees
        self.max_depth = max_depth

        # Allow only one-model plans
        self.only_pure = only_pure

        # How much one adapt step is weighted w.r.t the entire pre training
        self.extra_weight = extra_weight

        # Contains scores for all plans
        self.all_plans = {}

        # Number of subjects used by pre training
        self.n_pre_subjects = None

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __str__(self):
        s = str(self.name) + "\n"
        s += str([str(o) for o in self.operations]) + "\n"
        s += str(self.n_pre_subjects) + "\n"
        s += str(self.extra_weight) + "\n"
        s += str(self.all_plans) + "\n"
        return s

    def __deepcopy__(self, memodict={}):
        copied_object = AbstractModelBase(self.operations, self.only_pure, self.extra_weight, self.max_depth, self.name, False)
        copied_object.all_plans = copy.deepcopy(self.all_plans)
        copied_object.n_pre_subjects = self.n_pre_subjects
        copied_object.plan_syl_to_concl = self.plan_syl_to_concl
        copied_object.concl_syl_to_plans = self.concl_syl_to_plans
        copied_object.ops = self.ops
        return copied_object

    def start_participant(self, **kwargs):
        pass

    def adapt(self, item, target, **kwargs):
        syl = ccobra.syllogistic.encode_task(item.task)
        # Adapt plan model by increasing score of paths that lead from item to target
        for plan in self.find_plans(syl, ccobra.syllogistic.encode_response(target, item.task)):
            self.all_plans[plan] += self.n_pre_subjects * self.extra_weight

    def recursive_follow(self, root, plan):
        node = self.follow_plan(root, plan)
        if node is not None and node.is_leaf:
            return plan
        if "/" in plan:
            new_plan = plan[:plan.rfind("/")]
            return self.recursive_follow(root, new_plan)
        return None

    def get_best_plan(self, item):
        """ Get the plan with the highest score that appears in item's tree """

        for plan in sorted(self.all_plans, key=self.all_plans.get, reverse=True):
            return plan
        raise Exception

    def check_pureness(self, plan):
        plan_segments = [p for p in plan.split("/") if "root" != p and "resp_" not in p]
        plan_segments = [p[:re.search(r"[\(\[]", p).start()] for p in plan_segments]
        return len(set([self.ops.op2model[strop] for strop in plan_segments])) == 1

    def pre_train(self, dataset):
        # Pre-train single models
        self.ops.mreasoner.pre_train(dataset)
        self.ops.vm.pre_train(dataset)
        self.ops.psycop.pre_train(dataset)
        self.ops.phm.pre_train(dataset)

        # Round paramters that would lead to stochastic operation results to True or False
        for model in [self.ops.mreasoner, self.ops.vm, self.ops.psycop, self.ops.phm]:
            params = model.model.params
            for p in params:
                if p in ["guess", "epsilon", "System 2", "Weaken", "p-entailment", "confidenceA", "confidenceI", "confidenceE", "confidenceO"]:  # probability parameters
                    model.model.params[p] = 0.0 if params[p] < 0.5 else 1.0

        self.plan_syl_to_concl, self.concl_syl_to_plans = self.span_all_trees()

        # Store number of subjects for later as weight for adapt
        self.n_pre_subjects = len(dataset)

        # Store score of every plan
        self.all_plans = {}

        all_plans = []
        for plan, syl in self.plan_syl_to_concl:
            if syl == "AA1":
                all_plans.append(plan)

        # Initialize score of all plans to 0
        for plan in all_plans:
            self.all_plans[plan] = 0

        # Count correct predicitons
        for participant in dataset:
            for answer in participant:
                syl = ccobra.syllogistic.encode_task(answer["item"].task)
                target = ccobra.syllogistic.encode_response(answer["response"], answer["item"].task)
                target_nodes = self.find_plans(syl, target)
                for tn in target_nodes:
                    plan = tn
                    self.all_plans[plan] += 1

    def applicable(self, operation, current_node):
        """ Check if operation is applicable in current_node """

        if operation.pre_state == current_node.state and operation.eval_preconditions(current_node):
            return True
        return False

    def update_variables(self, operation, current_node, post_content):
        """ Apply variable changes in state transition defined by operation """

        new_vars = copy.deepcopy(current_node.vars)
        if operation.fnc.__name__ == "psycop_check":
            new_vars["num_psycop_checks"] += 1
        if operation.fnc.__name__ == "mm_falsify" or operation.fnc.__name__ == "vm_reencode":
            new_vars["num_concludes"] -= 1
        if operation.optype == OpType.PREENCODE:
            new_vars["num_preencodes"] += 1
        elif operation.optype == OpType.CONCLUDE:
            new_vars["num_concludes"] += 1
        elif operation.optype == OpType.REENCODE:
            new_vars["num_reencodes"] += 1
        if operation.fnc.__name__ == "phm_p_entailment":
            new_vars["num_entails"] += 1
        if operation.fnc.__name__ == "mreasoner_falsify" or operation.fnc.__name__ == "mreasoner_falsify_with_model":
            new_vars["mreasoner_last_falsify"] = operation.args[0]
        elif operation.optype == OpType.FALSIFY:
            new_vars["num_falsifies"] += 1
            if operation.fnc.__name__ == "mm_falsify":
                new_vars["mm_last_falsify"] = operation.args[0]
        if operation.post_state == State.MR and post_content[1] is not None:
            new_vars["n_conclusions"] = len(post_content[1])
        if operation.fnc.__name__ == "weaken_conclusion":
            new_vars["num_weakens"] += 1
        if operation.fnc.__name__ == "phm_reply":
            new_vars["num_phm_replies"] += 1

        return new_vars

    def follow_plan_no_root(self, syl, plan):
        return self.plan_syl_to_concl[(plan, syl)]

    def follow_plan(self, root, plan):
        """ Return node you arrive at when following plan from root. """

        plan_segments = plan.split("/")
        current_node = root
        for i, strop in enumerate(plan_segments):
            part_plan = "/".join(plan_segments[:i+1])
            try:
                current_node = search.findall_by_attr(current_node, value=part_plan, name="name", maxlevel=2)[0]
            except IndexError:
                return None
        return current_node

    def plan_vote(self, item):
        syl = ccobra.syllogistic.encode_task(item.task)

        conclusions = ccobra.syllogistic.RESPONSES
        scores = []

        for concl in conclusions:
            target_nodes = self.find_plans(syl, concl)
            scores.append(sum([self.all_plans[tn] for tn in target_nodes]))
        return conclusions[scores.index(max(scores))]

    def plan_vote2(self, item):
        conclusions = ccobra.syllogistic.RESPONSES
        scores = []

        for concl in conclusions:
            target_nodes = self.find_plans(item, concl)
            concl_score = 0
            for tn in target_nodes:
                additional_weight = np.prod([self.param_bevorzugung[f] for f in self.param_bevorzugung if f in tn.name])
                concl_score += self.all_plans[tn.name] * additional_weight

            scores.append(concl_score)

        n = sum(scores)
        scores = [s / n for s in scores]
        return conclusions[scores.index(max(scores))]

    def find_plans(self, syl, target):
        return self.concl_syl_to_plans[(syl, target)]

    def new_child(self, operation, current_node):
        """ Return a new child node by applying an operation to current_node. """

        # Different operation types = different placement of the function output
        if operation.optype == OpType.NONE:
            post_content = (operation.fnc(self.ops, current_node.content, *operation.args), current_node.content[1])
        elif operation.optype == OpType.PREENCODE:
            post_content = operation.fnc(self.ops, current_node.content, *operation.args)
        elif operation.optype == OpType.HEURISTIC:
            post_content = (None, operation.fnc(self.ops, current_node.content, *operation.args))
        elif operation.optype == OpType.ENCODE or operation.optype == OpType.ENCODE_PREMISES:
            post_content = (operation.fnc(self.ops, current_node.content, *operation.args), None)
        elif operation.optype == OpType.CONCLUDE or operation.optype == OpType.REENCODE or operation.optype == OpType.ENTAIL:
            if operation.fnc.__name__ == "vm_reencode":
                post_content = operation.fnc(self.ops, current_node.content, *operation.args)
            else:
                post_content = (current_node.content[0], operation.fnc(self.ops, current_node.content, *operation.args))
        elif operation.optype == OpType.FALSIFY:
            post_content = (operation.fnc(self.ops, current_node.content, *operation.args), current_node.content[1])
        elif operation.optype == OpType.RESPOND:
            post_content = operation.fnc(self.ops, current_node.content, *operation.args)
        elif operation.optype == OpType.XYZ:
            post_content = (None, operation.fnc(self.ops, current_node.content, *operation.args))
        else:
            raise Exception

        # new child node
        return AnyNode(name="{0}/{1}{2}".format(current_node.name, operation.fnc.__name__, operation.args),
                       content=post_content,
                       state=operation.post_state,
                       vars=self.update_variables(operation, current_node, post_content),
                       parent=current_node)

    def print_tree(self, root):
        for pre, _, node in RenderTree(root):
            print("%s%s\t%s" % (pre, node.name, node.content))

    def remove_leaf(self, leaf):
        leaf.parent.children = tuple(n for n in leaf.parent.children if n is not leaf)\

    def span(self, item):
        self.ops.current_syllogism = ccobra.syllogistic.encode_task(item.task)
        root = AnyNode(name="root", content=item, state=State.Item, vars={"num_preencodes": 0,
                                                                          "num_concludes": 0,
                                                                          "num_falsifies": 0,
                                                                          "num_entails": 0,
                                                                          "num_weakens": 0,
                                                                          "num_phm_replies": 0,
                                                                          "num_reencodes": 0,
                                                                          "num_psycop_checks": 0,
                                                                          "mm_last_falsify": None,
                                                                          "mreasoner_last_falsify": None,
                                                                          "n_conclusions": 0,
                                                                          })
        fringe = [root]

        while True:
            if len(fringe) == 0:
                break
            current_node = fringe.pop()
            if current_node.depth == self.max_depth:
                continue
            for op in self.operations:
                if self.only_pure and not self.check_pureness("{0}/{1}{2}".format(current_node.name, op.fnc.__name__, op.args)):
                    continue
                if self.applicable(op, current_node):
                    x = self.new_child(op, current_node)
                    fringe.append(x)

        # Remove all leaves that don't contain a conclusion
        while True:
            leaves_to_remove = [leaf for leaf in root.leaves if leaf.content not in ccobra.syllogistic.RESPONSES]
            if len(leaves_to_remove) == 0:
                break
            for leaf in leaves_to_remove:
                # remove leaf from tree
                self.remove_leaf(leaf)

        return root

    def span_all_trees(self):
        plan_syl_to_concl, concl_syl_to_plans = {}, {}
        for syllogism in ccobra.syllogistic.SYLLOGISMS:
            for concl in ccobra.syllogistic.RESPONSES:
                concl_syl_to_plans[(syllogism, concl)] = []
        for i, item in enumerate(sylutil.GENERIC_ITEMS):
            syl = ccobra.syllogistic.encode_task(item.task)
            root = self.span(item)
            for i, leaf in enumerate(root.leaves):
                concl = leaf.content
                plan = leaf.name
                plan_syl_to_concl[(plan, syl)] = concl
                concl_syl_to_plans[(syl, concl)].append(plan)

        return plan_syl_to_concl, concl_syl_to_plans

    def predict(self, item, **kwargs):
#        random.seed(1243)
        syl = ccobra.syllogistic.encode_task(item.task)
        prediction_node = self.follow_plan_no_root(syl, self.get_best_plan(item))
        return ccobra.syllogistic.decode_response(prediction_node, item.task)
