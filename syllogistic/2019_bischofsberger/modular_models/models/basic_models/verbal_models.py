import copy
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.models.basic_models.interface import SyllogisticReasoningModel
from modular_models.util import sylutil


class VerbalModels(SyllogisticReasoningModel):
    def __init__(self):
        SyllogisticReasoningModel.__init__(self)

        # For meaning of parameteres see Polk & Newell 1995.
        self.params["p1"] = "b"
        self.params["p2"] = "a"
        self.params["p3"] = "a"
        self.params["p4"] = "a"
        self.params["p5"] = "b"
        self.params["p6"] = "a"

        self.params["p10"] = "b"
        self.params["p11"] = "b"
        self.params["p12"] = "b"
        self.params["p13"] = "c"

        self.params["p14"] = "b"
        self.params["p15"] = "c"
        self.params["p16"] = "c"
        self.params["p17"] = "c"

        self.params["p18"] = "b"
        self.params["p19"] = "c"
        self.params["p20"] = "b"
        self.params["p21"] = "c"

        # Commented out parameter ranges are implemented but left out to reduce parameter space.
        self.param_grid["p1"] = ["a", "b", "c"]  # ["a", "b", "c", "d", "e"]
        self.param_grid["p2"] = ["a", "b", "c"]  # ["a", "b", "c", "d", "e"]
        self.param_grid["p3"] = ["a", "b"]
        self.param_grid["p4"] = ["a", "b", "c"]  # ["a", "b", "c", "d"]
        self.param_grid["p5"] = ["a", "b"]
        self.param_grid["p6"] = ["a", "b", "c"]  # ["a", "b", "c", "d"]

        self.param_grid["p10"] = ["a", "b"]  # VR1/2: "a", VR3: "b"
        self.param_grid["p11"] = ["a", "b"]  # VR1: "a", VR2/3: "b"
        self.param_grid["p12"] = ["a", "b"]  # VR1: "a", VR2/3: "b"
        self.param_grid["p13"] = ["a", "c"]  # VR1/2: "a", VR3: "c"

        self.param_grid["p14"] = ["a", "b"]  # VR1/2: "a", VR3: "b"
        self.param_grid["p15"] = ["a", "c"]  # VR1/2: "a", VR3: "c"
        self.param_grid["p16"] = ["a", "c"]  # VR1/2: "a", VR3: "c"
        self.param_grid["p17"] = ["a", "c"]  # VR1/2: "a", VR3: "c"

        self.param_grid["p18"] = ["a", "b"]  # VR1/2: "a", VR3: "b"
        self.param_grid["p19"] = ["a", "c"]  # VR1/2: "a", VR3: "c"
        self.param_grid["p20"] = ["a", "b"]  # VR1/2: "a", VR3: "b"
        self.param_grid["p21"] = ["a", "c"]  # VR1/2: "a", VR3: "c"

        self.is_stochastic = False

        # global counter to measure access time
        self.t = -1

    def generate_param_configurations(self):
        """ Custom parameter grid to reduce parameter space """

        configurations = []
        for p1 in self.param_grid["p1"]:
            for p2 in self.param_grid["p2"]:
                for p3 in self.param_grid["p3"]:
                    for p4 in self.param_grid["p4"]:
                        for p5 in self.param_grid["p5"]:
                            for p6 in self.param_grid["p6"]:
                                for i in [0, 1, 2]:
                                    if i == 0:
                                        # VR1
                                        p10to21 = ["a"] * 12
                                    elif i == 1:
                                        # VR2
                                        p10to21 = ["a", "b", "b", "a"] + ["a"]*8
                                    else:
                                        # VR3
                                        p10to21 = ["b", "b", "b", "c", "b", "c", "c", "c", "b", "c", "b", "c"]

                                    param_dict = {"p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6}
                                    d = dict([(p, val) for p, val in zip(["p"+str(i) for i in range(10, 22)], p10to21)])
                                    param_dict.update(d)
                                    configurations.append(param_dict)

        return configurations



    class Prop:
        """ Class to represent a property of an individual in a VM """

        def __init__(self, name, neg=False, identifying=False, t=-1):
            # name, typically one of "a", "b" and "c"
            self.name = name

            # negation flag ("-")
            self.neg = neg

            # identifying flag ("'")
            self.identifying = identifying

            # access time of the proposition
            self.t = t

        def __eq__(self, other):
            return self.name == other.name and self.neg == other.neg

        def __repr__(self):
            prefix = "-" if self.neg else ""
            suffix = "'" if self.identifying else ""
            return prefix + self.name + suffix + "(" + str(self.t) + ")"

    class Individual:
        """ Class to represent an individual (= a row) in a VM """

        def __init__(self, props, t):
            # list of properties
            self.props = props

            # access time of the individual
            self.t = t

        def contains(self, p):
            """ Check if individual contains a certain property (respecting only name and negation
            but not access time and identifying flag) """

            for sp in self.props:
                if sp.name == p.name and sp.neg == p.neg:
                    return True
            return False

        def most_recent_prop(self):
            """ Returns the individual's most recently accessed property """

            return sorted(self.props, key=lambda x: x.t, reverse=True)[0]

        def __repr__(self):
            return "[" + "".join([str(p) + ", " for p in self.props])[:-2] + "](" + str(
                self.t) + ")"

    def set_version(self, version):
        """ Adjust parameters to version "VR1", "VR2" or "VR3", see Polk & Newell 1995 """

        if version == "VR1":
            params_vr1 = {"p10": "a", "p11": "a", "p12": "a", "p13": "a",
                          "p14": "a", "p15": "a", "p16": "a", "p17": "a",
                          "p18": "a", "p19": "a", "p20": "a", "p21": "a"}
            self.params.update(params_vr1)

        elif version == "VR2":
            params_vr2 = {"p10": "a", "p11": "b", "p12": "b", "p13": "a",
                          "p14": "a", "p15": "a", "p16": "a", "p17": "a",
                          "p18": "a", "p19": "a", "p20": "a", "p21": "a"}
            self.params.update(params_vr2)

        elif version == "VR3":
            params_vr3 = {"p10": "b", "p11": "b", "p12": "b", "p13": "c",
                          "p14": "b", "p15": "c", "p16": "c", "p17": "c",
                          "p18": "b", "p19": "c", "p20": "b", "p21": "c"}
            self.params.update(params_vr3)

    def timestamp(self):
        """ get timestamp by increasing a counter """
        self.t = self.t + 1
        return self.t

    def encode(self, syllogism):
        """ Returns a VM by initially encoding a syllogism

        >>> vm = VerbalModels()
        >>> vm.params.update(dict.fromkeys(["p1", "p2", "p3", "p4", "p5", "p6"], "a"))
        >>> vm.encode("IA4") # p. 539
        [[b'(0), a(0), c(1)](0), [b'(0), c(1)](0)]
        >>> vm.encode("IA3") # p. 542
        [[a'(2), b(2)](2), [a'(2)](2), [c'(3), b(3)](3)]
        >>> vm.encode("AI1") # p. 542
        [[a'(4), b'(4), c(5)](5), [a'(4), b'(4)](5)]

        >>> vm.encode("EI1") # own
        [[a'(6), -b(6)](6), [b'(7), c(7)](7), [b'(7)](7)]
        >>> vm.encode("AA1") # own
        [[a'(8), b'(8), c(9)](8)]
        >>> vm.encode("EE1") # own
        [[a'(10), -b(10)](10), [b'(11), -c(11)](11)]
        """

        vm = []
        premises = [syllogism[i] + sylutil.term_order(syllogism[2])[i] for i in [0, 1]]
        for premise in premises:
            vm = self.extend_vm(vm, premise, reencoding=False)

        return vm

    def extend_vm(self, vm, premise, reencoding=True, subj_neg=False):
        """ Basically implements Polk & Newell 1995, p. 538, Figure 4

        >>> m = VerbalModels()
        >>> vm = m.encode("OA1")
        >>> vm
        [[a'(0), -b(0)](0), [a'(0)](0), [b'(1), c(1)](1)]
        >>> m.extend_vm(vm, "Acb", False)
        [[a'(0), -b(0)](0), [a'(0)](0), [b'(1), c'(1)](1)]
        >>> m.extend_vm(vm, "Oba", False)
        [[a'(0), -b(0)](0), [a'(0)](0), [b'(1), c(1), -a(3)](3), [b'(1), c(1)](3)]
        """

        t = self.timestamp()

        prem_subj = self.Prop(name=premise[1], neg=subj_neg, identifying=True, t=t)
        prem_obj = self.Prop(name=premise[2], neg=True if premise[0] in ["O", "E"]  else False,
                             identifying=False, t=t)
        prem_obj_neg = self.Prop(name=premise[2], neg=False if premise[0] in ["O", "E"] else True,
                                 identifying=False, t=t)
        vm_copy = copy.deepcopy(vm)

        if premise[0] in ["A", "E"]:
            param35 = [self.params["p3"], self.params["p3"]][reencoding] if premise[0] == "A" else [self.params["p5"], self.params["p5"]][reencoding]
            none_found = True
            for indiv in vm_copy:
                if indiv.contains(prem_subj):
                    none_found = False

                    # set identifying flag
                    for p in indiv.props:
                        if p == prem_subj:
                            p.identifying = True

                    # augment
                    if not indiv.contains(prem_obj):
                        indiv.props.append(prem_obj)

            if none_found or param35 == "b":
                vm_copy.append(self.Individual(props=[prem_subj, prem_obj], t=t))
            return vm_copy

        try:
            param46 = [self.params["p4"], self.params["p4"]][reencoding] if premise[0] == "I" else [self.params["p6"], self.params["p6"]][reencoding]
        except:
            print("GI", self.params)
        param12 = [self.params["p1"], self.params["p1"]][reencoding] if premise[0] == "I" else [self.params["p2"], self.params["p2"]][reencoding]

        candidates = []

        # Search a candidate to be changed (for the right row of Table 4), see Appendix C
        # First try: Search for a candidate with X and Y
        if param46 in ["a", "c"]:
            for indiv in vm_copy:
                if indiv.contains(prem_subj) and indiv.contains(prem_obj):
                    candidates.append(indiv)

        # Second try: Search for a candidate with X (with or without -Y)
        if param46 in ["a", "b"]:
            if len(candidates) == 0:
                for indiv in vm_copy:
                    if indiv.contains(prem_subj):
                        candidates.append(indiv)

        # Third try: Append an object and return without augmentation
        if len(candidates) == 0 or param46 == "d":
            vm_copy.append(self.Individual(props=[prem_subj, prem_obj], t=t))
            vm_copy.append(self.Individual(props=[prem_subj], t=t))
            return vm_copy

        # find most recent (MR) individual among the candidates
        most_recent_individual = sorted(candidates, key=lambda x: x.t, reverse=True)[0]

        # operation 1: set identiying flag
        for p in most_recent_individual.props:
            if p == prem_subj:
                p.identifying = True

        # operation 2: augment most recent individual
        if not most_recent_individual.contains(prem_obj):
            most_recent_individual.props.append(prem_obj)
            most_recent_individual.t = t

        # operation 3: create new individual - TODO: getrickst
        if param12 == "a":
            new_inds = [self.Individual(props=[p for p in most_recent_individual.props if p != prem_obj], t=t)]
        elif param12 == "b":
            new_inds = [self.Individual(props=[prem_subj], t=t)]
        elif param12 == "c":
            new_inds = [self.Individual(props=[prem_subj, prem_obj_neg], t=t)]
        elif param12 == "d":
            new_inds = [self.Individual(props=[p if p != prem_obj else prem_obj_neg for p in most_recent_individual.props], t=t),
                        self.Individual(props=[p for p in most_recent_individual.props if p != prem_obj], t=t)]
        elif param12 == "e":
            new_inds = []

        vm_copy.extend(new_inds)

        return vm_copy

    def identifying_props(self, vm):
        """ Get identifying properties from vm.

        >>> m = VerbalModels()
        >>> a = m.Prop("a", False, False, 0)
        >>> a_ = m.Prop("a", False, True, 0)
        >>> b = m.Prop("b", False, False, 0)
        >>> b_ = m.Prop("b", False, True, 0)
        >>> c = m.Prop("c", False, False, 0)
        >>> c_ = m.Prop("c", False, True, 0)
        >>> m.identifying_props([m.Individual([a, b_, c_], 0)])
        [b'(0), c'(0)]
        >>> m.identifying_props([m.Individual([a_, b_], 0), m.Individual([a_, b_, c], 0)])
        [a'(0), b'(0)]
        """

        ip = []
        for indiv in vm:
            for p in indiv.props:
                if p.identifying and all([p_el.name != p.name for p_el in ip]):
                    ip.append(p)
        return ip

    def conclude(self, vm):
        """ Draw conclusions from vm.

        >>> m = VerbalModels()
        >>> a = m.Prop("a", False, False, 0)
        >>> a_ = m.Prop("a", False, True, 0)
        >>> b = m.Prop("b", False, False, 0)
        >>> b_ = m.Prop("b", False, True, 0)
        >>> bn = m.Prop("b", True, False, 0)
        >>> c = m.Prop("c", False, False, 0)
        >>> c_ = m.Prop("c", False, True, 0)
        >>> cn_ = m.Prop("c", True, True, 0)
        >>> m.conclude([m.Individual(props=[b_, c], t=0),
        ...             m.Individual(props=[b_, a, c], t=1)]) # p.539
        ['NVC']
        >>> m.conclude([m.Individual(props=[b_, c], t=0),
        ...             m.Individual(props=[a_], t=1),
        ...             m.Individual(props=[b_, a_, c], t=2)]) # p.539
        ['Iac']
        >>> m.conclude([m.Individual(props=[b_, a, c], t=0)]) # p.545
        ['NVC']
        >>> m.conclude([m.Individual(props=[b_, a, c_], t=0)]) # p.545
        ['Aca']
        >>> m.conclude([m.Individual(props=[a_], t=0),
        ...             m.Individual(props=[a_, bn], t=1),
        ...             m.Individual(props=[b_, a, c], t=2),
        ...             m.Individual(props=[c_], t=3),
        ...             m.Individual(props=[c_, bn], t=4)]) # p.545
        ['Iac', 'Ica']
        >>> m.conclude([m.Individual(props=[b_, a, cn_], t=0)]) # own EA4
        ['NVC']
        """

        conclusions = []
        for p in self.identifying_props(vm):
            if p.name == "b" or p.neg:
                # don't draw conclusions with b, -b, -a or -c as subject
                continue
            other_p = "a" if p.name == "c" else "c"
            other_p_neg = self.Prop(name=other_p, neg=True, identifying=False, t=-1)
            other_p = self.Prop(name=other_p, neg=False, identifying=False, t=-1)
            a, i, e, o = True, False, True, False

            # Check conditions for drawing conclusions of type A, I, E, O
            for indiv in vm:
                if indiv.contains(p):
                    if indiv.contains(other_p):
                        i = True
                    else:
                        a = False
                    if indiv.contains(other_p_neg):
                        o = True
                    else:
                        e = False

            # build conclusions
            for x, s in [(a, "A"), (i, "I"), (e, "E"), (o, "O")]:
                if x:
                    # exclude weaker conclusions
                    if (s == "I" and "A" + p.name + other_p.name in conclusions) or \
                            (s == "O" and "E" + p.name + other_p.name in conclusions):
                        continue
                    conclusions.append(s + p.name + other_p.name)
        if len(conclusions) > 0:
            return sorted(conclusions)
        return ["NVC"]

    def get_additional_premises(self, target_premise, reference_property):
        """ Takes a Reference Property and a Target Proposition and returns additional information
        encoded as an additional premise. For VR3 this function corresponds to Polk & Newell (1995),
        Figure A1, p. 565.

        :param target_premise: Target Premise in string encoding.
        :param reference_property: Reference Property as Prop object
        :returns (prem, neg_subj): with prem being a string encoding the "additional premise" and
            neg_subj telling whether or not the subject of this premise is negative.

        >>> m = VerbalModels()
        >>> a, b = m.Prop("a", False, False), m.Prop("b", False, False)
        >>> na, nb = m.Prop("a", True, False), m.Prop("b", True, False)
        >>> m.set_version("VR1")
        >>> m.get_additional_premises("Aab", b)
        (None, None)
        >>> m.set_version("VR2")
        >>> m.get_additional_premises("Aab", b)
        (None, None)
        >>> m.get_additional_premises("Eab", b)
        ('Eba', False)
        >>> m.get_additional_premises("Iab", b)
        ('Iba', False)
        >>> m.get_additional_premises("Aab", na)
        (None, None)
        >>> m.get_additional_premises("Iab", na)
        (None, None)
        >>> m.get_additional_premises("Eab", na)
        (None, None)
        >>> m.get_additional_premises("Oab", na)
        (None, None)
        >>> m.get_additional_premises("Aab", nb)
        (None, None)
        >>> m.get_additional_premises("Iab", nb)
        (None, None)
        >>> m.get_additional_premises("Eab", nb)
        (None, None)
        >>> m.get_additional_premises("Oab", nb)
        (None, None)

        See Polk & Newell (1995), Figure A1, p. 565
        >>> m.set_version("VR3")
        >>> m.get_additional_premises("Aab", b)
        ('Aba', False)
        >>> m.get_additional_premises("Iab", b)
        ('Iba', False)
        >>> m.get_additional_premises("Eab", b)
        ('Eba', False)
        >>> m.get_additional_premises("Oab", b)
        ('Oba', False)

        >>> m.get_additional_premises("Aab", na)
        ('Eab', True)
        >>> m.get_additional_premises("Iab", na)
        ('Iab', True)
        >>> m.get_additional_premises("Eab", na)
        ('Eab', True)
        >>> m.get_additional_premises("Oab", na)
        ('Oab', True)

        >>> m.get_additional_premises("Aab", nb)
        ('Eba', True)
        >>> m.get_additional_premises("Iab", nb)
        ('Iba', True)
        >>> m.get_additional_premises("Eab", nb)
        ('Aba', True)
        >>> m.get_additional_premises("Oab", nb)
        ('Oba', True)
        """

        # derive the additional premise's terms
        rp_term = reference_property.name
        rp_is_negative = reference_property.neg
        other_term = target_premise[1] if rp_term == target_premise[2] else target_premise[2]

        # Y column
        if target_premise[2] == rp_term and not rp_is_negative:
            if target_premise[0] == "A" and self.params["p10"] == "b":
                return "A" + rp_term + other_term, False
            elif target_premise[0] == "I" and self.params["p11"] == "b":
                return "I" + rp_term + other_term, False
            elif target_premise[0] == "E" and self.params["p12"] == "b":
                return "E" + rp_term + other_term, False
            elif target_premise[0] == "O" and self.params["p13"] == "c":
                return "O" + rp_term + other_term, False

        # -X column
        elif target_premise[1] == rp_term and rp_is_negative:
            if target_premise[0] == "A" and self.params["p14"] == "b":
                return "E" + rp_term + other_term, True
            elif target_premise[0] == "I" and self.params["p15"] == "c":
                return "I" + rp_term + other_term, True
            elif target_premise[0] == "E" and self.params["p16"] == "c":
                return "E" + rp_term + other_term, True
            elif target_premise[0] == "O" and self.params["p17"] == "c":
                return "O" + rp_term + other_term, True

        # -Y column
        elif target_premise[2] == rp_term and rp_is_negative:
            if target_premise[0] == "A" and self.params["p18"] == "b":
                return "E" + rp_term + other_term, True
            elif target_premise[0] == "I" and self.params["p19"] == "c":
                return "I" + rp_term + other_term, True
            elif target_premise[0] == "E" and self.params["p20"] == "b":
                return "A" + rp_term + other_term, True
            elif target_premise[0] == "O" and self.params["p21"] == "c":
                return "O" + rp_term + other_term, True

        return None, None

    def reencode(self, syllogism, vm, conclusions):
        # sort rows (individuals) of VM by access time (most recent first)
        sorted_inds = sorted(vm, key=lambda x: x.t, reverse=True)

        # sort properties per individual by access time (most recent first), flatten and uniquify
        sorted_props = [sorted(ind.props, key=lambda p: p.t, reverse=True) for ind in sorted_inds]
        sorted_props = sylutil.uniquify_keep_order([p for l in sorted_props for p in l])

        p1_terms, p2_terms = sylutil.term_order(syllogism[2])
        for prop in sorted_props:
            for target_premise in [syllogism[0] + p1_terms, syllogism[1] + p2_terms]:
                if prop.name in target_premise:
                    # reencode the target premise without additional information
                    if not prop.neg:
                        vm = self.extend_vm(vm, target_premise, reencoding=True)
                        new_conclusions = self.conclude(vm)
                        if any([c not in conclusions for c in new_conclusions]):
                            return new_conclusions

                    # get indirect information encoded as additional premise
                    prem, subj_neg = self.get_additional_premises(target_premise, prop)
                    if prem is not None:
                        # reencode using the additional premise
                        vm = self.extend_vm(vm, prem, reencoding=True, subj_neg=subj_neg)
                        new_conclusions = self.conclude(vm)
                        if any([c not in conclusions for c in new_conclusions]):
                            return new_conclusions
        return ["NVC"]

    def predict(self, syllogism):
        vm = self.encode(syllogism)
        conclusions = self.conclude(vm)
        if conclusions == ["NVC"]:
            conclusions = self.reencode(syllogism, vm, conclusions)
        return conclusions



vm = VerbalModels()
vm.generate_param_configurations()
