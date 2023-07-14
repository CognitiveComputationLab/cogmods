"""Reasoner based on the 'model theory'
"""
# from assertion_parser import parse_all, facts
# from model_builder import premises_model, remove_duplicates, MentalModel, not_model
import numpy as np
# from logger import logging

from .assertion_parser import parse_all
from .model_builder import premises_model, remove_duplicates, MentalModel, not_model
from .logger import logging

def model(premises):
    """Turn premises into one model

    Arguments:
        premises {list} -- list of premise strings

    Keyword Arguments:
        system {int} -- system 1 or 2 (default: {1})

    Returns:
        MentalModel -- the model
    """
    if not isinstance(premises, list):
        premises = [premises]
    parsed = parse_all(premises)
    return premises_model(parsed)

def all_val(arr, val):
    """Check if all values in array have specific value

    Arguments:
        arr {np.array} -- 1D numpy array of ints
        val {int} -- value to check

    Returns:
        bool -- yes/no
    """
    for el in arr:
        if el != val:
            return False
    return True

def some_val(arr, val):
    """Check if at least one value in array has a specific value

    Arguments:
        arr {np.array} -- 1D numpy array of ints
        val {int} -- value to check

    Returns:
        bool -- yes/no
    """
    for el in arr:
        if el == val:
            return True
    return False

def what_follows(premises, system=1):
    """What follows from a set of premises?

    Facts already in the premises are dismissed from return value.

    Arguments:
        premises {list} -- list of premise strings

    Keyword Arguments:
        system {int} -- system 1 or 2 (default: {1})

    Returns:
        tuple -- necessary and possible clauses that follow: (nec, pos)
    """
    f = facts(premises)
    nec, pos = nec_and_pos(premises, system)
    nec_without_facts = [n for n in nec if n not in f]
    pos_without_facts_and_nec = [p for p in pos if p not in f and p not in nec]
    return (nec_without_facts, pos_without_facts_and_nec)


def nec_and_pos(premises, system=1):
    """Return clauses that are necessary and possible

    Arguments:
        premises {list} -- list of premise strings

    Keyword Arguments:
        system {int} -- system 1 or 2 (default: {1})

    Raises:
        Exception: if system other that 1 or 2

    Returns:
        tuple -- necessary and possible clauses that follow: (nec, pos)
    """
    m = model(premises)
    if system == 1:
        nec = []
        pos = []
        for c in m.clauses:
            column = m.get_column(c)
            if all_val(column, 1):
                nec.append(c)
            elif all_val(column, -1):
                nec.append('¬' + c)
            if some_val(column, 1):
                pos.append(c)
            if some_val(column, -1):
                pos.append('¬' + c)
        return (nec, pos)
    elif system == 2:
        nec = []
        pos = []
        for c in m.full_clauses:
            column = m.full_get_column(c)
            if all_val(column, 1):
                nec.append(c)
            elif all_val(column, -1):
                nec.append('¬' + c)
            if some_val(column, 1):
                pos.append(c)
            if some_val(column, -1):
                pos.append('¬' + c)
        return (nec, pos)
    else:
        raise Exception

def how_possible(premises, conclusion, system=1):
    """Return how possible the conclusion is given the premisses

    Arguments:
        premises {list} -- list of assertion strings
        conclusion {str} -- assertion string

    Keyword Arguments:
        system {int} -- system 1 or 2 (default: {1})

    Returns:
        str -- the description of how possible
    """
    p = probability(premises, conclusion, 2)
    if p == 0:
        return "impossible"
    elif p < 0.1:
        return "almost impossible"
    elif p < 0.3:
        return "less possible"
    elif p <= 0.7:
        return "possible"
    elif p <= 0.9:
        return "very possible"
    elif p < 1:
        return "almost certain"
    else:
        return "certain"

def probability(premises, conclusion, system=1):
    """Return probability of an assertion given the premises

    Based on an "assumption of equal possibilities": The number of models of
    the conclusion that are also models of the premises divided by the number
    of models of the premises.

    Arguments:
        premises {list} -- list of premise strings
        conclusion {str} -- conclusion string

    Keyword Arguments:
        system {int} -- system 1 or 2 (default: {1})

    Returns:
        float -- probability
    """
    if system == 1:
        return None
    m1 = model(premises)
    m2 = model(conclusion)
    common = in_common(m1, m2, system)

    if not common:
        return None

    poss_1, poss_2 = poss_in_common(m1, m2, system, common)

    matches = 0
    for row_2 in poss_2:
        for row_1 in poss_1:
            if np.array_equal(row_1, row_2):
                matches += 1

    return round(matches / len(m1.full_poss), 2)

def poss_in_common(m1, m2, system=1, common=None, keep_duplicates=True):
    """Return only those parts of the possibilities for which the two models
    have clauses in common

    Arguments:
        m1 {MentalModel} -- model 1
        m2 {MentalModel} -- model 2

    Keyword Arguments:
        system {int} -- system 1 or 2 (default: {1})
        common {(str,int,int)} -- (clause, index_1, index_2) (default: {None})
        keep_duplicates {bool} -- if True keep duplicate rows else discard (default: {True})

    Returns:
        (np.array, np.array) -- the reduced possibilities of the models
    """
    if not common:
        common = in_common(m1, m2, system)

    n_columns = len(common)
    if system == 1:
        n_rows = len(m1.poss)
    else:
        n_rows = len(m1.full_poss)
    poss_1 = np.zeros((n_rows, n_columns), dtype=int)
    for i, cl in enumerate(common):
        if system == 1:
            poss_1[:, i] = m1.get_column(cl[0])
        else:
            poss_1[:, i] = m1.full_get_column(cl[0])

    n_columns = len(common)
    if system == 1:
        n_rows = len(m2.poss)
    else:
        n_rows = len(m2.full_poss)
    poss_2 = np.zeros((n_rows, n_columns), dtype=int)
    for i, cl in enumerate(common):
        if system == 1:
            poss_2[:, i] = m2.get_column(cl[0])
        else:
            poss_2[:, i] = m2.full_get_column(cl[0])

    if not keep_duplicates:
        poss_1 = remove_duplicates(poss_1)
        poss_2 = remove_duplicates(poss_2)

    return (poss_1, poss_2)


def matching_poss(poss_1, poss_2):
    """Count how many rows the possibilities have in common.

    Arguments:
        poss_1 {np.array} -- possibilities 1
        poss_2 {np.array} -- possibilities 2

    Returns:
        int -- the count/matches
    """
    matches = 0
    for row_2 in poss_2:
        for row_1 in poss_1:
            if np.array_equal(row_1, row_2):
                matches += 1
    return matches

def verify(premises, evidence, system=1):
    """Verify premisses given the evidence.

    Arguments:
        premises {list} -- list of assertion strings
        evidence {list} -- list of assertion strings

    Keyword Arguments:
        system {int} -- system 1 or 2 (default: {1})

    Raises:
        NotImplementedError
        Exception: invalid system

    Returns:
        bool -- True/False
        str  -- Undetermined/Possibly True
    """
    logging("Given evidence '" +  evidence + "', verify premisses '" + str(premises) + "' (system " + str(system) + ")")
    p = model(premises)
    e = model(evidence)
    common = in_common(p, e, system)
    if system == 1:
        if len(common) != len(e.clauses):
            logging("Evidence lacks information in premises")
            return "Undetermined"
        else:
            poss_1, poss_2 = poss_in_common(p, e, system, common, False)
            matches = matching_poss(poss_1, poss_2)
            neg_p = not_model(p)
            neg_poss_1, neg_poss_2 = poss_in_common(neg_p, e, system, in_common(neg_p, e), False)
            neg_matches = matching_poss(neg_poss_1, neg_poss_2)
            if neg_matches and not matches:
                return False
            elif neg_matches and matches:
                return "Undetermined"
            elif not neg_matches and matches:
                return True
            else:
                return "Undetermined"

            # if all and only those poss in premisses are supported by evidence, then true
            # if all poss in premisses are supported by evidence but evidence has more models, then undetermined
            # if not all poss in premisses are supported by evidence, then false
    elif system == 2:
        if len(common) != len(e.full_clauses):
            logging("Evidence lacks information in premises")
            return "Undetermined"
        else:
            poss_1, poss_2 = poss_in_common(p, e, system, common, False)

            matches = 0
            for row_2 in poss_2:
                for row_1 in poss_1:
                    if np.array_equal(row_1, row_2):
                        matches += 1

            # if all and only those poss in premisses are supported by evidence, then true
            if matches == len(poss_1) and len(poss_1) == len(poss_2):
                return True
            elif matches == len(poss_1):
                return "Undetermined"
            # if some evidence supports some premisses, then possibly true
            elif matches > 0:
                return "Possibly True"
            elif matches == 0:
                return False
            else:
                raise NotImplementedError
    else:
        raise Exception

def in_common(m1, m2, system=1):
    """Return clauses in common

    Arguments:
        m1 {MentalModel} -- model 1
        m2 {MentalModel} -- model 2

    Keyword Arguments:
        system {int} -- system 1 or 2 (default: {1})

    Raises:
        Exception: if system not 1 or 2

    Returns:
        tuple -- (clause in common, index in model 1, index in model 2)
    """
    if system == 1:
        clauses_1 = m1.clauses
        clauses_2 = m2.clauses
    elif system == 2:
        clauses_1 = m1.full_clauses
        clauses_2 = m2.full_clauses
    else:
        raise Exception
    return [
        (cl1, i1, i2)
        for i1, cl1 in enumerate(clauses_1)
        for i2, cl2 in enumerate(clauses_2)
        if cl1 == cl2]



def necessary(premises, conclusion, system=1, weak=False):
    """Is conclusion necessary given the premises?

    Arguments:
        premises {list} -- list of premise strings
        conclusion {str} -- conclusion string

    Keyword Arguments:
        system {int} -- system 1 or 2 (default: {1})
        weak {bool} -- weak necessity (default: {False})

    Raises:
        Exception: if not system 1 or 2

    Returns:
        bool -- yes or no
    """
    m1 = model(premises)
    m2 = model(conclusion)
    common = in_common(m1, m2, system)

    if not common:
        return False

    poss_1, poss_2 = poss_in_common(m1, m2, system, common, False)

    matches = 0
    for row_2 in poss_2:
        for row_1 in poss_1:
            if np.array_equal(row_1, row_2):
                matches += 1

    if matches != len(poss_2):
        return False
    elif matches == len(poss_1):
        return True
    elif weak and matches < len(poss_1):
        return True
    else:
        return False


def possible(premises, conclusion, system=1):
    """Is conclusion possible given the premises?

    Arguments:
        premises {list} -- list of premise strings
        conclusion {str} -- conclusion string

    Keyword Arguments:
        system {int} -- system 1 or 2 (default: {1})

    Raises:
        Exception: if not system 1 or 2

    Returns:
        bool -- yes or no
    """
    m1 = model(premises)
    m2 = model(conclusion)
    common = in_common(m1, m2, system)

    if not common:
        return False

    poss_1, poss_2 = poss_in_common(m1, m2, system, common, False)

    matches = 0
    for row_2 in poss_2:
        for row_1 in poss_1:
            if np.array_equal(row_1, row_2):
                matches += 1

    if matches == 0:
        return False
    elif matches == len(poss_2) and matches == len(poss_1):
        return True
    elif matches == len(poss_2) and matches != len(poss_1):
        return True
    elif matches != len(poss_2) and matches == len(poss_1):
        return True
    elif matches != len(poss_2) and matches != len(poss_1):
        return True

def defeasance(premises, fact, system=1):
    """Revise premises given the fact.

    Arguments:
        premises {list} -- list of assertion strings
        fact {str} -- assertion string

    Keyword Arguments:
        system {int} -- system 1 or 2 (default: {1})

    Returns:
        MentalModel -- revised model of premisses
    """
    fact_model = model(fact)
    premisses_models = [model(p) for p in premises]
    keep = []
    reject = []
    not_in_common = []
    for i, m in enumerate(premisses_models):
        common = in_common(m, fact_model, system)
        if common:
            if part_of_model(m, fact_model, common, system):
                logging("fact model MATCHES premisse model")
                keep.append(premises[i])
            else:
                logging("fact model MISMATCHES premisse model")
                reject.append(premises[i])
        else:
            not_in_common.append(premises[i])
    logging("premisses to reject:")
    for p in reject:
        logging(p)
    logging("premisses to keep:")
    for p in keep:
        logging(p)
    logging("premisses not in common:")
    for p in not_in_common:
        logging(p)

    logging("new model that needs explaining:")
    if reject:
        keep.extend(not_in_common)
        keep.append(fact)
        new_model = model(keep)
        logging(new_model)
    else:
        keep.append(fact)
        new_model = model(keep)
        logging(new_model)


    new_model = match_knowledge(new_model, system)

    return new_model


def part_of_model(m, fact_model, common, system=1):
    """Check if a model is part of another model.

    If all possibility rows of the fact model are also part of the other model
    then return True, else False.

    Arguments:
        m {MentalModel} -- model
        fact_model {MentalModel} -- fact model
        common {(str,int,int)} -- clauses in common with indices

    Keyword Arguments:
        system {int} -- system 1 or 2 (default: {1})

    Returns:
        bool -- True if fact is part, else False
    """
    # if all rows of fact are in a model then return True, else False
    poss_1, poss_2 = poss_in_common(m, fact_model, system, common)

    for p2 in poss_2:
        match = False
        for p1 in poss_1:
            if np.array_equal(p1, p2):
                match = True
        if not match:
            return False
    return True


def match_knowledge(m, system=1):
    """Return knowledge model if it matches the model, else return back the model

    Arguments:
        m {MentalModel} -- the model

    Keyword Arguments:
        system {int} -- system 1 or 2 (default: {1})

    Returns:
        MentalModel -- either the matching knowledge model or the unchanged input model
    """
    knowledge = []
    # knowledge.append(model(['a poisonous snake bites her & she dies']))
    # knowledge.append(model(['~a poisonous snake bites her & ~she dies']))
    knowledge.append(model(['a poisonous snake bites her & she takes antidote & ~ she dies']))
    knowledge.append(model(['~a poisonous snake bites her & the snake has a weak jaw & ~ she dies']))

    for k in knowledge:
        common = in_common(k, m, system)
        if part_of_model(k, m, common, system):
            logging("knowledge did match")
            # print(k)
            return k
    logging("knowledge did not match")
    return m




    # (((  a-poisonous-snake-bites-her)                                   (  she-dies))
    #  ((- a-poisonous-snake-bites-her)                                   (- she-dies))
    #  ((- a-poisonous-snake-bites-her)(the-snake-has-a-weak-jaw)         (- she-dies))
    #  ((- a-poisonous-snake-bites her)(the-snake-is-blind)               (- she-dies)))

    # (((  a-poisonous-snake-bites-her) (she-takes-antidote)              (- she-dies))
    #  ((  a-poisonous-snake-bites-her) (the-tourniquet-blocks-the-poison)(- she-dies))
    #  ((  a-poisonous-snake-bites-her) (someone-sucks-out-the-poison)    (- she-dies))
    #  ((  a-poisonous-snake-bites her) (its-venom-lacks-potency)         (- she-dies)))

    # (((  she-anticipates-bite)        (  she-takes-antidote))
    #  ((- she-anticipates-bite)        (- she-takes-antidote)))

    # (((  she-uses-a-tourniquet)       (  the-tourniquet-blocks-the-poison))
    #  ((- she-uses-a-tourniquet)       (- the-tourniquet-blocks-the-poison)))

    # (((  someone-knows-what-to-do)    (  someone-sucks-out-the-poison))
    #  ((- someone-knows-what-to-do)    (- someone-sucks-out-the-poison)))

    # (((  the-snake-has-a-disease)     (  its-venom-lacks-potency))
    #  ((- the-snake-has-a-disease)     (- its-venom-lacks-potency)))

    # (((  the-snake-is-tired)          (  the-snake-has-a-weak-jaw))
    #  ((- the-snake-is-tired)          (- the-snake-has-a-weak-jaw)))

    # (((  the-snake-is-diseased)       (  the-snake-is-blind))
    #  ((- the-snake-is-diseased)       (- the-snake-is-blind)))

def original_mSentential():
    # Examples from the original lisp program:
    #     (inference '((if a or b then c)(a)))
    #     (inference '((God exists or atheism is right)))
    #     (inference '((if a or b then c)(a)) 'what-follows?)
    #     (inference '((a)(a or b)) 'necessary?)
    #     (inference '((if a then b)(not b)(not a)) 'necessary?)
    #     (inference '((if a poisonous snake bites her then she dies)(A poisonous snake bites her)(not she dies)) 'necessary?)
    #     (inference '((a)(a or b)) 'possible?)
    #     (inference '((it is hot or it is humid)(it is hot)) 'probability?)
    #     (inference '((if a then b)(not a and not b)) 'verify?)
    print("model(['(a | b) -> c', 'a'])")
    print(model(['(a | b) -> c', 'a']))
    print()
    print()


    print("model(['God exists | atheism is right'])")
    print(model(['God exists | atheism is right']))
    print()
    print()

    print("what_follows(['a | b -> c', 'a'])")
    print(what_follows(['a | b -> c', 'a']))
    print()
    print()

    print("what_follows(['a | b -> c', 'a'], 2)")
    print(what_follows(['a | b -> c', 'a'], 2))
    print()
    print()

    print("necessary(['a'], 'a|b')")
    print(necessary(['a'], 'a|b'))
    print()
    print()

    print("necessary(['a'], 'a|b', 2)")
    print(necessary(['a'], 'a|b', 2))
    print()
    print()

    print("necessary(['a -> b', '~b'], '~a')")
    print(necessary(['a -> b', '~b'], '~a'))
    print()
    print()

    print("necessary(['a -> b', '~b'], '~a', 2)")
    print(necessary(['a -> b', '~b'], '~a', 2))
    print()
    print()

    print("necessary(['a poisonous snake bites her -> she dies', 'a poisonous snake bites her'], '~she dies')")
    print(necessary(['a poisonous snake bites her -> she dies', 'a poisonous snake bites her'], '~she dies'))
    print()
    print()

    print("necessary(['a poisonous snake bites her -> she dies', 'a poisonous snake bites her'], '~she dies', 2)")
    print(necessary(['a poisonous snake bites her -> she dies', 'a poisonous snake bites her'], '~she dies', 2))
    print()
    print()

    print("possible(['a'], 'a|b')")
    print(possible(['a'], 'a|b'))
    print()
    print()

    print("possible(['a'], 'a|b', 2)")
    print(possible(['a'], 'a|b', 2))
    print()
    print()

    print("probability(['it is hot | it is humid'], 'it is hot')")
    print(probability(['it is hot | it is humid'], 'it is hot'))
    print()
    print()

    print("probability(['it is hot | it is humid'], 'it is hot', 2)")
    print(probability(['it is hot | it is humid'], 'it is hot', 2))
    print()
    print()

    print("verify(['a -> b'], '~a & ~b')")
    print(verify(['a -> b'], '~a & ~b'))
    print()
    print()

    print("verify(['a -> b'], '~a & ~b', 2)")
    print(verify(['a -> b'], '~a & ~b', 2))
    print()
    print()


def weak_necessity():
    # weak necessity
    print("necessary(['a|b'], 'a^b', weak=False)")
    print(necessary(['a|b'], 'a^b', weak=False))
    print()
    print()

    print("necessary(['a|b'], 'a^b', 2, weak=False)")
    print(necessary(['a|b'], 'a^b', 2, weak=False))
    print()
    print()

    print("necessary(['a|b'], 'a^b', weak=True)")
    print(necessary(['a|b'], 'a^b', weak=True))
    print()
    print()

    print("necessary(['a|b'], 'a^b', 2, weak=True)")
    print(necessary(['a|b'], 'a^b', 2, weak=True))
    print()
    print()


def from_paper():

    ### New tests
    print("possible('trump | ~trump', '~trump')")
    print(possible('trump | ~trump', '~trump'))
    print()
    print()


    print("how_possible('<e:0.9> snow', 'snow', 2)")
    print(how_possible('<e:0.9> snow', 'snow', 2))
    print()
    print()

    print("possible('<>pat & <>~viv', 'pat & ~viv')")
    print(possible('<>pat & <>~viv', 'pat & ~viv'))
    print()
    print()

    print("model('<>(Ivanka | Jared)')")
    print(model('<>(Ivanka | Jared)'))
    print()
    print()

    print("probability('<e:0.9> snow', 'snow', 2)")
    print(probability('<e:0.9> snow', 'snow', 2))
    print()
    print()


    print("how_possible('<>pat & <>~viv', 'pat & ~viv', 2)")
    print(how_possible('<>pat & <>~viv', 'pat & ~viv', 2))
    print()
    print()

    print("model('pie ^ cake', 'pie ^ ~cake')")
    print(model(['pie ^ cake', 'pie ^ ~cake']))
    print()
    print()

    print("model(['<>A', 'A->B'])")
    print(model(['<>A', 'A->B']))
    print()
    print()

    print("necessary(['cold & (snowing ^ raining)'], 'snowing ^ raining', 2)")
    print(necessary(['cold & (snowing ^ raining)'], 'snowing ^ raining', 2))
    print()
    print()

    print("model(['canal -> [a] flooding'])")
    print(model(['canal -> [a] flooding']))
    print()
    print()

    print("model(['canal -> <a> flooding'])")
    print(model(['canal -> <a> flooding']))
    print()
    print()

    print("model(['children -> [d] taking care', 'taking care -> [d] ~leaving'])")
    print(model(['children -> [d] taking care', 'taking care -> [d] ~leaving']))
    print()
    print()

    print("what_follows(['children -> [d] taking care', 'taking care -> [d] ~leaving'])")
    print(what_follows(['children -> [d] taking care', 'taking care -> [d] ~leaving']))
    print()
    print()

    print("model(['[d] (children -> taking care)', '[d] (taking care -> ~leaving)'])")
    print(model(['[d] (children -> taking care)', '[d] (taking care -> ~leaving)']))
    print()
    print()

    print("what_follows(['[d] (children -> taking care)', '[d] (taking care -> ~leaving)'], 2)")
    print(what_follows(['[d] (children -> taking care)', '[d] (taking care -> ~leaving)'], 2))
    print()
    print()

def open_questions():

    print("model(['[d] ich sage immer die wahrheit', '~ich sage immer die wahrheit'])")
    print(model(['[d] ich sage immer die wahrheit', '~ich sage immer die wahrheit']))
    print()
    print()

    print("model(['[e] ich sage immer die wahrheit', '~ich sage immer die wahrheit'])")
    print(model(['[e] ich sage immer die wahrheit', '~ich sage immer die wahrheit']))
    print()
    print()

    print("model(['[a] ich sage immer die wahrheit', '~ich sage immer die wahrheit'])")
    print(model(['[a] ich sage immer die wahrheit', '~ich sage immer die wahrheit']))
    print()
    print()

    print("model(['~<e:0.9>a'])")
    print(model(['~<e:0.9>a']))
    print()
    print()

    print(model('<e:0.9>snow'))

    print(how_possible('<e:0.9>snow', 'snow'))
    print(how_possible('<e:1>snow', 'snow'))
    print(how_possible('<e:0.5>snow', 'snow'))
    print(how_possible('<e:0.2>snow', 'snow'))
    print(probability('<e:0.2>snow', 'snow', 2))

def testing_defeasance():
    print("defeasance(['a poisonous snake bites her -> she dies', 'a poisonous snake bites her'], '~she dies')")
    print(defeasance(['a poisonous snake bites her -> she dies', 'a poisonous snake bites her'], '~she dies'))

    print("defeasance(['a poisonous snake bites her -> she dies', 'a poisonous snake bites her'], '~she dies', 2)")
    print(defeasance(['a poisonous snake bites her -> she dies', 'a poisonous snake bites her'], '~she dies', 2))

def testing_verify():
    print(verify('a | b', 'a & b'))
    print(verify('a ^ b', 'a & b'))
    print(verify('a -> b', 'a & b'))
    print(verify('a | b', 'a & b', 2))
    print(verify('a ^ b', 'a & b', 2))
    print(verify('a -> b', 'a & b', 2))

    print(verify('a | b', '~a & ~b'))
    print(verify('a ^ b', '~a & ~b'))
    print(verify('a -> b', '~a & ~b'))
    print(verify('a | b', '~a & ~b', 2))
    print(verify('a ^ b', '~a & ~b', 2))
    print(verify('a -> b', '~a & ~b', 2))

    print(verify('a -> b', 'a & ~b'))
    print(verify('a -> b', 'a & ~b', 2))

    print("######################################################")

    print(verify('a -> b', 'a & b', 2))
    print(verify('a -> b', 'a & ~b', 2))
    print(verify('a -> b', '~a & b', 2))
    print(verify('a -> b', '~a & ~b', 2))

    print(verify('a <-> b', 'a & b', 2))
    print(verify('a <-> b', 'a & ~b', 2))
    print(verify('a <-> b', '~a & b', 2))
    print(verify('a <-> b', '~a & ~b', 2))

    print(verify('a | b', 'a & b', 2))
    print(verify('a | b', 'a & ~b', 2))
    print(verify('a | b', '~a & b', 2))
    print(verify('a | b', '~a & ~b', 2))

    print(verify('a ^ b', 'a & b', 2))
    print(verify('a ^ b', 'a & ~b', 2))
    print(verify('a ^ b', '~a & b', 2))
    print(verify('a ^ b', '~a & ~b', 2))

    print(verify('a ^ b', '~a & ~b', 2))

if __name__ == "__main__":
    print('############################### original examples ####################################\n\n')
    original_mSentential()
    print('############################### weak necessity ######################################\n\n')
    weak_necessity()
    print('############################### examples from paper ######################################\n\n')
    from_paper()
    print('############################### open questions ######################################\n\n')
    open_questions()
    print('############################### TESTING ######################################\n\n')
    testing_defeasance()
    testing_verify()


