import os
import sys

import ccobra

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")))
from modular_models.models.basic_models.verbal_models import VerbalModels


# ---- GENERIC SYLLOGISMS FOR CCOBRA ---- #

GENERIC_TASKS = [
                 "All;x;y/All;y;z",
                 "All;y;x/All;z;y",
                 "All;x;y/All;z;y",
                 "All;y;x/All;y;z",
                 "All;x;y/Some;y;z",
                 "All;y;x/Some;z;y",
                 "All;x;y/Some;z;y",
                 "All;y;x/Some;y;z",
                 "All;x;y/No;y;z",
                 "All;y;x/No;z;y",
                 "All;x;y/No;z;y",
                 "All;y;x/No;y;z",
                 "All;x;y/Some not;y;z",
                 "All;y;x/Some not;z;y",
                 "All;x;y/Some not;z;y",
                 "All;y;x/Some not;y;z",

                 "Some;x;y/All;y;z",
                 "Some;y;x/All;z;y",
                 "Some;x;y/All;z;y",
                 "Some;y;x/All;y;z",
                 "Some;x;y/Some;y;z",
                 "Some;y;x/Some;z;y",
                 "Some;x;y/Some;z;y",
                 "Some;y;x/Some;y;z",
                 "Some;x;y/No;y;z",
                 "Some;y;x/No;z;y",
                 "Some;x;y/No;z;y",
                 "Some;y;x/No;y;z",
                 "Some;x;y/Some not;y;z",
                 "Some;y;x/Some not;z;y",
                 "Some;x;y/Some not;z;y",
                 "Some;y;x/Some not;y;z",

                 "No;x;y/All;y;z",
                 "No;y;x/All;z;y",
                 "No;x;y/All;z;y",
                 "No;y;x/All;y;z",
                 "No;x;y/Some;y;z",
                 "No;y;x/Some;z;y",
                 "No;x;y/Some;z;y",
                 "No;y;x/Some;y;z",
                 "No;x;y/No;y;z",
                 "No;y;x/No;z;y",
                 "No;x;y/No;z;y",
                 "No;y;x/No;y;z",
                 "No;x;y/Some not;y;z",
                 "No;y;x/Some not;z;y",
                 "No;x;y/Some not;z;y",
                 "No;y;x/Some not;y;z",

                 "Some not;x;y/All;y;z",
                 "Some not;y;x/All;z;y",
                 "Some not;x;y/All;z;y",
                 "Some not;y;x/All;y;z",
                 "Some not;x;y/Some;y;z",
                 "Some not;y;x/Some;z;y",
                 "Some not;x;y/Some;z;y",
                 "Some not;y;x/Some;y;z",
                 "Some not;x;y/No;y;z",
                 "Some not;y;x/No;z;y",
                 "Some not;x;y/No;z;y",
                 "Some not;y;x/No;y;z",
                 "Some not;x;y/Some not;y;z",
                 "Some not;y;x/Some not;z;y",
                 "Some not;x;y/Some not;z;y",
                 "Some not;y;x/Some not;y;z"
                ]
GENERIC_CHOICES = "All/x/z|All/z/x|Some/x/z|Some/z/x|Some not/x/z|Some not/z/x|No/x/z|No/z/x|NVC"
GENERIC_ITEMS = [ccobra.Item(0, "syllogistic", task, "single-choice", GENERIC_CHOICES) for task in GENERIC_TASKS]


# ---- SYLLOGISTIC FUNCTIONS ---- #

def add_implicatures(conclusions, existential=True, gricean=True):
    """
    existential: A -> I, E -> O
    gricean:     I -> O, O -> I

    >>> add_implicatures(["Aac", "Aca"], gricean=False)
    ['Aac', 'Aca', 'Iac', 'Ica']
    >>> add_implicatures(["Aac", "Eac", "NVC"], gricean=False)
    ['Aac', 'Eac', 'NVC', 'Iac', 'Oac']
    >>> add_implicatures(["Ica", "Iac", "Oca", "NVC", "Oac"], gricean=False)
    ['Ica', 'Iac', 'Oca', 'NVC', 'Oac']
    """
    added_conclusions = []
    for conclusion in conclusions:
        if existential:
            if conclusion[0] == "A":
                added_conclusions.append("I" + conclusion[1:])
            elif conclusion[0] == "E":
                added_conclusions.append("O" + conclusion[1:])
        if gricean:
            if conclusion[0] == "I":
                added_conclusions.append("O" + conclusion[1:])
            elif conclusion[0] == "O":
                added_conclusions.append("I" + conclusion[1:])
    return conclusions + added_conclusions


def term_order(figure):
    """ Return the order of terms a,b,c in a syllogism by figure

    >>> term_order("1")
    ['ab', 'bc']
    """

    if figure == "1":
        return ["ab", "bc"]
    if figure == "2":
        return ["ba", "cb"]
    if figure == "3":
        return ["ab", "cb"]
    if figure == "4":
        return ["ba", "bc"]


def premises_to_syllogism(premises):
    """
    >>> premises_to_syllogism(["Aab", "Ebc"])
    'AE1'
    """

    figure = {"abbc": "1", "bacb": "2", "abcb": "3", "babc": "4"}[premises[0][1:] + premises[1][1:]]
    return premises[0][0] + premises[1][0] + figure


def syllogism_to_premises(syllogism):
    """
    >>> syllogism_to_premises("AA1")
    ['Aab', 'Abc']
    """
    return [syllogism[i] + term_order(syllogism[2])[i] for i in [0, 1]]


def syllogism_to_item(syllogism):
    """
    >>> syllogism_to_item("AA1").task
    [['All', 'x', 'y'], ['All', 'y', 'z']]
    >>> syllogism_to_item("EO4").task
    [['No', 'y', 'x'], ['Some not', 'y', 'z']]
    >>> syllogism_to_item("II2").task
    [['Some', 'y', 'x'], ['Some', 'z', 'y']]
    """

    q1 = {"A": "All", "I": "Some", "E": "No", "O": "Some not"}[syllogism[0]]
    q2 = {"A": "All", "I": "Some", "E": "No", "O": "Some not"}[syllogism[1]]
    to = {"1": ["x;y", "y;z"], "2": ["y;x", "z;y"], "3": ["x;y", "z;y"], "4": ["y;x", "y;z"]}[syllogism[2]]

    # task = e.g. "All;x;y/All;y;z",
    task = q1 + ";" + to[0] + "/" + q2 + ";" + to[1]
    return ccobra.Item(0, "syllogistic", task, "single-choice", GENERIC_CHOICES)


def encode_proposition(prop, item):
    """
    >>> encode_proposition(["All", "x", "y"], GENERIC_ITEMS[0])
    'Aab'
    """
    quantor = {"All": "A", "Some": "I", "No": "E", "Some not": "O"}[prop[0]]
    middle_term = list(set(item.task[0][1:]).intersection(item.task[1][1:]))[0]
    a_term = list(set(item.task[0][1:]) - set(middle_term))[0]

    i = prop[1:].index(middle_term)
    end_term = "a" if a_term in prop else "c"

    pr_enc = quantor + "b" + end_term if i == 0 else quantor + end_term + "b"
    return pr_enc


# ---- GENERAL FUNCTIONS ---- #

def uniquify_keep_order(l):
    """ uniquify list keeping elements in order

    >>> uniquify_keep_order('abracadabra')
    ['a', 'b', 'r', 'c', 'd']
    """
    return [l[i] for i in range(len(l)) if l.index(l[i]) == i]


t_mm = -1
def get_time():
    """ Get "timestamp" via counter """
    global t_mm
    t_mm += 1
    return t_mm


# ---- DATA WRANGLING FUNCTIONS ---- #

def aggregate_data(dataset):
    """ Take full syllogistic dataset and return response statistics per syllogism aggregated over
    all subjects and items.
    """
    # e.g. data["AA1"]["Aac"] = 0
    data = {syl: {} for syl in ccobra.syllogistic.SYLLOGISMS}
    for syl in data:
        for response in ccobra.syllogistic.RESPONSES:
            data[syl][response] = 0

    # Aggregate # of responses per syllogism over all subjects and their items
    for subject in dataset:
        for answer in subject:
            syllogism = ccobra.syllogistic.encode_task(answer["item"].task)
            response = ccobra.syllogistic.encode_response(answer["response"], answer["item"].task)
            data[syllogism][response] += 1

    # Standardize response frequency
    for syllogism in data:
        row_sum = sum([data[syllogism][response] for response in data[syllogism]])
        for response in data[syllogism]:
            num = data[syllogism][response]
            data[syllogism][response] = num / row_sum

    return data


def persubjectify(dataframe):
    """ Convert dataframe to iterable over subjects and their items nested (similar as taken by pre_train)

    :param dataframe: pandas dataframe. Required fields: "id", "domain", "task", "response_type", "choices"
    :return:
    """
    dataset = [[] for i in dataframe["id"].unique()]

    # handle the case where ids start at integers other than 0 (e.g. 1)
    dataframe["id_normalized"] = dataframe["id"] - dataframe["id"].min()

    for i, row in dataframe.iterrows():
        subject_id = row["id_normalized"]
        item = ccobra.Item(identifier=subject_id, domain=row["domain"], task=row["task"], resp_type=row["response_type"],
                           choices=row["choices"])
        response = [row["response"].split(";")]
        dataset[subject_id].append({"item": item, "response": response})
    return dataset


# ---- CONVERSION BETWEEN VERBAL MODELS AND MENTAL MODELS ---- #

def vm_to_mm(verbal_model):
    """ Convert VM to MM. Time information will be lost.

    >>> a = VerbalModels.Prop(name="a", neg=False, identifying=False, t=123)
    >>> non_b = VerbalModels.Prop(name="b", neg=True, identifying=False, t=456)
    >>> vm = [VerbalModels.Individual(props=[a], t=789), VerbalModels.Individual(props=[a, non_b], t=159)]
    >>> vm_to_mm(vm)
    [['a'], ['a', '-b']]
    """

    mental_model = []
    for i, ind in enumerate(verbal_model):
        mental_model.append([])
        for p in ind.props:
            p_str = "-"+p.name if p.neg else p.name
            mental_model[i].append(p_str)
    return [sorted(row, key=lambda e: e[-1]) for row in mental_model]


def mm_to_vm(mental_model):
    """ Convert MM to VM.

    >>> mm_to_vm([['a', 'b'], ['a']])
    [[a(-1), b(-1)](-1), [a(-1)](-1)]
    """

    verbal_model = []
    for row in mental_model:
        props = []
        for p in row:
            neg = True if p[0] == "-" else False
            props.append(VerbalModels.Prop(name=p[-1], neg=neg, identifying=False, t=-1))
        verbal_model.append(VerbalModels.Individual(props=props, t=-1))
    return verbal_model
