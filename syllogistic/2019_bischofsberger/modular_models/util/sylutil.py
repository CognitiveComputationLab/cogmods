import ccobra.syllogistic
import ccobra
import os
import json


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


# ---- Decorator for storing and retrieving function results to file ---- #

# https://stackoverflow.com/questions/16463582/memoize-to-disk-python-persistent-memoization
# makes no difference between different function parameters!
def persistent_memoize(file_name):

    def decorator(original_func):

        def new_func(param):
            try:
                cache = json.load(open(file_name, 'r'))
            except (IOError, ValueError):
                cache = {}

            if cache == {}:
                cache = original_func(param)
                json.dump(cache, open(file_name, 'w'))
            return cache

        return new_func

    return decorator

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


# ---- GENERAL FUNCTIONS ---- #

def uniquify_keep_order(l):
    """ uniquify list keeping elements in order

    >>> uniquify_keep_order('abracadabra')
    ['a', 'b', 'r', 'c', 'd']
    """
    return [l[i] for i in range(len(l)) if l.index(l[i]) == i]


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
