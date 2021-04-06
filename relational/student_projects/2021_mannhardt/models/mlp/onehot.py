import numpy as np

import ccobra

allTasks = {
    "RightAB RightCA CABBAC" : 0,
    "LeftABLeftBCABCCBA" : 1,
    "RightABLeftACBACCAB" : 2,
    "LeftABRightCBABCCBA" : 3,
    "RightABRightCABACCAB" : 4,
    "LeftABRightCBCBAABC" : 5,
    "RightABLeftACCABBAC" : 6,
    "LeftABLeftBCCBAABC" : 7,
    "BACRightBC" : 8,
    "ABCLeftCA" : 9,
    "ABCRightCA" : 10,
    "ABCLeftAC" : 11,
    "BACLeftCB" : 12,
    "BACRightCB" : 13,
    "BACLeftBC" : 14,
    "ABCRightAC" : 15,
    "BACLeftCBACBCBA" : 16,
    "ABCLeftCABCACAB" : 17,
    "BACRightBCACBCBA" : 18,
    "BACLeftCBCBAACB" : 19,
    "BACRightBCCBAACB" : 20,
    "ABCLeftCACABBCA" : 21,
    "ABCRightACBCACAB" : 22,
    "ABCRightACCABBCA" : 23,
    "ABCLeftCACABBAC" : 24,
    "ABCRightACBCAACB" : 25
}

allItems = {
    "RightAB" : 0,
    "RightCA" : 1,
    "RightCB" : 2,
    "RightBC" : 3,
    "RightAC" : 4,
    "LeftAB" : 5,
    "LeftBC" : 6,
    "LeftCB" : 7,
    "LeftAC" : 8,
    "LeftCA" : 9,
    "ABC" : 10,
    "ACB" : 11,
    "BAC" : 12,
    "BCA" : 13,
    "CAB" : 14,
    "CBA" : 15
}

allOutputs = {
    "ABC" : 0,
    "ACB" : 1,
    "BAC" : 2,
    "BCA" : 3,
    "CAB" : 4,
    "CBA" : 5,
    True : 6,
    False : 7
}

def onehot_syllogism(syl):
    result = np.zeros((64,), dtype='float')
    result[ccobra.syllogistic.SYLLOGISMS.index(syl)] = 1
    return result

def onehot_syllogism_content(syl):
    """
    >>> onehot_syllogism('AA1')
    array([1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.])
    >>> onehot_syllogism('OI3')
    array([0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0.])
    >>> onehot_syllogism('IE4')
    array([0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.])

    """

    task = np.zeros((12,), dtype='float')
    quants = quants = ['A', 'I', 'E', 'O']
    task[quants.index(syl[0])] = 1
    task[4 + quants.index(syl[1])] = 1
    task[8 + int(syl[2]) - 1] = 1
    return task

def onehot_response(response):
    """
    >>> onehot_response('Aac')
    array([1., 0., 0., 0., 0., 0., 0., 0., 0.])
    >>> onehot_response('NVC')
    array([0., 0., 0., 0., 0., 0., 0., 0., 1.])
    >>> onehot_response('Oca')
    array([0., 0., 0., 0., 0., 0., 0., 1., 0.])

    """

    resp = np.zeros((9,), dtype='float')
    resp[ccobra.syllogistic.RESPONSES.index(response)] = 1
    return resp

def create_input_output(task, seq, response=""):
    """
    task is list of premises/models
    """
    if response == "weiter":
        return None, None
    inp = [0 for i in range(len(allItems)+1)]
    for prem in task:
        inp[allItems[prem]] = 1
        
    if seq == 2:
        inp[-1] = 1

    output = [0 for i in range(len(allOutputs))]
    try:
        output[allOutputs[response]] = 1
    except:
        pass
    return inp, output

def task_to_string(syntaxTree):
    """
    Changes e.g. ["a", "And", ["b", "Or", ["Not", "c"]]] to "a And b Or -c"
    Args:
        syntaxTree
    Returns:
        string - syntaxTree written as string
    """
    if isinstance(syntaxTree[0][0], bool):
        return syntaxTree[0][0]

    if isinstance(syntaxTree, str) or isinstance(syntaxTree, bool):
        return syntaxTree

    all = ""
    for elem in syntaxTree:
        all += task_to_string(elem)
    return all