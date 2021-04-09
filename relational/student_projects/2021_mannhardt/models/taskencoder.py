
'''
inclues helper functions used by some models
'''

import numpy as np
import math

def list_to_string_help(stringList):
    for i in stringList:
        if isinstance(i, str):
            yield i
        elif isinstance(i, list):
            yield list_to_string(i)


def list_to_string(stringList):
    fullStr = ""
    for s in list_to_string_help(stringList):
        fullStr += s
    return fullStr


def keywithmaxval(d):
     """ return the key with the max value
         src: stackoverflow
         """  
     v=list(d.values())
     k=list(d.keys())
     return k[v.index(max(v))]

def task_to_string(syntaxTree):
    """
    Changes e.g. ["Left;A;B", "Right";"C";"B"] to "LeftABRightCB"
    Args:
        task
    Returns:
        string - task written as string
    """
    if isinstance(syntaxTree[0][0], bool):
        return syntaxTree[0][0]

    if isinstance(syntaxTree, str) or isinstance(syntaxTree, bool):
        return syntaxTree

    all = ""
    for elem in syntaxTree:
        all += task_to_string(elem)
    return all


def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    normalized_v1 = v1/np.linalg.norm(v1)
    normalized_v2 = v2/np.linalg.norm(v2)
    return round(math.acos(dotproduct(normalized_v1, normalized_v2) / (length(normalized_v1) * length(normalized_v2))), 5)
