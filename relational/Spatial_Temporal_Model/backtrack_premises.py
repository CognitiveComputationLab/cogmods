'''
Module for searching for the relevant premises regarding a given question.

Created on 16.07.2018

@author: Christian Breu <breuch@web.de>, Julia Mertesdorf<julia.mertesdorf@gmail.com>

Note: This module is only used by the Temporal Model!
'''

import copy

import low_level_functions as helper

import parser_spatial_temporal as parser


PRINT_BACKTRACK = False


# ---------------------------- SEARCHING FOR RELEVANT PREMISES - WHEN CAPACITY IS EXCEEDED --------

# ONLY USED BY TEMPORAL MODEL
def work_back(premisses):
    """
    This function is called when the capacity is exceeded to just search
    for the relevant premises in order to solve a simplified version of the problem.
    Function extracts the relevant subject and object of the question and
    calls p_search with it in order to find a path from the subject of the question
    to the object of the question.
    p_search returns all premises which are necessary to solve the problem.
    The question is added to this new premise set and interpret is called again
    with the reduced problem.
    Interpret should now be able to construct a model out of the problem that does
    not exceed the capacity again.
    """
    if PRINT_BACKTRACK:
        print("Function-call: Work_back")
    rev_prem = premisses[::-1]
    question = rev_prem[0]
    premises = premisses[:-1]
    subj = [[question[1]]]
    obj = [question[2]]
    print("work-back: Memory capacity exceeded! Examining only premises relevant to"
          " question: Relation between", subj[0], "and", obj)
    search_obj = p_search(subj, obj, premises)
    if PRINT_BACKTRACK:
        print("work-back: search resulted in: ", search_obj)
    search_obj.append(question)
    print("work_back: Search successfull! Try to construct model again with prems:", search_obj)
    return search_obj

# ONLY USED BY TEMPORAL MODEL
def p_search(paths, goal, prems, output=None):
    """
    Function starts breadth-first search for path(s) which lead from the initial member
    of the path until the goal, using update to find the relevant premises to continue
    the paths.
    If the goal was found and there were no more paths after calling "pSearch" recursively
    again, call longest_path to decide which path to take to go from start to goal
    element (longest path is chosen because all premises that do not contain extraneous
    referents are relevant to the temporal inference). After "longest_path" was called,
    rec_prems is called which extracts all relevant premises that are used in the longest
    path. These premises are returned to work_back.
    """
    if PRINT_BACKTRACK:
        print("Function-call: pSearch with paths", paths, "goal", goal, "output", output)
    if not paths:
        if PRINT_BACKTRACK:
            print("pSearch: no paths! Return premises for longest path!")
        return rec_prems(longest_path(output), prems)
    tmp = paths[0]
    if tmp[len(tmp)-1] == goal[0]:
        if PRINT_BACKTRACK:
            print("pSearch: FOUND GOAL!!!!!!!!!")
        path_tmp = [paths[0], output]
        return p_search(paths[1:], goal, prems, path_tmp)
    if PRINT_BACKTRACK:
        print("pSearch: current paths:", paths, "--> update path", paths[0])
    new_paths = p_update(paths[0], prems)
    if new_paths:
        new_path_tmp = new_paths
        for path in paths[1:]:
            new_path_tmp.append(path)
        return p_search(new_path_tmp, goal, prems, output)
    if PRINT_BACKTRACK:
        print("pSearch: else, pop first element and check next path")
    return p_search(paths[1:], goal, prems, output)

# ONLY USED BY TEMPORAL MODEL
def p_update(path, prems):
    """
    Function that updates the current path with the next found and fitting premises.
    First, the function calls prem_lis in order to find all premises, which contain
    the current last element of "path". These premises are saved in "outlis".
    In the next step, check for each premise in "outlis", whether the other
    element of the premise (which was not the last element in path), is already
    a member of the path. All of these elements, which are already part of the path,
    are saved in a list, which is afterwards subtracted from "outlis", which
    means that only those premises remain in outlis, which contain elements that
    are not already a part of the path.
    Afterwards, the function creates a list (here tmp_list2), where all next
    potential path-elements are appended.
    In the fourth step of the function, each of the potential following path
    elments is added after the current last path element and with that,
    creates a new (or several new) path(s). The new path(s) are returned to pSearch.

    Example:
    path = ["A", "B"] and premises "A happens before B", "B happens before C",
    "C happens before D" --> function returns new path = ["A","B","C"]
    """
    if PRINT_BACKTRACK:
        print("Function call - p_update with path", path)
    elm = path
    if len(path) > 1:
        elm = path[len(path)-1]
    outlis = prem_lis(elm, prems)
    if PRINT_BACKTRACK:
        print("p-update: All important premises for current last path is:", outlis)

    tmp_list = []
    last_path = path[len(path)-1]
    for prem in outlis:
        other_ref_el = other_ref_than(last_path, prem)
        mem = member(other_ref_el, path)
        if mem:
            tmp_list.append(mem)
    if tmp_list:
        for prem in outlis:
            for elm2 in tmp_list:
                if joint_refers(elm2[0], elm2[1], prem):
                    outlis.remove(prem)
    if PRINT_BACKTRACK:
        print("p-update: outlis after deletion of irrelevant premises is:", outlis)

    tmp_list2 = []
    for prem in outlis:
        other_ref_el = other_ref_than(last_path, prem)
        if other_ref_el != None:
            tmp_list2.append(other_ref_el)

    tmp_list3 = []
    for item in tmp_list2:
        if item[0] not in path:
            new_path = copy.deepcopy(path)
            new_path.append(item[0])
            tmp_list3.append(new_path)
    if PRINT_BACKTRACK:
        print("p-update: New paths are:", tmp_list3)
    return tmp_list3

# ONLY USED BY TEMPORAL MODEL
def member(item, lis):
    """
    Function, which is a built-in function in Lisp.
    For a given list and item, check if the item is in the list, and return the
    list beginning with the found item and cut off the rest of the initial list
    until that item was found. Checks only for the first appearance of item
    in list.
    Example:
    member("a", ["i", "h", "a", "e"]) returns ["a", "e"].
    """
    if PRINT_BACKTRACK:
        print("function call - member with item", item, "and list", lis)
    return_list = []
    copy_lis = copy.deepcopy(lis)
    if isinstance(item, list):
        item = item[0]
    while copy_lis:
        if copy_lis[0] != item:
            del copy_lis[0]
        else:
            for elm in copy_lis:
                return_list.append(elm)
            if PRINT_BACKTRACK:
                print("member: return the list", return_list)
            return return_list
    if PRINT_BACKTRACK:
        print("member: no members found! Return the list empty!")
    return return_list

# ONLY USED BY TEMPORAL MODEL
def rec_prems(path, prems):
    """
    Function returns the premises corresponding to path.
    Example: For path = ["A", "B", "C"], returns the premises which are
    needed to construct that path (like "A happens before B", "B happens before C").
    """
    if not path[1:]:
        if PRINT_BACKTRACK:
            print("rec_prems: Paths are empty")
        return None
    prem_list = []
    for i in enumerate(path):
        if i[0]+1 <= len(path)-1:
            tmp = find_one([path[i[0]]], [path[i[0]+1]], prems)
            prem_list.append(tmp)
    if PRINT_BACKTRACK:
        print("rec_prems: Premises corresponding to path", path, "are", prem_list)
    return prem_list

# ONLY USED BY TEMPORAL MODEL
def longest_path(paths, long=None):
    """
    Function returns the longest path in paths.
    Recursive function, long is always the currently longest found path.
    """
    if PRINT_BACKTRACK:
        print("Function call - longest_path with paths", paths, "long:", long)
    if not paths:
        return long
    if ((long is None) or (long == [None])):
        len_long = 0
    else: len_long = len(long)

    if ((paths[0] is None) or(paths[0] == [None])):
        len_path = 0
    else: len_path = len(paths[0])

    if len_path > len_long:
        return longest_path(paths[1:], paths[0])
    return longest_path(paths[1:], long)

# ONLY USED BY TEMPORAL MODEL
def find_one(el1, el2, prems):
    """
    Function returns the first premise in prems, that contains both item el1 and
    item el2, which are both lists (like ["A"]). Otherwise returns None.
    """
    if PRINT_BACKTRACK:
        print("Function call - find_one with e1", el1, "e2", el2, "and prems", prems)
    if not prems:
        return None
    for prem in prems:
        if joint_refers(el1, el2, prem):
            if PRINT_BACKTRACK:
                print("Find_one: first premise to refer both items is: ", prem)
            return prem
    if PRINT_BACKTRACK:
        print("Find_one: Couldn´t find premise which refers both items. Return None")
    return None

# ONLY USED BY TEMPORAL MODEL
def joint_refers(item1, item2, prem):
    """
    Function returns true, if item 1 and item 2 are both referred to by the
    premise prem.
    """
    tmp = (refers(item1, prem)) and (refers(item2, prem))
    if PRINT_BACKTRACK:
        print("joint_refers: Both items", item1, item2, "referred to in prem", prem, "?", tmp)
    return tmp

# ONLY USED BY TEMPORAL MODEL
def other_ref_than(term, prem):
    """
    Function returns reference from premise other than the given term, and None
    if the Term doesn´t occur in the premise.
    Example: For prem "A happens before B" and term ["B"], return ["A"].
    """
    if PRINT_BACKTRACK:
        print("function call - other_ref_than with term", term, "and prem", prem)
    prop = parser.Parser(False).parse(prem)
    subj = [helper.get_subject(prop)]
    obj = [helper.get_object(prop)]
    if not isinstance(term, list):
        term = [term]
    if term == subj:
        return obj
    if term == obj:
        return subj
    else:
        if PRINT_BACKTRACK:
            print("other_ref_than: Nothing found, return None")
        return None

# ONLY USED BY TEMPORAL MODEL
def prem_lis(term, prems):
    """
    Function creates a list of all premises containing references to
    the specified term (like ["A"]).
    """
    if PRINT_BACKTRACK:
        print("Function call - prem_lis with term", term)
    if not prems:
        return None
    prem_list = []
    for prem in prems:
        if refers(term, prem):
            prem_list.append(prem)
    if PRINT_BACKTRACK:
        print("prem_lis: Return List with premises:", prem_list)
    return prem_list

# ONLY USED BY TEMPORAL MODEL
def refers(item, prem):
    """
    Function returns True if item, which is a list (like ["A"]), is a
    subject or object of the premise, otherwise False.
    """
    prop = parser.Parser(False).parse(prem)
    subj = [helper.get_subject(prop)]
    obj = [helper.get_object(prop)]
    if not isinstance(item, list):
        item = [item]
    if ((item == subj) or (item == obj)):
        return True
    return False
