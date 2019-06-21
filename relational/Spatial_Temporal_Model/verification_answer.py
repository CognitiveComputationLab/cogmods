'''
Module for methods to verify problems and answer a question. Functions are used
while the model is constructed or after the model construction has finished.

Created on 16.07.2018

@author: Christian Breu <breuch@web.de>, Julia Mertesdorf<julia.mertesdorf@gmail.com>
'''

import low_level_functions as helper

import parser_spatial_temporal as parser

import modify_model as modify


PRINT_VERIFICATION = False


# USED BY BOTH MODELS
def normalize(vector):
    """OK [66]
    Normalizes each element of the given list to 1 for positive values and
    -1 for negative values.
    """
    if PRINT_VERIFICATION:
        print("normalize vector: ", vector)
    for count, value in enumerate(vector):
        if value > 0:
            vector[count] = 1
        elif value < 0:
            vector[count] = -1
        else: vector[count] = 0
    if PRINT_VERIFICATION:
        print("normalized vector: ", vector)
    return vector


# ---------------------------- FUNCTIONS FOR THE SPATIAL MODEL -----------------------------------

# ONLY USED IN SPATIAL
def verify_spatial(proposition, model):
    """
    Extracts the relation, subject and object from the proposition. Then
    searches the subj + obj in the model. Returns the model if the relation
    between the subj and obj is correctly represented in the model. Iterates
    through the relation and checks the corresponding subj and obj coordinates
    for the axis of the relation.
    e.g. for relation[0] = 1, checks if the subj_coords and obj_coords at the
    corresponding index do satisfy the relation.
    If the relation is 0 but the obj and subj coordinates are different,
    verification fails aswell.
    Returns None if the relation does not hold.
    """
    if PRINT_VERIFICATION:
        print("call verify_spatial with prop, model: ", proposition, model)
    relation = helper.get_relation(proposition)
    subj = helper.get_subject(proposition)
    obj = helper.get_object(proposition)
    subj_coords = helper.find_first_item(subj, [model])[0]
    obj_coords = helper.find_first_item(obj, [model])[0]
    if PRINT_VERIFICATION:
        print("verify_spatial: subj_coords, obj_coords, relation",
              subj_coords, obj_coords, relation)
    # iterate through the relation and the coordinates of the objects.
    for index, value in enumerate(relation):
        # if the relation is != 0, check if the relation holds in this axis
        if (value > 0) and (subj_coords[index] <= obj_coords[index]):
            # if the subject coords are < than obj coords in rel axis, relation
            # does not hold!
            if PRINT_VERIFICATION:
                print("verify_spatial: relation does not hold, return None")
            return None
        if (value < 0) and (subj_coords[index] >= obj_coords[index]):
            # the same for the opposite relation, this is the case when verify_temporal fails
            if PRINT_VERIFICATION:
                print("verify_spatial: relation does not hold, return None")
            return None
        if (value == 0) and (subj_coords[index] != obj_coords[index]):
            # the items must not be at the same position!
            if PRINT_VERIFICATION:
                print("verify_spatial: objects are on a different line in another axis ")
            return None
    if PRINT_VERIFICATION:
        print("verify_spatial: succesfully verified, return the model")
    return model

# ONLY USED IN SPATIAL
def conflict(premises, model, spatial_parser):
    """
    Finds all premises that are conflicting with the given model.
    Iterates over premises and parses them each. If the premises can't be
    parsed or the subject and the object are in the model, try to verify_temporal the
    premise(prop) with verify_spatial.
    If it can't be verified, add the premise(prop) to the result list of
    conflicted props. Returns a list of conflicting premises.
    """
    if PRINT_VERIFICATION:
        print("conflict: prems, model: ", premises, model)
    if spatial_parser:
        pars = parser.Parser(True)
    else:
        pars = parser.Parser(False)
    if not premises:
        return None
    result_list = []
    for prem in premises:
        prop = pars.parse(prem)
        subj = helper.get_subject(prop)
        obj = helper.get_object(prop)
        if PRINT_VERIFICATION:
            print("conflict: subj, obj", subj, obj)
        #check if the premise can be parsed(should be always the case)
        # and the subject  and object are in the model.
        # call new_find_item with a list!!!
        if(prop is None) or ((helper.find_first_item(subj, [model]))
                             and (helper.find_first_item(obj, [model]))):
            #if subj + obj are in the model, try to verify_temporal. if verify_temporal
            # returns false, add the proposition to the conflicted props.
            if not verify_spatial(prop, model):
                if PRINT_VERIFICATION:
                    print("conflicted premise in prems: with model", prop, model)
                result_list.append(prop)
    return result_list

# ONLY USED IN SPATIAL
def conflict_props(propositions, model):
    """
    Returns list of conflicted propositions in model. Works similiar to the
    conflict method, but it uses a list of already parsed propositions.
    Also uses verify_spatial to check for conflicted propositions.
    """
    if PRINT_VERIFICATION:
        print("conflict_props with prop, model: ", propositions, model)
    if propositions is None:
        return None
    conflict_list = []
    for prop in propositions:
        if not verify_spatial(prop, model):
            conflict_list.append(prop)
    return conflict_list

# ONLY USED IN SPATIAL
def make(prop_list, fix_props, model, premises, spatial_parser):
    """
    Iterates over the given prop-list and tries to make the props true by
    calling switch. If the resulting model is not None, switch was able to
    create a model in which prop holds. If thats the case, add the prop to
    the fix_props that should always hold. If the result of switch is None,
    return None, it is not possible to create a model with all props = true
    with this prop_list. After each iteration through the prop_list, set the
    prop_list to all the conflicting props in the current model.
    If there are no conflicts, return the model.
    """
    if PRINT_VERIFICATION:
        print("make with prop_list, fix_props, model, premises", prop_list,
              fix_props, model, premises)
    while prop_list:
        #first, iterate over the prop list and call switch on the props
        for prop in prop_list:#for each proposition, call switch and change the model with this
            model = switch(prop, fix_props, model)
            # if switch could make the prop hold in the model, add it to the fix props
            if model != None:
                if PRINT_VERIFICATION:
                    print("make: switch worked, insert prop into fix-props")
                fix_props.insert(0, prop)
            else: return None#returns None, if the model becomes None
        #when all the props are through, check if there are any conflicts in the new model.
        #if there are no conflicts, the loop is over
        prop_list = conflict(premises, model, spatial_parser)
        if PRINT_VERIFICATION:
            print("current prop_list after conflict:", prop_list)
    return model

# ONLY USED IN SPATIAL
def remove_prem(proposition, premises):
    """
    Iterates over all the premises and returns a list of the premises
    without the given proposition(premise). Adds premises to the result
    list if they aren't equal to the proposition.
    """
    if((premises is None) or (not premises)):
        return None#return None if the premises list is empty or None
    result = []
    for prem in premises:
        #add premises != proposition in order to remove proposition from the list.
        if proposition != parser.Parser(True).parse(prem):
            result.append(prem)
    return result

# ONLY USED IN SPATIAL
def make_false(proposition, model, premises, spatial_parser):
    """
    Tries to make the model hold with a negated relation from the premise.
    If this is possible, the proposition is falsified. If not, the premise
    is valid in the model. The original model is returned with a statement.
    """
    if PRINT_VERIFICATION:
        print("make-false with prop: ", proposition)
    prems = remove_prem(proposition, premises)
    neg_prop = negate_prop(proposition)#negate the proposition
    new_mod = make([neg_prop], [neg_prop], model, prems, spatial_parser)
    if (new_mod != None) and (verify_spatial(neg_prop, new_mod)):
        print("Premise was previously possibly true")
        return model
    print("Premise follows validly from previous premises")
    return model

# ONLY USED IN SPATIAL
def make_true(proposition, model, premises, spatial_parser):
    """
    Tries to find a way to make the proposition hold in model.
    Modifies the model in different ways to see if the proposition and all
    the other premises do hold then. If this suceeds, returns the new model.
    Calls make to modify the model.
    """
    if PRINT_VERIFICATION:
        print("make true with premise, model: ", proposition, model)
    prems = remove_prem(proposition, premises)
    new_mod = make([proposition], [proposition], model, prems, spatial_parser)
    if (new_mod != None) and (verify_spatial(proposition, new_mod)):
        print("Premise was previously possibly false")
        return new_mod
    print("Premise is inconsistent with previous premises")
    return model

# ONLY USED IN SPATIAL
def negate_prop(proposition):
    """
    Negates relation-part of the proposition and returns the changed proposition.
    (before --> after, after --> before, while --> while).
    """
    #proposition needs to be List!
    relation = helper.get_relation(proposition)
    proposition[0] = helper.convert(relation)
    if PRINT_VERIFICATION:
        print("negate prop: ", proposition)
    return proposition

# ONLY USED IN SPATIAL
def switch(new_prop, fix_props, model):
    """new version of switch for dictionaries.
    First, tries to swap the object and subject if the relation is the
    opposite of the required relation.(find_rel_prop will return the
    relation between the subj and object.)
    Calls swap with the subject and the object. Checks if the resulting
    model has any conflicts, if not returns it. If there were any conflicts,
    set the new_mod to the result of move with the subject.
    move will change the position of the subject to make the premise true in
    the model. Returns new_mod if conflict-free.
    After that it tries the same thing with moving the object.
    If nothing works, returns None
    """
    if PRINT_VERIFICATION:
        print("switch with new_prop, fixprops, model: ", new_prop, fix_props, model)
    relation = helper.get_relation(new_prop)
    # just use the string of the item
    subj = helper.get_subject(new_prop)
    obj = helper.get_object(new_prop)
    # call new_find_item with a list!!!
    s_coord = helper.find_first_item(subj, [model])[0] # s_coord + o_coord are tuples
    o_coord = helper.find_first_item(obj, [model])[0]# only get the coordinates
    # check if the relation of subj and obj is converse to the relation of the proposition.
    if find_rel_prop(s_coord, o_coord) == helper.convert(relation):
        # only if first condition holds, try to swap the items.
        new_mod = modify.swap(subj, s_coord, obj, o_coord, model)
        if PRINT_VERIFICATION:
            print("switch: model, new_mod after swap:", model, new_mod)
        if new_mod != None:
            # if there are now conflicting props in the new model, return it
            if not conflict_props(fix_props, new_mod):
                if PRINT_VERIFICATION:
                    print("no conflicts found in the model")
                return new_mod
            if PRINT_VERIFICATION:
                print("model + new_model after swap+conflict:", model, new_mod)
    # move the subject and check if there are any conflicting propositions
    new_mod = modify.move(subj, s_coord, relation, o_coord, model)
    if PRINT_VERIFICATION:
        print("new_mod after move: ", new_mod)
    # revise condition!!!
    if new_mod and (new_mod != None) and not conflict_props(fix_props, new_mod):
        return new_mod
    if PRINT_VERIFICATION:
        print("model + new_model after move subject:", model, new_mod)
    # move the subject and check if there are any conflicting props
    new_mod = modify.move(obj, o_coord, helper.convert(relation), s_coord, model)
    if PRINT_VERIFICATION:
        print("new_mod after move: ", new_mod)
    if (new_mod != None) and not conflict_props(fix_props, new_mod):
        return new_mod
    return None  # nothing worked

# ONLY USED IN SPATIAL
def find_rel_prop(s_coords, o_coords):
    """OK [65]
    Returns the normalized difference of the subject and object coordinates.
    The normalization is the semantic relation between the two coordinates.
    Calls list_substract with the two coordinate lists and normalizes the
    result.
    Example: for s_coords = [2, 0, 1] and o_coords = [0, 0, 1]
    returns [1, 0, 0]
    """
    if PRINT_VERIFICATION:
        print("find rel prop with s_coords, o_coords: ", s_coords, o_coords)
    vector = list_substract(s_coords, o_coords)
    return normalize(vector)

# ONLY USED IN SPATIAL
def list_substract(list1, list2):
    """OK [52]
    Returns list1 where each element is substracted by the corresponding
    list2 element. Returns None if the lenght of the lists is not the same.
    """
    if not list1:
        return None
    result_list = []
    if len(list1) != len(list2):
        #print("list substract with lists of different length!!, abort")
        return None
    for count, value in enumerate(list1):
        result_list.append(value - list2[count])
    return result_list


# ---------------------------- FUNCTIONS FOR THE TEMPORAL MODEL -----------------------------------

# ONLY USED IN TEMPORAL
def is_question(premises):
    """
    Returns True if the last premise in premises is a question
    (has the form ["?", "A", "B"]), otherwise False.
    """
    if PRINT_VERIFICATION:
        print("Function call - is_question")
    rev_prem = premises[::-1]
    if rev_prem[0][0] == "?":
        return True
    return False

# ONLY USED IN TEMPORAL
def answer(subj, obj, models, spatial_parser):
    """
    Function which is called when all premises were already parsed and the last
    premise contains a question.
    The function firsts computes all models out of the given models, which support
    the relation "before" between the two items in the question "subj" and "obj".
    The same is done for the relations "after" and "while".
    Afterwards, the function compares these resulting lists of models. If only
    one of the lists is not empty, then the answer is clearly that model-set
    with the certain relation (For instance: before_mods contains some models,
    after_mods and while_mods are empty. Then the relation between the two items
    of the question is clearly that item1 (subj) happens before item2 (obj)).
    If however two ore more of these lists are not empty, then the conclusion
    is that there is no definite relation between the two events, since some models
    support a different relation than some of the other models (indeterminacies).
    """
    before_mods = verify_mods((-1, 0, 0), subj, obj, models, spatial_parser)
    after_mods = verify_mods((1, 0, 0), subj, obj, models, spatial_parser)
    while_mods = verify_mods((0, 1, 0), subj, obj, models, spatial_parser)
    print("Question: What is the relation between", subj, "and", obj, "?")
    if PRINT_VERIFICATION:
        print("Answer: before_mods are:", before_mods, "after_mods are:", after_mods,
              "while_mods are:", while_mods)
    if before_mods:
        if (after_mods or while_mods):
            print("Answer(1a): There is no definite relation between the two events")
        else:
            print("Answer(1b):", subj, "happens before", obj)
    elif after_mods:
        if while_mods:
            print("Answer(2a): There is no definite relation between the two events")
        else:
            print("Answer(2b):", subj, "happens after", obj)
    elif while_mods:
        print("Answer(3):", subj, "happens while", obj)
    else:
        print("Answer(4): Wakaranai..... Nani o surukanaaa... (I don´t know)")

# ONLY USED IN TEMPORAL
def verify_models(rel, subj, obj, models, spatial_parser):
    """
    Function is called by decide in order to verify_temporal whether the given models
    still hold with the new premise with rel, subj and obj.
    Function calls verify_temporal on all models, if all models are succesfully verified,
    the new_mods list should contain the exact same models as the models-list.
    If that´s not the case, some of the models have not been successfuly verified.
    IF none of the models could be successfully verified with the new premise,
    this new premise is inconsistent with the previous ones. If some could be verified
    and some not, then the premise was hitherto possibly false.
    """
    if PRINT_VERIFICATION:
        print("Function call - Verify_models")
    new_mods = verify_mods(rel, subj, obj, models, spatial_parser)
    if not new_mods:
        print("The premise is inconsistent with the previous ones")
        return None
    if new_mods == models:
        print("The premise follows from the previous ones.")
        return models
    print("The premise was hitherto possibly false")
    return new_mods

# ONLY USED IN TEMPORAL
def verify_mods(rel, subj, obj, mods, spatial_parser):
    """
    Function returns all models in which the given relation, subject and object holds.
    The function can therefore be used to remove models corresponding to
    indeterminacies that turn out to be false.
    """
    if PRINT_VERIFICATION:
        print("Function call - verify_mods")
    if not mods:
        return None
    return_list = []
    for mod in mods:
        # Append all models to the result list that are successfully verified.
        if verify_temporal(rel, subj, obj, mod, spatial_parser):
            return_list.append(mod)
    return return_list

# ONLY USED IN TEMPORAL
def verify_temporal(rel, subj, obj, mod, spatial_parser):
    """
    Extracts the relation, subject and object from the proposition. Then
    searches the subj + obj in the model. After that, the function actually checks and verifies
    whether the given relation holds between the the given subject and object.
    For instance in the relation (1, 0, 0) between a subject a and an object b
    (f.i. in a temporal model, the relation "A happens after B"), the x-coordinate of
    the subject a needs to be bigger than the x-coordinate of the object b.
    (Note: is different to verification in spatial models, since in temporal relations,
    items do not need to be on the same axis in order to be successfully verified.
    So the verification-function of the Spatial Model is more strict).
    """
    subj_co = helper.find_item_in_model(subj, mod)
    obj_co = helper.find_item_in_model(obj, mod)
    if PRINT_VERIFICATION:
        print("Verify: subject coordinates are:", subj_co, "object coordinates are:", obj_co)
    if ((subj_co is None) or (obj_co is None)):
        print("Verify: Trying to verify_temporal relation between entities",
              "in two unrelated models", mod)

    rel_holds = True
    for index, value in enumerate(rel):
        if (value > 0) and (subj_co[index] <= obj_co[index]):
            rel_holds = False
        if (value < 0) and (subj_co[index] >= obj_co[index]):
            rel_holds = False
    # exceptional case: For the while-relation in temporal models, the only criteria is
    if rel == (0, 1, 0) and not spatial_parser: # that the two objects have the same x-value.
        rel_holds = (subj_co[0] == obj_co[0])

    if PRINT_VERIFICATION:
        print("verify_temporal returned:", rel_holds)
    return rel_holds
