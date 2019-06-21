'''
Module for the construction and combination of spatial and temporal models.

Created on 16.07.2018

@author: Christian Breu <breuch@web.de>, Julia Mertesdorf<julia.mertesdorf@gmail.com>
'''

import copy

import low_level_functions as helper


PRINT_MODEL = False


# ---------------------------- FUNCTIONS FOR MODEL CONSTRUCTION -----------------------------------

# USED BY BOTH MODELS
def startmod(relation, subj, obj):
    """
    Function constructs a new model out of a given subject, object and relation from a
    premise. It creates an empty dictionary, then adds the object and the subject of the
    proposition as a new dictionary entry with "add_item" (was add_it in v.1).
    Returns the resulting Model.
    """
    if PRINT_MODEL:
        print("startmod with rel,sub, obj:", relation, subj, obj)
    model = {(0, 0, 0): obj} # add the obj to the origin directly
    # put subject at appropriate relation to object
    model = add_item((0, 0, 0), relation, subj, model)
    if PRINT_MODEL:
        print("startmod: Full startmodel with object and subject is:", model)
    return model

# USED BY BOTH MODELS
def add_item(coordinates, relation, item, model):
    """
    New version of add_item + add_it
    Function adds the given item to the model by using the coordinates and the relation
    (relation and coordinates need to be tuples). Adds the relation to the coordinates
    in order to get the real coordinates that the item should have in the model by calling
    tuple_add.
    If the relation is (0, 0, 0), gets the item at the target coordinates, if there is
    an element, and concatenates item and this element. If the slot is still empty, the item
    is simply inserted at this slot and afterwards the model is returned.
    If the relation wasn´t (0, 0, 0), search for the first empty slot (by adding the relation
    to the target coordinates until a free slot is reached), and insert the item.
    (Note: Only the Temporal Model needs to work on the copy in order to construct all
    possible models, but it works that way as well for the Spatial Model.)
    """
    target_coords = helper.tuple_add(coordinates, relation)
    if PRINT_MODEL:
        print("add_item: coords:", coordinates, "rel", relation, "--> target",
              target_coords, "item", item, "mod", model)
    if relation == (0, 0, 0):
        # check if there is already an object at the given coords.
        if model.get(target_coords) is not None:
            item = [item, model.get(target_coords)] # add the item to the already existing one
        model[target_coords] = item                 # add the item to the model at the coords
        return model
    # check if there is another object at the current coords.
    while model.get(target_coords) is not None:
        # search the first free spot in the correlating axis in the model.
        # add the relation to coords until a free spot is found.
        target_coords = helper.tuple_add(target_coords, relation)
    if PRINT_MODEL:
        print("add the item at coords:, ", item, target_coords)
    model2 = copy.deepcopy(model) # FOR TEMPORAL
    model2[target_coords] = item # add the item to the model
    return model2

# ONLY USED BY TEMPORAL MODEL
def add_item_models(rel, item_to_add, item_in_mods, mods, spatial_parser):
    """
    Function for adding seperate item to models (f.i. if new subject/object is not found in
    existing models, but there already are existing models).
    item_to_add is the new item that needs to be added; item_in_mods is the item,
    which is in relation to item_to_add in the current premsie and already existing in one
    or several models.
    In the while Loop, the function calls "find_first_item" to find the next model in mods which
    already contains "item_in_mods". The function then calls add_item_mod with the found
    model, and adds item_to_add to this model. If there are indeterminacies, the function
    add_item_mod returns several new potential models. All of these new models are appended to
    the resulting model list which is returned at the end of the function.
    In order to retain all models again (also the input models which didn´t contain
    "item_in_mods" and weren´t found by the function "find_first_item") are
    added to the resulting model list.
    If "item_in_mods" is not contained in any of the mods, returns model-list unchanged.
    """
    if PRINT_MODEL:
        print("add_item_models with rel", rel, "item to add", item_to_add,
              "item in mods", item_in_mods)
    if not mods:
        return None
    model_list = []
    while mods:
        if PRINT_MODEL:
            print("add_item_models: LOOP- Full mods list is:", mods, "returnList:", model_list)
        co_mod = helper.find_first_item(item_in_mods, mods) #returns coords and model
        if co_mod != None:
            co_ = co_mod[0]
            mod = co_mod[1]

            tmp_list = [] # Is needed to prevent bracket-errors
            tmp_list = add_item_mod(co_, rel, item_to_add, mod, spatial_parser)
            for item in tmp_list:
                if item not in model_list:
                    model_list.append(item)
        # to ensure that model is retained in output if itemInMods is not in a particular model
        for i in enumerate(mods):
            if mods[i[0]] != co_mod[1]:
                model_list.append(mods[i[0]])
                mods.pop(0)
            if mods[i[0]] == co_mod[1]:
                break
        mods.pop(0)
        if PRINT_MODEL:
            print("add_item_mod: END LOOP- mods is now:", mods, "returnList is:", model_list)
    # If itemInMods is not in any of the models, return models unchanged.
    if PRINT_MODEL:
        print("add-item-models: Finsihed. Return all models: ", model_list)
    return model_list

# ONLY USED BY TEMPORAL MODEL
def add_item_mod(coords, rel, item, mod, spatial_parser):
    """
    Function adds the specified item to the model. In case there is an indeterminacy,
    the function returns a list of models corresponding to each possible interpretation,
    otherwise a list of one model.
    (Note: Indeterminacies do not occur in case of "while" so far)
    (Note2: The Temporal model shall construct all possible models for a given Spatial or
    Temporal problem. However, in case of a temporal problem with the relation "while",
    the order between the items which are in a while-relation to each other, does not matter.
    In that case (and no other case), the function "add_item", which is mainly used by the
    Spatial Model and implements a first-fit-strategy, can be used to insert the item).
    """
    if PRINT_MODEL:
        print("add_item_mod; rel", rel, "co", coords, "item", item, "mod", mod)
    # order is only irrelevant in temporal models for a "while" relation.
    if not spatial_parser and (rel == (0, 1, 0)):
        if PRINT_MODEL:
            print("add_item_mod: relation is while, call add-it!")
        return [add_item(coords, rel, item, mod)]
    else:
        tmp = insert_it(rel, item, coords, mod)
        tmp2 = make_indets(rel, item, helper.tuple_add(coords, rel), mod, spatial_parser)
        if PRINT_MODEL:
            print("Add_item_mod: Insert leaded to:", tmp, "make_indets leaded to:", tmp2)
        if tmp2 != None:
            print("add_item_mod: Inserting item", item, "leaded to different models!")
            return_list = []
            return_list.append(tmp)
            for model in tmp2:
                return_list.append(model)
            if PRINT_MODEL:
                print("add_item_mod: Return the following list:", return_list)
            return return_list
        else:
            return [tmp]

# ONLY USED BY TEMPORAL MODEL
def make_indets(rel, item, co_, mod, spatial_parser):
    """
    Function creates set of different potential models when a temporal relation is
    indeterminate, allowing each indeterminacy between two items x and y to yield a model
    in which x is before y, a model in which x is after y and a model where x is while y.

    Example: rel = (-1,0,0), item a is before d. mod = b c d.
    Leads to five different models (in this case, the direction left-right is
    before/after, and elements beneath other elements illustrate "while"):
    a b c d,   b c d,     b a c d,   b c d,    b c a d
               a                       a
    (b and c in this example are called the "interveners".)

    (Note: Function was adapted to also work for Spatial relations, except the case of
    the relation (0,-1,0) which would interfere with the while-relation in case the problem
    is a temporal problem. (this is due to the fact that for temporal relations, the order of
    items in a while-relation doesn´t matter, so (0, 1, 0) can be added without problems
    to the coordinates of a temporal problem to get the while-intervener. Since spatial problems
    work differently and the order always matters, this approach does not work for spatial problems,
    so an exception is need to be made)).
    """
    if PRINT_MODEL:
        print("Function call - make_indets; rel", rel, "item", item, "co", co_, "mod", mod)
    # Check if coordinates are already outside current dimensions.
    dims = helper.dict_dimensions(mod)
    if rel == (0, 1, 0) or rel == (0, -1, 0):            # while relation --> y-coord relevant!
        if (co_[1] > dims[1]) or (co_[1] < -dims[1]):
            return None
    elif rel == (0, 0, 1) or rel == (0, 0, -1):          # Third dimension, only for spatial
        if (co_[2] > dims[2]) or (co_[2] < -dims[2]):
            return None
    else:                                       # before or after relation --> x-coord relevant!
        if (co_[0] > dims[0]) or (co_[0] < -dims[0]):
            return None
    if mod.get(co_) != None:
        ins = insert_it(rel, item, co_, mod)  # item relation intervener
        add = None
        if rel != (0, -1, 0): # EXCEPTION FOR SPATIAL PROBLEMS (see docstring above)
            add = add_item(co_, (0, 1, 0), item, mod) # item while intervener
        ind = make_indets(rel, item, helper.tuple_add(co_, rel), mod, spatial_parser)
        if PRINT_MODEL:
            print("---Make indets: Insert it is: ", ins, " add is: ", add, " indets is: ", ind)
        if add != None:
            if ind != None:
                return_list = [ins, add]
                for model in ind:
                    return_list.append(model)
                return return_list
            return [ins, add]
        elif ind != None:
            return_list = [ins]
            for model in ind:
                return_list.append(model)
            return return_list
        return [ins]
    ind2 = make_indets(rel, item, helper.tuple_add(co_, rel), mod, spatial_parser)
    if PRINT_MODEL:
        print("Make indets (außerhalb if):", ind2)
    return ind2

# ONLY USED BY TEMPORAL MODEL
def insert_it(rel, item, coords, mod):
    """
    Function inserts an item according to the relation. If the target coordinates are not
    contained in the dictionary, function simply inserts the target coordinates as a key with
    the new value (item) in the dictionary. If the target spot is already occupied and
    if the relation contains a positive number, the function adds the item at the according
    target spot (or with a negative number, at the coords spot) and shifts all other items,
    that are affected by the new insertion, one spot to the right/top (+1).
    (Note: The function works on copies of the model since in case a problem can lead to many
    models, all of these models should be created and thus the insertion process should be
    continued on a copied version of the original model).
    """
    target = helper.tuple_add(rel, coords)
    if mod.get(target) is None: # add item to empty target cell
        mod2 = copy.deepcopy(mod)
        mod2[target] = item
        return mod2
    else:                       # target cell already occupied
        if mod.get(coords) is not None:
            mod3 = copy.deepcopy(mod)
            if rel[0] == 1:            # after relation --> shift elements
                mod3 = shift_coordinates(mod3, target, 0)
                mod3[target] = item
            elif rel[0] == -1:         # before relation --> shift elements
                mod3 = shift_coordinates(mod3, coords, 0)
                mod3[coords] = item
            elif rel[1] == 1:
                mod3 = shift_coordinates(mod3, target, 1)
                mod3[target] = item
            elif rel[1] == -1:
                mod3 = shift_coordinates(mod3, coords, 1)
                mod3[coords] = item
            elif rel[2] == 1:
                mod3 = shift_coordinates(mod3, target, 2)
                mod3[target] = item
            elif rel[2] == -1:
                mod3 = shift_coordinates(mod3, coords, 2)
                mod3[coords] = item
            return mod3
    return mod

# ONLY USED BY TEMPORAL MODEL
def shift_coordinates(mod, coords, index):
    """
    Function to shift all items with their coordinates in a model
    that are affected by the insertion of a new item. "coords" are the coordinates,
    where the new item is going to be inserted, so all items that have coordinates
    that equal to "coords" or are bigger than "coords" need to be shifted to the right.
    Function first determines on which axis the affected items need to be shifted to,
    then saves all of the affected keys with the new key-coordinates in the list
    "keys_to_delete", and after finishing the for-loop, deletes this keys and inserts
    the new key-values for the shifted items.
    """
    if PRINT_MODEL:
        print("shift_coordinates: mod", mod, "coords", coords, "index", index)
    # Determine axis to shift the key-coordinates.
    tuple_to_add = ()
    if index == 0:
        tuple_to_add = (1, 0, 0)
    elif index == 1:
        tuple_to_add = (0, 1, 0)
    elif index == 2:
        tuple_to_add = (0, 0, 1)
    # Determine all keys that need to be replaced by another key.
    keys_to_delete = [] # First is the key to delete, second the new key
    for key in mod.keys():
        if key[index] >= coords[index]:
            new_coords = helper.tuple_add(key, tuple_to_add)
            keys_to_delete.append([key, new_coords])
    keys_to_delete = sorted(keys_to_delete, key=lambda x: x[0])
    if PRINT_MODEL:
        print("keys to update/delete are", reversed(keys_to_delete))
    # Replace the old key-coordinates by the new ones.
    for key in reversed(keys_to_delete):
        new_val = mod[key[0]]
        mod[key[1]] = new_val
        del mod[key[0]]
    if PRINT_MODEL:
        print("model after shifting is", mod)
    return mod


# ---------------------------- FUNCTIONS FOR COMBINING MODELS ------------------------------------

# ONLY USED BY TEMPORAL MODEL
def combine_mods(rel, subj, obj, subj_mods, s_mods, o_mods):
    """
    Function combines each model in s-mods with each model in o-mods according to the
    relation between subj (in s-mods) with obj (in o-mods).
    This function makes one combination of each pair.
    (Note: Function doesn´t compute indeterminacies)
    """
    if PRINT_MODEL:
        print("Function call - combine_mods with rel", rel, "subj", subj, "obj", obj,
              "subj_mods", subj_mods, "s_mods", s_mods, "and o_mods", o_mods)
    if not s_mods:
        if not o_mods:
            return None
        else:
            if PRINT_MODEL:
                print("combine-mods: o_mods not null, call combine_mods again")
            return combine_mods(rel, subj, obj, subj_mods, subj_mods, o_mods[1:])
    if not o_mods:
        return None
    tmp1 = combine(rel, helper.find_item_in_model(subj, s_mods[0]),
                   helper.find_item_in_model(obj, o_mods[0]), s_mods[0], o_mods[0])
    tmp2 = combine_mods(rel, subj, obj, subj_mods, s_mods[1:], o_mods)
    if PRINT_MODEL:
        print("Combine_mods: combine leaded to:", tmp1, "combine_mods leaded to:", tmp2)
    if tmp2 != None:
        return [tmp1, tmp2]
    return [tmp1]

# USED BY BOTH MODELS
def combine(relation, s_co, o_co, subj_mod, obj_mod):
    """
    New version of combine with dictionaries.
    Function combines the subject model and the object model in a way that the relation
    between the subject and the object is satisfied.
    Calls dimensions_n_orig to find out what the new dimensions and origins
    need to be. Then the coordinates of all objects in both models are
    shifted according to the new dimensions.
    returns the combined model.
    """
    if PRINT_MODEL:
        print("function call - combine with rel", relation, "s_co", s_co, "o_co", o_co,
              "subj_mod", subj_mod, "and obj_mod", obj_mod)
    # use the dimension to determine the size of the models
    sub_dims = helper.dict_dimensions(subj_mod)
    obj_dims = helper.dict_dimensions(obj_mod)
    tmp = find_new_origin_dict(relation, sub_dims, obj_dims, s_co, o_co)
    if PRINT_MODEL:
        print("Combine: Dimensions of new model is (find_new_origin_dict): ", tmp)
    # find out by which delta the origin needs to be shifted
    new_sub_orig = [y[0] for y in tmp] # build the new origins from sub and obj
    new_obj_orig = [z[1] for z in tmp]
    if PRINT_MODEL:
        print("new origins of subj is:", new_sub_orig, "new origins of obj is:", new_obj_orig)
    # update the coordinates on both models.
    new_subj_mod = shift_origin_dict(subj_mod, new_sub_orig)
    new_obj_mod = shift_origin_dict(obj_mod, new_obj_orig)
    if PRINT_MODEL:
        print("Combine: after origin update, subj is: ", new_subj_mod, "obj is", new_obj_mod)
    # just put the two dicts together to get the full combined model
    new_subj_mod.update(new_obj_mod)
    if PRINT_MODEL:
        print("Combine: combined model is: ", new_subj_mod)
    return new_subj_mod

# USED BY BOTH MODELS
def find_new_origin_dict(relation, sub_dims, obj_dims, s_co, o_co):
    """
    New version for dimensions_n_orig, only returns new origins.
    Iterates through the relation and coordinates of the two
    models. Returns a list of 3 lists consisting of the subject and
    the object new origin.
    If the relation of an axis is > 0: add the object dimension to the
    subject-origin part.
    -> because the subject is right or in front of the object and hence need
    to have the origin value of the obj aswell.
    If the relation is < 0: add the subject dimension to the object-origin part.
    if the relation is 0 at a certain point, call ortho (with the coordinates of sub+obj)
    and add the result of ortho to the result.
    """
    if PRINT_MODEL:
        print("Function call - dimensions_n_orig with rel", relation, "s_dims", sub_dims,
              "o_dims", obj_dims, "s_co", s_co, "o_co", o_co)
    if not relation:
        return None
    result_list = []
    for count, value in enumerate(relation):
        if value > 0:
            # the subject needs a new origin
            result_list.append([obj_dims[count], 0])
        elif value < 0:
            #the object needs a new origin.
            result_list.append([0, sub_dims[count]])
        else:
            result_list.append(ortho(s_co[count], o_co[count]))
    if PRINT_MODEL:
        print("dimensions_n_orig: returnList is:", result_list)
    return result_list

# USED BY BOTH MODELS
def ortho(sub_cord, obj_cord):
    """
    New version of orhto method. returns the new origins for the subject and object.
    Takes coordinate components of the subject and object.
    """
    if PRINT_MODEL:
        print("Function call - ortho with sub_cord", sub_cord, "obj_cord", obj_cord)
    if sub_cord > obj_cord:
        return [0, (sub_cord - obj_cord)]
    elif sub_cord < obj_cord:
        return [(obj_cord - sub_cord), 0]
    return [0, 0]

# USED BY BOTH MODELS
def shift_origin_dict(dictionary1, origin_list):
    """
    New version of "new_origin" used for dictionaries.
    Shifts all coordinates of the elements in the model by the number given in origin_list.
    Returns the model with all items shifted according to new origin.
    """
    new_dict = {}
    for (x_co, y_co, z_co) in dictionary1.keys():
        new_dict[x_co+origin_list[0], y_co+origin_list[1], z_co+origin_list[2]
                ] = dictionary1[(x_co, y_co, z_co)]
    return new_dict
