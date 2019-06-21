'''
Module for low level helper functions.

Created on 16.07.2018

@author: Christian Breu <breuch@web.de>, Julia Mertesdorf<julia.mertesdorf@gmail.com>
'''
import copy

import numpy

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d # Is needed even though Pylint doesn´t like it


PRINT_LOW_LEVEL = False


# ---------------------------- FUNCTIONS FOR FINDING ITEMS IN MODELS ------------------------------

# ONLY USED BY TEMPORAL MODEL
def find_item_mods(item, mods):
    """
    Function searches for an item in all models by calling "find_item_in_model" for each model.
    For each found model which contains the given item, adds this model to the resulting list.
    Returns the list at the end of the function. In case no model was found, return None.
    """
    if PRINT_LOW_LEVEL:
        print("Function call - find_item_mods: item", item, "mods", mods)
    if not mods:
        return None
    mod_list = []
    for mod in mods:
        coords = find_item_in_model(item, mod)
        if coords != None:
            mod_list.append(mod)
    if mod_list:
        return mod_list
    return None

# ONLY USED BY TEMPORAL MODEL
def find_item_in_model(item, model):
    """
    This function replaces "finders" of version v.1. Returns coordinates of one item.
    Function iterates through the key-value pairs of the model (dictionary) and compares
    each value with "item". When a fitting key-value pair is found which contains this item
    as a value, save the key-coordinates and return them. If item wasn´t found, return None.
    """
    if PRINT_LOW_LEVEL:
        print("find-item-in-model - item", item, "model", model)
    if not model:
        return None
    coordinates = []
    for key, val in model.items():
        if val == item:
            coordinates = key
    if PRINT_LOW_LEVEL:
        print("find-item-in-model, item", item, "coordinates is:", coordinates)
    if coordinates != []:
        return coordinates
    return None

# USED BY BOTH MODELS
def find_first_item(item, models):
    """
    new version of find item with dictionary (replaces find_first_item, finders, finds,
    eq_or_inc and check_member of v.1)
    Searches an item in all the models given as a dictionary.
    Iterates over all models and checks if the item is in one of them.
    Function works for both when the value is a list of items or when the
    value itself is a single item.
    Returns a list with a tuple of the coordinates of the item and the model
    where the item was found in. if it couldn´t be found, returns None.
    """
    if PRINT_LOW_LEVEL:
        print("Function call - find_first_item", item, "in models", models)
    if not models:
        return None
    if not isinstance(models, list):
        models = [models]
    for model in models:
        #need to check if the model is None!!
        if model != None:
            #checks if the item is in the current model, and if yes, where
            coordinates = [key for key, val in model.items() if item in val]
            if coordinates == []:
                coordinates = [key for key, val in model.items() if item == val] # NEW FOR TEMPORAL
            if coordinates != []:
                #return the coordinates and the corresponding model when item found
                if PRINT_LOW_LEVEL:
                    print("find_first_item returns:", coordinates[0])
                return [coordinates[0], model]
    if PRINT_LOW_LEVEL:
        print("find_first_item failed, nothing found")
    return None


# ---------------------------- GENERAL LOW LEVEL FUNCTIONS ----------------------------------------

# ONLY USED BY TEMPORAL MODEL
def converse(relation):
    """
    Converts a given relation (negation of a relation, f.i. (1, 0, 0) --> (-1, 0, 0))
    (Note: is converted to list in order to use the shared convert-function, and back
    to tuples, since the Temporal Model works with Tuples and the Spatial Model
    with lists).
    """
    conv_rel = convert([relation[0], relation[1], relation[2]])
    if PRINT_LOW_LEVEL:
        print("converted relation to", conv_rel)
    return (conv_rel[0], conv_rel[1], conv_rel[2])

# USED BY BOTH MODELS
def convert(relation):
    """
    Function inverts all numbers in relation, changing positive numbers
    to negative numbers and vice versa, leaving the 0´s unchanged.
    """
    if relation is None:
        return None
    neg_rel = [-x for x in relation]
    return neg_rel

# USED BY BOTH MODELS
def get_relation(proposition):
    """
    Returns the relation as a list of ints from the given premise
    """
    relation_string = proposition[0]
    if PRINT_LOW_LEVEL:
        print("get_relation: ", proposition)
    if isinstance(relation_string, list):
        # print("refln: relation is already a list")
        # the relation has already been converted
        return relation_string
    # re-format the string into a list for coordinate use
    relation_string = relation_string.strip('(')
    relation_string = relation_string.strip(')')
    relation = relation_string.split()
    relation = [int(relation[0]), int(relation[1]), int(relation[2])]
    return relation

# USED BY BOTH MODELS
def get_subject(proposition):
    """
    Returns the subject of a given proposition
    """
    return proposition[1][0]

# USED BY BOTH MODELS
def get_object(proposition):
    """
    Returns the object of a given proposition
    """
    return proposition[2][0]

# USED BY BOTH MODELS
def tuple_add(tuple1, tuple2):
    """
    new version of list_add, used for dictionaries.
    Adds all single elements of the tuples. Returns the tuple of the sums.
    Returns None if the tuples do not have the same length.
    """
    if (not tuple1) or (not tuple2) or (len(tuple1) != len(tuple2)):
        return None
    return (tuple1[0] + tuple2[0], tuple1[1] + tuple2[1], tuple1[2] + tuple2[2])

# USED BY BOTH MODELS
def dict_dimensions(dict1):
    """
    Determines the dimensions of a model in dictionary form.
    Returns a tuple with the max values of indices from the coords.
    """
    x_dim = []
    y_dim = []
    z_dim = []
    for (i, j, k) in dict1.keys():
        y_dim.append(j)
        x_dim.append(i)
        z_dim.append(k)
    # add 1 to actually get the size of the model.
    return tuple([max(x_dim)+1, max(y_dim)+1, max(z_dim)+1])

# USED BY BOTH MODELS
def dict_mins(dict1):
    """
    Determines the minima of all coordinates. returns the minima for the 3
    axes separately. Returns a tuple with min x, y and z
    """
    x_dim = []
    y_dim = []
    z_dim = []
    for (i, j, k) in dict1.keys():
        y_dim.append(j)
        x_dim.append(i)
        z_dim.append(k)
    # add 1 to actually get the size of the model.
    return tuple([min(x_dim), min(y_dim), min(z_dim)])

# USED BY BOTH MODELS
def normalize_coords(model):
    """
    Function which normalizes a given model (dictionary) to get a model with only
    positive coordinates.
    Function takes the minimum of the 3 coordinate components. For each of them,
    shift all coordinates to make the smallest number 0(if the minimum is
    smaller than 0 only). Returns the normalized model.
    e.g. shifts the items at (1, 1, 0) and (-1, 2, 0) to (2, 1, 0) and (0, 2, 0)
    """
    if PRINT_LOW_LEVEL:
        print("normalize model with model: ", model)
    min_coords = dict_mins(model) # get all minima of the coordinates
    shift_vals = [0, 0, 0] # a list for the values by which the coords will be shifted.
    if min_coords[0] < 0: # shift all x coordinates by the absolute value of the minimum.
        shift_vals[0] = abs(min_coords[0])
    if min_coords[1] < 0:
        shift_vals[1] = abs(min_coords[1])
    if min_coords[2] < 0:
        shift_vals[2] = abs(min_coords[2])
    # iterate through the model and copy the items into another dict with updated coordinates.
    shifted_model = {}
    for (x_co, y_co, z_co), value in model.items():
        shifted_model[x_co + shift_vals[0], y_co + shift_vals[1], z_co + shift_vals[2]] = value
    if PRINT_LOW_LEVEL:
        print("normalized model: ", shifted_model)
    return shifted_model


# ---------------------------- FUNCTIONS FOR PRINTING / VISUALIZING MODELS ------------------------

# USED BY BOTH MODELS
def print_models(model_list):
    """
    Prints all models in a given model_list the way they should look.
    Uses matplotlib scatterplot.
    """
    plt.ioff()
    fig = plt.figure()
    model_objects = {'[]':'s', 'V': '^', 'O': 'o', 'I': '|', '+': 'X',
                     'L': '$L$', '^': '$V$', '*': '*', 'S': '$S$',
                     '[A]': '$A$', '[B]': '$B$', '[C]': '$C$',
                     '[D]': '$D$', '[E]': '$E$', '[F]': '$F$',
                     '[G]': '$G$', '[H]': '$H$', '[J]': '$J$',
                     'A': '$A$', 'B': '$B$', 'C': '$C$',
                     'D': '$D$', 'E': '$E$', 'F': '$F$',
                     'G': '$G$', 'H': '$H$', 'J': '$J$'}
    # compute the square root from the number of elements and then adjust the
    # size for the grid to fit the number of models.
    rows_cols = int(numpy.sqrt(len(model_list))+0.5)+1
    #print("rows_cols ", rows_cols)
    # iterate through the models
    for index, model in enumerate(model_list):
        ax_ = fig.add_subplot(rows_cols, rows_cols, index+1, projection='3d')
        ax_.set_xlabel('X-Axis')
        ax_.set_ylabel('Y-Axis')
        ax_.set_zlabel('Z-Axis')
        for (x_co, y_co, z_co), value in model.items():
            if isinstance(value, list):
                #model_obj = ''
                for item in value:
                    ax_.scatter(x_co, y_co, z_co, marker=model_objects[item])
                # print("double item found in model:", model_obj)
            else:
                ax_.scatter(x_co, y_co, z_co, marker=model_objects[value])
    # print an asnwer or sth. like that
    #fig.text(.5, .05, answer, ha='center')
    fig.text(.5, .05, "all created models in their creation order", ha='center')
    plt.show()

# ONLY USED BY TEMPORAL MODEL; DOES NOT WORK FOR SPATIAL PROBLEMS
def format_model_dictionary(model):
    """
    Function to print the model in a visually more appealing way.
    Events in one row happen at the same time, rows after each other visualize that
    the lower event happend after the upper event.
    (Note: This function is only applicable for temporal-problems since they are
    two-dimensional. The spatial model would need a completely different
    function to print them in form of lists in the console. This is however
    redundant since there is a 3D-plot function which can be used by both models,
    and there are only unit-tests for temporal models anyway which need this function.)
    """
    translated_model = translate_dict_to_list(model)
    if PRINT_LOW_LEVEL:
        print("New_format_model: model_list is", translated_model)
    for x__ in translated_model:
        listy = []
        for y__ in x__:
            if y__ != [None]:
                listy.append(y__)
            else:
                listy.append(["."])
        print(listy)

# ONLY USED BY TEMPORAL MODEL; DOES NOT WORK FOR SPATIAL PROBLEMS
def translate_all_dicts(dict_list):
    """
    Function takes a list of dictionaries and normalizes and translates all of them
    to the nested-lists-pattern of v.1 of this program by calling "translate_dict_to_list".
    Returns a list of all models which are lists as well.
    (Note: function only works for temporal problems, since they are two-dimensional).
    """
    if not dict_list:
        return None
    dict_to_model_list = []
    for dictionary in dict_list:
        normalized_dict = normalize_coords(dictionary)
        translated_dict = translate_dict_to_list(normalized_dict)
        dict_to_model_list.append(translated_dict)
    return dict_to_model_list

# ONLY USED BY TEMPORAL MODE; DOES NOT WORK FOR SPATIAL PROBLEMS
def translate_dict_to_list(dictionary):
    """
    Function takes the normalized dictionary, and translates it into the
    nested-lists-pattern of the first version of the Temporal Reasoning Model
    (in order to compare it to the results of the first version, to be able
    to run the same doc-tests on them and to have another option to print the models
    besides 3D plotting with the matplotlib).
    (Note: function works only for temporal deduction problems, since they are only
    two-dimensional.)
    """
    if not dictionary:
        return None
    original_dims = dict_dimensions(dictionary) # dimensions of original model
    copy_model = copy.deepcopy(dictionary)
    model_list = []       # Final translated model to return
    while copy_model:
        if PRINT_LOW_LEVEL:
            print("while loop - copy_model is", copy_model)
        # compute next row
        key_row_list = []
        print_list = []
        min_coords = dict_mins(copy_model)
        if PRINT_LOW_LEVEL:
            print("min_coords are", min_coords, "dims are", original_dims)

        # find global minimum of current (choped) model and append it
        min_key_value = copy_model.get(min_coords)
        if min_key_value is not None:
            if min_coords[1] != 0: # Special case for when there´s a space before the next item
                key_row_list.append([None, [None]])
            key_row_list.append([min_coords, min_key_value])
        else:
            key_row_list.append([None, [None]])
        if PRINT_LOW_LEVEL:
            print("Smallest key is:", key_row_list)

        # append all items which happen at the same time (while-relation).
        for num in range(1, original_dims[1]):
            found_key = False
            for key, value in copy_model.items():
                if (key[0] == min_coords[0]) and (key[1] == num):
                    if [key, value] not in key_row_list:
                        key_row_list.append([key, value])
                    found_key = True
            if not found_key:
                key_row_list.append([None, [None]])
        if PRINT_LOW_LEVEL:
            print("keylist after while-relation loop - list is now", key_row_list)
        # append all values of the keys to the resulting row.
        for elm in key_row_list:
            print_list.append(elm[1])
        # delete all keys in dictionary that were found in this time-step (row).
        for elm in key_row_list:
            if elm[0] is not None:
                del copy_model[elm[0]]
        # append this row to the hole model.
        model_list.append(print_list)
    if PRINT_LOW_LEVEL:
        print("modellist is", model_list)
    return model_list
