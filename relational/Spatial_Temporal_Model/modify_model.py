'''
Module for modifiyng models.

Created on 16.07.2018

@author: Christian Breu <breuch@web.de>, Julia Mertesdorf<julia.mertesdorf@gmail.com>

Note: This module is only used by the Spatial Model!
'''

from copy import deepcopy

import model_construction as model_builder

import low_level_functions as helper


PRINT_MODIFY = False

# ONLY USED BY SPATIAL MODEL
def move(item, item_coord, relation, other_coord, model):
    """new version of move for dictionaries. item_coord and other coord must
    be tuples.
    Removes the item at the specified coordinates from the model, then computes
    a new position by calling new_posn. The new position satisfies the relation,
    so the item is set to a position to be in the relation with the other item.
    In new_posn the coordinates of the item are changed to the coordinates of
    the other item, but the dimension of the relation is changed(to make the relation hold).
    Now it adds the item to the model at the new_position.
    Returns the resulting model.
    """
    if PRINT_MODIFY:
        print("move with: item, item_coord, relation, other_coord, model: ", item,
              item_coord, relation, other_coord, model)
    # use deepcopy to not manipulate the original model
    move_model = deepcopy(model)
    # remove the item
    item_in_mod = move_model[item_coord]
    if isinstance(item_in_mod, list):
        if PRINT_MODIFY:
            print("move: special case: the item at the item_coords are a list")
        item_in_mod.remove(item)
        # if the list contains only one item now, remove the list aswell.
        if len(item_in_mod) == 1:
            move_model[item_coord] = item_in_mod[0]
        else:
            # in this case there are still more items in at the specified coordinates.
            move_model[item_coord] = item_in_mod
    else:
        #the usual case, the item of the model is just a single item(string)
        move_model.pop(item_coord)
    # maybe basic idea: just add the item to the model closest to the other coords,
    # if relation doesn't already hold in this dimension.
    new_position = new_posn(item_coord, relation, other_coord,
                            update_coords(other_coord, relation))
    # convert the new position coordinates to a tuple.
    new_pos_tuple = (new_position[0], new_position[1], new_position[2])
    move_model = model_builder.add_item(new_pos_tuple, (0, 0, 0), item, move_model)
    #move_model = self.new_add_item(new_pos_tuple, (0,0,0), item, move_model)
    return move_model

# ONLY USED BY SPATIAL MODEL
def new_posn(item_coord, relation, other_coord, coordinates):
    """
    Works out to which coordinates the item should be moved.
    Calls compare_ints with all the components of the given coordinates.
    compare_int decides if the coordinates need to be set to the coordinates
    that are given as the last argument or if the coordinates of the item stay
    the same. The item is moved along only one dimension to satisfy the
    relation with the other coordinates.
    """
    if item_coord is None:
        return None
    result = []
    #iterate over all the coordinates
    for count, value in enumerate(item_coord):
        result.append(compare_ints(value, relation[count],
                                   other_coord[count], coordinates[count]))
    return result

# ONLY USED BY SPATIAL MODEL
def compare_ints(item_co_int, rel_int, other_co_int, coord_int):
    """
    Returns the int of the item_coordinates, if the number of the relation
    is > 0 and the item_int is > other_co_int aswell or the relation is < 0
    and the item_co_int < other_co_int respectively.
    If the relation is positive and the item_coords are positive, the
    relation holds with the item_coords, hence returns item_coords.
    The other way around for negative values in relation.
    If those conditions do not hold, the relation is not
    satisfied with these coordinates. In this case, returns the coord_int,
    which is given.
    (obj_coords + relation -> the first place to satisfy the relation)
    """
    if ((rel_int > 0) and (item_co_int > other_co_int)) or ((rel_int < 0)
                                                            and (item_co_int < other_co_int)):
        return item_co_int
    return coord_int

# ONLY USED BY SPATIAL MODEL
def shrink_dict(model):
    """
    Tets the dict dimensions and then finds the empty cols, rows and plas.
    Then iterates through these and reasigns the coordinates in the model.
    If there is an emtpy col between an item an the origin, substract 1 from
    the item coordinates to shift the items closer together and get rid of
    the gap. Returns the shrinked model. Only changes coordinates.
    """
    # print("shrink_dict with model: ", model)
    dict_dims = helper.dict_dimensions(model)
    dim1 = dict_dims[0]-1 # substract 1 because its about the indices.
    dim2 = dict_dims[1]-1
    dim3 = dict_dims[2]-1
    cols = list_ints(dim1)
    rows = list_ints(dim2)
    plas = list_ints(dim3)
    # print("prepared cols, rows, plas for shrink:", cols, rows, plas)
    empty_cols = emtpy_cols(cols, rows, plas, model)
    # iterate through the model and check how many empty cols, rows and pals
    # are infront of the coordinates. Changes coords accordingly.
    shrink_mod = {}
    for (x_co, y_co, z_co), item in model.items():
        new_x = x_co
        new_y = y_co
        new_z = z_co
        for col in empty_cols[0]:
            if x_co > col:
                new_x -= 1
        for row in empty_cols[1]:
            if y_co > row:
                new_y -= 1
        for pla in empty_cols[2]:
            if z_co > pla:
                new_z -= 1
        shrink_mod[new_x, new_y, new_z] = item
    if PRINT_MODIFY:
        print("shrinked model: ", shrink_mod)
    return shrink_mod

# ONLY USED BY SPATIAL MODEL
def emtpy_cols(cols, rows, plas, model):
    """
    Helper function for shrink_dict.
    Iterates over the given model and removes all numbers of the coordinates
    from the items in the model. this way there will be remaining lists of
    empty cols, rows and plas. Returns a list of the 3 lists.
    """
    for (x_co, y_co, z_co) in model.keys():
        # remove all coordinate components from the corresponding list.
        if x_co in cols:
            cols.remove(x_co)
        if y_co in rows:
            rows.remove(y_co)
        if z_co in plas:
            plas.remove(z_co)
    #print("empty cols returns cols, rows, plas:", cols, rows, plas)
    return [cols, rows, plas]

# ONLY USED BY SPATIAL MODEL
def swap(subj, s_coord, obj, o_coord, model):
    """
    First removes the subject and then the object from the model. Then adds
    the subject to the object coordinates and the object to the subject coordinates.
    If the subj or obj is in a list at the coordinates, just swap the subj/obj
    string inside the list.
    """
    if PRINT_MODIFY:
        print("swap with subj, obj, model: ", subj, obj, model)
    new_model = deepcopy(model)
    #check if the items at the coords are lists!
    # if they are, just remove the obj/subj from the list and append the
    # other item. check for both obj and subj.
    if isinstance(new_model[s_coord], list):
        new_model[s_coord].remove(subj)
        new_model[s_coord].append(obj)
    else:
        #usual case
        #update the values to swap the subject and object
        new_model[s_coord] = obj
    if isinstance(new_model[o_coord], list):
        new_model[o_coord].remove(obj)
        new_model[o_coord].append(subj)
    else:
        new_model[o_coord] = subj
    if PRINT_MODIFY:
        print("swap model with swapped items: ", new_model)
    return new_model

# ONLY USED BY SPATIAL MODEL
def list_ints(num):
    """
    creates a list of integers from num to zero, starting from num.
    """
    if num == 0:
        return [0]
    # range returns list excluding num, so just use num+1
    return list(reversed(list(range(num+1))))

# ONLY USED BY SPATIAL MODEL
def update_coords(coords, relation):
    """
    Returns the added coords and relation as a list where each element is
    the sum of the two corresponding numbers.
    If relation is None, returns None.
    """
    if not relation:
        return coords
    return helper.tuple_add(coords, relation)
