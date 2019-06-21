'''
Main Module for the spatial and temporal Models, contains the high-level functions,
all examples (deduction problem-sets) and some unit-tests.

Created on 16.07.2018

@author: Christian Breu <breuch@web.de>, Julia Mertesdorf<julia.mertesdorf@gmail.com>
'''

from copy import deepcopy

import unittest

from parser_spatial_temporal import Parser

import model_construction as construct

import low_level_functions as helper

import backtrack_premises as back

import modify_model as modify

import verification_answer as ver_ans


# GLOBAL VARIABLES

# Two global variables enabling function-prints. Mainly used for debugging purposes.
PRINT_PARSING = False # global variable for whether to print parsing process or not

PRINT_MODEL = False # global variable for whether to print model construction process or not.



class MainModule:
    """
    Main module of the Temporal and Spatial Model. This Class contains the high-level functions
    like "process_problem_spatial/temporal" which the user can call on a specific problem set.
    Moreover, it contains the different "interpret" and "decide" - functions for each the Spatial
    and the Temporal Model, which are called depending on whether
    "process_problem spatial / temporal" was called.

    Short Spatial and Temporal Model Documentation:
    This version of the Spatial Model is semantical (almost) equal to the space6 lisp program
    from 1989. In this version the program can be called with interpret_spatial(problem) directly
    or by calling process_problem / process_all_problems. The Temporal Model can be called likewise
    to the Spatial Model.
    The Temporal model is as well semantically equal to the first version of the python
    translation, besides the fact that it uses dictionaries and is now able to also process
    spatial deduction problems.

    Quick description of the program:
    First it goes through all given PREMISES, parses them and adds them to
    the current model or creates a new one. Premises can also combine two exising
    models when both items of a premise are in different models.
    If The two items from a premise are in the same model, the program will
    check if this premise holds in the model. If yes, tries to falsify and
    if no, tries to verify the premise an the given model. In order to do this,
    the program takes the last added premise(which triggered verify) and tries
    to make it hold in the model. If there is a model that satisfies the premise,
    it will be added to a list of PREMISES that always have to hold. Now it will
    iteratively check if there are conflicts in the current model(that is changed
    every time when a premise is made true f.i.) and then try to make them true
    in the model. If there are no more conficting PREMISES, the program will
    return the new model. If there is a premise that cannot be satisfied in the
    model, the program will terminate with the opposite result(e.g. if a certain
    premise cannot be made false in the model, it is verified. If it can be made
    false, the previous model might be wrong/false.
    There is only one order for verifying the conflicting PREMISES that occur,
    so there are different models with a different outcome probably left out.
    """

    # Global variable capacity illustrating the working memory (how many different models can be
    # kept in the working memory). Is usually set to 4.
    capacity = 4

    def process_all_problems(self, problems, problem_type, spatial_parser=None):
        """
        The function takes a set of problems, a problem_type and optional a desired
        parser as input. The problem_type can be either the string "spatial" or "temporal",
        depending on that the interpret-function of either the temporal or spatial model is called.
        The boolean spatial_parser is set to None if not explicitly written in the function-call,
        which enables the default parser for each of the models (temporal or spatial parser).
        If temporal deduction problems should be processed by the Spatial Model, then one has to
        type "spatial" for the problem-type and "False" for spatial_parser. For spatial
        deduction problems computed in the temporal model, the problem_type needs to be "temporal"
        and spatial_parser needs to be set to "True".
        All problems in the problem set are executed by calling the appropriate
        interpret-function for each problem. Before each new problem,
        the function inserts two dotted lines to separate the problems clearly.
        """
        if not problems:
            print("No problem given to process!")
        for problem in problems:
            print("-----------------------------------------------------------------------")
            print("-----------------------------------------------------------------------")
            print("Process-all-problems: Interpret next problem!")
            if problem_type == "spatial":
                if spatial_parser is None:
                    spatial_parser = True
                self.interpret_spatial(problem, spatial_parser)
            elif problem_type == "temporal":
                if spatial_parser is None:
                    spatial_parser = False
                self.interpret_temporal(problem, spatial_parser)
            else:
                print("No such problem type exists. Try >spatial< or >temporal<.")

    def process_problem(self, number, problems, problem_type, spatial_parser=None):
        """
        The function takes a number, a set of problems, a problem_type and optional a desired
        parser as input. The problem_type can be either the string "spatial" or "temporal",
        depending on that the interpret-function of either the temporal or spatial model is called.
        The boolean spatial_parser is set to None if not explicitly written in the function-call,
        which enables the default parser for each of the models (temporal or spatial parser).
        If a temporal deduction problem should be processed by the Spatial Model, then one has to
        type "spatial" for the problem-type and "False" for spatial_parser. For a spatial
        deduction problem computed in the temporal model, the problem_type needs to be "temporal"
        and spatial_parser needs to be set to "True".
        The function Function processes the problem number n of the given problem-set.
        Every premise of this particular problem is interpreted with the appropriate
        interpret-function.
        """
        if not problems:
            print("No problem given to process!")
            return
        if number > len(problems):
            print("There is no problem with that number in this problem-set. Try a smaller number!")
            return
        if number < 1:
            print("Try a positive number")
            return
        print("Problem to execute is number", number, "\n",
              problems[number-1])
        if problem_type == "spatial":
            if spatial_parser is None:
                spatial_parser = True
            self.interpret_spatial(problems[number-1], spatial_parser)
        elif problem_type == "temporal":
            if spatial_parser is None:
                spatial_parser = False
            self.interpret_temporal(problems[number-1], spatial_parser)
        else:
            print("No such problem type exists. Try >spatial< or >temporal<.")


# ---------------------------- SPATIAL MODEL FUNCTIONS --------------------------------------------

    def interpret_spatial(self, prem, spatial_parser):
        """
        interpret_spatial iterates over the given premise and parses it.
        calls the parse function on the PREMISES, then calls the decide_spatial function
        on the results.returns the resulting models and the deduction that was
        made after parsing every premise.
        The argument "spatial_parser" defines which parser shall be used for processing the
        premises. For a temporal problem, spatial_parser should be set to False, and for a
        spatial problem to True.
        """
        mods = []  # list of models
        all_mods = [] # to save each construction step

        # Check whether the problem is a question problem, which is the case when a Temporal
        # problem is executed with the Spatial Model. This is however not possible at the moment.
        is_question = ver_ans.is_question(prem)
        if is_question:
            print("The Spatial Model cannot solve problems with questions.",
                  "Try a problem without a question.")
            return mods

        # create the desired parser from the Parser Module
        if spatial_parser:
            pars = Parser(True)
        else:
            pars = Parser(False)

        # iterate over the list of PREMISES, return models when done
        for pre_ in prem:
            if PRINT_MODEL:
                print(pre_, "premise")
            premise = pars.parse(pre_)  # the currently parsed premise
            if PRINT_MODEL:
                print("parsed premise: ", prem)
            mods = self.decide_spatial(premise, mods, prem, spatial_parser)
            mods[0] = helper.normalize_coords(mods[0])
            mods[0] = modify.shrink_dict(mods[0])
            # list for all models
            all_mods.append(deepcopy(mods[0]))
            if PRINT_MODEL:
                print("current model after decide_spatial: ", mods)
        # print out models in the list.
        print("list of all resulting Models")
        print(all_mods)
        # print all models in the model list.
        helper.print_models(all_mods)
        return mods

    def decide_spatial(self, proposition, models, premises, spatial_parser):
        """[2]
        takes the parsed premise and the list of current models.
        extracts the subject and object of the premise, then checks if they
        can be found in any model.
        deletes the models from the models list, if they contain the subj. or obj.
        calls helper function choose_function_spatial to decide_spatial what should
        be done depending on the
        premise and the current models.(see documentation of ddci for more detail)
        returns list of current models as a result of choose_function_spatial.
        """
        relation = helper.get_relation(proposition)
        subject = helper.get_subject(proposition)
        object1 = helper.get_object(proposition)
        s_co = None
        o_co = None
        subj_mod = None
        obj_mod = None
        if PRINT_MODEL:
            print("call decide_spatial with rel-subj-obj:", relation, subject, object1)
        # retrieve the subject and the object from the models
        subj_co_mod = helper.find_first_item(subject, models)
        if subj_co_mod is not None:
            s_co = subj_co_mod[0]
            subj_mod = subj_co_mod[1]
        obj_co_mod = helper.find_first_item(object1, models)
        if obj_co_mod is not None:
            o_co = obj_co_mod[0]
            obj_mod = obj_co_mod[1]
        if subj_mod in models:
            models.remove(subj_mod)
        if obj_mod in models:
            models.remove(obj_mod)
        #print("s_co and o_co:", s_co, o_co)
        if not models:
            return [self.choose_function_spatial(proposition, s_co, o_co, relation, subject,
                                                 object1, subj_mod, obj_mod, premises,
                                                 spatial_parser)]

        models.insert(0, self.choose_function_spatial(proposition, s_co, o_co, relation,
                                                      subject, object1, subj_mod, obj_mod,
                                                      premises, spatial_parser))
        return models

    @staticmethod
    def choose_function_spatial(proposition, s_co, o_co, relation, subject, object1,
                                subj_mod, obj_mod, premises, spatial_parser):
        """
        takes a premise(proposition), subject-and object coordinates, a subject and
        an object and their models in which they are contained.
        deletes the models from the models list, if they contain the subj. or obj.
        creates a new model if the subj. and obj. both aren't in any model.
        if one of them is in a model, add the new item to the corresponding model.
        if they both are in the same model, verify the model. depending on the
        result of that, calls make_true or make_false to find counterexamples.
        """
        if s_co is not None:
            if o_co is not None:
                # whole premise already in model, check if everything holds
                if subj_mod == obj_mod:
                    if PRINT_MODEL:
                        print("verify, whether subj. and obj. are in same model")
                    # verify returns the model in case the premise holds
                    if ver_ans.verify_spatial(proposition, subj_mod) is not None:
                        if PRINT_MODEL:
                            print("verify returns true, the premise holds")
                        # try to make falsify the result
                        return ver_ans.make_false(proposition, subj_mod, premises, spatial_parser)
                    # try to make all PREMISES hold
                    return ver_ans.make_true(proposition, subj_mod, premises, spatial_parser)
                # subj and obj both already exist, but in different models
                if PRINT_MODEL:
                    print("combine")
                # commented out for testing
                return construct.combine(relation, s_co, o_co, subj_mod, obj_mod)
            if PRINT_MODEL:
                print("add object to the model")
                # convert relation because the object is added
                print("relation before convert: ", relation, "after convert: ",
                      helper.convert(relation))
            return construct.add_item(s_co, helper.convert(relation), object1, subj_mod)
        # object != Null but subject doesn't exist at this point
        elif o_co is not None:
            if PRINT_MODEL:
                print("add subject to the model")
            return construct.add_item(o_co, relation, subject, obj_mod)
        else:
            # sub and ob doesn't exist at the moment
            if PRINT_MODEL:
                print("startmod")
            return construct.startmod(relation, subject, object1)


# ---------------------------- TEMPORAL MODEL FUNCTIONS -------------------------------------------

    def interpret_temporal(self, premises, spatial_parser, unit_test=False, worked_back=False):
        """
        High level function that is called for each problem consisting of several premises.
        The argument "spatial_parser" defines which parser shall be used for processing the
        premises. For a temporal problem, spatial_parser should be set to False, and for a
        spatial problem to True.
        The argument unit_test is usually set to False except in the test-functions of the
        unit-test. This argument will then disable the 3D-plot of the models.

        The factor capacity describes the working memory, specifically how many models
        the program can remember. Is the capacity of models exceeded (len(mods) > capacity),
        and if the last premise of the problem is a question, "work_back" is called
        (function which extracts only the premises which are relevant to answer the question).
        If the last premise wasn´t a question or if worked_back is set to True when calling
        interpret_temporal, the capacity is just set to a high number, making it possible to
        continue solving the problem.
        While iterating over the premises of the problem, the function checks for each
        premise whether capacity was exceeded and whether the current premise is a question.
        If thats the case, "answer" is called, which will give a statement about the
        relation between the two items in the questions. After checking for capacity and
        question, function parses the current premise and calls decide_temporal on it afterwards
        in order to use this premise to continue constructing the mental model(s).
        The resulting models for the problem are returned either after answering a
        question or at the end of the function.
        Additionally, in case the processed problem is a temporal problem (spatial_parser is set
        to False), interpret_temporal calls "format_model_print" for each premise in order to
        print the current models in a formated and visually better way.
        """
        capacity = self.capacity
        is_question = ver_ans.is_question(premises) # boolean whether we have a question or not
        mods = []                                      # List of models
        mod_length = 0

        if spatial_parser:                         # Set the desired parser.
            parser = Parser(True)
        else:
            parser = Parser(False)

        if not premises:
            print("No PREMISES found! Return empty list of models")
            return mods
        for prem in premises:
            # Print all current models.
            if mods:
                print("INTERPRET-LOOP: THE CURRENT NUMBER OF MODELS IS:", mod_length)
                if not spatial_parser:
                    for mod in mods:  # print all models as nested lists (Works only for temporal!)
                        print("-------------------------------------------------------------")
                        mod = helper.normalize_coords(mod)
                        helper.format_model_dictionary(mod)
                # Uncomment the following line in order to print each construction step in 3D
                # helper.print_models(mods)

            if mod_length > capacity:                   # Capacity is exceeded, too many models!
                if ((worked_back is False) and is_question):
                    print("interpret_temporal: CAPACITY exceeded - work back PREMISES!")
                    reversed_prems = back.work_back(premises)
                    return self.interpret_temporal(reversed_prems, spatial_parser, unit_test)
                print("interpret_temporal: Memory capacity exceeded, increase it to 500!")
                self.capacity = 500
                return self.interpret_temporal(premises, spatial_parser, unit_test)

            if prem[0] == "?":                         # Found a question at the end, answer it!
                print("interpret_temporal: Answer question!")
                ver_ans.answer([prem[1]], [prem[2]], mods, spatial_parser)
                if not unit_test:
                    helper.print_models(mods)           # print all models in a 3D - plot
                return mods

            print("Interpret: Premise is: ", prem)
            parsed_prem = parser.parse(prem)             # The current parsed premise
            print("Interpret: Parsed Premise is: ", parsed_prem)
            # Continue constructing models with new premise
            mods = self.decide_temporal(parsed_prem, mods, spatial_parser)
            if mods != None:                        # calculate length of models (amount of models)
                mod_length = len(mods)
            else: mod_length = 0
        # Construction of models is done. Print the models and the amount of models and return mods.
        if mods != None:
            print("THE FINAL NUMBER OF MODELS IS:", mod_length)
            if not spatial_parser:
                print("INTERPRET FINISHED; RESULTING MODELS ARE (FORMATED):")
                for mod in mods: # print all models as nested lists (Works only for temporal!)
                    print("-------------------------------------------------------------")
                    mod = helper.normalize_coords(mod)
                    helper.format_model_dictionary(mod)
            if not unit_test:
                helper.print_models(mods) # print all models in a 3D - plot
        else:
            print("No model for this premise set!")
        return mods

    @staticmethod
    def decide_temporal(proposition, models, spatial_parser):
        """
        Function takes a premise (parsed proposition) and first extracts the subject,
        relation and object of the proposition.
        Afterwards, the function calls "find_item_mods" on the subject and on the object
        in order to find all models which already contain the subject or the object.

        Depending on those two model-lists, decide_temporal how to handle the new proposition:
        - if both the subject and object are already contained in the existing models:
            - if those model lists contain the same models: Try to verify the new proposition.
            - if the model lists contain different models: Try to combine existing models with
              the new proposition.
        - if either the subject or the object is already contained in the existing models
          (but not both): call add_item_models and add the new item (either subject or object)
          to the existing models.
        - if neither subject nor object is already contained in the existing models:
          Start a new model with the given subject, object and relation of the proposition.
        """
        relation = helper.get_relation(proposition)
        relation = (relation[0], relation[1], relation[2])
        subj = [helper.get_subject(proposition)]
        obj = [helper.get_object(proposition)]

        subj_mods = helper.find_item_mods(subj, models) # Find set of models in which subj occurs
        obj_mods = helper.find_item_mods(obj, models) # Find set of models in which obj occurs

        if PRINT_MODEL:
            print("Function call - DECIDE with subj_mods =", subj_mods, " obj_mods =", obj_mods)
        if subj_mods != None:
            if obj_mods != None:
                if subj_mods == obj_mods: # VERIFY
                    print("-----------Verify premise!")
                    models = ver_ans.verify_models(relation, subj, obj, models, spatial_parser)
                    return models
                print("-----------Combine models!") # COMBINE
                models = construct.combine_mods(relation, subj, obj, subj_mods, subj_mods, obj_mods)
                return models
            print("-----------Add Object to model!") # ADD
            # The following is needed to make Spatial Problems work with the Temporal model.
            if not spatial_parser and relation == (0, 1, 0): # do not alter relation "while"
                models = construct.add_item_models(relation, obj, subj, models, spatial_parser)
            else:
                models = construct.add_item_models(helper.converse(relation), obj, subj,
                                                   models, spatial_parser)
            return models
        elif obj_mods != None:
            print("-----------Add Subject to model!") # ADD
            models = construct.add_item_models(relation, subj, obj, models, spatial_parser)
            return models
        else:
            print("-----------Start new Model!") # START
            models.append(construct.startmod(relation, subj, obj))
            return models


# ----------------------------SPATIAL REASONING PROBLEM SETS ------------------------------------

# SPATIAL COMBINATION PROBLEMS
COMBO_PROBLEMS = [
    [["the", "square", "is", "behind", "the", "circle"],
     ["the", "cross", "is", "in", "front", "of", "the", "triangle"],
     ["the", "square", "is", "on", "the", "left", "of", "the", "cross"]],
    [["the", "circle", "is", "in", "front", "of", "the", "square"],
     ["the", "triangle", "is", "behind", "the", "cross"],
     ["the", "cross", "is", "on", "the", "right", "of", "the", "square"]],
    [["the", "square", "is", "behind", "the", "circle"],
     ["the", "triangle", "is", "behind", "the", "cross"],
     ["the", "cross", "is", "on", "the", "left", "of", "the", "square"]],
    [["the", "square", "is", "behind", "the", "circle"],
     ["the", "triangle", "is", "behind", "the", "cross"],
     ["the", "line", "is", "above", "the", "triangle"],
     ["the", "cross", "is", "on", "the", "left", "of", "the", "square"]]]

# SPATIAL DEDUCTION PROBLEMS
# (correct: 1, 2, 3, 4, 5, 6 (for 5 and 6 only checked important intermediate results)
DEDUCTIVE_PROBLEMS = [
    [["the", "circle", "is", "on", "the", "right", "of", "the", "square"],
     ["the", "triangle", "is", "on", "the", "left", "of", "the", "circle"],
     ["the", "cross", "is", "in", "front", "of", "the", "triangle"],
     ["the", "line", "is", "in", "front", "of", "the", "circle"],
     ["the", "cross", "is", "on", "the", "left", "of", "the", "line"]],    # Premise follows validly
    [["the", "cross", "is", "in", "front", "of", "the", "circle"],
     ["the", "circle", "is", "in", "front", "of", "the", "triangle"],
     ["the", "cross", "is", "in", "front", "of", "the", "triangle"]],      # Premise follows validly
    [["the", "square", "is", "on", "the", "right", "of", "the", "circle"],
     ["the", "circle", "is", "on", "the", "right", "of", "the", "triangle"],
     ["the", "square", "is", "on", "the", "right", "of", "the", "triangle"]], # Premise fol. validly
    [["the", "square", "is", "on", "the", "right", "of", "the", "circle"],
     ["the", "triangle", "is", "on", "the", "left", "of", "the", "circle"],
     ["the", "square", "is", "on", "the", "right", "of", "the", "triangle"]], # Premise fol. validly
    [["the", "square", "is", "on", "the", "right", "of", "the", "circle"],
     ["the", "cross", "is", "in", "front", "of", "the", "triangle"],
     ["the", "triangle", "is", "on", "the", "left", "of", "the", "square"],
     ["the", "square", "is", "behind", "the", "line"],
     ["the", "line", "is", "on", "the", "right", "of", "the", "cross"]],   # Premise follows validly
    [["the", "triangle", "is", "on", "the", "right", "of", "the", "square"],
     ["the", "circle", "is", "in", "front", "of", "the", "square"],
     ["the", "cross", "is", "on", "the", "left", "of", "the", "square"],
     ["the", "line", "is", "in", "front", "of", "the", "cross"],
     ["the", "line", "is", "on", "the", "right", "of", "the", "ell"],
     ["the", "star", "is", "in", "front", "of", "the", "ell"],
     ["the", "circle", "is", "on", "the", "left", "of", "the", "vee"],
     ["the", "ess", "is", "in", "front", "of", "the", "vee"],
     ["the", "star", "is", "on", "the", "left", "of", "the", "ess"]]]      # Premise follows validly

# SPATIAL INDETERMINATE PROBLEMS --> all correct
INDETERMINATE_PROBLEMS = [
    [["the", "circle", "is", "on", "the", "right", "of", "the", "square"],
     ["the", "triangle", "is", "on", "the", "left", "of", "the", "circle"],
     ["the", "cross", "is", "in", "front", "of", "the", "triangle"],
     ["the", "line", "is", "in", "front", "of", "the", "square"],
     ["the", "cross", "is", "on", "the", "left", "of", "the", "line"]],   # previously possibly true
    [["the", "triangle", "is", "on", "the", "right", "of", "the", "square"],
     ["the", "circle", "is", "in", "front", "of", "the", "square"],
     ["the", "cross", "is", "on", "the", "left", "of", "the", "triangle"],
     ["the", "line", "is", "in", "front", "of", "the", "cross"],
     ["the", "line", "is", "on", "the", "right", "of", "the", "ell"],
     ["the", "star", "is", "in", "front", "of", "the", "ell"],
     ["the", "circle", "is", "on", "the", "left", "of", "the", "vee"],
     ["the", "ess", "is", "in", "front", "of", "the", "vee"],
     ["the", "star", "is", "on", "the", "right", "of", "the", "ess"]],   # previously possibly false
    [["the", "square", "is", "on", "the", "right", "of", "the", "circle"],
     ["the", "triangle", "is", "on", "the", "left", "of", "the", "square"],
     ["the", "triangle", "is", "on", "the", "right", "of", "the", "circle"]],# previously pos. false
    [["the", "square", "is", "on", "the", "right", "of", "the", "circle"],
     ["the", "triangle", "is", "on", "the", "left", "of", "the", "square"],
     ["the", "cross", "is", "in", "front", "of", "the", "triangle"],
     ["the", "line", "is", "in", "front", "of", "the", "circle"],
     ["the", "cross", "is", "on", "the", "right", "of", "the", "line"]], # previously possibly false
    [["the", "square", "is", "on", "the", "right", "of", "the", "circle"],
     ["the", "triangle", "is", "on", "the", "left", "of", "the", "square"],
     ["the", "cross", "is", "in", "front", "of", "the", "triangle"],
     ["the", "line", "is", "in", "front", "of", "the", "circle"],
     ["the", "triangle", "is", "on", "the", "right", "of", "the", "circle"]],# previously pos. false
    [["the", "circle", "is", "on", "the", "right", "of", "the", "square"],
     ["the", "triangle", "is", "on", "the", "left", "of", "the", "circle"],
     ["the", "cross", "is", "in", "front", "of", "the", "triangle"],
     ["the", "line", "is", "in", "front", "of", "the", "square"],
     ["the", "cross", "is", "on", "the", "right", "of", "the", "line"]], # previously possibly false
    [["the", "triangle", "is", "in", "front", "of", "the", "square"],
     ["the", "circle", "is", "on", "the", "right", "of", "the", "square"],
     ["the", "cross", "is", "behind", "the", "triangle"],
     ["the", "line", "is", "on", "the", "right", "of", "the", "cross"],
     ["the", "line", "is", "in", "front", "of", "the", "ell"],
     ["the", "star", "is", "on", "the", "right", "of", "the", "ell"],
     ["the", "circle", "is", "behind", "the", "vee"],
     ["the", "ess", "is", "on", "the", "right", "of", "the", "vee"],
     ["the", "star", "is", "in", "front", "of", "the", "ess"]],          # previously possibly false
    [["the", "triangle", "is", "on", "top", "of", "the", "square"],
     ["the", "circle", "is", "on", "the", "right", "of", "the", "square"],
     ["the", "cross", "is", "below", "the", "triangle"],
     ["the", "line", "is", "on", "the", "right", "of", "the", "cross"],
     ["the", "line", "is", "on", "top", "of", "the", "ell"],
     ["the", "star", "is", "on", "the", "right", "of", "the", "ell"],
     ["the", "circle", "is", "below", "the", "vee"],
     ["the", "ess", "is", "on", "the", "right", "of", "the", "vee"],
     ["the", "star", "is", "on", "top", "of", "the", "ess"]],            # previously possibly false
    [["the", "square", "is", "on", "the", "right", "of", "the", "triangle"],
     ["the", "circle", "is", "on", "the", "left", "of", "the", "square"],
     ["the", "circle", "is", "behind", "the", "star"],
     ["the", "ell", "is", "in", "front", "of", "the", "circle"],
     ["the", "line", "is", "in", "front", "of", "the", "triangle"],
     ["the", "vee", "is", "in", "front", "of", "the", "triangle"],
     ["the", "star", "is", "on", "the", "right", "of", "the", "vee"]]]   # previously possibly false

# SPATIAL PROBLEMS WITH INCONSISTENT PREMISES --> all correct
INCONSISTENT_PROBLEMS = [
    [["the", "square", "is", "on", "the", "left", "of", "the", "circle"],
     ["the", "cross", "is", "in", "front", "of", "the", "square"],
     ["the", "triangle", "is", "on", "the", "right", "of", "the", "circle"],
     ["the", "triangle", "is", "behind", "the", "line"],
     ["the", "line", "is", "on", "the", "left", "of", "the", "cross"]],    # premise is inconsistent
    [["the", "square", "is", "in", "front", "of", "the", "circle"],
     ["the", "triangle", "is", "behind", "the", "circle"],
     ["the", "triangle", "is", "in", "front", "of", "the", "square"]],     # premise is inconsistent
    [["the", "triangle", "is", "on", "the", "right", "of", "the", "square"],
     ["the", "circle", "is", "in", "front", "of", "the", "square"],
     ["the", "cross", "is", "on", "the", "left", "of", "the", "square"],
     ["the", "line", "is", "in", "front", "of", "the", "cross"],
     ["the", "line", "is", "on", "the", "right", "of", "the", "ell"],
     ["the", "star", "is", "in", "front", "of", "the", "ell"],
     ["the", "circle", "is", "on", "the", "left", "of", "the", "vee"],
     ["the", "ess", "is", "in", "front", "of", "the", "vee"],
     ["the", "star", "is", "on", "the", "right", "of", "the", "ess"]]]     # premise is inconsistent


# ----------------------------TEMPORAL REASONING PROBLEM SETS ------------------------------------

# PROBLEMS WITH QUESTIONS
TRANSITIVE_ONE_MODEL_PROBLEMS = [
    [["the", "A", "happens", "before", "the", "B"],
     ["the", "B", "happens", "before", "the", "C"],
     ["the", "D", "happens", "while", "the", "A"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "D", "E"]],                               # D happens before E
    [["the", "A", "happens", "before", "the", "B"],
     ["the", "C", "happens", "after", "the", "B"],
     ["the", "D", "happens", "while", "the", "A"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "D", "E"]],                               # D happens before E
    [["the", "B", "happens", "after", "the", "A"],
     ["the", "B", "happens", "before", "the", "C"],
     ["the", "D", "happens", "while", "the", "A"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "D", "E"]],                               # D happens before E
    [["the", "B", "happens", "after", "the", "A"],
     ["the", "C", "happens", "after", "the", "B"],
     ["the", "D", "happens", "while", "the", "A"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "D", "E"]],                               # D happens before E
    [["the", "A", "happens", "before", "the", "B"],
     ["the", "B", "happens", "before", "the", "C"],
     ["the", "D", "happens", "while", "the", "A"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "E", "D"]],                               # E happens after D
    [["the", "A", "happens", "before", "the", "B"],
     ["the", "C", "happens", "after", "the", "B"],
     ["the", "D", "happens", "while", "the", "A"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "E", "D"]],                               # E happens after D
    [["the", "B", "happens", "after", "the", "A"],
     ["the", "B", "happens", "before", "the", "C"],
     ["the", "D", "happens", "while", "the", "A"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "E", "D"]],                               # E happens after D
    [["the", "B", "happens", "after", "the", "A"],
     ["the", "C", "happens", "after", "the", "B"],
     ["the", "D", "happens", "while", "the", "A"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "E", "D"]]]                               # E happens after D

NON_TRANSITIVE_ONE_MODEL_PROBLEMS = [
    [["the", "A", "happens", "before", "the", "B"],
     ["the", "B", "happens", "before", "the", "C"],
     ["the", "D", "happens", "while", "the", "B"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "D", "E"]],                               # D happens before E
    [["the", "A", "happens", "before", "the", "B"],
     ["the", "C", "happens", "after", "the", "B"],
     ["the", "D", "happens", "while", "the", "B"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "D", "E"]],                               # D happens before E
    [["the", "B", "happens", "after", "the", "A"],
     ["the", "B", "happens", "before", "the", "C"],
     ["the", "D", "happens", "while", "the", "B"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "D", "E"]],                               # D happens before E
    [["the", "B", "happens", "after", "the", "A"],
     ["the", "C", "happens", "after", "the", "B"],
     ["the", "D", "happens", "while", "the", "B"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "D", "E"]],                               # D happens before E
    [["the", "A", "happens", "before", "the", "B"],
     ["the", "B", "happens", "before", "the", "C"],
     ["the", "D", "happens", "while", "the", "B"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "E", "D"]],                               # E happens after D
    [["the", "A", "happens", "before", "the", "B"],
     ["the", "C", "happens", "after", "the", "B"],
     ["the", "D", "happens", "while", "the", "B"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "E", "D"]],                               # E happens after D
    [["the", "B", "happens", "after", "the", "A"],
     ["the", "B", "happens", "before", "the", "C"],
     ["the", "D", "happens", "while", "the", "B"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "E", "D"]],                               # E happens after D
    [["the", "B", "happens", "after", "the", "A"],
     ["the", "C", "happens", "after", "the", "B"],
     ["the", "D", "happens", "while", "the", "B"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "E", "D"]]]                               # E happens after D

MULTIPLE_MODEL_WITH_VALID_ANSWER_PROBLEMS = [
    [["the", "A", "happens", "before", "the", "B"],
     ["the", "C", "happens", "before", "the", "B"],
     ["the", "D", "happens", "while", "the", "B"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "D", "E"]],                               # D happens after E
    [["the", "A", "happens", "before", "the", "B"],
     ["the", "B", "happens", "after", "the", "C"],
     ["the", "D", "happens", "while", "the", "B"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "D", "E"]],                               # D happens after E
    [["the", "B", "happens", "after", "the", "A"],
     ["the", "C", "happens", "before", "the", "B"],
     ["the", "D", "happens", "while", "the", "B"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "D", "E"]],                               # D happens after E
    [["the", "B", "happens", "after", "the", "A"],
     ["the", "B", "happens", "after", "the", "C"],
     ["the", "D", "happens", "while", "the", "B"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "D", "E"]],                               # D happens after E
    [["the", "A", "happens", "before", "the", "B"],
     ["the", "C", "happens", "before", "the", "B"],
     ["the", "D", "happens", "while", "the", "B"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "E", "D"]],                               # E happens before D
    [["the", "A", "happens", "before", "the", "B"],
     ["the", "B", "happens", "after", "the", "C"],
     ["the", "D", "happens", "while", "the", "B"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "E", "D"]],                               # E happens before D
    [["the", "B", "happens", "after", "the", "A"],
     ["the", "C", "happens", "before", "the", "B"],
     ["the", "D", "happens", "while", "the", "B"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "E", "D"]],                               # E happens before D
    [["the", "B", "happens", "after", "the", "A"],
     ["the", "B", "happens", "after", "the", "C"],
     ["the", "D", "happens", "while", "the", "B"],
     ["the", "E", "happens", "while", "the", "C"],
     ["?", "E", "D"]]]                               # E happens before D

MULTIPLE_MODEL_WITH_NO_VALID_ANSWER_PROBLEMS = [
    [["the", "A", "happens", "before", "the", "B"],
     ["the", "C", "happens", "before", "the", "B"],
     ["the", "D", "happens", "while", "the", "C"],
     ["the", "E", "happens", "while", "the", "A"],
     ["?", "D", "E"]],                                 # No definite relation
    [["the", "A", "happens", "before", "the", "B"],
     ["the", "B", "happens", "after", "the", "C"],
     ["the", "D", "happens", "while", "the", "C"],
     ["the", "E", "happens", "while", "the", "A"],
     ["?", "D", "E"]],                                 # No definite relation
    [["the", "B", "happens", "after", "the", "A"],
     ["the", "C", "happens", "before", "the", "B"],
     ["the", "D", "happens", "while", "the", "C"],
     ["the", "E", "happens", "while", "the", "A"],
     ["?", "D", "E"]],                                 # No definite relation
    [["the", "B", "happens", "after", "the", "A"],
     ["the", "B", "happens", "after", "the", "C"],
     ["the", "D", "happens", "while", "the", "C"],
     ["the", "E", "happens", "while", "the", "A"],
     ["?", "D", "E"]],                                 # No definite relation
    [["the", "A", "happens", "before", "the", "B"],
     ["the", "C", "happens", "before", "the", "B"],
     ["the", "D", "happens", "while", "the", "C"],
     ["the", "E", "happens", "while", "the", "A"],
     ["?", "E", "D"]],                                 # No definite relation
    [["the", "A", "happens", "before", "the", "B"],
     ["the", "B", "happens", "after", "the", "C"],
     ["the", "D", "happens", "while", "the", "C"],
     ["the", "E", "happens", "while", "the", "A"],
     ["?", "E", "D"]],                                 # No definite relation
    [["the", "B", "happens", "after", "the", "A"],
     ["the", "C", "happens", "before", "the", "B"],
     ["the", "D", "happens", "while", "the", "C"],
     ["the", "E", "happens", "while", "the", "A"],
     ["?", "E", "D"]],                                 # No definite relation
    [["the", "B", "happens", "after", "the", "A"],
     ["the", "B", "happens", "after", "the", "C"],
     ["the", "D", "happens", "while", "the", "C"],
     ["the", "E", "happens", "while", "the", "A"],
     ["?", "E", "D"]]]                                 # No definite relation

WORKING_BACKWARDS_PROBLEMS = [
    [["the", "X", "happens", "before", "the", "B"],
     ["the", "A", "happens", "before", "the", "B"],
     ["the", "B", "happens", "before", "the", "C"],
     ["the", "C", "happens", "before", "the", "D"],
     ["the", "E", "happens", "before", "the", "D"],
     ["the", "F", "happens", "before", "the", "D"],
     ["?", "A", "D"]],                              # A happens before D
    [["the", "A", "happens", "after", "the", "Z"],
     ["the", "B", "happens", "after", "the", "Z"],
     ["the", "C", "happens", "after", "the", "Z"],
     ["the", "D", "happens", "after", "the", "Z"],
     ["?", "A", "D"]],                              # No definite relation
    [["the", "A", "happens", "before", "the", "B"],
     ["the", "B", "happens", "before", "the", "C"],
     ["the", "D", "happens", "before", "the", "C"],
     ["the", "E", "happens", "before", "the", "C"],
     ["the", "F", "happens", "before", "the", "C"],
     ["the", "G", "happens", "before", "the", "C"],
     ["?", "A", "C"]],                              # A happens before C
    [["the", "A", "happens", "after", "the", "B"],
     ["the", "B", "happens", "after", "the", "C"],
     ["the", "C", "happens", "after", "the", "D"],
     ["the", "E", "happens", "after", "the", "D"],
     ["the", "F", "happens", "after", "the", "D"],
     ["the", "G", "happens", "after", "the", "D"],
     ["?", "A", "D"]]]                              # A happens after D

# PROBLEMS WITHOUT QUESTIONS
COMBINATION_PROBLEMS = [
    [["the", "A", "happens", "while", "the", "B"],
     ["the", "C", "happens", "while", "the", "D"],
     ["the", "A", "happens", "before", "the", "C"]],
    [["the", "B", "happens", "while", "the", "A"],
     ["the", "D", "happens", "while", "the", "C"],
     ["the", "C", "happens", "after", "the", "A"]],
    [["the", "A", "happens", "while", "the", "B"],
     ["the", "D", "happens", "while", "the", "C"],
     ["the", "C", "happens", "before", "the", "A"]],
    [["the", "A", "happens", "while", "the", "B"],
     ["the", "D", "happens", "while", "the", "C"],
     ["the", "E", "happens", "while", "the", "D"],
     ["the", "C", "happens", "before", "the", "A"]]]

DEDUCTION_PROBLEMS = [
    [["the", "B", "happens", "after", "the", "A"],
     ["the", "D", "happens", "before", "the", "B"],
     ["the", "C", "happens", "while", "the", "D"],
     ["the", "E", "happens", "while", "the", "B"],
     ["the", "C", "happens", "before", "the", "E"]], # Premise follows from previous ones
    [["the", "C", "happens", "while", "the", "B"],
     ["the", "B", "happens", "while", "the", "D"],
     ["the", "C", "happens", "while", "the", "D"]],  # Premise follows from previous ones
    [["the", "A", "happens", "after", "the", "B"],
     ["the", "B", "happens", "after", "the", "D"],
     ["the", "A", "happens", "after", "the", "D"]],  # Premise follows from previous ones
    [["the", "A", "happens", "after", "the", "B"],
     ["the", "D", "happens", "before", "the", "B"],
     ["the", "A", "happens", "after", "the", "D"]],  # Premise follows from previous ones
    [["the", "A", "happens", "after", "the", "B"],
     ["the", "C", "happens", "while", "the", "D"],
     ["the", "D", "happens", "before", "the", "A"],
     ["the", "A", "happens", "while", "the", "E"],
     ["the", "E", "happens", "after", "the", "C"]],  # Premise follows from previous ones
    [["the", "D", "happens", "after", "the", "A"],
     ["the", "B", "happens", "while", "the", "A"],
     ["the", "C", "happens", "before", "the", "A"],
     ["the", "E", "happens", "while", "the", "C"],
     ["the", "E", "happens", "after", "the", "F"],
     ["the", "G", "happens", "while", "the", "F"],
     ["the", "B", "happens", "before", "the", "H"],
     ["the", "J", "happens", "while", "the", "H"],
     ["the", "G", "happens", "before", "the", "J"]]] # Premise follows from previous ones

INDETERMINACIES_PROBLEMS = [
    [["the", "B", "happens", "after", "the", "A"],
     ["the", "D", "happens", "before", "the", "B"],
     ["the", "C", "happens", "while", "the", "D"],
     ["the", "E", "happens", "while", "the", "A"],
     ["the", "C", "happens", "before", "the", "E"]], # Premise was hitherto possibly false
    [["the", "D", "happens", "after", "the", "A"],
     ["the", "B", "happens", "while", "the", "A"],
     ["the", "C", "happens", "before", "the", "D"],
     ["the", "E", "happens", "while", "the", "C"],
     ["the", "E", "happens", "after", "the", "F"],
     ["the", "G", "happens", "while", "the", "F"],
     ["the", "B", "happens", "before", "the", "H"],
     ["the", "J", "happens", "while", "the", "H"],
     ["the", "G", "happens", "after", "the", "J"]],  # Premise was hitherto possibly false
    [["the", "A", "happens", "after", "the", "B"],
     ["the", "D", "happens", "before", "the", "A"],
     ["the", "D", "happens", "after", "the", "B"]],  # Premise was hitherto possibly false
    [["the", "A", "happens", "after", "the", "B"],
     ["the", "D", "happens", "before", "the", "A"],
     ["the", "C", "happens", "while", "the", "D"],
     ["the", "E", "happens", "while", "the", "B"],
     ["the", "C", "happens", "after", "the", "E"]],  # Premise was hitherto possibly false
    [["the", "A", "happens", "after", "the", "B"],
     ["the", "D", "happens", "before", "the", "A"],
     ["the", "C", "happens", "while", "the", "D"],
     ["the", "E", "happens", "while", "the", "B"],
     ["the", "D", "happens", "after", "the", "B"]],  # Premise was hitherto possibly false
    [["the", "B", "happens", "after", "the", "A"],
     ["the", "D", "happens", "before", "the", "B"],
     ["the", "C", "happens", "while", "the", "D"],
     ["the", "E", "happens", "while", "the", "A"],
     ["the", "C", "happens", "after", "the", "E"]],  # Premise was hitherto possibly false
    [["the", "D", "happens", "while", "the", "A"],
     ["the", "B", "happens", "after", "the", "A"],
     ["the", "C", "happens", "while", "the", "D"],
     ["the", "E", "happens", "after", "the", "C"],
     ["the", "E", "happens", "while", "the", "F"],
     ["the", "G", "happens", "after", "the", "F"],
     ["the", "B", "happens", "while", "the", "H"],
     ["the", "J", "happens", "after", "the", "H"],
     ["the", "G", "happens", "while", "the", "J"]],  # Premise follows from the previous ones
    [["the", "A", "happens", "after", "the", "D"],
     ["the", "B", "happens", "before", "the", "A"],
     ["the", "B", "happens", "while", "the", "G"],
     ["the", "F", "happens", "while", "the", "B"],
     ["the", "E", "happens", "while", "the", "D"],
     ["the", "H", "happens", "while", "the", "D"],
     ["the", "G", "happens", "after", "the", "H"]]]  # Premise was hitherto possibly false

INCONSISTENT_PREMISES_PROBLEMS = [
    [["the", "A", "happens", "before", "the", "B"],
     ["the", "C", "happens", "while", "the", "A"],
     ["the", "D", "happens", "after", "the", "B"],
     ["the", "D", "happens", "while", "the", "E"],
     ["the", "E", "happens", "before", "the", "C"]], # premise is inconsistent
    [["the", "A", "happens", "while", "the", "B"],
     ["the", "D", "happens", "while", "the", "B"],
     ["the", "D", "happens", "while", "the", "A"]],  # Premise follows from the previous ones
    [["the", "B", "happens", "after", "the", "A"],
     ["the", "D", "happens", "while", "the", "A"],
     ["the", "C", "happens", "before", "the", "A"],
     ["the", "E", "happens", "while", "the", "C"],
     ["the", "E", "happens", "after", "the", "F"],
     ["the", "G", "happens", "while", "the", "F"],
     ["the", "B", "happens", "before", "the", "H"],
     ["the", "J", "happens", "while", "the", "H"],
     ["the", "G", "happens", "after", "the", "J"]]]  # premise is inconsistent


# ---------------------------- UNIT TEST FOR TEMPORAL MODEL ------------------------------------

class TestInterpret(unittest.TestCase):
    """
    Unittest-Class which tests each of the 9 different problem types
    in a seperate test-function.
    (Note: The Unit-Tests are almost the same as in program version v1.
    The new thing is that there is a function "translate_all_dicts" which
    evaluates the output of "interpret" before comparing it in the assertions.
    The new function simply translates the new output dictionaries to nested lists,
    like the models were stored in version v1.)
    (Note 2: In the unit-tests, the models are not 3D-plotted, otherwise the unit-test
    stops after each finished model)
    """

    def test_transitive_one_model(self):
        """
        Unittest consisting of eight problems of the problem-type transitive
        one-model problems (These problem-types lead to only one model which does
        require transitive relations in order to solve the problem-question).
        Model should always answer with a definite relation like before/ after/ while
        between the two events.
        """
        model = MainModule()
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            TRANSITIVE_ONE_MODEL_PROBLEMS[0], False, True)),
                         [[[['A'], ['D']], [['B'], [None]], [['C'], ['E']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            TRANSITIVE_ONE_MODEL_PROBLEMS[1], False, True)),
                         [[[['A'], ['D']], [['B'], [None]], [['C'], ['E']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            TRANSITIVE_ONE_MODEL_PROBLEMS[2], False, True)),
                         [[[['A'], ['D']], [['B'], [None]], [['C'], ['E']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            TRANSITIVE_ONE_MODEL_PROBLEMS[3], False, True)),
                         [[[['A'], ['D']], [['B'], [None]], [['C'], ['E']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            TRANSITIVE_ONE_MODEL_PROBLEMS[4], False, True)),
                         [[[['A'], ['D']], [['B'], [None]], [['C'], ['E']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            TRANSITIVE_ONE_MODEL_PROBLEMS[5], False, True)),
                         [[[['A'], ['D']], [['B'], [None]], [['C'], ['E']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            TRANSITIVE_ONE_MODEL_PROBLEMS[6], False, True)),
                         [[[['A'], ['D']], [['B'], [None]], [['C'], ['E']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            TRANSITIVE_ONE_MODEL_PROBLEMS[7], False, True)),
                         [[[['A'], ['D']], [['B'], [None]], [['C'], ['E']]]])

    def test_non_transitive_one_model(self):
        """
        Unittest consisting of eight problems of the problem-type non-transitive
        one-model problems (These problem-types lead to only one model which doesn´t
        require transitive relations in order to solve the problem-question).
        Model should always answer with a definite relation like before/ after/ while
        between the two events.
        """
        model = MainModule()
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            NON_TRANSITIVE_ONE_MODEL_PROBLEMS[0], False, True)),
                         [[[['A'], [None]], [['B'], ['D']], [['C'], ['E']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            NON_TRANSITIVE_ONE_MODEL_PROBLEMS[1], False, True)),
                         [[[['A'], [None]], [['B'], ['D']], [['C'], ['E']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            NON_TRANSITIVE_ONE_MODEL_PROBLEMS[2], False, True)),
                         [[[['A'], [None]], [['B'], ['D']], [['C'], ['E']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            NON_TRANSITIVE_ONE_MODEL_PROBLEMS[3], False, True)),
                         [[[['A'], [None]], [['B'], ['D']], [['C'], ['E']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            NON_TRANSITIVE_ONE_MODEL_PROBLEMS[4], False, True)),
                         [[[['A'], [None]], [['B'], ['D']], [['C'], ['E']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            NON_TRANSITIVE_ONE_MODEL_PROBLEMS[5], False, True)),
                         [[[['A'], [None]], [['B'], ['D']], [['C'], ['E']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            NON_TRANSITIVE_ONE_MODEL_PROBLEMS[6], False, True)),
                         [[[['A'], [None]], [['B'], ['D']], [['C'], ['E']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            NON_TRANSITIVE_ONE_MODEL_PROBLEMS[7], False, True)),
                         [[[['A'], [None]], [['B'], ['D']], [['C'], ['E']]]])

    def test_multiple_model_with_answer(self):
        """
        Unittest consisting of eight problems of the problem-type multiple models with
        valid answer (These problem-types lead to different models, which all lead to the
        same conclusion for the problem-question).
        Model should always answer with a definite relation like before/ after/ while
        between the two events.
        """
        model = MainModule()
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            MULTIPLE_MODEL_WITH_VALID_ANSWER_PROBLEMS[0], False, True)),
                         [[[['A'], [None]], [['C'], ['E']], [['B'], ['D']]],
                          [[['C'], ['E']], [['A'], [None]], [['B'], ['D']]],
                          [[['A'], ['C'], ['E']], [['B'], ['D'], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            MULTIPLE_MODEL_WITH_VALID_ANSWER_PROBLEMS[1], False, True)),
                         [[[['A'], [None]], [['C'], ['E']], [['B'], ['D']]],
                          [[['C'], ['E']], [['A'], [None]], [['B'], ['D']]],
                          [[['A'], ['C'], ['E']], [['B'], ['D'], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            MULTIPLE_MODEL_WITH_VALID_ANSWER_PROBLEMS[2], False, True)),
                         [[[['A'], [None]], [['C'], ['E']], [['B'], ['D']]],
                          [[['C'], ['E']], [['A'], [None]], [['B'], ['D']]],
                          [[['A'], ['C'], ['E']], [['B'], ['D'], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            MULTIPLE_MODEL_WITH_VALID_ANSWER_PROBLEMS[3], False, True)),
                         [[[['A'], [None]], [['C'], ['E']], [['B'], ['D']]],
                          [[['C'], ['E']], [['A'], [None]], [['B'], ['D']]],
                          [[['A'], ['C'], ['E']], [['B'], ['D'], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            MULTIPLE_MODEL_WITH_VALID_ANSWER_PROBLEMS[4], False, True)),
                         [[[['A'], [None]], [['C'], ['E']], [['B'], ['D']]],
                          [[['C'], ['E']], [['A'], [None]], [['B'], ['D']]],
                          [[['A'], ['C'], ['E']], [['B'], ['D'], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            MULTIPLE_MODEL_WITH_VALID_ANSWER_PROBLEMS[5], False, True)),
                         [[[['A'], [None]], [['C'], ['E']], [['B'], ['D']]],
                          [[['C'], ['E']], [['A'], [None]], [['B'], ['D']]],
                          [[['A'], ['C'], ['E']], [['B'], ['D'], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            MULTIPLE_MODEL_WITH_VALID_ANSWER_PROBLEMS[6], False, True)),
                         [[[['A'], [None]], [['C'], ['E']], [['B'], ['D']]],
                          [[['C'], ['E']], [['A'], [None]], [['B'], ['D']]],
                          [[['A'], ['C'], ['E']], [['B'], ['D'], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            MULTIPLE_MODEL_WITH_VALID_ANSWER_PROBLEMS[7], False, True)),
                         [[[['A'], [None]], [['C'], ['E']], [['B'], ['D']]],
                          [[['C'], ['E']], [['A'], [None]], [['B'], ['D']]],
                          [[['A'], ['C'], ['E']], [['B'], ['D'], [None]]]])

    def test_multiple_model_with_no_ans(self):
        """
        Unittest consisting of eight problems of the problem-type multiple models with
        no valid answer (These problem-types lead to different models, which lead
        to different conclusions for the problem-question, which is why there is
        no definite answer).
        Model should always answer with "There is no definite relation between the two events."
        """
        model = MainModule()
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            MULTIPLE_MODEL_WITH_NO_VALID_ANSWER_PROBLEMS[0], False, True)),
                         [[[['A'], ['E']], [['C'], ['D']], [['B'], [None]]],
                          [[['C'], ['D']], [['A'], ['E']], [['B'], [None]]],
                          [[['A'], ['C'], ['D'], ['E']], [['B'], [None], [None], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            MULTIPLE_MODEL_WITH_NO_VALID_ANSWER_PROBLEMS[1], False, True)),
                         [[[['A'], ['E']], [['C'], ['D']], [['B'], [None]]],
                          [[['C'], ['D']], [['A'], ['E']], [['B'], [None]]],
                          [[['A'], ['C'], ['D'], ['E']], [['B'], [None], [None], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            MULTIPLE_MODEL_WITH_NO_VALID_ANSWER_PROBLEMS[2], False, True)),
                         [[[['A'], ['E']], [['C'], ['D']], [['B'], [None]]],
                          [[['C'], ['D']], [['A'], ['E']], [['B'], [None]]],
                          [[['A'], ['C'], ['D'], ['E']], [['B'], [None], [None], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            MULTIPLE_MODEL_WITH_NO_VALID_ANSWER_PROBLEMS[3], False, True)),
                         [[[['A'], ['E']], [['C'], ['D']], [['B'], [None]]],
                          [[['C'], ['D']], [['A'], ['E']], [['B'], [None]]],
                          [[['A'], ['C'], ['D'], ['E']], [['B'], [None], [None], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            MULTIPLE_MODEL_WITH_NO_VALID_ANSWER_PROBLEMS[4], False, True)),
                         [[[['A'], ['E']], [['C'], ['D']], [['B'], [None]]],
                          [[['C'], ['D']], [['A'], ['E']], [['B'], [None]]],
                          [[['A'], ['C'], ['D'], ['E']], [['B'], [None], [None], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            MULTIPLE_MODEL_WITH_NO_VALID_ANSWER_PROBLEMS[5], False, True)),
                         [[[['A'], ['E']], [['C'], ['D']], [['B'], [None]]],
                          [[['C'], ['D']], [['A'], ['E']], [['B'], [None]]],
                          [[['A'], ['C'], ['D'], ['E']], [['B'], [None], [None], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            MULTIPLE_MODEL_WITH_NO_VALID_ANSWER_PROBLEMS[6], False, True)),
                         [[[['A'], ['E']], [['C'], ['D']], [['B'], [None]]],
                          [[['C'], ['D']], [['A'], ['E']], [['B'], [None]]],
                          [[['A'], ['C'], ['D'], ['E']], [['B'], [None], [None], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            MULTIPLE_MODEL_WITH_NO_VALID_ANSWER_PROBLEMS[7], False, True)),
                         [[[['A'], ['E']], [['C'], ['D']], [['B'], [None]]],
                          [[['C'], ['D']], [['A'], ['E']], [['B'], [None]]],
                          [[['A'], ['C'], ['D'], ['E']], [['B'], [None], [None], [None]]]])

    def test_working_backwards_problems(self):
        """
        Unittest consisting of four problems of the problem-type which requires
        re-working the premise set (while constructing models, capacity (working
        memory) is exceeded and interpret needs to search for only the relevant
        premises in the problem and start model-constructing process again with these).
        """
        model = MainModule()
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            WORKING_BACKWARDS_PROBLEMS[0], False, True)),
                         [[[['A']], [['B']], [['C']], [['D']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            WORKING_BACKWARDS_PROBLEMS[1], False, True)),
                         [[[['Z']], [['D']], [['A']]], [[['Z']], [['A']], [['D']]],
                          [[['Z'], [None]], [['A'], ['D']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            WORKING_BACKWARDS_PROBLEMS[2], False, True)),
                         [[[['A']], [['B']], [['C']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            WORKING_BACKWARDS_PROBLEMS[3], False, True)),
                         [[[['D']], [['C']], [['B']], [['A']]]])

    def test_combination_problems(self):
        """
        Unittest consisting of four problems of the problem-type combination problems
        (combine different existing models to one big model).
        """
        model = MainModule()
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            COMBINATION_PROBLEMS[0], False, True)),
                         [[[['B'], ['A']], [['D'], ['C']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            COMBINATION_PROBLEMS[1], False, True)),
                         [[[['A'], ['B']], [['C'], ['D']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            COMBINATION_PROBLEMS[2], False, True)),
                         [[[[None], ['C'], ['D']], [['B'], ['A'], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            COMBINATION_PROBLEMS[3], False, True)),
                         [[[[None], ['C'], ['D'], ['E']], [['B'], ['A'], [None], [None]]]])

    def test_deduction_problems(self):
        """
        Unittest consisting of six problems of the problem-type deduction problems.
        Since there is no question contained in the problem, the programm simply
        tests if the existing models all hold with the last premise in the problem
        (verify --> last premise should follow from the previous ones if it is
        consistent with the model(s)).
        """
        model = MainModule()
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            DEDUCTION_PROBLEMS[0], False, True)),
                         [[[['A'], [None]], [['D'], ['C']], [['B'], ['E']]],
                          [[['D'], ['C']], [['A'], [None]], [['B'], ['E']]],
                          [[['A'], ['D'], ['C']], [['B'], ['E'], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            DEDUCTION_PROBLEMS[1], False, True)),
                         [[[['B'], ['C'], ['D']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            DEDUCTION_PROBLEMS[2], False, True)),
                         [[[['D']], [['B']], [['A']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            DEDUCTION_PROBLEMS[3], False, True)),
                         [[[['D']], [['B']], [['A']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            DEDUCTION_PROBLEMS[4], False, True)),
                         [[[['D'], ['C']], [['B'], [None]], [['A'], ['E']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            DEDUCTION_PROBLEMS[5], False, True)),
                         [[[[None], ['F'], ['G']], [['C'], ['E'], [None]],
                           [['A'], ['B'], [None]], [['D'], ['H'], ['J']]]])

    def test_indeterminacies_problems(self):
        """
        Unittest consisting of eight problems of the problem-type
        problems with indeterminacies.
        """
        model = MainModule()
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            INDETERMINACIES_PROBLEMS[0], False, True)),
                         [[[['D'], ['C']], [['A'], ['E']], [['B'], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            INDETERMINACIES_PROBLEMS[1], False, True)),
                         [[[['A'], ['B'], [None]], [[None], ['H'], ['J']], [[None], ['F'], ['G']],
                           [['C'], ['E'], [None]], [['D'], [None], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            INDETERMINACIES_PROBLEMS[2], False, True)),
                         [[[['B']], [['D']], [['A']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            INDETERMINACIES_PROBLEMS[3], False, True)),
                         [[[['B'], ['E']], [['D'], ['C']], [['A'], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            INDETERMINACIES_PROBLEMS[4], False, True)),
                         [[[['B'], ['E']], [['D'], ['C']], [['A'], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            INDETERMINACIES_PROBLEMS[5], False, True)),
                         [[[['A'], ['E']], [['D'], ['C']], [['B'], [None]]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            INDETERMINACIES_PROBLEMS[6], False, True)),
                         [[[['A'], ['D'], ['C'], [None]], [['B'], ['H'], ['E'], ['F']],
                           [[None], ['J'], [None], ['G']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            INDETERMINACIES_PROBLEMS[7], False, True)),
                         [[[['D'], ['E'], ['H']], [['B'], ['G'], ['F']], [['A'], [None], [None]]]])

    def test_inconsistent_premises(self):
        """
        Unittest consisting of three problems of the problem-type inconsistent premisses.
        These kind of problems are problems where the last premise is inconsistent with the
        previous ones, leading to no solution for the problem.
        """
        model = MainModule()
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            INCONSISTENT_PREMISES_PROBLEMS[0], False, True)), None)
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            INCONSISTENT_PREMISES_PROBLEMS[1], False, True)), [[[['B'], ['A'], ['D']]]])
        self.assertEqual(helper.translate_all_dicts(model.interpret_temporal(
            INCONSISTENT_PREMISES_PROBLEMS[2], False, True)), None)


# ---------------------------- MAIN FUNCTION ------------------------------------------------------

def main():
    """
    Main-function.

    Quickstart:
    1) To run a Spatial Problem with the Spatial Model:
    - process_problem(problem-number, name-of-spatial-problem-set, "spatial")
    2) To run a Spatial Problem with the Temporal Model:
    - process_problem(problem-number, name-of-spatial-problem-set, "temporal", True)
    3) To run a Temporal Problem with the Temporal Model:
    - process_problem(problem-number, name-of-temporal-problem-set, "temporal")
    4) To run a Temporal Problem with the Spatial Model:
    - process_problem(problem-number, name-of-temporal-problem-set, "spatial", False)
    (See examples below)

    Instead of process_problem, process_all_problems can be called without a number
    as explained above to process all problems of a given problem-set.
    """

    spatial_model = MainModule()    
    spatial_model.process_problem(1, INDETERMINATE_PROBLEMS, "spatial")                 # 1)
    #spatial_model.process_all_problems(COMBINATION_PROBLEMS, "temporal")               # 5)

    #temporal_model = MainModule()
    #temporal_model.process_problem(1, NON_TRANSITIVE_ONE_MODEL_PROBLEMS, "temporal")   # 3)

    #temporal_spatial_model = MainModule()
    #temporal_spatial_model.process_problem(1, COMBO_PROBLEMS, "temporal", True)        # 2)
    #temporal_spatial_model.process_problem(3, DEDUCTION_PROBLEMS, "spatial", False)    # 4)


if __name__ == "__main__":
    main()
    unittest.main()
