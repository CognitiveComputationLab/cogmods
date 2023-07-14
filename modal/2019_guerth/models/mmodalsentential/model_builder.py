"""Builder for Mental Models and Fully Explicit Models

This module implements a model builder based on the 'Model Theory'.
"""
from itertools import product
import numpy as np
from .assertion_parser import Expr, preprocess_modals, parse_one, parse_all
from .logger import logging


class MentalModel():
    """Class for Mental Models and Fully Explicit Models.

    This class combines both Mental Models and Fully Explicit Models.

    The possibilities are represented by the rows of 2D numpy arrays:
        1 means positive clause
       -1 means negative clause
        0 means gap (only in Mental Models)

    Example:
        clauses = ['A', 'B']
        poss = [[1,0], [0,1], [-1,-1]]

        These two arguments represent the following Mental Model:
         A
             B
        ~A  ~B

    Arguments:
        clauses {list} -- clauses/atoms of Mental Model
        poss {np.array} -- possibilities of Mental Model
        ell {int} -- counter of ellipsis of Mental Model
        full_clauses {list} -- clauses/atoms of Fully Explicit Model
        full_poss {np.array} -- possibilities of Fully Explicit Model
    """

    def __init__(self, clauses, poss, ell, full_clauses, full_poss, modality='epistemic by default'):
        self.clauses = clauses
        self.poss = poss
        self.ell = ell

        self.full_poss = full_poss
        self.full_clauses = full_clauses

        self.modality = modality


    def __str__(self):
        s = "Mental Model (" + self.modality + "):\n"
        if len(self.clauses) == 0:
            s+= "nil\n"
        for p in self.poss:
            for i, cl in enumerate(p.tolist()):
                if i != 0:
                    # s += "\t"
                    if last_negation:
                        s += " " * 7
                    else:
                        s += " " * 8


                last_negation = False
                if cl == 1:
                    s += self.clauses[i]
                elif cl == -1:
                    s += "¬" + self.clauses[i]
                    last_negation = True
                elif cl == 0:
                    s += " " * (len(self.clauses[i]))
                else:
                    raise Exception
            s += "\n"

        for _ in range(self.ell):
            # s += "\t" * ((len(self.poss[0])//2)-1)
            s += " " * ((sum(len(s) for s in self.clauses) + (len(self.clauses) - 1) * 8) // 2 - 1)
            s += "...\n"

        s += "\nFully Explicit Model (" + self.modality + "):\n"
        if len(self.full_clauses) == 0:
            s+= "nil\n"
        for p in self.full_poss:
            for i, cl in enumerate(p.tolist()):
                if i != 0:
                    # s += "\t"
                    if last_negation:
                        s += " " * 7
                    else:
                        s += " " * 8

                last_negation = False
                if cl == 1:
                    s += self.full_clauses[i]
                elif cl == -1:
                    s += "¬" + self.full_clauses[i]
                    last_negation = True
                elif cl == 0:
                    raise Exception
                else:
                    raise Exception
            s += "\n"

        return s

    def get_column(self, clause):
        """Get column of clause.

        Arguments:
            clause {str} -- the clause to retrieve

        Raises:
            Exception: if clause does not exist

        Returns:
            np.array -- column
        """
        if clause in self.clauses:
            i = self.clauses.index(clause)
            return self.poss[:, i]
        else:
            raise Exception

    def full_get_column(self, clause):
        """Get column of clause.

        Arguments:
            clause {str} -- the clause to retrieve

        Raises:
            Exception: if clause does not exist

        Returns:
            np.array -- column
        """
        if clause in self.full_clauses:
            i = self.full_clauses.index(clause)
            return self.full_poss[:, i]
        else:
            raise Exception

def mental_model(parsed):
    """Turn parsed Expr into Mental Model (recursive function)

    Arguments:
        parsed {Expr} -- [description]

    Raises:
        Exception: input is not an Expr
        NotImplementedError: unknown operator
        Exception: if argument of Expr is not Expr or Mental Model

    Returns:
        MentalModel -- Mental Model of input
    """
    if isinstance(parsed, Expr):
        if parsed.op == "id":
            return id_model(parsed.args)

        args = []
        for arg in parsed.args:
            if isinstance(arg, Expr):
                args.append(mental_model(arg))
            elif isinstance(arg, MentalModel):
                args.append(arg)
            else:
                raise Exception

        if parsed.op == '~':
            return not_model(args[0])
        elif parsed.op[0] == '<' and parsed.op[-1] == '>' and parsed.op[1] != '-':
            return poss_model(args[0], parsed.op)
        elif parsed.op[0] == '[' and parsed.op[-1] == ']':
            return nec_model(args[0], parsed.op)
        # elif parsed.op == '<>':
        #     return poss_model(args[0])
        # elif parsed.op == '[]':
        #     return nec_model(args[0])
        elif parsed.op == '&':
            return and_model(args[0], args[1])
        elif parsed.op == '|':
            return or_model(args[0], args[1])
        elif parsed.op == '^':
            return xor_model(args[0], args[1])
        elif parsed.op == '->':
            return cond_model(args[0], args[1])
        elif parsed.op == '<->':
            return bicond_model(args[0], args[1])
        else:
            raise NotImplementedError
    else:
        raise Exception

def remove_duplicates(arr):
    """Remove identical rows.

    Arguments:
        arr {np.array} -- 2D array

    Returns:
        np.array -- array without identical rows
    """
    if arr.size == 0:
        return arr
    _, idx = np.unique(arr, return_index=True, axis=0)
    return arr[np.sort(idx)]

def id_model(args):
    """Create Mental Model of clause.

    Arguments:
        args {list} -- clauses (should be 1)

    Returns:
        MentalModel -- created Model
    """
    return MentalModel([args[0]], np.array([[1]]), 0, [args[0]], np.array([[1]]))

def and_model(model_1, model_2):
    """Conjunction two models.

    Arguments:
        model_1 {MentalModel} -- operand 1
        model_2 {MentalModel} -- operand 2

    Returns:
        MentalModel -- [description]
    """
    if not same_modalities(model_1.modality, model_2.modality):
        raise ValueError("modalities of models do not match: " + model_1.modality + " and " + model_2.modality)
    if model_1.modality != 'epistemic by default':
        modality = model_1.modality
    elif model_2.modality != 'epistemic by default':
        modality = model_2.modality
    else:
        modality = 'epistemic by default'

    # Mental Model:
    common = [
        (cl1, i1, i2)
        for i1, cl1 in enumerate(model_1.clauses)
        for i2, cl2 in enumerate(model_2.clauses)
        if cl1 == cl2]
    new_poss = [
        conjoin(p1, p2, common)
        for p1 in model_1.poss
        for p2 in model_2.poss
        if match(p1, p2, common)]
    # new_poss = remove_duplicates(np.array(new_poss))
    new_poss = np.array(new_poss)
    new_ell = model_1.ell + model_2.ell
    if new_poss.size == 0:
        new_clauses = []
    else:
        cl2 = [cl for cl in model_2.clauses if cl not in model_1.clauses]
        new_clauses = model_1.clauses + cl2

    # Fully Explcit Model:
    full_common = [
        (cl1, i1, i2)
        for i1, cl1 in enumerate(model_1.full_clauses)
        for i2, cl2 in enumerate(model_2.full_clauses)
        if cl1 == cl2]
    new_full_poss = [
        conjoin(p1, p2, full_common)
        for p1 in model_1.full_poss
        for p2 in model_2.full_poss
        if match(p1, p2, full_common)]
    # new_full_poss = remove_duplicates(np.array(new_full_poss))
    new_full_poss = np.array(new_full_poss)
    if new_full_poss.size == 0:
        new_full_clauses = []
    else:
        cl2 = [cl for cl in model_2.full_clauses if cl not in model_1.full_clauses]
        new_full_clauses = model_1.full_clauses + cl2

    return MentalModel(new_clauses, new_poss, new_ell, new_full_clauses, new_full_poss, modality)

def conjoin(p1, p2, common):
    """Conjoin two possibilities.

    Arguments:
        p1 {np.array} -- first possibility
        p2 {np.array} -- second possibility
        common {(str,int,int)} -- common clauses

    Returns:
        np.array -- conjoined possibility
    """
    to_delete = [c for _,_,c in common]
    p2 = np.delete(p2, to_delete)
    new_possibility = np.hstack((p1, p2))
    return new_possibility

def match(p1, p2, common):
    """Return whether clauses in common match.

    Arguments:
        p1 {np.array} -- first possibility
        p2 {np.array} -- second possibility
        common {(str,int,int)} -- common clauses

    Returns:
        True/False -- Do clauses in common match?
    """
    for c in common:
        if p1[c[1]] != p2[c[2]]:
            return False
    return True

def not_model(model):
    """Negate model.

    If the model is not a simple model of one clause, then the negation of the
    Mental Model is the same as the negation of the Fully Explicit Model.

    Arguments:
        model {MentalModel} -- model to negate

    Returns:
        MentalModel -- negated model
    """
    # Fully Explicit Model:
    full_partition = list(product([1, -1], repeat=len(model.full_clauses)))
    full_complement = np.array([list(p) for p in full_partition if list(p) not in model.full_poss.tolist()])
    if full_complement.size == 0:
        full_clauses = []
    else:
        full_clauses = model.full_clauses

    # Mental Model:
    if len(model.clauses) == 1 and model.ell == 0:
        new_clauses = model.clauses
        complement = model.poss * -1
        new_ell = model.ell
    else:
        new_clauses = full_clauses
        complement = full_complement
        new_ell = 0

    return MentalModel(new_clauses, complement, new_ell, full_clauses, full_complement)

def merge_mental_and_full(mental, full):
    """Merge Mental Model and Fully Explicit Model.

    Arguments:
        mental {MentalModel} -- from this Model the Mental Model part is used
        full {MentalModel} -- from this Model the Fully Explicit Model part is used

    Returns:
        MentalModel -- merged from inputs
    """
    mental.full_clauses = full.full_clauses
    mental.full_poss = full.full_poss
    return mental

def or_model(model1, model2):
    """Disjunction of two models.

    Arguments:
        model1 {MentalModel} -- first operand
        model2 {MentalModel} -- second operand

    Returns:
        MentalModel -- the disjunction model
    """
    mental = merge_models(model1, model2, and_model(model1, model2))
    fully = merge_fullex(
        and_model(model1, not_model(model2)),
        and_model(not_model(model1), model2),
        and_model(model1, model2)
    )
    return merge_mental_and_full(mental, fully)

def xor_model(model1, model2):
    """Exclusive disjunction of two models.

    Arguments:
        model1 {MentalModel} -- first operand
        model2 {MentalModel} -- second operand

    Returns:
        MentalModel -- the exclusive disjunction model
    """
    mental = merge_models(model1, model2)
    fully = merge_fullex(
        and_model(model1, not_model(model2)),
        and_model(not_model(model1), model2)
    )
    return merge_mental_and_full(mental, fully)

def cond_model(model1, model2):
    """Conditional.

    Arguments:
        model1 {MentalModel} -- antecedent
        model2 {MentalModel} -- consequent

    Returns:
        MentalModel -- the conditional model
    """
    mental = and_model(model1, model2)
    mental.ell += 1
    fully = merge_fullex(
        and_model(model1, model2),
        and_model(not_model(model1), not_model(model2)),
        and_model(not_model(model1), model2)
    )
    return merge_mental_and_full(mental, fully)


def bicond_model(model1, model2):
    """Biconditional.

    Arguments:
        model1 {MentalModel} -- antecedent
        model2 {MentalModel} -- consequent

    Returns:
        MentalModel -- the biconditional model
    """
    mental = and_model(model1, model2)
    mental.ell += 1
    fully = merge_fullex(
        and_model(model1, model2),
        and_model(not_model(model1), not_model(model2))
    )
    return merge_mental_and_full(mental, fully)

def same_modalities(m1, m2):
    """Return if modalities are valid/legal together.

    Arguments:
        m1 {str} -- modality
        m2 {str} -- modality

    Returns:
        bool -- True if valid else False
    """
    if m1 == 'epistemic by default' or m2 == 'epistemic by default':
        return True
    return m1 == m2

def poss_model(model1, modal_op):
    """Possible (modal) of model

    Arguments:
        model1 {MentalModel} -- operand

    Returns:
        MentalModel -- result
    """
    weight_set = False
    if 'e' in modal_op:
        modality = 'epistemic'
        if ':' in modal_op:
            weight = round(float(modal_op[3:-1]), 1)
            weight_set = True
    elif 'a' in modal_op:
        modality = 'alethic'
    elif 'd' in modal_op:
        modality = 'deontic'
    else:
        modality = 'epistemic by default'

    if not same_modalities(model1.modality, modality):
        raise ValueError("modalities do not match")
    if modality != 'epistemic by default':
        model1.modality = modality


    if not weight_set or weight == 0.5:
        model1.ell += 1
        fully = merge_fullex(
            model1,
            not_model(model1)
        )
    elif weight == 0:
        model1 = not_model(model1)
        fully = model1
    elif weight == 1:
        fully = model1
    else:
        weight = int(10 * weight)
        mental = model1
        fully = model1
        for i in range(weight - 1):
            fully = merge_fullex(fully, model1)
        for i in range(10 - weight):
            fully = merge_fullex(fully, not_model(model1))
        for i in range(weight - 1):
            mental = merge_models(mental, model1)
        mental.ell += 1
        model1 = mental

    return merge_mental_and_full(model1, fully)


def nec_model(model1, modal_op):
    """Necessary (modal) of model

    Arguments:
        model1 {MentalModel} -- operand

    Returns:
        MentalModel -- result
    """
    if 'e' in modal_op:
        modality = 'epistemic'
    elif 'a' in modal_op:
        modality = 'alethic'
    elif 'd' in modal_op:
        modality = 'deontic'
    else:
        modality = 'epistemic by default'
    model1.modality = modality
    return model1


def clauses_in_common(*models):
    """Return all clauses the models have in common.

    Returns:
        list -- clauses
    """
    common = []
    for m in models:
        for cl in m.clauses:
            if cl not in common:
                common.append(cl)
    return common

def full_clauses_in_common(*models):
    """Return all clauses the models have in common.

    Returns:
        list -- clauses
    """
    common = []
    for m in models:
        for cl in m.full_clauses:
            if cl not in common:
                common.append(cl)
    return common


def merge_models(*models):
    """Merge Mental Model parts of Models.

    Returns:
        MentalModel -- merged model
    """
    modality = models[0].modality
    for m in models:
        if not same_modalities(m.modality, modality):
            raise ValueError("modalities do not match")

    common = clauses_in_common(*models)
    padded = [padd_model(m, common) for m in models]
    poss = [p.poss for p in padded]
    # merged_poss = remove_duplicates(np.vstack(poss))
    merged_poss = np.vstack(poss)
    merged_ell = sum([p.ell for p in padded])
    if merged_poss.size == 0:
        common = []
    return MentalModel(common, merged_poss, merged_ell, None, None, modality)

def merge_fullex(*models):
    """Merge Fully Explicit Model parts of Models.

    Returns:
        MentalModel -- merged model
    """
    modality = models[0].modality
    for m in models:
        if not same_modalities(m.modality, modality):
            raise ValueError("modalities do not match: " + m.modality + modality)

    common = full_clauses_in_common(*models)
    padded = [padd_fully(m, common) for m in models]
    poss = [p.full_poss for p in padded]
    # merged_poss = remove_duplicates(np.vstack(poss))
    merged_poss = np.vstack(poss)
    if merged_poss.size == 0:
        common = []
    return MentalModel(None, None, None, common, merged_poss, modality)

def padd_model(model, common):
    """Extend Mental Model to include clauses in common.

    The new columns get initialized by 0, indicating gaps in Mental Models.

    Arguments:
        model {MentalModel} -- model to extend
        common {list} -- list of clauses

    Returns:
        MentalModel -- extended model
    """
    n_columns = len(common)
    n_rows = len(model.poss)
    new_poss = np.zeros((n_rows, n_columns), dtype=int)
    for i, cl in enumerate(common):
        if cl in model.clauses:
            new_poss[:, i] = model.get_column(cl)
    return MentalModel(common, new_poss, model.ell, None, None)

def padd_fully(model, common):
    """Extend Fully Explicit Model to include clauses in common.

    Arguments:
        model {MentalModel} -- model to extend
        common {list} -- list of clauses

    Returns:
        MentalModel -- extended model
    """
    n_columns = len(common)
    n_rows = len(model.full_poss)
    new_poss = np.zeros((n_rows, n_columns), dtype=int)
    for i, cl in enumerate(common):
        if cl in model.full_clauses:
            new_poss[:, i] = model.full_get_column(cl)
    return MentalModel(None, None, None, common, new_poss)

def premises_model(premises):
    """Turn premises into one model. Apply modulation.

    Arguments:
        premises {list} -- list of premise strings

    Returns:
        MentalModel -- the model
    """
    if len(premises) == 1:
        m = mental_model(premises[0])
    elif len(premises) == 2:
        m = and_model(mental_model(premises[0]), mental_model(premises[1]))
    else:
        m = and_model(mental_model(premises[0]), mental_model(premises[1]))
        for i in range(2, len(premises)):
            m = and_model(m, mental_model(premises[i]))
    return apply_modulation(m)

knowledge = []

clauses = ['God exists', 'atheism is right']
poss = np.array([[1, -1], [-1, 1]])
knowledge.append(MentalModel([], [], 0, clauses, poss))



def apply_modulation(model, knowledge=knowledge):
    """Apply modulation to model

    Arguments:
        model {MentalModel} -- model to apply modulation to

    Keyword Arguments:
        knowledge {list} -- list of MentalModels (default: {knowledge})

    Returns:
        MentalModel -- the modified model
    """
    if len(knowledge) == 0:
        return model

    applied = []
    not_finished = True
    while not_finished:
        not_finished = False
        for k in knowledge:
            if k in applied:
                continue
            for c in k.full_clauses:
                if c in model.full_clauses:
                    #logging(model)
                    model = merge_mental_and_full(model, and_model(model, k))
                    applied.append(k)
                    logging("applied modulation")
                    #logging(model)
                    # logging(k)
                    not_finished = True
                    break
            if not_finished:
                break
    return model


if __name__ == "__main__":
    a = input('Enter assertion: ')
    p = parse_one(a)
    p = preprocess_modals(p)
    m = mental_model(p)
    # print(m)
    m = apply_modulation(m)
    print(m)



    # parsed = parse_all(['a ^ b', 'a', 'a | b & c -> []d', 'Atheism is true | God exists', '<>(a|b)', '~<>l', '~[]~a'])
    # parsed = [preprocess_modals(p) for p in parsed]
    # # parsed = parse_all(['a<->b'])
    # for p in parsed:
    #     print(mental_model(p))


    # examples = []
    # examples.append('trump | ~trump')
    # examples.append('<e:0.9> snow')
    # examples.append('<>pat & <>~viv')
    # examples.append('sanders lost & <>~sanders lost')
    # examples.append("biden ^ sanders")
    # examples.append("canal -> [a] flooding")
    # examples.append("canal -> <a> flooding")
    # examples.append("t -> [d] c")
    # examples.append("t -> <d> c")
    # examples.append("<>A")
    # examples.append("A -> B")
    # examples.append("Donald in the office -> Kellyanne in the office")
    # examples.append("<>(Ivanka | Jared)")
    # parsed = parse_all(examples)
    # for p in parsed:
    #     print(mental_model(p))





