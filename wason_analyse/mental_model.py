#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

import argparse
import logging

from sympy.logic.boolalg import truth_table, sympify, Not

""" Implementation of the selection task algorithm """


def selection_task(insight_level="NONE", converse=False, cards=["p", "~p", "q", "~q"], model_builder="MM"):
    """
    selects the right cards to turn around given some evidence and a rule to falsify

    I. The functionality of the JL&W algorithm using models is as shown in Figure 2.
    So:
    1.  With no insight, the program consults the mm’s,
        but of course only one model has content:
            p  q.
        It selects p iff it scans the model only from p to q,
        but it selects p  q  iff it scans the model in both directions.
    2.  With partial insight alone, it adds q iff it had previously selected only p,
        but it adds not-q iff it had previously selected p and q.
    3.  With complete insight, its selections are governed solely by a model
        of the counterexample:
           p  not-q  is impossible
    And so it selects p and not-q.


    Parameters
    ----------
    cards: list of tuples
        list of the available evidence/cards to be used to falsify the rule
        each evidence is binary and has values p/~p and q/~q

    insight_level: str
        None, Partial or Full

    converse:   bool
        Determines if S scans implication in both directions

    Returns
    -------
    set
        Cards to turn

    """

    # first generate truth_table/mental models for rule
    sympy_rule = sympify("p >> q")
    antecedent = sympy_rule.args[0]
    consequent = sympy_rule.args[1]
    logging.info("RULE IS IMPLICATION WITH:\n\t\tANTECEDENT {}\n\t\tCONSEQUENT {}".format(antecedent, consequent))

    sorted_atoms = sorted(sympy_rule.atoms(), key=str)

    """
    if model_builder == "TT":
        # sort atoms alphabetically and get truth_table
        models_rule = {
            (tuple(input_combination), val)
            for input_combination, val in truth_table(sympy_rule, sorted_atoms)
        }

        all_models = {
            input_combination
            for input_combination, val in models_rule
        }

        # get both the verifying models and the falsifying models
        verifying_models = {(1, 1)}
        falsifying_models = {(0, 1)}
    """

    atom_index_mapping = {atom: i for i, atom in enumerate(sorted_atoms)}
    index_atom_mapping = dict(enumerate(sorted_atoms))

    if model_builder == "MM":
        if insight_level == "NONE":
            all_models = {(1, 1)}
            verifying_models = {(1, 1)}
        elif insight_level == "PARTIAL" or insight_level == "FULL":  # deliberate and get full model
            all_models = {
                (1, 1), (0, 1), (0, 0),  # principle of truth
                (1, 0)  # only false model
            }
            verifying_models = {(1, 1)}
            falsifying_models = {(1, 0)}

    literals_in_models = {index_atom_mapping[i] if literal else Not(index_atom_mapping[i]) for model in all_models for i, literal in enumerate(model)}

    # set partial, full insight to the appropriate
    if insight_level == "NONE":
            partial_insight = False
            falsification_insight = False
    elif insight_level == "PARTIAL":
            partial_insight = True
            falsification_insight = False
    else:
            partial_insight = falsification_insight = True

    logging.info("S HAS PARTIAL INSIGHT: \t{}".format(partial_insight))
    logging.info("S HAS FULL INSIGHT: \t\t{}".format(falsification_insight))
    logging.info("The verifying models are: \t{}".format(verifying_models))

    if insight_level != "NONE":
        logging.info("The falsifiying models are: \t{}".format(falsifying_models))

    def verifies(card, verifying_models):
        """
        checks if a card's back has the potential to verify the rule.

        Parameters
        ----------
        card
            the card that verifies or does not verify the rule
        rule
            the rule to be verified

        Returns
        -------
            True or False
        """

        if type(card) == Not:
            value = False
            arg = card.args[0]
        else:
            value = True
            arg = card

        models_with_arg = set(filter(lambda model: model[atom_index_mapping[arg]] == value, all_models))

        logging.debug("--------------Verifies----------------")
        logging.debug("CARD is: {}".format(card))
        logging.debug("Models with card: {}".format(models_with_arg))
        logging.debug("verifying models are: {}".format(verifying_models))
        logging.debug("Models with arg that are not in verifying_models: {}".format(models_with_arg.difference(verifying_models)))

        models_not_in_verifying = models_with_arg.difference(verifying_models)
        if models_not_in_verifying == models_with_arg:
            logging.debug("{} does NOT verify".format(card))
            return False
        else:
            logging.debug("{} DOES verify".format(card))
            return True

    def falsifies(card, falsifying_models):
        """
        checks if a card has the potential to falsify the rule

        Parameters
        ----------
        card
            the card to be ckecked as either being able to falsify the rule or not
        rule
            rule to be falisied
        Returns
        -------
        str
            "by_itself", "not_by_itself", False
        """

        if type(card) == Not:
            value = False
            arg = card.args[0]
        else:
            value = True
            arg = card

        models_with_arg = set(filter(lambda model: model[atom_index_mapping[arg]] == value, all_models))
        models_not_in_falsifying = models_with_arg.difference(falsifying_models)

        logging.debug("--------------FALSIFIES----------------")
        logging.debug("Models with card: {}".format(models_with_arg))
        logging.debug("falsifying models are: {}".format(falsifying_models))
        logging.debug("Models with arg that are not in falsifying_models: {}".format(models_with_arg.difference(falsifying_models)))

        if models_not_in_falsifying == models_with_arg:
            logging.debug("DOES NOT FALSIFY")
            return False
        elif not models_with_arg:
            logging.debug("DOES FALSIFY BY ITSELF")
            return "by_itself"
        else:
            logging.debug("DOES NOT FALSIFY BY ITSELF")
            return "not_by_itself"

    if converse:
        logging.info("CONVERSE HAS BEEN SET TO TRUE")  # S scans model in both directions and adds both p and q
        set_of_potentials = {antecedent, consequent}
        set_of_remaining = literals_in_models.difference(set_of_potentials)

    else:
        logging.info("CONVERSE HAS BEEN SET TO FALSE")
        set_of_potentials = {antecedent}
        set_of_remaining = literals_in_models.difference(set_of_potentials)

    logging.info("SET OF POTENTIALS: \t\t{}".format(set_of_potentials))
    logging.info("SET OF REMAINING: \t\t{}".format(set_of_remaining))

    removed_cards = set()
    flipped_cards = set()

    while True:
        if set_of_potentials:
            current_card = set_of_potentials.pop()
            logging.debug("REMAINING CARDS IN POTENTIALS:{}".format(set_of_potentials))
            logging.debug("CURRENT CARD: \t\t{}".format(current_card))
            if verifies(current_card, verifying_models):
                print("verifies")
                if falsification_insight:
                    falsifying_capability = falsifies(current_card, falsifying_models)
                    if not falsifying_capability:
                        # card irrelevant remove from cards to consider
                        logging.debug("REMOVING IRRELEVANT CARD: \t\t{}".format(current_card))
                        removed_cards.add(current_card)
                        continue

                    if falsifying_capability == "by_itself":
                        return "Rule False"

                # card should be "turned over". Remove from cards to be considered
                logging.info("ADDING CARD TO FLIPPED CARDS: {}".format(current_card))
                flipped_cards.add(current_card)

            else:
                falsifying_capability = falsifies(current_card, falsifying_models)
                if not falsifying_capability:
                    # card irrelevant remove from cards to consider
                    removed_cards.add(current_card)
                    continue

                if falsifying_capability == "by_itself":
                    return "Rule False"

                # card should be "turned over". Remove from cards to be considered
                logging.info("ADDING CARD TO FLIPPED CARDS: {}".format(current_card))
                flipped_cards.add(current_card)
        else:
            # check if there are any cards on remaining cards
            # With partial insight alone, it adds q iff it had previously selected only p,
            # but it adds not-q iff it had previously selected p and q.
            if set_of_remaining and partial_insight:
                if converse:
                    set_of_potentials.update(set_of_remaining)  # adding both ~q and ~p
                    set_of_remaining.clear()
                else:
                    if not falsification_insight:
                        set_of_remaining.discard(Not(consequent))  # discard ~q for S has no insight in its use
                    set_of_potentials.update(set_of_remaining)  # add q and ~p but not ~q
                    set_of_remaining.clear()
                logging.debug("SET OF POTENTIALS: \t\t\t{}".format(set_of_potentials))
            else:
                return flipped_cards


def main(insight_level, converse, model_builder):

    selected_cards = selection_task(insight_level=insight_level, converse=converse, model_builder=model_builder)
    print("THE FOLLOWING CARDS HAVE BEEN SELECTED: \t\t{}".format(selected_cards))


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser("")
    cmdline_parser.add_argument("-i", "--insight_level", choices=['NONE', 'PARTIAL', 'FULL'], default='FULL', help='Insight Level that S has')
    cmdline_parser.add_argument("-c", '--converse', action="store_true", default=False, help="Sets converse to True. Determines if S scans mental model in both directions")
    cmdline_parser.add_argument("-m", '--model_builder', choices=['TT', 'MM'], default='MM', help='Choices are: TT: Truth Table MM: Mental Models')
    cmdline_parser.add_argument('-v', '--verbose', choices=['DEBUG', 'INFO'], default='INFO', help='Log-lvl')

    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')
    main(args.insight_level, args.converse, args.model_builder)
