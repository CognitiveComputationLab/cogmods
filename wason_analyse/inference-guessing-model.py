from sympy import sympify
from sympy.logic.boolalg import Not
import random

"""Implementation of the inference guessing algorithm"""


def inference_model(c="CONDITIONAL", d="FORWARD", s="SUFFICIENT", i="IRREVERSIBLE", x="BIDIRECTIONAL", rule="p>>q"):
    sympy_rule = sympify(rule)
    antecedent = sympy_rule.args[0]
    consequent = sympy_rule.args[1]

    selected_cards = set()

    if c == "CONDITIONAL":
        if d == "FORWARD":
            if s == "SUFFICIENT":
                # modus ponens
                selected_cards.add(antecedent)
            else:
                # denial antecedent
                selected_cards.add(Not(antecedent))
        else:
            if s == "SUFFICIENT":
                # affirmation_consequent
                selected_cards.add(consequent)
            else:
                # modus tollens
                selected_cards.add(Not(consequent))
    else:
        if x == "BIDIRECTIONAL":
            if s == "SUFFICIENT":
                # modus ponens bidirectional
                selected_cards.add(antecedent)
                selected_cards.add(consequent)
            else:
                # modus tollens bidirectional
                selected_cards.add(Not(consequent))
                selected_cards.add(Not(antecedent))
        else:
            if d == "FORWARD":
                # modus ponens and denial antecedent
                selected_cards.add(antecedent)
                selected_cards.add(Not(antecedent))
            else:
                # modus tollens and affirmation consequent
                selected_cards.add(Not(consequent))
                selected_cards.add(consequent)
    if i == "REVERSIBLE":
        """
        Select opposite card
        e.g. if p selected choice ~q
        e.g. if ~p selected choice q
        e.g. if p, ~p selected choice q, ~q
        """
        tmp_selected = {i for i in selected_cards}
        for i in tmp_selected:
            if i == antecedent:
                selected_cards.add(Not(consequent))
            elif Not(i) == antecedent:
                selected_cards.add(consequent)
            elif i == consequent:
                selected_cards.add(Not(antecedent))
            elif Not(i) == consequent:
                selected_cards.add(antecedent)
    return selected_cards


def independence_model():
    cards = {"p", "~p", "q", "~q"}
    selected_cards = set(random.sample(cards, random.randint(0, 4)))
    return selected_cards


def inference_guessing_model(c="CONDITIONAL", d="FORWARD", s="SUFFICIENT", i="IRREVERSIBLE", x="BIDIRECTIONAL",
                             rule="p>>q", a="INFERENCE"):
    if a == "INFERENCE":
        return inference_model(c, d, s, i, x, rule)
    else:
        return independence_model()


print(inference_guessing_model())
