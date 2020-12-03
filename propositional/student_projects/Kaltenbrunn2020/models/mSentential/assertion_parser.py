"""Sentential and Modal Assertion Parser by Guerth:
https://github.com/CognitiveComputationLab/cogmods/blob/master/modal/student_projects/2019_guerth/models/mmodalsentential/assertion_parser.py

Modified by Kaltenbl for propositional reasoning

This module implements a parser for assertions using the following symbols
as operators:

symbol  meaning
--------------------
~       not
[]      possible
<>      necessary
&       and
|       or
^       xor
->      if then
<->     iff then

Special cases: The possible operator and the necessary operator can signal a
modality (epistemic, alethic, deontic). This can be done by writing the first
character of the modality (e, a, d) between the operator. For example: [d]
signals a necessary of the deontic modality (also called an obligation).
In addition, the possible of the epistemic modality can be given a weight by
adding ':weight' (the weight must be a value between 0 and 1).
For example, <e:0.2> signals a rather lowly weighted possible.

Operator Precedence:
1: ~, [], <>   necessary, possibly, negation (Unary operators)
2: &           and
3: |, ^        or, xor
4: ->, <->     conditional, biconditional
If there are operators of same precedence,
then they are evaluated from left to right.
Parentheses can enforce arbitrary orders.

The top level function is 'parse_all'.
Example: parse_all(['a | b -> c', 'a'])

To visualize a nested Expr call 'pretty_print_expr'.
Example: pretty_print_expr(Expr('&', [Expr('id',['a']), Expr('id',['b'])]))
"""
import collections
import re

Token = collections.namedtuple('Token', ['type', 'value', 'pos'])
Expr = collections.namedtuple('Expr', ['op', 'args'])
CCobra_expr = collections.namedtuple('CCobra_expr', ['str'])


def parse_all(assertions):
    """Parse list of assertions.

    Arguments:
        assertions {list} -- list of assertion strings

    Returns:
        list -- list of nested Expr

    Example usage:
        parse_all(['a | b', '~a'])
    """
    return [parse_one(a) for a in assertions]


def parse_one(assertion):
    """Parse an assertion string.

    Arguments:
        assertion {str} -- assertion to parse

    Returns:
        Expr -- the parsed assertion as one nested Expr
    """
    return preprocess_modals(parse_tokens(list(tokenize(assertion))))


def tokenize(assertion):
    """Turn an assertion into a generator that yields the Tokens.

    Arguments:
        assertion {str} -- The assertion.

    Raises:
        RuntimeError: Gets raised if the assertion is not valid.
    """
    token_specification = [
        ('PAR', r'[()]'),               # Parentheses.
        ('VAR', r'( *[A-Za-z\'] *)+'),  # Variables.
        ('UNA', r'~|<(a|d|e(:(0|1|0\.[0-9]+))?)?>|\[(a|d|e)?\]'),    # Unary operators.
        ('BIN', r'&|\||<->|\^|->'),     # Binary operators.
        ('WHI', r'\s+'),                # Whitespaces.
        ('ERR', r'.'),                  # Any other character.
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    for match in re.finditer(tok_regex, assertion):
        kind = match.lastgroup
        value = match.group()
        pos = match.start()
        # Reduce whitespaces between words to one space.
        if kind == 'VAR':
            value = ' '.join(value.split())
        # Skip whitespaces.
        elif kind == 'WHI':
            continue
        elif kind == 'ERR':
            raise RuntimeError(f'{value!r} unexpected on position {pos}')
        yield Token(kind, value, pos)


def remove_parentheses(t):
    """Remove first instance of valid parentheses.

    Arguments:
        t {list} -- list of Tokens and Expr

    Returns:
        True -- if parentheses were removed
        False -- if no parentheses were removed
    """
    for i in range(len(t)-2):
        if isinstance(t[i], Token) and t[i].value == '(' and isinstance(t[i+1], Expr) and isinstance(t[i+2], Token) and t[i+2].value == ')':
            del t[i+2], t[i]
            return True
    return False


def convert_unary(t):
    """Merge first instance of valid unary operator.

    Arguments:
        t {list} -- list of Tokens and Expr

    Returns:
        True -- if unary operator were merged
        False -- if no unary operator were merged
    """
    for i in range(len(t)-1):
        if isinstance(t[i], Token) and t[i].type == 'UNA' and isinstance(t[i+1], Expr):
            t[i] = Expr(t[i].value, [t[i+1]])
            del t[i+1]
            return True
    return False


def convert_binary(t):
    """Merge first instance of valid binary operator and obey operator precedence.

    Arguments:
        t {list} -- list of Tokens and Expr

    Returns:
        True -- if binary operator were merged
        False -- if no binary operator were merged
    """
    matches = []
    for i in range(1, len(t)-1):
        if isinstance(t[i-1], Expr) and isinstance(t[i], Token) and t[i].type == 'BIN' and isinstance(t[i+1], Expr):
            matches.append((t[i].value, i))
    for value, i in matches:
        if value == '&':
            t[i-1] = Expr(t[i].value, [t[i-1], t[i+1]])
            del t[i], t[i]
            return True
    for value, i in matches:
        if value == '|' or value == '^':
            t[i-1] = Expr(t[i].value, [t[i-1], t[i+1]])
            del t[i], t[i]
            return True
    for value, i in matches:
        if value == '->' or value == '<->':
            t[i-1] = Expr(t[i].value, [t[i-1], t[i+1]])
            del t[i], t[i]
            return True
    return False


def parse_tokens(t):
    """Convert Token list into parsed Expr.

    Arguments:
        t {list} -- list of Tokens

    Raises:
        Exception: if list does not get parsed completely

    Returns:
        Expr -- one nested Expr
    """
    if not t:
        raise Exception("empty assertion")

    # First, turn all names/varaiables/atoms/non-operators into an Expr.
    for i in range(len(t)):
        if t[i].type == 'VAR':
            t[i] = Expr('id', [t[i].value])

    # Loop until no more changes are made to t.
    # If a change gets made, start from top.
    while True:
        if remove_parentheses(t):
            continue
        if convert_unary(t):
            continue
        if convert_binary(t):
            continue
        break

    # Invalid Token list.
    # Parsing failed because list could not be turned in one Expr.
    if len(t) != 1:
        raise Exception("PARSING ERROR:", t)

    return t[0]


def pretty_print_expr(expr, indent=0):
    """Print Expr in a pretty way by recursion and indentation.

    Arguments:
        expr {Expr} -- the Expr to print

    Keyword Arguments:
        indent {int} -- spaces to indent (default: {0})
    """
    if expr.op == 'id':
        print(indent*'\t', expr.args[0])
    else:
        print(indent*'\t', expr.op)
        for e in expr.args:
            pretty_print_expr(e, indent+1)


def expr_to_polish(expr):
    """Turn an Expr into a polish notation generator.

    Arguments:
        expr {Expr} -- the Expr to to change
    """
    if expr.op == 'id':
        yield expr.args[0]
    else:
        yield expr.op
    for a in expr.args:
        if isinstance(a, Expr):
            yield from expr_to_polish(a)


def polish_to_expr(p):
    """Turn a list of polish notation into an Expr.

    Arguments:
        p {list} -- a list of operators

    Returns:
        Expr -- the result
    """
    una_ops = ['<>', '[]', '~']
    bin_ops = ['&', '|', '^', '->', '<->']
    for i, el in enumerate(p):
        if el not in una_ops and el not in bin_ops:
            if not ((el[0] == '<' and el[-1] == '>' and el[1] != '-') or (el[0] == '[' and el[-1] == ']')):
                p[i] = Expr('id', [el])

    not_finished = True
    while not_finished:
        not_finished = False
        for i in range(len(p)-1, -1, -1):
            if (p[i] in una_ops or ((p[i][0] == '<' and p[i][-1] == '>' and p[i][1] != '-') or (p[i][0] == '[' and p[i][-1] == ']'))) and isinstance(p[i+1], Expr):
                p[i] = Expr(p[i], [p[i+1]])
                del p[i+1]
                not_finished = True
                break
            elif p[i] in bin_ops and isinstance(p[i+1], Expr) and isinstance(p[i+2], Expr):
                p[i] = Expr(p[i], [p[i+1], p[i+2]])
                del p[i+2], p[i+1]
                not_finished = True
                break
    return p[0]


def preprocess_modals_polish(p):
    """Transform negated modals in polish notation.

    ~ <> gets transformed to [] ~
    ~ [] gets transformed to <> ~

    Negated weighted epistemic modals get their weight inverted, meaning the
    new weight is 1 - old weight.
    For example: ~<e:0.2> gets transformed to <e:0.8>.


    Arguments:
        p {list} -- list of operators
    """
    not_finished = True
    while not_finished:
        not_finished = False
        for i in range(len(p)-1):
            if p[i] == '~' and p[i+1] == '<>':
                p[i] = '[]'
                p[i+1] = '~'
                not_finished = True
                break
            elif p[i] == '~' and p[i+1] == '[]':
                p[i] = '<>'
                p[i+1] = '~'
                not_finished = True
                break
            elif p[i] == '~' and p[i+1][0:3] == '<e:':
                weight = float(p[i+1][3:-1])
                if weight == 0 or weight == 1:
                    p[i] = '<e>'
                    del p[i+1]
                else:
                    weight = 1 - weight
                    p[i] = '<e:' + str(weight) + '>'
                    del p[i+1]
                not_finished = True
                break
            elif p[i] == '~' and p[i+1][0] == '<' and p[i+1][1] != '-' and p[i+1][-1] == '>':
                modal = p[i+1][2]
                p[i] = '[' + modal + ']'
                p[i+1] = '~'
                not_finished = True
                break
            elif p[i] == '~' and p[i+1][0] == '[' and p[i+1][-1] == ']':
                modal = p[i+1][2]
                p[i] = '<' + modal + '>'
                p[i+1] = '~'
                not_finished = True
                break


def preprocess_modals(expr):
    """Transform negated modals.

    Arguments:
        expr {Expr} -- the Expr to transform

    Returns:
        Expr -- Expr with transformed modals.
    """
    p = list(expr_to_polish(expr))
    preprocess_modals_polish(p)
    return polish_to_expr(p)


def facts(premises):
    """Return all facts of premises.

    Arguments:
        premises {list} -- list of premise strings

    Returns:
        list -- list of facts (clauses, negated or non-negated)
    """
    if not isinstance(premises, list):
        premises = [premises]
    parsed = parse_all(premises)
    f = []
    for p in parsed:
        if p.op == 'id':
            f.append(p.args[0])
        elif p.op == '~' and p.args[0].op == 'id':
            f.append('not' + p.args[0].args[0])
        else:
            pass
    return f


def ccobra_unary(ccobra_list):
    """Merge first instance of valid unary operator.

    Arguments:
        ccobra_list {list} -- list of operators and CCobra_expr

    Returns:
        True -- if unary operator were merged
        False -- if no unary operator were merged
    """
    ops = ['~', '[]', '<>']
    for i in range(len(ccobra_list)-1):
        if ccobra_list[i] in ops and isinstance(ccobra_list[i+1], CCobra_expr):
            ccobra_list[i] = CCobra_expr(ccobra_list[i] + ccobra_list[i+1].str)
            del ccobra_list[i+1]
            return True
    return False


def ccobra_binary(ccobra_list):
    """Merge first instance of valid binary operator and obey operator precedence.

    Arguments:
        ccobra_list {list} -- list of operators and CCobra_expr

    Returns:
        True -- if binary operator were merged
        False -- if no binary operator were merged
    """
    ops = ['&', '|', '^', '->', '<->']
    matches = []
    for i in range(1, len(ccobra_list)-1):
        if ccobra_list[i-1] in ops and isinstance(ccobra_list[i], CCobra_expr) and isinstance(ccobra_list[i+1], CCobra_expr):
            matches.append((ccobra_list[i-1], i-1))
    for value, i in matches:
        if value == '&':
            ccobra_list[i] = CCobra_expr('(' + ccobra_list[i+1].str + ' ' + value + ' ' + ccobra_list[i+2].str + ')')
            del ccobra_list[i+1], ccobra_list[i+1]
            return True
    for value, i in matches:
        if value == '|' or value == '^':
            ccobra_list[i] = CCobra_expr('(' + ccobra_list[i+1].str + ' ' + value + ' ' + ccobra_list[i+2].str + ')')
            del ccobra_list[i+1], ccobra_list[i+1]
            return True
    for value, i in matches:
        if value == '->' or value == '<->':
            ccobra_list[i] = CCobra_expr('(' + ccobra_list[i+1].str + ' ' + value + ' ' + ccobra_list[i+2].str + ')')
            del ccobra_list[i+1], ccobra_list[i+1]
            return True
    return False


def symbols(el):
    """Map words of operators to symbols of operators.

    Arguments:
        el {str} -- word of operator

    Returns:
        str -- symbol of operator
    """
    el_lower = el.lower()
    if el_lower == "and":
        return "&"
    elif el_lower == "or":
        return "|"
    elif el_lower == "xor":
        return "^"
    elif el_lower == "possible":
        return "<>"
    elif el_lower == "necessary":
        return "[]"
    elif el_lower == "not":
        return "~"
    elif el_lower == "if":
        return "->"
    elif el_lower == "iff":
        return "<->"
    else:
        return el


def ccobra_to_assertion(ccobra_list):
    """Transform an assertion in CCobra syntax to an assertion in the parser syntax

    Arguments:
        ccobra_list {list} -- list of clauses & operators in polnish notation (CCobra style)

    Raises:
        Exception: empty assertion
        Exception: invalid list

    Returns:
        str -- an assertion the parser understands
    """
    if not ccobra_list:
        raise Exception("empty assertion")

    ccobra_list = [symbols(el) for el in ccobra_list]

    ops = ['~', '[]', '<>', '&', '|', '^', '->', '<->']

    # First, turn all names/varaiables/atoms/non-operators into a CCobra_expr.
    for i, el in enumerate(ccobra_list):
        if el not in ops:
            ccobra_list[i] = CCobra_expr(el)

    # Loop until no more changes are made to ccobra_list.
    # If a change gets made, start from top.
    while True:
        if ccobra_unary(ccobra_list):
            continue
        if ccobra_binary(ccobra_list):
            continue
        break

    # Invalid Token list.
    # Parsing failed because list could not be turned in one CCobra_expr.
    if len(ccobra_list) != 1:
        raise Exception("PARSING ERROR:", ccobra_list)

    return ccobra_list[0].str


if __name__ == "__main__":
    a = input('Enter assertion: ')
    e = parse_one(a)
    print('\nExpr:')
    print(e)
    print('\nPretty:')
    pretty_print_expr(e)

    # print(list(expr_to_polnish(Expr(op='|', args=[Expr(op='~', args=[Expr(op='id', args=['A'])]), Expr(op='&', args=[Expr(op='id', args=['B']), Expr(op='~', args=[Expr(op='<>', args=[Expr(op='id', args=['C'])])])])]))))
    # print(polnish_to_expr(['|', '~', 'A', '&', 'B', '~', '<>', 'C']))

    # p = ['|', '~', '<>','~','[]','~','~','A', '&', 'B', '~','~','~', '<>', 'C']
    # print(p)
    # preprocess_modals_polish(p)
    # print(p)
    # print(polish_to_expr(p))

