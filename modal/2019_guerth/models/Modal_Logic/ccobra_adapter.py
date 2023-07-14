import collections

CCobra_expr = collections.namedtuple('CCobra_expr', ['str'])

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
		return ">"
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

    ops = ['~', '[]', '<>', '&', '|', '^', '>', '<->']


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

    a = ccobra_list[0].str
    a = a.lower()

    return a


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
    ops = ['&', '|', '^', '>', '<->']
    matches = []
    for i in range(1, len(ccobra_list)-1):
        if ccobra_list[i-1] in ops and isinstance(ccobra_list[i], CCobra_expr) and isinstance(ccobra_list[i+1], CCobra_expr):
            matches.append((ccobra_list[i-1], i-1))
    for value, i in matches:
        if value == '&':
            ccobra_list[i] = CCobra_expr('(' + ccobra_list[i+1].str +value +ccobra_list[i+2].str + ')')
            del ccobra_list[i+1], ccobra_list[i+1]
            return True
    for value, i in matches:
        if value == '|': # or value == '^':
            ccobra_list[i] = CCobra_expr('(' + ccobra_list[i+1].str + value +ccobra_list[i+2].str + ')')
            del ccobra_list[i+1], ccobra_list[i+1]
            return True
        elif value == '^':
            a = ccobra_list[i+1].str
            b = ccobra_list[i+2].str
            ccobra_list[i] = CCobra_expr('(((' + a + ')&(~(' + b + ')))|((~(' + a + '))&(' + b + ')))')
            del ccobra_list[i+1], ccobra_list[i+1]
            return True

    for value, i in matches:
        if value == '->':
            ccobra_list[i] = CCobra_expr('(' + ccobra_list[i+1].str + value + ccobra_list[i+2].str + ')')
            del ccobra_list[i+1], ccobra_list[i+1]
            return True
        elif value == '<->':
            a = ccobra_list[i+1].str
            b = ccobra_list[i+2].str
            ccobra_list[i] = CCobra_expr('(((' + a + ')>(' + b + '))&((' + b + ')>(' + b + ')))')
            del ccobra_list[i+1], ccobra_list[i+1]
            return True
    return False