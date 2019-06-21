'''
Module for parsing spatial and temporal premises.

Created on 16.07.2018

@author: Christian Breu <breuch@web.de>, Julia Mertesdorf <julia.mertesdorf@gmail.com>
'''

# Global variable enabling function-prints. Mainly used for debugging purposes.
PRINT_PARSING = False # global variable for whether to print parsing process or not


# GRAMMAR FOR SPATIAL PREMISES
GRAMMAR_SPATIAL = [[["NP-sing", "PRED"], "S 1", "s-prop"],
                   [["NP-sing", "NEGPRED"], "S 2", "sneg-sem"],
                   [["art-def", "N-sing"], "NP-sing 1", "npfun"],
                   [["rel"], "reln 1", "npfun"],
                   [["next-p", "to-p"], "reln 1", "npfun"],
                   [["in-p", "rel front-p", "of-p"], "reln 1", "npfun"],
                   [["in-p", "art-def", "front-p", "of-p"], "reln 1", "npfun"],
                   [["on-p", "art-def", "rel horiz", "of-p"], "reln 1", "npfun"],
                   [["on-p", "rel vert", "of-p"], "reln 1", "npfun"],
                   [["in-p", "art-def", "adj-same", "n-loc", "as-p"], "reln 1", "npfun"],
                   [["in-p", "art-indef", "adj-different", "n-loc", "to-p"], "reln 1", "npfun"],
                   [["in-p", "art-indef", "adj-different", "n-loc", "from-p"], "reln 1", "npfun"],
                   [["V-cop", "reln", "NP-sing"], "PRED 1", "pred"],
                   [["V-cop", "NEG", "reln", "NP-sing"], "NEGPRED 1", "neg-pred-sem"]]

# LEXICON FOR SPATIAL PREMISES
LEXICON_SPATIAL = [["a", ["art-indef", ["dummy"]]], ["the", ["art-def", ["dummy"]]],
                   ["not", ["neg", ["neg-semantics"]]], ["of", ["of-p", ["dummy"]]],
                   ["as", ["as-p", ["dummy"]]], ["is", ["V-cop", ["dummy"]]],
                   ["in", ["in-p", ["dummy"]]],
                   ["next", ["next-p", ["next-semantics"]]],
                   ["to", ["to-p", ["dummy"]]], ["from", ["from-p", ["dummy"]]],
                   ["on", ["on-p", ["dummy"]]], ["right", ["rel horiz", ["(1 0 0)"]]],
                   ["left", ["rel horiz", ["(-1 0 0)"]]],
                   ["front", ["rel front-p", ["(0 1 0)"]]],
                   ["behind", ["rel", ["(0 -1 0)"]]], ["above", ["rel", ["(0 0 -1)"]]],
                   ["top", ["rel vert", ["(0 0 -1)"]]], ["below", ["rel", ["(0 0 1)"]]],
                   ["between", ["relat", ["between-semantics"]]],
                   ["among", ["relat", ["among-semantics"]]],
                   ["beside", ["relat", ["beside-semantics"]]],
                   ["square", ["N-sing", ["[]"]]],
                   ["triangle", ["N-sing", ["V"]]],
                   ["circle", ["N-sing", ["O"]]], ["line", ["N-sing", ["I"]]],
                   ["cross", ["N-sing", ["+"]]],
                   ["ell", ["N-sing", ["L"]]], ["vee", ["N-sing", ["^"]]],
                   ["star", ["N-sing", ["*"]]], ["ess", ["N-sing", ["S"]]]]

# GRAMMAR FOR TEMPORAL PREMISES
GRAMMAR_TEMPORAL = [[["NP-sing", "PRED"], "S 1", "s-prop"],
                    [["art-def", "N-sing"], "NP-sing 1", "npfun"],
                    [["N-sing"], "NP-sing 1", "npfun"],
                    [["rel"], "reln 1", "npfun"],
                    [["V-cop", "reln", "NP-sing"], "PRED 1", "pred"]]

# LEXICON FOR TEMPORAL PREMISES
LEXICON_TEMPORAL = [["the", ["art-def", ["dummy"]]],
                    ["happens", ["V-cop", ["dummy"]]],
                    ["is", ["V-cop", ["dummy"]]],
                    ["after", ["rel", ["(1 0 0)"]]],
                    ["before", ["rel", ["(-1 0 0)"]]],
                    ["while", ["rel", ["(0 1 0)"]]],
                    ["during", ["rel", ["(0 1 0)"]]]]


class Parser:
    """Class for parsing spatial or temporal premises. It can be set up for one
    of the two premise types. Depending on this type, it uses a different grammar
    and parsing algorithm.
    """
    #global variables for parsing
    sentence = [] # Currently parsed sentence (premise).
    pstack = [] # Stack where all words and syntactic/semantic interpretations are put on.
    rel_grammar = [] # the grammar that will be used for parsing
    rel_lexicon = [] # the lexicon that will be used for parsing.
    spatial = True # variable used to decide which reduces word should be used.

    def __init__(self, spatial_parser=True):
        """constructor for the parser class. Takes a boolean argument which decides
        what kind of premises the parser should be able to parse. If spatial is
        set to False, the temporal grammar and lexicon will be used.
        According to the boolean value, sets the correct grammar for the parsing.
        """
        if spatial_parser:
            self.rel_grammar = GRAMMAR_SPATIAL
            self.rel_lexicon = LEXICON_SPATIAL
        else:
            self.rel_grammar = GRAMMAR_TEMPORAL
            self.rel_lexicon = LEXICON_TEMPORAL
        # keep the value as attribute within the class.
        self.spatial = spatial_parser

    def parse(self, premise):
        """
        Function parses a given premise.
        Works through the premise and replaces the words by their lexical category
        and finally their semantics. Calls analyze to process words on the stack.
        If that is not possible, shift to the next word of the premise and put it
        on top of the stack. If that didn´t work either, use backtracking to go back
        if nothing else worked (backtrack is never used apparently).
        At the end of the function, return the parsed premise.

        Example:
        returns [[0, 1, 0], ['A'], ['B']] for the premise:
        ["the", A", "happens", "while", "the", "B"]
        """
        # Initialize all global variables for the parsing process
        self.sentence = premise
        gram = self.rel_grammar
        lex = self.rel_lexicon
        self.pstack = []

        anything_worked = True
        while anything_worked:
            if PRINT_PARSING:
                print("-------------------------LOOP--------------------------")
                print("pStack contains: ", self.pstack, "\n", "Rest of phrase is: ", self.sentence)
            anything_worked = False
            if (not self.sentence) and (len(self.pstack) >= 1) and (self.pstack[0][0] == "S"):
                return self.pstack[0][1]
            # Always try to analyze first.
            if self.analyze(gram, lex, None):
                anything_worked = True # Continue Parsing
            # If analyze didn´t work, try to shift to the next word (works if sentence not empty).
            elif self.shift():
                anything_worked = True # Continue Parsing
        print("Parsing process fails!")
        return None

    def analyze(self, gram, lex, lhs):
        """
        The function first trys to analyze the word on top of the pstack by replaccing it's
        lexical category through passing it to reduce word. If this doesn t work,
        the function trys to reduce the syntax to their respective semantics by
        calling reduces_syntax instead.
        If the word reduction worked, the result of it is added to the history, else
        return None.
        """
        if not self.pstack:
            if PRINT_PARSING:
                print("Analyze doesn´t work - stack is emtpy")
            return None
        if self.spatial:
            tmp = self.reduces_word_spatial(lex, lhs)# Contains word, if it exists.
        else:
            tmp = self.reduces_word_temporal(lex, lhs)# Contains word, if it exists.
        if tmp != None:
            if PRINT_PARSING:
                print("ANALYZE WORKED WITH WORD REDUCTION")
            return True
        else:
            tmp = self.reduces_syntax(gram, lhs)
            if tmp != None:
                if PRINT_PARSING:
                    print("ANALYZE WORKED WITH SYNTAX REDUCTION")
                return True
        return None

    def reduces_word_temporal(self, lexicon, lhs):
        """
        This function checks if the word at the top of the current stack has an entry in the
        lexicon. If so, retrieves the lexical category of the current word.
        Returns the lexical category of the word, or manually assigns the lexical category
        "n-sing" to it via function-call of "check_var", in case the word is not in the lexicon.
        The current word is removed from the stack and the list containing the word and the found
        lexical category (or the manually assigned category) is inserted at the top of the stack.
        Return the list of the current word and its lexical category.
        """
        if not self.pstack:
            return None
        pstack_at0 = self.pstack[0]
        len_pstack_at0 = 1
        if isinstance(pstack_at0, list):
            len_pstack_at0 = len(pstack_at0)
        if len_pstack_at0 == 2:
            if PRINT_PARSING:
                print("+++++++++++++++++++++REDUCE WORD FAILED+++++++++++++++++")
            return None
        tmp = self.lexical_category(self.pstack[0], lexicon, lhs)
        # Found lexical category! Add it to the top of the pstack.
        if tmp != None:
            if PRINT_PARSING:
                print("REDUCE WORD", self.pstack[0], " - lexical category is", tmp)
            tmp2 = [tmp[0], tmp[1]]
            self.pstack = self.pstack[1:]
            self.pstack.insert(0, tmp2)
            return tmp2
        # Couldn´t find lexical category! Manually assign "n-sing" to it, add to pstack and return.
        tmp3 = self.check_var(self.pstack[0])
        self.pstack = self.pstack[1:]
        self.pstack.insert(0, tmp3)
        if PRINT_PARSING:
            print("REDUCE WORD", self.pstack[0], "-- lexical category is", tmp3)
        return tmp3

    def reduces_word_spatial(self, lexicon, lhs):
        """OK [6]
        this function checks if the top of the current stack has an entry in the
        lexicon. If so, retrieves the lexical category for the word.
        The top of the stack gets sliced of and a list containing the two
        elements of the lexical category are put on the stack.
        Returns the lexical category or None, if the word is not in the lexicon.

        Example:
        top of the pstack is "in". The method will replace this by
        ["in-p", ["dummy"]] at the top of the pstack.

        """
        if not self.pstack:
            return None#or False
        #print(self.pstack)
        if self.word(self.pstack[0], lexicon):
            tmp = self.lexical_category(self.pstack[0], lexicon, lhs)
            #print("lexical cat:", tmp)
            if tmp != None:  # if top of stack is word in lexicon
                # only use first two entries(if there are more)
                new_tmp = [tmp[0], tmp[1]]
                # print("new tmp:", new_tmp)
                self.pstack = self.pstack[1:]
                self.pstack.insert(0, new_tmp)
                # print("reduces_word worked", new_tmp)
                return new_tmp
        return None

    @staticmethod
    def word(item, lex):
        """only used for spatial parsing.
        Takes an item and a lexicon as a list. Returns true, if the item has an
        entry in the lexicon, else returns false.
        """
        if not lex:
            return False
        #iterate over the lexicon and check if something is matchinf with item
        for entry in lex:
            if entry[0] == item:#entry is another list
                return True
        return False

    @staticmethod
    def check_var(list_item):
        """only used for temporal parsing
        Function manually assigns the current observed listItem the lexical category "n-sing"
        and returns the list of both the item and the category to the function "reduces_word".
        """
        var_list = ["N-sing", [list_item]]
        return var_list

    def reduces_syntax(self, gram, lhs):
        """
        The function calls strip_semantic, rule_list and syntax_rule and with that,
        gets the first applicable syntax rule which fits the information on the stack.
        If there is no such rule, "none" is returned, otherwise the function calls compose
        with the found rule. This way the semantic elemts of the pstack will be
        further processed with each step.
        """
        if PRINT_PARSING:
            print("TRY Reduce Syntax with gram", gram, "and lhs", lhs)
        stripped_semantic = self.strip_semantic(self.pstack)
        appl_rules = self.rule_list(stripped_semantic, gram)
        current_rule = self.syntax_rule(appl_rules, lhs, gram) # Usually returns first rule of list
        if current_rule != None:
            if PRINT_PARSING:
                print("REDUCES SYNTAX WORKED")
            return self.compose(current_rule, gram)
        else:
            if PRINT_PARSING:
                print("+++++++++++++++++++++++++REDUCES SYNTAX FAILED+++++++++++++++++++++++++++")
            return None

    def compose(self, rule, gram):
        """
        This function first computes new_syntax (the new syntax which can be
        applied to the stack, f.i. art-def & N-Sing = NP-sing), new_sem (the
        corresponding semantics to that rule) and the arguments (contains the
        semantic of all words which are relevant for the rule).
        Depending from the value of new_sem, it calls a specific function to
        get a certain part of these arguments. new_sem is then replaced by the
        outcome of its function which was called.
        The result of the composition (new syntax and new semantics) is then
        placed on top of the pstack, while all old symbols on the pstack which
        were used for the syntax reduction are deleted.
        The function then returns the complete lhs of the rule and the new
        semantics.
        """
        if PRINT_PARSING:
            print("\n COMPOSE with rule: ", rule)
        new_syntax = self.lhs_of_rule(rule, gram).split()[0] # removes the " 1"
        new_sem = self.sem_of_rule(rule)
        reversed_rhs = list(reversed(self.rhs_of_rule(rule)))
        arguments = self.args_from_stack(reversed_rhs, self.pstack)
        if PRINT_PARSING:
            print("New_syntax:", new_syntax, " new_semantic:", new_sem, " Arguments:", arguments)

        # Call function with name given in new_sem & replace new_sem by outcome of called function
        if new_sem == "npfun":
            new_sem = self.npfun(arguments) # Returns first non-dummy function from arguments.
            if PRINT_PARSING:
                print("new_sem after npfun:", new_sem)
        elif new_sem == "pred":
            new_sem = self.pred(arguments) # Shifts argument behind relation.
            if PRINT_PARSING:
                print("new_sem after pred:", new_sem)
        elif new_sem == "s-prop":
            new_sem = self.s_proposition(arguments) # Assembles premise to desired pattern.
            if PRINT_PARSING:
                print("new_sem after s_propositon", new_sem)

        self.pstack = self.chop(self.rhs_of_rule(rule), self.pstack)
        self.pstack.insert(0, [new_syntax, new_sem])
        return [self.lhs_of_rule(rule, gram), new_sem]

    @staticmethod
    def args_from_stack(rev_rhs, stack):
        """
        Function takes the reversed right hand side of a rule and the stack as input and iterates
        over them. It returns a list of all the corresponding semantic part of elements of
        the stack that match an element of the reversed rhs (at correct position)
        Example:
        rev_rhs is of the form [a, b] and stack of the form [[a, 1], [b, 2]]
        (1 and 2 can be lists as well); a and b are phrase functions like
        "N-sing", "art-def" (and 1 & 2 the semantics).
        The function appends the second element of each list of the stack, which fits the
        element of rev_rhs (a and b) to a list and returns this list when the iteration ends
        ([1, 2] is returned).
        """
        if PRINT_PARSING:
            print("ARGS_FROM_STACK: rev rhs is: ", rev_rhs, " stack is ", stack)
        if not rev_rhs:
            if PRINT_PARSING:
                print("no rev rhs, stop args_from_stack")
            return None
        result = []
        for count, obj in enumerate(rev_rhs):
            if obj == stack[count][0]:
                result.append(stack[count][1])
        return result

    def shift(self):
        """
        Adds the current(first) word of the sentence to the pstack and history,
        then deletes it from the sentence. Returns True.
        If the sentence is empty, returns None.
        """
        if PRINT_PARSING:
            print("--------------------------SHIFT to the next word------------------------")
        if not self.sentence:
            return None
        self.pstack = [self.sentence[0]] + self.pstack
        self.sentence = self.sentence[1:]
        return True

    def lexical_category(self, item, lexicon, lc_):
        """
        Returns the lexical category of a given word, if the word is in the
        lexicon. Iterates through the lexicon and checks, if the item equals
        a word in the lexicon and returns the entry for that word in the
        lexicon.
        """
        for lex_entry in lexicon:
            if item == lex_entry[0]:
                return self.legalcategory(lc_, lex_entry[1])#the next item after lc in lex_entry[1]
        if PRINT_PARSING:
            print("symbol not in lexicon")
        return None #if item is not found

    def legalcategory(self, lc_, lis):
        """
        Takes a list of a lexical category from the lexicon and an argument lc_.
        Returns the next item in the list after lc_. This way there can be more
        lexical categories than one. If lc_ is None, returns the whole lexical
        category.
        """
        if not lc_:
            return lis
        print("lc is not empty!")
        return self.member_lis(lc_, lis)[0] #the next item after lc in lis

    def syntax_rule(self, rules_list, lhs, gram):
        """
        Returns the rhs to a given lhs in the rule_list. If lhs is None, returns
        the first elemt of the given rule_list. In parsing the method is only
        called with lhs = None.
        If lhs is not None, returns the complete grammatical rule after the item
        that matches with lhs.
        """
        if not rules_list:
            return None
        if lhs is None: # Lhs usually none in parsing
            return rules_list[0]
        list_p1 = self.expand(lhs, gram)
        list_p2 = self.rule_semantics(lhs, gram)
        list_complete = list_p1+lhs+list_p2
        return self.member_lis(list_complete, rules_list)[0]

    def rule_list(self, syn_stack, gram):
        """
        Function iterates trough all grammar rules and searches for matches between the
        stack and the grammar rules. All rules that fit the information on the stack
        are added to the result list.
        """
        if not gram:
            return None
        result_list = []
        for gra in gram:
            if self.match_rule(list(reversed(self.rhs_of_rule(gra))), syn_stack):
                result_list.append(gra)
        return result_list

    @staticmethod
    def match_rule(reversed_rule, syn_stack):
        """
        Function returns True if a given rule (which is reversed before) matches
        the information on top of the stack. Returns False otherwise.
        """
        if((not syn_stack) or (len(syn_stack) < len(reversed_rule))):
            return False
        for counter, value in enumerate(reversed_rule): # Iterate over the reversed rule
            if value != syn_stack[counter]:
                return False # The rules don't match!
        return True # No differences found between the two rules, hence the loop went through

    @staticmethod
    def member_lis(list1, list2):
        """
        Function takes two lists. Iterates over list2 and returns the remainder of list2
        if list1 equals the current element of list2.
        If list2 is None or list1 couldn´t be found, return False.
        """
        if PRINT_PARSING:
            print("member lis - list1 is ", list1, "list2 is ", list2)
        if list2 is None:
            return False
        for count, value in enumerate(list2):
            if list1 == value:
                return list2[count+1:]#return the rest of list2
        return False

    @staticmethod
    def chop(list1, stack):
        """
        Function returns the stack minus the same number of items as in list1.
        """
        return stack[len(list1):] # Deletes the first len(list) elements from stack.

    def expand(self, lhs, gram):
        """
        For a given left hand side of a rule, find the matching right
        hand side of this rule in the grammar, and return it.
        Return False if there is no fitting right hand side.
        """
        for count, value in enumerate(gram):
            if lhs == self.lhs_of_rule(value, gram[count:]):
                return self.rhs_of_rule(value)
        if PRINT_PARSING:
            print("reduction not in grammar")
        return False

    @staticmethod
    def strip_semantic(stack):
        """
        Iterates through the stack and returns all the syntactic elements of the
        stack(the first item in the lists from the stack).
        """
        result = []
        for item in stack:
            result.append(item[0])
        return result

    def lhs_of_rule(self, rule, gram):
        """
        Function takes a rule and the grammar and checks, whether this rule is contained
        in the grammar. If that´s the case, it returns the left-hand-side of the rule.
        """
        if rule == self.rule_in_grammar(rule, gram):
            return rule[1]
        return None

    @staticmethod
    def rule_in_grammar(rule, grammar):
        """
        Function taks a rule and the grammar and searches for this rule in the grammar.
        If the rule is found, it is returned, otherwise the function returns "None".
        """
        for gram in grammar:
            if rule == gram:
                return rule
        print("rule not in grammar")
        return None

    @staticmethod
    def rhs_of_rule(rule):
        """
        Returns the right hand side of the given rule.
        """
        return rule[0]

    @staticmethod
    def sem_of_rule(rule):
        """
        Function returns the semantics of the rule, which is the 3rd element of the list.
        """
        return rule[2]

    def rule_semantics(self, lhs, gram):
        """
        Function takes an lhs of a rule, retrieves the whole rule and then returns the
        semantics of the rule (third item of the rhs).
        """
        return self.sem_of_rule(self.expand(lhs, gram))

    @staticmethod
    def s_proposition(list1):
        """
        This function assembles relation, and the two arguments in list, for
        instance [[1, 0, 0], ["A"], ["B"]].
        This is The final pattern of the parsed premise that will be returned.
        """
        return [list1[0][0][0], list1[1], list1[0][1][0]]

    @staticmethod
    def npfun(list1):
        """
        Function returns the first item that is not "dummy" from a given list.
        """
        if list1 is None:
            return None
        for list_item in list1:
            if list_item[0] != "dummy": # List items are lists, so look at the first element
                return list_item
        return None

    @staticmethod
    def pred(list1):
        """
        Shifts argument in list behind the relation.
        """
        return [list1[1], [list1[0]]]
