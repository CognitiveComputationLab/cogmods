import ccobra

def monotonicity_all(Q, A, B, subset, superset):
    if (Q == "All"):
        # Possible conclusions with All:
        # All A B
        # All C A => All C B by the upward entailing position of A in the first and second premise
        # Here B(A) must be equal to subset
        #-----------------------------------------------------------------------------
        # All A B
        # All B C => All A C by the downward entailing position of B in the first and second premise
        # Here A(B) must be equal to superset
        if (B == subset):
            return "All,"+A+","+superset
        elif (A == superset):
            return "All,"+subset+","+B
        else:
            return "NVC"
    elif (Q == "No"):
        # Possible conclusions with No:
        # All A D
        # No D C => No A C by the downward entailing position of D in the first and second premise
        # Here A(D) must be equal to superset
        #-----------------------------------------------------------------------------
        # All A D
        # No C D => No C A by the downward entailing position of D in the first and second premise
        # Here B(D) must be equal to superset
        if (A == superset):
            return "No,"+subset+","+B
        elif (B == superset):
            return "No,"+A+","+subset
        else:
            return "NVC"
    elif (Q == "Some"):
        # Possible conclusions with Some:
        # All A B
        # Some A C => Some B C by the upward entailing position of A in the first and second premise
        # Here A(A) must be equal to subset
        #-----------------------------------------------------------------------------
        # All A B
        # Some C A => Some C B by the upward entailing position of A in the first and second premise
        # Here B(A) must be equal to subset
        if (A == subset):
            return "Some,"+superset+","+B
        elif (B == subset):
            return "Some,"+A+","+superset
        else:
            return "NVC"
    elif (Q == "Some not"):
        # Possible conclusions with Some not:
        # All A B
        # Some not A C => Some not B C by the upward entailing position of A in the first and second premise
        # Here A(A) must be equal to subset
        #-----------------------------------------------------------------------------
        # All A B
        # Some not C B => Some not C A by the downward entailing position of B in the first and second premise
        # Here B(B) must be equal to superset
        if (A == subset):
            return "Some not,"+superset+","+B
        elif (B == superset):
            return "Some not,"+A+","+subset
        else:
            return "NVC"
    elif (Q == "Most"):
        # Possible conclusions with Most:
        # All B C
        # Most A B => Most A C by the upward entailing position of B in the first and second premise
        # Here B(B) must be equal to subset
        #-----------------------------------------------------------------------------        
        if (B == subset):
            return "Most,"+A+","+superset
        else:
            return "NVC"
    elif (Q == "Few"):
        # Possible conclusions with Few:
        # All C B
        # Few A B => Few A C by the downward entailing position of B in the first and second premise
        # Here B(B) must be equal to superset
        #-----------------------------------------------------------------------------        
        if (B == superset):
            return "Few,"+A+","+subset
        else:
            return "NVC"
    elif (Q == "Most not"):
        # Possible conclusions with Most not:
        # All A B
        # Most not C B => Most not C A by the downward entailing position of B in the first and second premise
        # Here B(B) must be equal to superset
        #-----------------------------------------------------------------------------        
        if (B == superset):
            return "Most not,"+A+","+subset
        else:
            return "NVC"
    elif (Q == "Few not"):
        # Possible conclusions with Few not:
        # All A B
        # Few not C A => Few not C B by the upward entailing position of A in the first and second premise
        # Here B(A) must be equal to subset
        #-----------------------------------------------------------------------------        
        if (B == subset):
            return "Few not,"+A+","+superset
        else:
            return "NVC"
    else:
        return "NVC"
        
def monotonicity_no(Q, A, B, not_subset, not_superset):
    if (Q == "Some"):
        # Possible conclusions with Some:
        # No A B == All A notB
        # All A notB
        # Some C A => Some C notB == Some not C B by the upward entailing position of A in the first and second premise
        # Here B(A) must be equal to not_subset
        #-----------------------------------------------------------------------------
        # All A notB
        # Some A C == Some C A => Some C notB by the upward entailing position of A in the first and second premise and symmetry of some
        # Here A(A) must be equal to not_subset
        #-----------------------------------------------------------------------------
        # because of the symmetry of No and Some all possibilities can lead to the same conclusion:
        # No B A == No A B
        # All A notB
        # Some C A => Some C notB by the symmetry of no and the upward entailing position of A in the first and second premise
        # Here B(A) must be equal to not_superset
        #-----------------------------------------------------------------------------
        # No B A == No A B
        # All A notB
        # Some A C == Some C A => Some C notB by the symmetry of no and the upward entailing position of A in the first and second premise and symmetry of some
        # Here A(A) must be equal to not_superset
        if (B == not_subset):
            return "Some not,"+A+","+not_superset
        elif (A == not_subset):
            return "Some not,"+B+","+not_superset
        elif (B == not_superset):
            return "Some not,"+A+","+not_subset
        elif (A == not_superset):
            return "Some not,"+B+","+not_subset
    elif (Q == "Most"):
        # Possible conclusions with Most:
        # No B C == All B notC
        # Most A B => Most A notC by the upward entailing position of B in the first and second premise
        # Here B(B) must be equal to not_subset
        #-----------------------------------------------------------------------------
        # No C B == No B C
        # Most A B => Most A notC by the upward entailing position of B in the first and second premise
        # Here B(B) must be equal to not_superset
        if (B == not_subset):
            return "Most not,"+A+","+not_superset
        elif (B == not_superset):
            return "Most not,"+A+","+not_subset
        else:
            return "NVC"
    elif (Q == "Few"):
        # Possible conclusions with Few:
        # No A B == All A notB
        # Few C (notnot)B => Few C notA by the upward entailing position of notB in the first and second premise
        # Here B(B) must be equal to not_superset
        #------------------------------------------------------------------------------
        # No B A == No A B == All B notC
        # Few C (not not)B => Few C notA by the upward entailing position of notB in the first and second premise and symmetry of No
        # Here B(B) must be equal to not_subset
        if (B == not_superset):
            return "Few not,"+A+","+not_subset
        elif (B == not_subset):
            return "Few not,"+A+","+not_superset
        else:
            return "NVC"
    else:
        return "NVC"

def solve(Q1, A, B, Q2, C, D):
    # one Q is All
    if (Q1 == "All"): # A is subset of B
        return monotonicity_all(Q2, C, D, A, B)
    elif (Q2 == "All"): # C is subset of D
        return monotonicity_all(Q1, A, B, C, D)
    # one Q is No
    elif (Q1 == "No"):
        return monotonicity_no(Q2, C, D, A, B)
    elif (Q2 == "No"):
        return monotonicity_no(Q1, A, B, C, D)
    else:
        return "NVC"
        

assert solve("All", "A", "B", "All", "B", "C") == "All,A,C"
assert solve("All", "A", "B", "All", "C", "A") == "All,C,B"
assert solve("All", "A", "D", "No", "D", "C") == "No,A,C"
assert solve("All", "A", "D", "No", "C", "D") == "No,C,A"
assert solve("All", "A", "B", "Some", "A", "C") == "Some,B,C"
assert solve("All", "A", "B", "Some", "C", "A") == "Some,C,B"
assert solve("All", "A", "B", "Some not", "A", "C") == "Some not,B,C"
assert solve("All", "A", "B", "Some not", "C", "B") == "Some not,C,A"
assert solve("No", "A", "B", "Some", "C", "A") == "Some not,C,B"
assert solve("No", "A", "B", "Some", "A", "C") == "Some not,C,B"
assert solve("No", "B", "A", "Some", "C", "A") == "Some not,C,B"
assert solve("No", "B", "A", "Some", "A", "C") == "Some not,C,B"
assert solve("All", "A", "B", "Most", "C", "A") == "Most,C,B"
assert solve("All", "A", "B", "Few", "C", "B") == "Few,C,A"
assert solve("All", "A", "B", "Most not", "C", "B") == "Most not,C,A"
assert solve("All", "A", "B", "Few not", "C", "A") == "Few not,C,B"
assert solve("No", "A", "B", "Most", "C", "A") == "Most not,C,B"
assert solve("No", "A", "B", "Few", "C", "B") == "Few not,C,A"
assert solve("No", "B", "A", "Most", "C", "A") == "Most not,C,B"
assert solve("No", "B", "A", "Few", "C", "B") == "Few not,C,A"
assert solve("All", "A", "B", "All", "C", "B") == "NVC"

class MonotonicityModel(ccobra.CCobraModel):
    def __init__(self, name='MonotonicityModel'):
        super(MonotonicityModel, self).__init__(name, ['syllogistic-generalized'], ["single-choice"])

    def predict(self, item, **kwargs):
        syllogism = ccobra.syllogistic_generalized.GeneralizedSyllogism(item)
        response = solve(syllogism.p1[0], syllogism.p1[1], syllogism.p1[2], syllogism.p2[0], syllogism.p2[1], syllogism.p2[2]).split(",")
        response_list = list()
        response_list.append(response)
        if (response_list in item.choices):
        	pred_idx = item.choices.index(response_list)
        	return item.choices[pred_idx]
        else:
        	return response

