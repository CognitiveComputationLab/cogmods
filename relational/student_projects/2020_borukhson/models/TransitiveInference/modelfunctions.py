
""" Transitive-Inference model implementation.
"""

def rewardedStimulus(stim, pair):
    return min([int(a) for a in pair]) == int(stim)    
def sortedPair(pair):
    return str(min([int(a) for a in pair])), str(max([int(a) for a in pair]))
def correctReply(pair):        
    return str(min([int(a) for a in pair]))
