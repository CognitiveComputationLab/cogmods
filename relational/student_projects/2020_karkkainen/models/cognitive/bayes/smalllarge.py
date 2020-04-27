import random

pos = [0,1,2,3,4]

def checkconstellation(c, t):
    for constraint in t:
        if constraint[0] == "West" or constraint[0] == "Left":
            if c[constraint[1]] >= c[constraint[2]]:
                return False
        if constraint[0] == "East" or constraint[0] == "Right":
            if c[constraint[1]] <= c[constraint[2]]:
                return False
    return True
    

def computeconstellations(task):
    correct = []
    constellation = {}
    trees = []
    for c in task:
        constellation[c[1]] = 0
        constellation[c[2]] = 0
    for tree in constellation:
        if not tree in trees:
            trees.append(tree)
    for a in pos:
        for b in pos:
            for c in pos:
                for d in pos:
                    for e in pos:
                        constellation[trees[0]] = a
                        constellation[trees[1]] = b
                        constellation[trees[2]] = c
                        constellation[trees[3]] = d
                        if len(trees) == 5:
                            constellation[trees[4]] = e
                        if checkconstellation(constellation, task):
                            if len(trees) == 5 and {trees[0]:a, trees[1]:b, trees[2]:c, trees[3]:d, trees[4]:e} not in correct:
                                correct.append({trees[0]:a, trees[1]:b, trees[2]:c, trees[3]:d, trees[4]:e})
                            elif len(trees) == 4 and {trees[0]:a, trees[1]:b, trees[2]:c, trees[3]:d} not in correct:
                                correct.append({trees[0]:a, trees[1]:b, trees[2]:c, trees[3]:d})
    return correct

def giveanswer(task, choice):
    t = computeconstellations(task)
    c = computeconstellations(choice)
    for l1 in t:
         for l2 in c:
             v = True
             for e in l2:
                if l2[e] != l1[e]:
                    v = False
                    break
             if v:
                return True
                
    return False

def spatialPredictor(task, choice, cond, marginal, unit, constellationnumber, constellationprob, constellationgain):
    if len(choice) == 1:
        choice = choice[0]
    constellation = {}
    trees = []
    for c in task:
        constellation[c[1]] = 0
        constellation[c[2]] = 0
    for tree in constellation:
        if not tree in trees:
            trees.append(tree)
    t = computeconstellations(task)
    c = computeconstellations(choice)
    simpleOrderForTask = False
    simpleOrderForChoice = False
    for con in t:
        d = con.values()
        k = []
        for e in d:
            if e not in k:
                k.append(e)
        if len(k) == len(trees):
            simpleOrderForTask = True
    for con in c:
        d = con.values()
        k = []
        for e in d:
            if e not in k:
                k.append(e)
        if len(k) == len(trees):
            simpleOrderForChoice = True
    if simpleOrderForChoice and simpleOrderForTask:
        return giveanswer(task, choice)
    possibleConstellations = len(t) * len(c)
    if possibleConstellations > constellationnumber:
        return oppositebool(giveanswer(task, choice))
    prob = (cond * (marginal + constellationgain * possibleConstellations)/unit)
    if prob > constellationprob:
        return oppositebool(giveanswer(task, choice))
    return giveanswer(task, choice)
    
        
    
    
    

def trainer(dataset, cond, marginal, unit):
    bestscore = 0
    constellationnumber = 50
    constellationprob = 0
    constellationgain = -0.05
    while constellationnumber < 150:
        constellationprob = 0
        while constellationprob < 1.0:
            constellationgain = -0.05
            while constellationgain < 0.05:
                correctanswer = 0
                for subj_train_data in dataset:
                    for seq_train_data in subj_train_data:
                        real = seq_train_data['response']
                        task = getlist(seq_train_data['task'])
                        choices = getlist(seq_train_data['choices'])
                        prediction = spatialPredictor(task, choices, cond, marginal, unit, constellationnumber, constellationprob, constellationgain)
                        if prediction == real:
                            correctanswer += 1
                if correctanswer >= bestscore:
                    bestscore = correctanswer
                    bestnumber = constellationnumber
                    bestprob = constellationprob
                    bestgain = constellationgain
                print(str(constellationgain) + " + " + str(constellationprob) + " + " + str(constellationnumber) + "-->" + str(bestscore))
                constellationgain += 0.01
            constellationprob += 0.1
        constellationnumber += 5
    best = [bestnumber, bestprob, bestgain]
    print(best)
    return best
    

def predictor(task, choice, dt, cond, marginal, unit, fast=False):
    if len(choice) == 1:
        choice = choice[0]
    correct = giveanswer(task, choice)
    if fast:
        return correct
    if task in dt:
        prob = (cond * marginal) / unit
        #print(prob)random.uniform(0,0.5) < prob:
        if random.uniform(0,0.5) < prob:
            #print("opposite")
            #print(prob)
            return oppositebool(correct)
        else:
            return correct
    else:
        return correct

def analyse(dataset):
    d = {}
    h = {}
    wa = {True:0, False:0}
    answers = {True:0, False:0}
    direction = {"West":0, "Left":0}
    number = 0
    wrong = 0
    diff = []
    for subj_train_data in dataset:
        for seq_train_data in subj_train_data:
            number += 1
            real = seq_train_data['response']
            task = getlist(seq_train_data['task'])
            choices = getlist(seq_train_data['choices'])
            correct = giveanswer(task, choices)
            if real:
                answers[real] += 1
            else:
                answers[False] += 1
            if correct != real:
                if task[0][0] == "West":
                    direction["West"] += 1
                else:
                    direction["Left"] += 1
                wrong += 1
                wa[real] += 1
                if seq_train_data['Task-ID'] in d:
                    d[seq_train_data['Task-ID']] += 1                  
                else:
                    d[seq_train_data['Task-ID']] = 1
                    h[seq_train_data['Task-ID']] = task
    maxi = {}
    while len(maxi) < 10:
        maxielement = 0
        maxivalue = 0
        for e in d:
            if d[e] > maxivalue:
                maxivalue = d[e]
                maxielement = e
        if maxivalue < 10:
            break
        maxi[maxielement] = maxivalue
        d[maxielement]=-1000
    for subj_train_data in dataset:
        for seq_train_data in subj_train_data:
            ID = seq_train_data['Task-ID']
            if ID in maxi and ID not in diff:
                diff.append(getlist(seq_train_data['task']))
    m = 0
    for e in maxi:
        m += maxi[e]
    cond = (m / len(maxi))/wrong
    marginal = wrong / number
    unit = 1 / 51
    return diff, cond, marginal, unit



def getlist(s):
    l = []
    i = 0
    first = None
    second = None
    verbose = True
    start = 0
    while i < len(s):
        if s[i] == ";" and first is None:
            first = i
            verbose = False
        if s[i] == ";" and first is not None:
            second = i
            verbose = True
        if s[i] == "/":
            l.append([s[start:first], s[first+1:second], s[second+1:i]])
            first = None
            second = None
            start = i+1
        i += 1
    return l
                     
            

def numberoftrees(l):
    trees = []
    for e in l:
        if e[1] not in trees:
            trees.append(e[1])
        if e[2] not in trees:
            trees.append(e[2])
    return len(trees)

def oppositebool(b):
    if b:
        return False
    else:
        return True




            

