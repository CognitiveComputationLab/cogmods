import random

pos = {"A" : 0, "B" : 1, "C" : 2, "D" : 3}
directions = ["left", "Left", "right", "Right"]
counter3ps = [2, 2, 4]
counter4ps = [6, 6, 6, 4]
def analyse(dataset):
    d = {}
    f = {}
    for subj_train_data in dataset:
        for seq_train_data in subj_train_data:
            if parsebool(seq_train_data['korrekt']) != seq_train_data['response']:
                if seq_train_data['Task-ID'] in d:
                    d[seq_train_data['Task-ID']] += 1
                else:
                    d[seq_train_data['Task-ID']] = 1
                if seq_train_data['Figur'] in f:
                    f[seq_train_data['Figur']] += 1
                else:
                    f[seq_train_data['Figur']] = 1
    print(d)
    print(f)

def simpleanalyser(choices):
    while len(choices) == 1:
        choices = choices[0]
    choice = choices
    defined = False
    for e in choice:
        if e in directions:
            operator = e
            defined = True
    if not defined:
        return True
    choice.remove(operator)
    if operator == "Right" or operator == "right":
        if pos[choice[0]] > pos[choice[1]]:
            return True
        else:
            return False
    elif operator == "Left" or operator == "left":
        if pos[choice[0]] < pos[choice[1]]:
            return True
        else:
            return False
    
def predictor(choices, task, r, s, w, right, switch, hardtasks, fast=True):
    choice = choices[0]
    correct = simpleanalyser(choices)
    if fast:
        return correct
    n0 = 0
    n1 = 0
    if switch == 0:
        switch = 0.0000000000000001
    if right == 0:
        right = 0.0000000000000001
    if type(task) == list:
        for t in task:
            if hasright(t):
                n0 += 1
            if hasswitch(t):
                n1 += 1
    if type(task) == str:
        t = []
        lastsplit = 0
        i = 0
        while i < len(seq_train_data['task']):
            if seq_train_data['task'][i] == "/":
                t.append(seq_train_data['task'][:i])
                lastsplit = i
            i += 1
        t.append(seq_train_data['task'][lastsplit+1:])
        for tt in t:
            if hasright(tt):
                n0 += 1
            if hasswitch(tt):
                n1 += 1
    if n0 != 0 and n1 != 0:
        #k = (n0 * r * w) / right + (n1 * s * w) / switch
        k = ( r * w) / right + (s * w) / switch
    elif n0 != 0 and n1 == 0:
        #k = (n0 * r * w) / right
        k = (r * w) / right
    elif n0 == 0 and n1 != 0:
        #k = (n1 * s * w) / switch
        k = (s * w) / switch
    else:
        k = 0
    #print(k)
    #print(task)
    #print(hardtasks)
    #print("--------------")
    if k > 0.5:
        return oppositebool(correct)
    else:
        return correct
    
def oppositebool(b):
    if b:
        return False
    else:
        return True

def getalist(s):
    counter = 0
    for letter in s:
        if letter == "/":
            counter += 1
    start = 0
    stop = 0
    part = 0
    if counter == 1:
        l = [[],[]]
        for e in s:
            #print(s[start: stop])
            #print(start)
            #print(stop)
            #print(l)
            if e == ";":
                l[part].append(s[start : stop])
                start = stop + 1
            elif e == "/":
                l[part].append(s[start : stop])
                part += 1
                start = stop + 1
            stop += 1
        l[part].append(s[start : ])
    elif counter == 2:
        l = [[],[], []]
        for e in s:
            #print(s[start: stop])
            #print(start)
            #print(stop)
            #print(l)
            if e == ";":
                l[part].append(s[start : stop])
                start = stop + 1
            elif e == "/":
                l[part].append(s[start : stop])
                part += 1
                start = stop + 1
            stop += 1
        l[part].append(s[start : ])
    return l

def getastring(l):
    count = len(l)
    i = 0
    string = ""
    while i < count:
        count2 = len(l[i])
        j = 0
        while j < count2:
            if j == 0:
                string += l[i][j] + ";"
            if j == 1:
                string += l[i][j] + ";"
            if j == 2:
                string += l[i][j] + "/"
            j += 1
        i += 1
    string = string[:-1]
    return string

def parsebool(s):
    if s == 1:
        return True
    if s == 0:
        return False

def hasright(option):
    while type(option)==list:
        option = option[0]
    if "Right" in option:
        return True
    else:
        return False

def hasswitch(option):
    if type(option) == list:
        if pos[option[1]] > pos[option[2]]:
            return True
        else:
            return False
    else:
        i = 0
        while i < len(option):
            if option[i] in pos:
                a = option[i]
                break
            i += 1
        option = option[:i-1]+option[i+1:]
        i = 0
        while i < len(option):
            if option[i] in pos:
                b = option[i]
                break
            i += 1
        if pos[a] > pos[b]:
            return True
        else:
            return False
    

def train(dataset):
    c = 0
    rights = [0, 0]
    rightsinchoice = [0, 0]
    hardtasks = {}
    if len(dataset[0][0]['task']) <= 3:
        c = len(dataset[0][0]['task'])
    else:
        for letter in dataset[0][0]['task']:
            if letter == "/":
                c += 1
        c += 1
    wronganwers = 0
    total = 0
    switch = 0
    alloverr = 0
    allovers = 0
    for subj_train_data in dataset:
        for seq_train_data in subj_train_data:
            if 'korrekt' in seq_train_data:
                t = []
                lastsplit = 0
                i = 0
                while i < len(seq_train_data['task']):
                    if seq_train_data['task'][i] == "/":
                        t.append(seq_train_data['task'][:i])
                        lastsplit = i
                    i += 1
                t.append(seq_train_data['task'][lastsplit+1:])
                if not hasright(t[1]) and not hasright(t[0]):
                    alloverr += 1
                for e in t:
                    if hasswitch(e):
                        allovers += 1
                if parsebool(seq_train_data['korrekt']) != seq_train_data['response']:
                    wronganwers += 1
                    if seq_train_data['task'] in hardtasks:
                        hardtasks[seq_train_data['task']] += 1
                    else:
                        hardtasks[seq_train_data['task']] = 1
                    if not hasright(t[1]) and not hasright(t[0]):
                        rights[1] += 1
                    if hasright(seq_train_data['choices']) and not hasright(t[0]) and not hasright(t[1]):
                        rightsinchoice[0] += 1
                    if hasright(seq_train_data['choices']):
                        rightsinchoice[1] += 1
                    for e in t:
                        if hasswitch(e):
                            switch += 1
                total += 1
            elif 'logicallycorrect' in seq_train_data:
                #print(seq_train_data['task'])
                if parsebool(seq_train_data['logicallycorrect']) != seq_train_data['response']:
                    wronganwers += 1
                    return [0, 0.1708542713567839, 0.18090452261306533, 0.24623115577889448,1.0050251256281406, hardtasks]
                total += 1
            else:
                print("data-error")
    rintask = rights[0] / total
    sintask = switch / total
    p_falscheantwort = wronganwers / total
    return [rintask, sintask, p_falscheantwort, alloverr / total, allovers / total, hardtasks]


    
    




















    

