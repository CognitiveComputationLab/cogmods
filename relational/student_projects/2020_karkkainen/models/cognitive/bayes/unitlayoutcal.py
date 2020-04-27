
import math
import numpy as np

def optimize(dataset, maxdist=200, maxwestgain=0.9, maxcardinalgain=0.9):
    CM = 0
    EM = 0
    NM = 0
    run = 1
    runw = 0
    runc = 0
    bestdist = 0
    initdist = 1
    initcard = 0.1
    initwest = 0.1
    bestrun = [100, initcard, initwest]
    s = run
    runc =0
    runw = 0
    change = 0
    while runc <= maxcardinalgain:
        print("Training ...")
        runw = 0
        while runw <= maxwestgain:
            NM = 0
            for subj_train_data in dataset:
                for seq_train_data in subj_train_data:
                    R1 = parser(seq_train_data['item'].task[0][0])
                    R2 = parser(seq_train_data['item'].task[1][0])
                    real = seq_train_data['response'][0][0]
                    R3 = predictor(R1, R2, dist=bestrun[0], cardinalgain=runc, westgain=runw)[0]
                    if parser(R3) == real:
                        NM += 1
            if NM > bestdist:
                bestdist = NM
                bestrun[1] = runc
                bestrun[2] = runw
            else:
                change += 1
                if change >= ((maxcardinalgain * maxwestgain * 100) - (initcard * initwest*100))*0.25:
                    break
            runw += 0.1
        runc += 0.1
    change = 0
    while run <= maxdist:
        print("Training ..." )
        NM = 0
        for subj_train_data in dataset:
            for seq_train_data in subj_train_data:
                R1 = parser(seq_train_data['item'].task[0][0])
                R2 = parser(seq_train_data['item'].task[1][0])
                real = seq_train_data['response'][0][0]
                R3 = predictor(R1, R2, dist=run, cardinalgain=bestrun[1], westgain=bestrun[2])[0]
                if parser(R3) == real:
                    NM += 1
        if NM > bestdist:
            bestdist = NM
            bestrun[0] = run
            change = 0
        else:
            change += 1
            if change >= (maxdist - initdist)*0.25:
                break
        run += 1
    print(bestdist)
    print("----------------Triaining ready-------------------")
    print(bestrun)
    return bestrun
    



def parser(a):
    if a == "N":
        return "north"
    if a == "NE":
        return "north-east"
    if a == "E":
        return "east"
    if a == "SE":
        return "south-east"
    if a == "S":
        return "south"
    if a == "SW":
        return "south-west"
    if a == "W":
        return "west"
    if a == "NW":
        return "north-west"
    if a == "north":
        return "N"
    if a == "north-east":
        return "NE"
    if a == "east":
        return "E"
    if a == "south-east":
        return "SE"
    if a == "south":
        return "S"
    if a == "south-west":
        return "SW"
    if a == "west":
        return "W"
    if a == "north-west":
        return "NW"

def predictor(R1, R2, R3="", dist=150, v=False, cardinalgain=0.2, westgain=0.2):
    if v:
        B = ["N", "E", "S", "W"]
    else:
        B = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    maincard = ["N", "E", "S", "W"]
    maximum = 0
    bestchoice = ""
    for r3 in B:
        cond = conditionalprobability(R1, R2, switcher(r3),dist=dist, verify=v)
        prob1 = 1 / len(B)
        prob = {}
        s = 0
        # better outside the r3 for loop
        for e in B:
            if e == "E":
                prob[e] = prob1 + westgain + cardinalgain
                s += prob1 + westgain + cardinalgain
            elif e in maincard and e != "E":
                prob[e] = prob1 + cardinalgain
                s += prob1 + cardinalgain
            else:
                prob[e] = prob1
                s += prob1
        for p in prob:
            prob[p] = prob[p] / s
        marginal = prob[r3]
        unit = 1 / (len(B) * len(B))
        choicevalue = (cond * marginal) / unit
        if choicevalue > maximum:
            maximum = choicevalue
            bestchoice = r3
    return [bestchoice, maximum]


def conditionalprobability(R1,R2,R3,dist=150,verify=False):
    if verify:
        B = ["N", "E", "S", "W"]
    else:
        B = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    c = unitlayoutcal(R3, R2, R1, distimposs=dist)
    summe = 0
    zw = 0
    for r1 in B:
        for r2 in B:
            zw = unitlayoutcal(R3, r1, r2, distimposs=dist)
            summe += 1 / zw
    return (1 / c) / summe
    

def euclideandistance(a,b):
    return math.sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]))

def switcher(a):
    if a == "N":
        return "S"
    if a == "S":
        return "N"
    if a == "W":
        return "E"
    if a == "E":
        return "W"
    if a == "NE":
        return "SW"
    if a == "SW":
        return "NE"
    if a == "SE":
        return "NW"
    if a == "NW":
        return "SE"
    

def unitlayoutcal(R3,R2,R1,distimposs=150):
    v = []
    #R3 = switcher(R3)
    if R3 == "N" or R3 == "E" or R3 == "S" or R3 == "W":
        if R3 == "N":
            a = [-1,0]
            c = [1,0]
            if R1 == "SE" and R2 == "NW":
                v = [-2,-1]
            elif R1 == "E" and R2 == "NW":
                v = [-1,-1]
            elif R1 == "NE" and R2 == "NW":
                v = [0,-1]
            elif R1 == "NE" and R2 == "W":
                v = [1,-1]
            elif R1 == "NE" and R2 == "SW":
                v = [2,-1]
            elif R1 == "S" and R2 == "N":
                v = [-2,0]
            elif R1 == "N" and R2 == "N":
                v = [0,0]
            elif R1 == "N" and R2 == "S":
                v = [2,0]
            elif R1 == "SW" and R2 == "NE":
                v = [-2,1]
            elif R1 == "W" and R2 == "NE":
                v = [-1,1]
            elif R1 == "NW" and R2 == "NE":
                v = [0,1]
            elif R1 == "NW" and R2 == "E":
                v = [1,1]
            elif R1 == "NW" and R2 == "SE":
                v = [2,1]
        if R3 == "E":
            a = [0,1]
            c = [0,-1]
            if R1 == "SE" and R2 == "NW":
                v = [-1,-2]
            elif R1 == "E" and R2 == "W":
                v = [0,-2]
            elif R1 == "NE" and R2 == "SW":
                v = [1,-2]
            elif R1 == "SE" and R2 == "N":
                v = [-1,-1]
            elif R1 == "NE" and R2 == "S":
                v = [1,-1]
            elif R1 == "SE" and R2 == "NE":
                v = [-1,0]
            elif R1 == "E" and R2 == "E":
                v = [0,0]
            elif R1 == "NE" and R2 == "SE":
                v = [1,0]
            elif R1 == "S" and R2 == "NE":
                v = [-1,1]
            elif R1 == "N" and R2 == "SE":
                v = [1,1]
            elif R1 == "SW" and R2 == "NE":
                v = [-1,2]
            elif R1 == "W" and R2 == "E":
                v = [0,2]
            elif R1 == "NW" and R2 == "SE":
                v = [1,2]
        if R3 == "S":
            a = [1,0]
            c = [-1,0]
            if R1 == "SE" and R2 == "NW":
                v = [-2,-1]
            elif R1 == "SE" and R2 == "W":
                v = [-1,-1]
            elif R1 == "SE" and R2 == "SW":
                v = [0,-1]
            elif R1 == "E" and R2 == "SW":
                v = [1,-1]
            elif R1 == "NE" and R2 == "SW":
                v = [2,-1]
            elif R1 == "S" and R2 == "N":
                v = [-2,0]
            elif R1 == "S" and R2 == "S":
                v = [0,0]
            elif R1 == "N" and R2 == "S":
                v = [2,0]
            elif R1 == "SW" and R2 == "NE":
                v = [-2,1]
            elif R1 == "SW" and R2 == "E":
                v = [-1,1]
            elif R1 == "SW" and R2 == "SE":
                v = [0,1]
            elif R1 == "W" and R2 == "SE":
                v = [1,1]
            elif R1 == "NW" and R2 == "SE":
                v = [2,1]
        if R3 == "W":
            a = [0,-1]
            c = [0,1]
            if R1 == "SE" and R2 == "NW":
                v = [-1,-2]
            elif R1 == "E" and R2 == "W":
                v = [0,-2]
            elif R1 == "NE" and R2 == "SW":
                v = [1,-2]
            elif R1 == "S" and R2 == "NW":
                v = [-1,-1]
            elif R1 == "N" and R2 == "SW":
                v = [1,-1]
            elif R1 == "SW" and R2 == "NW":
                v = [-1,0]
            elif R1 == "W" and R2 == "W":
                v = [0,0]
            elif R1 == "NW" and R2 == "SW":
                v = [1,0]
            elif R1 == "SW" and R2 == "N":
                v = [-1,1]
            elif R1 == "NW" and R2 == "S":
                v = [1,1]
            elif R1 == "SW" and R2 == "NE":
                v = [-1,2]
            elif R1 == "W" and R2 == "E":
                v = [0,2]
            elif R1 == "NW" and R2 == "SE":
                v = [1,2]
            
    else:
        if R3 == "NW":
            a = [-1,-1]
            c = [1,1]
            if R1 == "SE" and R2 == "NW":
                v = [-2,-2]
            elif R1 == "E" and R2 == "NW":
                v = [-1,-2]
            elif R1 == "NE" and R2 == "NW":
                v = [0,-2]
            elif R1 == "NE" and R2 == "W":
                v = [1,-2]
            elif R1 == "NE" and R2 == "SW":
                v = [2,-2]
            elif R1 == "S" and R2 == "NW":
                v = [-2,-1]
            elif R1 == "N" and R2 == "NW":
                v = [0,-1]
            elif R1 == "N" and R2 == "W":
                v = [1,-1]
            elif R1 == "N" and R2 == "SW":
                v = [2,-1]
            elif R1 == "SW" and R2 == "NW":
                v = [-2,0]
            elif R1 == "W" and R2 == "NW":
                v = [-1,0]
            elif R1 == "NW" and R2 == "NW":
                v = [0,0]
            elif R1 == "NW" and R2 == "W":
                v = [1,0]
            elif R1 == "NW" and R2 == "SW":
                v = [2,0]
            elif R1 == "SW" and R2 == "N":
                v = [-2,1]
            elif R1 == "W" and R2 == "N":
                v = [-1,1]
            elif R1 == "NW" and R2 == "N":
                v = [0,1]
            elif R1 == "NW" and R2 == "S":
                v = [2,1]
            elif R1 == "SW" and R2 == "NE":
                v = [-2,2]
            elif R1 == "W" and R2 == "NE":
                v = [-1,2]
            elif R1 == "NW" and R2 == "NE":
                v = [0,2]
            elif R1 == "NW" and R2 == "E":
                v = [1,2]
            elif R1 == "NW" and R2 == "SE":
                v = [2,2]
        if R3 == "NE":
            a = [-1,1]
            c = [1,-1]
            if R1 == "SE" and R2 == "NW":
                v = [-2,-2]
            elif R1 == "E" and R2 == "NW":
                v = [-1,-2]
            elif R1 == "NE" and R2 == "NW":
                v = [0,-2]
            elif R1 == "NE" and R2 == "W":
                v = [1,-2]
            elif R1 == "NE" and R2 == "SW":
                v = [2,-2]
            elif R1 == "SE" and R2 == "N":
                v = [-2,-1]
            elif R1 == "E" and R2 == "N":
                v = [-1,-1]
            elif R1 == "NE" and R2 == "N":
                v = [0,-1]
            elif R1 == "NE" and R2 == "S":
                v = [2,-1]
            elif R1 == "SE" and R2 == "NE":
                v = [-2,0]
            elif R1 == "E" and R2 == "NE":
                v = [-1,0]
            elif R1 == "NE" and R2 == "NE":
                v = [0,0]
            elif R1 == "NE" and R2 == "E":
                v = [1,0]
            elif R1 == "NE" and R2 == "SE":
                v = [2,0]
            elif R1 == "S" and R2 == "NE":
                v = [-2,1]
            elif R1 == "N" and R2 == "NE":
                v = [0,1]
            elif R1 == "N" and R2 == "E":
                v = [1,1]
            elif R1 == "N" and R2 == "SE":
                v = [2,1]
            elif R1 == "SW" and R2 == "NE":
                v = [-2,2]
            elif R1 == "W" and R2 == "NE":
                v = [-1,2]
            elif R1 == "NW" and R2 == "NE":
                v = [0,2]
            elif R1 == "NW" and R2 == "E":
                v = [1,2]
            elif R1 == "NW" and R2 == "SE":
                v = [2,2]
        if R3 == "SE":
            a = [1,1]
            c = [-1,-1]
            if R1 == "SE" and R2 == "NW":
                v = [-2,-2]
            elif R1 == "SE" and R2 == "W":
                v = [-1,-2]
            elif R1 == "SE" and R2 == "SW":
                v = [0,-2]
            elif R1 == "E" and R2 == "SW":
                v = [1,-2]
            elif R1 == "NE" and R2 == "SW":
                v = [2,-2]
            elif R1 == "SE" and R2 == "N":
                v = [-2,-1]
            elif R1 == "SE" and R2 == "S":
                v = [0,-1]
            elif R1 == "E" and R2 == "S":
                v = [1,-1]
            elif R1 == "NE" and R2 == "S":
                v = [2,-1]
            elif R1 == "SE" and R2 == "NE":
                v = [-2,0]
            elif R1 == "SE" and R2 == "E":
                v = [-1,0]
            elif R1 == "SE" and R2 == "SE":
                v = [0,0]
            elif R1 == "E" and R2 == "SE":
                v = [1,0]
            elif R1 == "NE" and R2 == "SE":
                v = [2,0]
            elif R1 == "S" and R2 == "NE":
                v = [-2,1]
            elif R1 == "S" and R2 == "E":
                v = [-1,1]
            elif R1 == "S" and R2 == "SE":
                v = [0,1]
            elif R1 == "N" and R2 == "SE":
                v = [2,1]
            elif R1 == "SW" and R2 == "NE":
                v = [-2,2]
            elif R1 == "SW" and R2 == "E":
                v = [-1,2]
            elif R1 == "SW" and R2 == "SE":
                v = [0,2]
            elif R1 == "W" and R2 == "SE":
                v = [1,2]
            elif R1 == "NW" and R2 == "SE":
                v = [2,2]
        if R3 == "SW":
            a = [1,-1]
            c = [-1,1]
            if R1 == "SE" and R2 == "NW":
                v = [-2,-2]
            elif R1 == "SE" and R2 == "W":
                v = [-1,-2]
            elif R1 == "SE" and R2 == "SW":
                v = [0,-2]
            elif R1 == "E" and R2 == "SW":
                v = [1,-2]
            elif R1 == "NE" and R2 == "SW":
                v = [2,-2]
            elif R1 == "S" and R2 == "NW":
                v = [-2,-1]
            elif R1 == "S" and R2 == "W":
                v = [-1,-1]
            elif R1 == "S" and R2 == "SW":
                v = [0,-1]
            elif R1 == "N" and R2 == "SW":
                v = [2,-1]
            elif R1 == "SW" and R2 == "NW":
                v = [-2,0]
            elif R1 == "SW" and R2 == "W":
                v = [-1,0]
            elif R1 == "SW" and R2 == "SW":
                v = [0,0]
            elif R1 == "W" and R2 == "SW":
                v = [1,0]
            elif R1 == "NW" and R2 == "SW":
                v = [2,0]
            elif R1 == "SW" and R2 == "N":
                v = [-2,1]
            elif R1 == "SW" and R2 == "S":
                v = [0,1]
            elif R1 == "W" and R2 == "S":
                v = [1,1]
            elif R1 == "NW" and R2 == "S":
                v = [2,1]
            elif R1 == "SW" and R2 == "NE":
                v = [-2,2]
            elif R1 == "SW" and R2 == "E":
                v = [-1,2]
            elif R1 == "SW" and R2 == "SE":
                v = [0,2]
            elif R1 == "W" and R2 == "SE":
                v = [1,2]
            elif R1 == "NW" and R2 == "SE":
                v = [2,2]
    """Compute now metric"""
    if not v:
        return distimposs
    else:
        return (euclideandistance(a,v)+euclideandistance(v,c))/euclideandistance(a,c)





def tablecal(dataset):
    B = ["NW","N", "NE", "E", "SE", "S", "SW", "W"]
    d = {}
    for d1 in B:
        for d2 in B:
            d[d1 + "-" + d2] = []
            for b in B:
                d[d1 + "-" + d2].append([b, 0])
    for subj_train_data in dataset:
        for seq_train_data in subj_train_data:
            R1 = parser(seq_train_data['item'].task[0][0])
            R2 = parser(seq_train_data['item'].task[1][0])
            real = parser(seq_train_data['response'][0][0])
            checker = R1 + "-" + R2
            i = 0
            while i < len(d[checker]):
                if d[checker][i][0] == real:
                    d[checker][i][1] += 1
                i += 1        
    maximizer = {}
    for objects in d:
        best = ["", 0]
        for k in d[objects]:
            if k[1] > best[1]:
                best[0] = k[0]
                best[1] = k[1]
        maximizer[objects] = [best[0], round((best[1] / 49) * 100, 2)]
    data = open("data_table.csv", "a")
    l = " ,"
    for b in B:
        l += b + ", "
    l = l[0:len(l)-1]
    data.write(l + "\n")
    for a in B:
        l = a + ", "
        for b in B:
            l += maximizer[a+"-"+b][0] + " (" + str(maximizer[a+"-"+b][1]) + "%),"
        l = l[0:len(l)-1]
        data.write(l + "\n")
    data.close()
    


def predictortable(di=150, cg=0.1, wg=0.1):
    B = ["NW","N", "NE", "E", "SE", "S", "SW", "W"]
    d = {}
    for d1 in B:
        for d2 in B:
            d[d1 + "-" + d2] = []
    #print(d)
    for a in B:
        for b in B:
            R1 = a
            R2 = b
            k = 0
            zw = []
            while k < 100:
                real = predictor(R1, R2,dist=di, cardinalgain=cg, westgain=wg)[0]
                verbose = True
                if not zw:
                    zw.append([real, 1])
                else:
                    for e in zw:
                        if e[0] == real:
                            e[1] += 1
                            verbose = False
                        if verbose:
                            zw.append([real, 1])
                k += 1
            maxi = 0
            maxie = ""
            #print(zw)
            for e in zw:
                if e[1] > maxi:
                    maxi = e[1]
                    maxie = e[0]
            checker = R1 + "-" + R2
            d[checker] = [maxie, maxi]
    data = open("predictor_table.csv", "a")
    l = " ,"
    for b in B:
        l += b + ", "
    l = l[0:len(l)-1]
    data.write(l + "\n")
    for a in B:
        l = a + ", "
        for b in B:
            l += d[a+"-"+b][0] + " (" + str(d[a+"-"+b][1]) + "%),"
        l = l[0:len(l)-1]
        data.write(l + "\n")
    data.close()
                    
            
            
    
    
                    
                        
                            
                    

























    
    
