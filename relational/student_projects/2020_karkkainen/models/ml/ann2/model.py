import collections

import numpy as np

from tqdm import tqdm

import ccobra

from ANN import NN


# mapping of ANN output to PA relation classes and from pairs of classes to CD relation

reason_mppng = {"[1, 0, 0]":"<",						# left or above
                "[0, 1, 0]":"=",						# equal
                "[0, 0, 1]":">",						# right or below
                "[1, 1, 1]":"<=>",						# all together (in one dimension)
                "[1, 1, 0]":"<=","[1, 0, 1]":"<>",		# error cases (ANN should not predict these values)
                "[0, 1, 1]":"=>","[0, 0, 0]":""}	

#output_mppng = {("<",">"):"NW",("<","="):"W", ("<","<"):"SW", 
#                ("=",">"):"N", ("=","="):"Eq",("=","<"):"S", 
#                (">",">"):"NE",(">","="):"E", (">","<"):"SE"}

output_mppng_x = {"north-west":[1,0,0],"west": [1,0,0],"south-west":[1,0,0], 
                "north": [0,1,0], "eq":[0,1,0], "south": [0,1,0], 
                "north-east":[0,0,1], "east": [0,0,1], "south-east":[0,0,1]}

output_mppng_y = {"north-west":[1,0,0],"west": [0,1,0],"south-west":[0,0,1], 
                "north": [1,0,0], "eq":[0,1,0], "south": [0,0,1], 
                "north-east":[1,0,0], "east": [0,1,0], "south-east":[0,0,1]}


# mapping of cardinal direction (CD) input
input_mppng_x = {"north-west":-1,"west": -1,"south-west":-1, 
                "north": 0, "eq":0, "south": 0, 
                "north-east":1, "east": 1, "south-east":1}

input_mppng_y = {"north-west":-1,"west": 0,"south-west":1, 
                "north": -1, "eq":0, "south": 1, 
                "north-east":-1, "east": 0, "south-east":1}


class ANNModel(ccobra.CCobraModel):
    def __init__(self, name='ANN', k=1):
        super(ANNModel, self).__init__(name, ["spatial-relational"], ["single-choice"])

        # Parameters
        self.k = k
        self.nn = NN(2,6,3)
        self.prediction = ""

    def pre_train(self, dataset):
        errors = []
        n_iterations = 10
        for i in tqdm(range(n_iterations)):
            error = 0.0
            for subj_train_data in dataset:
                for seq_train_data in subj_train_data:
                    task = seq_train_data['item'].task
                    input_a = [input_mppng_x[task[0][0]], input_mppng_x[task[1][0]]]
                    target_a = output_mppng_x[seq_train_data['response'][0][0]]

                    self.nn.update(input_a)
                    error = error + self.nn.backPropagate(target_a, 0.1, 0.1)

                    input_b = [input_mppng_y[task[0][0]], input_mppng_y[task[1][0]]]
                    target_b = output_mppng_y[seq_train_data['response'][0][0]]

                    self.nn.update(input_b)
                    error = error + self.nn.backPropagate(target_b, 0.1, 0.1)

            errors.append(error)


        #print("training done, error is {}".format(error))
        #plt.plot(errors)
        #plt.ylabel("Error")
        #plt.show()

    
    def reasoning_CD(self, cd_rel_1, cd_rel_2):
        
        # one-dimensional reasoning in PA with PA
        reason_x = [int(round(i,0))for i in self.nn.update([input_mppng_x[cd_rel_1],input_mppng_x[cd_rel_2]])] # x dim

        values_x = [round(i,3)for i in self.nn.update([input_mppng_x[cd_rel_1],input_mppng_x[cd_rel_2]])]

        reason_y = [int(round(i,0))for i in self.nn.update([input_mppng_y[cd_rel_1],input_mppng_y[cd_rel_2]])] # y dim

        values_y = [round(i,3)for i in self.nn.update([input_mppng_y[cd_rel_1],input_mppng_y[cd_rel_2]])]


        # mapping of ANN output to PA relation classes and from pairs of classes to CD relation
        """
        reason_mppng = {"[1, 0, 0]":"<",						# left or above
                        "[0, 1, 0]":"=",						# equal
                        "[0, 0, 1]":">",						# right or below
                        "[1, 1, 1]":"<=>",						# all together (in one dimension)
                        "[1, 1, 0]":"<=","[1, 0, 1]":"<>",		# error cases (ANN should not predict these values)
                        "[0, 1, 1]":"=>","[0, 0, 0]":""}
        
        output_mppng = {("<",">"):"NW",("<","="):"W", ("<","<"):"SW",
                        ("=",">"):"N", ("=","="):"Eq",("=","<"):"S",
                        (">",">"):"NE",(">","="):"E", (">","<"):"SE"}
        """
        
        reasoned_CD = []
        solved = []
        a=0
        while a < 3:
            b=0
            while b < 3:
                if a==0 and b==0 and reason_x[a]==1 and reason_y[b]==1:   
                    reasoned_CD.append("north-west")
                    solved.append(((values_x[a]+values_y[b])/2.0,"north-west"))
                    #print("Weight: \t",(values_x[a]+values_y[b])/2.0," for NW")
                elif a==0 and b==1 and reason_x[a]==1 and reason_y[b]==1: 
                    reasoned_CD.append("west")
                    solved.append(((values_x[a]+values_y[b])/2.0,"west"))
                    #print("Weight: \t",(values_x[a]+values_y[b])/2.0," for W")
                elif a==0 and b==2 and reason_x[a]==1 and reason_y[b]==1: 
                    reasoned_CD.append("south-west")
                    solved.append(((values_x[a]+values_y[b])/2.0,"south-west"))
                    #print("Weight: \t",(values_x[a]+values_y[b])/2.0," for SW")
                elif a==1 and b==0 and reason_x[a]==1 and reason_y[b]==1: 
                    reasoned_CD.append("north")
                    solved.append(((values_x[a]+values_y[b])/2.0,"north"))
                    #print("Weight: \t",(values_x[a]+values_y[b])/2.0," for N")
                elif a==1 and b==1 and reason_x[a]==1 and reason_y[b]==1: 
                    reasoned_CD.append(cd_rel_1)
                    solved.append(((values_x[a]+values_y[b])/2.0, cd_rel_1))
                    #print("Weight: \t",(values_x[a]+values_y[b])/2.0," for Eq, chose {}".format(cd_rel_1))
                elif a==1 and b==2 and reason_x[a]==1 and reason_y[b]==1: 
                    reasoned_CD.append("south")
                    solved.append(((values_x[a]+values_y[b])/2.0,"south"))
                    #print("Weight: \t",(values_x[a]+values_y[b])/2.0," for S")
                elif a==2 and b==0 and reason_x[a]==1 and reason_y[b]==1: 
                    reasoned_CD.append("north-east")
                    solved.append(((values_x[a]+values_y[b])/2.0,"north-east"))
                    #print("Weight: \t",(values_x[a]+values_y[b])/2.0," for NE")
                elif a==2 and b==1 and reason_x[a]==1 and reason_y[b]==1: 
                    reasoned_CD.append("east")
                    solved.append(((values_x[a]+values_y[b])/2.0,"east"))
                    #print("Weight: \t",(values_x[a]+values_y[b])/2.0," for E")
                elif a==2 and b==2 and reason_x[a]==1 and reason_y[b]==1: 
                    reasoned_CD.append("south-east")
                    solved.append(((values_x[a]+values_y[b])/2.0,"south-east"))
                    #print("Weight: \t",(values_x[a]+values_y[b])/2.0," for SE")
                b += 1
            a += 1

        return reasoned_CD


    def predict(self, item, **kwargs):
        task = item.task
        prediction = self.reasoning_CD(task[0][0], task[1][0])

        #print([prediction[0], task[-1][-1], task[0][-1]])
        self.prediction= [[prediction[0], task[-1][-1], task[0][1]]]

        return self.prediction

