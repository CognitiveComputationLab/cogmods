

import ccobra
import numpy as np
import unitlayoutcal as ulc
import verify as v
import smalllarge as sl

class ProbabilisticModel(ccobra.CCobraModel):
    def __init__(self, name='ProbabilisticModel'):
        """ Initializes the ProbabilisticModel by calling the parent-class constructor
        and passing information about the name as well as supported domains
        and response-types.

        Parameters
        ----------
        name : str
            Name of the model. Will be used as an identifier throughout the
            evaluation phase.

        """
        # general parameters
        self.typ1 = 'single-choice'
        self.typ2 = 'verify'
        super(ProbabilisticModel, self).__init__(
            name, ['spatial-relational'], [self.typ1, self.typ2])
        self.counter = 0
        self.firsttask = []
        self.wrongunderstanding = False
        self.wrongunderstandingcounter = 0
        
        # parameters for single-choice
        self.typ1 = 'single-choice'
        self.dip = 5
        self.wg = 0.1
        self.cg = 0.1
        self.adaptparameter = 12
        self.personalvalues = [self.dip, self.wg, self.cg]

        # parameters for verification - small large dataset
        self.difficulttasks = []
        self.cond = 0;
        self.marginal = 0
        self.unit = 1/51

        # parameters for verification - firgural & premiseorder
        self.r = 0
        self.sw = 0.1708542713567839
        self. wrong = 0.18090452261306533
        self.right = 0.24623115577889448
        self.switch = 1.0050251256281406
        self.total = 0
        self.number = 145
        self.prob = 1.0
        self.gain = 0
        self.hardtasks ={}
        
        


    def predict(self, item, **kwargs):
        """ Generates a prediction based on a given task item.

        Parameters
        ----------
        item : ccobra.data.Item
            Task item container. Holds information about the task, domain,
            response type and response choices.

        """
        self.s = item.response_type
        if self.s == self.typ1:
            # single-choice
            R1 = item.task[0][0]
            R1 = ulc.parser(R1)
            R2 = item.task[1][0]
            R2 = ulc.parser(R2)
            R3 = ulc.predictor(R1,R2,dist=self.dip, cardinalgain=self.cg, westgain=self.wg)[0]
            R3 = ulc.parser(R3)
            if self.wrongunderstanding:
                R3 = ulc.parser(ulc.switcher(ulc.parser(R3)))
            # Output have to be adapted (compared to Input)
            output = R3 +";"+ item.task[1][2] + ";" +item.task[0][1]
        elif self.s == self.typ2:
            verbose = False
            if len(item.task) >= 4:
                verbose = True
            if verbose:
                # small-large
                if [''] in item.task:
                    item.task.remove([''])
                #output = sl.predictor(item.task, item.choices, self.difficulttasks, self.cond, self.marginal, self.unit)
                output = sl.spatialPredictor(item.task, item.choices, self.cond, self.marginal, self.unit, self.number, self.prob, self.gain)
            else:
                # figural & premise-order
                output = v.predictor(item.choices, item.task, self.r, self.sw, self.wrong, self.right, self.switch, self.hardtasks, fast=False)
        return output
        

    def adapt(self, item, truth, **kwargs):
        """ Adapt the model to the individual currently being simulated.

        Parameters
        ----------
        item : ccobra.data.Item
            Task item container. Holds information about the task, domain,
            response type and response choices.

        """
        #print(item)
        if self.counter == 0:
            self.firsttask = item
        if self.firsttask == item:
            self.wrongunderstanding = False
            self.wrongunderstandingcounter = 0
            self.preferredwest = 0
            self.adaptwestgain = 0
        self.counter += 1
        if self.s == self.typ1:
            R1 = item.task[0][0]
            R1 = ulc.parser(R1)
            R2 = item.task[1][0]
            R2 = ulc.parser(R2)
            R3 = ulc.predictor(R1,R2,dist=self.dip, cardinalgain=self.cg, westgain=self.wg)[0]
            R3 = ulc.parser(R3)
            t = truth[0][0]
            if ulc.switcher(ulc.parser(t)) == ulc.parser(R3):
                self.wrongunderstandingcounter += 1
            if self.wrongunderstandingcounter >= self.adaptparameter:
                self.wrongunderstanding = True
                
                
            
        
        
        
        
    def pre_train(self, dataset, **kwargs):
        """ Pre-trains the model based on given information about other
        individuals. Uses the adaption scheme in this case, but could also
        implement a more elaborate pre-training mechanism.

        Parameters
        ----------
        dataset : list(list(dict))
            Pre-training dataset of shape (n_individuals, n_tasks, ...) with
            n_individuals being the number of human reasoners found in the
            dataset and n_tasks denoting the number of tasks they responded
            to 

        """

        self.s = dataset[0][0]['response_type']
        # Iterate over subjects in the data and the tasks they responded to
        if self.s == self.typ1:
            #next line for training of single-choice tasks, takes a long time
            #self.dip, self.cg, self.wg = ulc.optimize(dataset)
            #ulc.tablecal(dataset)
            #ulc.predictortable(di = self.dip, cg=self.cg, wg=self.wg)
            return
        else:
            c = 0
            for e in dataset[0][0]['task']:
                if e == "/":
                    c += 1
            if c >= 4:
                self.difficulttasks, self.cond, self.marginal, self.unit = sl.analyse(dataset)
                # next line for training -> takes a long time
                #self.number, self.prob, self.gain = sl.trainer(dataset, self.cond, self.marginal, self.unit)
            else:
                # v.analyse(dataset)
                self.r, self.sw, self.wrong, self.right, self.switch, self.hardtasks = v.train(dataset)
        




















        
        
