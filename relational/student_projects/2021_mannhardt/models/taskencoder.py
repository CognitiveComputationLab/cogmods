informations = ["id", "cat", "cons", "blv", "prems", "cormodel", "cfact", "revplausible", "revlo"]

class TaskEncoder():
    def __init__(self, task):
        self.task = task

        # initial premises
        self.prem1 = ""
        self.prem2 = ""

        # initial models
        self.modelLeft = ""
        self.modelRight = ""

        # counterfact
        self.counterfact = ""

        # model revision choices
        self.modelChoiceLeft = ""
        self.modelChoiceLeft = ""
    
    def encode_task(self):
        """Returns dictionary with all task information given in task string
        e.g. '45_N_Incon_ug_aufauf_modri_factab_revgle_lori_blank' turns to
        """
        taskList = self.task.split('_')
        taskDict = dict()
        for i in range(len(informations)):
            inf = informations[i]
            taskInf = taskList[i]
            if taskInf.endswith("le"):
                taskDict[inf] = "left"
                continue
            elif taskInf.endswith("ri"):
                taskDict[inf] = "right"
                continue
            else: 
                taskDict[inf] = taskInf
        return taskDict

def encode_task(task):
    """Returns dictionary with all task information given in task string
    e.g. '45_N_Incon_ug_aufauf_modri_factab_revgle_lori_blank' turns to
    """
    taskList = task.split('_')
    taskDict = dict()
    for i in range(len(informations)):
        inf = informations[i]
        taskInf = taskList[i]
        if taskInf.endswith("le"):
            taskDict[inf] = "left"
            continue
        elif taskInf.endswith("ri"):
            taskDict[inf] = "right"
            continue
        else: 
            taskDict[inf] = taskInf
    return taskDict

def list_to_string_help(stringList):
    for i in stringList:
        if isinstance(i, str):
            yield i
        elif isinstance(i, list):
            yield list_to_string(i)

def list_to_string(stringList):
    fullStr = ""
    for s in list_to_string_help(stringList):
        fullStr += s
    return fullStr