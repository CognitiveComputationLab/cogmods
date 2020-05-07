class Question:
    """
        represents a question asked by the ccobra item.
    """
    premises = []
    question = None
    experiment = 0
    question_id = 0

    def __init__(self, premises, question, experiment, id, org_question=None):
        self.premises = premises
        self.question = question
        self.experiment = experiment
        self.question_id = id
        self.org_question = org_question

    def __str__(self):
        string = ""
        i = 0
        for each in self.premises:
            string += "Prem" + str(i) + ": "
            string += each.visit_easy_read() + " "
            i += 1
        string += " Question" + str(self.question) + " Experiment: " + str(self.experiment) + " Question Nr." + \
                  str(self.question_id)
        return string
