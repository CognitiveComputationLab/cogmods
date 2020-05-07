from Model import Model


class ModelTestPerson:
    """ This class represents one test person of the experiment.
        It is in possession of of MMIs to answer the questions.
    """
    person_id = 0
    given_answers = []
    person_score = 0
    models = []

    def __init__(self, person_id, models, answers):
        self.person_id = person_id
        self.given_answers = answers
        self.models = models

    def answer_question_exp1(self, item):
        """
        answeres questions of the singelchoice benchmark.
        :param item: ccobra item
        :return: the answer to the task
        """
        answer = [["", None, None]]
        obj1 = item.choices[0][0][1]
        obj2 = item.choices[0][0][2]
        answer[0][1] = obj1
        answer[0][2] = obj2
        for model in self.models:
            model.build_model(item)
        for model in self.models:
            obj1_pos = model.find_object(obj1)
            obj2_pos = model.find_object(obj2)
            direction = ""
            # check if object is in model
            if obj1_pos[0] is not None and obj2_pos[0] is not None:
                marked = False
                if obj1_pos[1] > obj2_pos[1]:
                    direction += "south"
                    marked = True
                elif obj1_pos[1] < obj2_pos[1]:
                    direction += "north"
                    marked = True
                if obj1_pos[0] > obj2_pos[0]:
                    if marked:
                        direction += "-"
                    direction += "east"
                elif obj1_pos[0] < obj2_pos[0]:
                    if marked:
                        direction += "-"
                    direction += "west"
                answer[0][0] = direction
            return answer

    def answer_question_exp2(self, item):
        """
        answers the questions of the verification benchmark.
        :param item: ccobra item
        :return: true/false
        """
        for model in self.models:
            model.build_model(item)
            model.shrink_model()
        question_model = self.get_question_model(item).get_model_string()
        for model in self.models:
            # print("Question:", question_model, "Answer:", model.get_model_string())
            if model.get_model_string() == question_model:
                return True
        return False

    def answer_question_exp3(self, item):
        """
            answers the questions of the figural and premisorder benchmark.
            :param item: ccobra item
            :return: true/false
        """
        for model in self.models:
            model.build_model(item)
            model.shrink_model()
        question = [None, None]
        if item.choices[0][0][0] == "Left":
            question[0] = item.choices[0][0][2]
            question[1] = item.choices[0][0][1]
        elif item.choices[0][0][0] == "Right":
            question[0] = item.choices[0][0][1]
            question[1] = item.choices[0][0][2]
        else:
            print("ERROR IN ITEM THE QUESTION WAS:", item.choices[0][0])
        occured = False
        for model in self.models:
            row = model.model[int(len(model.model)/2)]
            if question[0] in row and question[1] in row:
                occured = True
                obj1_pos = row.index(question[1])
                obj2_pos = row.index(question[0])
                if obj1_pos > obj2_pos:
                    return False
        if occured:
            return True
        return False

    def get_question_model(self, item):
        """
        :param item: ccobra item
        :return: returns the arrangement of the question.
        """
        question_model = Model("0011011110111111001101111011111100110111101111110011011110111111", True)
        question_model.build_model(item)
        return question_model

    def __sortkey__(self):
        return self.person_score

