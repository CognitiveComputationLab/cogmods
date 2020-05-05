import random


class Model:
    """
    this class represents a model for the model approach.
    here all objects can be inserted  in an array based on the instructions of the coding.
    """
    def __init__(self, code, question=False):
        self.coding = code
        self.model = [[0 for x in range(16)]for y in range(16)]
        self.question = question
        # for 1% mutation probability in each bit use 10 for 10% use 100
        self.mutation_probability = 100

    def build_model(self, item):
        """
        Builds the model with the relations from the item.
        :param item: ccobra item
        """
        self.model = [[0 for x in range(16)] for y in range(16)]
        premise_number = 0
        if self.question:
            for premise in item.choices[0]:
                self.place(premise[2], premise[0], premise[1], premise_number)
                premise_number += 1
        else:
            for premise in item.task:
                self.place(premise[2], premise[0], premise[1], premise_number)
                premise_number += 1

    def place(self, object1, direction, object2, premise_number):
        """
        places one or two objects in relation (direction) of each other in the array based on the instruction
        in the coding for the premise number.
        :param object1: first object to place
        :param direction: general direction given by the item.
        :param object2: second object to place
        :param premise_number: number of the premise, needed to check which instructions shall be used.
        :return: true if it worked, false if not.
        """
        object1_pos = self.find_object(object1)
        object2_pos = self.find_object(object2)
        pos_x = int(len(self.model) / 2)
        pos_y = int(len(self.model[0]) / 2)

        directions = self.get_directions(direction)

        # both not in model
        if object1_pos[0] is None and object2_pos[0] is None:
            pos_x = int(len(self.model)/2)
            pos_y = int(len(self.model[0])/2)
            while not self.insert_object(object1, [pos_x, pos_y]):
                pos_x += 1
            object1_pos = self.find_object(object1)

        # obj 2 in model
        if object2_pos[0] is not None:
            if object1_pos[0] is not None:
                return
            y_shift = 0
            x_shift = 0
            if directions[0] is not None:
                x_shift = self.get_insertion_rule(premise_number, directions[0]) * (-1)
                pos_x = object2_pos[0] + x_shift
            else:
                pos_x = object2_pos[0]
            if directions[1] is not None:
                y_shift = self.get_insertion_rule(premise_number, directions[1]) * (-1)
                pos_y = object2_pos[1] + y_shift
            else:
                pos_y = object2_pos[1]
            cnt = 0
            insertion = self.insert_object(object1, [pos_x, pos_y])
            if x_shift == 0 and y_shift == 0:
                return
            while not insertion:
                if x_shift < 0 and cnt % 2 == 0:
                    pos_x -= 1
                if x_shift > 0 and cnt % 2 == 0:
                    pos_x += 1
                if y_shift < 0 and cnt % 2 == 1:
                    pos_x -= 1
                if y_shift > 0 and cnt % 2 == 1:
                    pos_x += 1
                insertion = self.insert_object(object1, [pos_x, pos_y])
                cnt += 1

        # obj 1 in model
        if object1_pos[0] is not None:
            if object2_pos[0] is not None:
                return
            y_shift = 0
            x_shift = 0
            if directions[0] is not None:
                x_shift = self.get_insertion_rule(premise_number, directions[0])
                pos_x = object1_pos[0] + x_shift
            else:
                pos_x = object1_pos[0]
            if directions[1] is not None:
                y_shift = self.get_insertion_rule(premise_number, directions[1])
                pos_y = object1_pos[1] + y_shift
            else:
                pos_y = object1_pos[1]
            cnt = 0
            insertion = self.insert_object(object2, [pos_x, pos_y])
            if x_shift == 0 and y_shift == 0:
                return
            while not insertion:
                if x_shift < 0 and cnt % 2 == 0:
                    pos_x -= 1
                if x_shift > 0 and cnt % 2 == 0:
                    pos_x += 1
                if y_shift < 0 and cnt % 2 == 1:
                    pos_x -= 1
                if y_shift > 0 and cnt % 2 == 1:
                    pos_x += 1
                insertion = self.insert_object(object2, [pos_x, pos_y])
                cnt += 1

    def get_insertion_rule(self, premise_number, direction):
        """
        gets the insertion rule for the given direction and the premise number out of the coding.
        """
        code = self.coding[premise_number*16:premise_number*16+16]
        return self.find_direction(direction, code)

    def find_direction(self, direction, code):
        """
        finds a given direction in the coding.
        """
        if direction == "left":
            i = 0
            while True:
                if code[i] == "0" and code[i + 1] == "1":
                    # case 00/11 direct or free
                    if code[i + 2] == "0" and code[i + 3] == "0":
                        return -1
                    if code[i + 2] == "1" and code[i + 3] == "1":
                        return -1
                    # case 01 one space
                    if code[i + 2] == "0" and code[i + 3] == "1":
                        return -2
                    # case 10 one space
                    if code[i + 2] == "1" and code[i + 3] == "0":
                        return -3
                    return 0
                else:
                    if i < len(code)-4:
                        i += 4
                    else:
                        return 0
        if direction == "front":
            i = 0
            while True:
                if code[i] == "1" and code[i + 1] == "0":
                    # case 00/11 direct or free
                    if code[i + 2] == "0" and code[i + 3] == "0":
                        return -1
                    if code[i + 2] == "1" and code[i + 3] == "1":
                        return -1
                    # case 01 one space
                    if code[i + 2] == "0" and code[i + 3] == "1":
                        return -2
                    # case 10 one space
                    if code[i + 2] == "1" and code[i + 3] == "0":
                        return -3
                    return 0
                else:
                    if i < len(code)-4:
                        i += 4
                    else:
                        return 0

        if direction == "right":
            i = 0
            while True:
                if code[i] == "0" and code[i + 1] == "0":
                    # case 00/11 direct or free
                    if code[i + 2] == "0" and code[i + 3] == "0":
                        return 1
                    if code[i + 2] == "1" and code[i + 3] == "1":
                        return 1
                    # case 01 one space
                    if code[i + 2] == "0" and code[i + 3] == "1":
                        return 2
                    # case 10 one space
                    if code[i + 2] == "1" and code[i + 3] == "0":
                        return 3
                    return 0
                else:
                    if i < len(code)-4:
                        i += 4
                    else:
                        return 0
        if direction == "back":
            i = 0
            while True:
                if code[i] == "1" and code[i + 1] == "1":
                    # case 00/11 direct or free
                    if code[i + 2] == "0" and code[i + 3] == "0":
                        return 1
                    if code[i + 2] == "1" and code[i + 3] == "1":
                        return 1
                    # case 01 one space
                    if code[i + 2] == "0" and code[i + 3] == "1":
                        return 2
                    # case 10 one space
                    if code[i + 2] == "1" and code[i + 3] == "0":
                        return 3
                    return 0
                else:
                    if i < len(code)-4:
                        i += 4
                    else:
                        return 0
        return 0

    def insert_object(self, obj, position):
        """
        inserts an object in the array on a certain possition.
        :return: true if insertion successful false if not.
        """
        if self.model[position[1]][position[0]] == 0:
            self.model[position[1]][position[0]] = obj
            return True
        else:
            return False

    def get_directions(self, direction):
        """
        :return: the direction where to place a object. if the direction is e.g.
        front-left it gets split to front and left.
        """
        directions = [None, None]
        # this is needed for the 2. experiment. There the only appearances are Left or West.
        if direction == "Left" or direction == "West":
            directions[0] = "left"

        # this is needed for the 1. experiment.
        if direction == "north":
            directions[1] = "front"
        if direction == "north-west":
            directions[1] = "front"
            directions[0] = "left"
        if direction == "north-east":
            directions[1] = "front"
            directions[0] = "right"

        if direction == "south":
            directions[1] = "back"
        if direction == "south-west":
            directions[0] = "left"
            directions[1] = "back"
        if direction == "south-east":
            directions[0] = "right"
            directions[1] = "back"

        if direction == "east":
            directions[0] = "right"
        if direction == "west":
            directions[0] = "left"

        return directions

    def find_object(self, obj):
        row_cnt = 0
        object_pos = [None, None]
        for row in self.model:
            if obj in row:
                object_pos[1] = row_cnt
                object_pos[0] = row.index(obj)
                return object_pos
            else:
                row_cnt += 1
        return object_pos

    def random_code_generator(self):
        """
        creates an random 64 bit long coding for a mental model instruction.
        :return:
        """
        code = ""
        for i in range(0, 64):
            if random.randint(0, 1) == 0:
                code += "0"
            else:
                code += "1"
        return code

    def mutate_coding(self):
        """
        mutates the codig by flipping some bits.
        :return:
        """
        old_code = self.coding
        new_code = ""
        for bit in old_code:
            if random.randint(0, 1000) < self.mutation_probability:
                new_code += self.flip_bit(bit)
            else:
                new_code += bit
        self.coding = new_code

    def flip_bit(self, bit):
        if bit == "1":
            return "0"
        return "1"

    def shrink_model(self):
        """
        shrinks the model array to a representation without free spaces between objects.
        :return:
        """
        center = [int(len(self.model[0])/2)+1, int(len(self.model)/2)]
        # move all what is left form center to right
        for i in range(int(len(self.model[center[1]])/2)-1, 1, -1):
            if self.model[center[1]][i] != 0:
                obj = self.model[center[1]][i]
                cnt = i+1
                while self.model[center[1]][cnt] == 0:
                    self.model[center[1]][cnt] = obj
                    self.model[center[1]][cnt-1] = 0
                    cnt += 1
        # move aöö what is right from center to left.
        for i in range(int(int(len(self.model[center[1]])/2)+1), len(self.model[center[1]])):
            if self.model[center[1]][i] != 0:
                obj = self.model[center[1]][i]
                cnt = i-1
                while self.model[center[1]][cnt] == 0:
                    self.model[center[1]][cnt] = obj
                    self.model[center[1]][cnt+1] = 0
                    cnt -= 1

    def get_model_string(self):
        center = int(len(self.model)/2)
        string = ""
        for each in self.model[center]:
            if each != 0:
                string += each
        return string

    def code_to_human_read(self, code):
        human_read = ""
        if code[0] == "0":
            if code[1] == "0":
                human_read += "right "
            else:
                human_read += "left "
        else:
            if code[1] == "0":
                human_read += "front "
            else:
                human_read += "back "
        if code[2] == "0":
            if code[3] == "0":
                human_read += "space1 "
            else:
                human_read += "space2 "
        else:
            if code[3] == "0":
                human_read += "space3 "
            else:
                human_read += "free "
        if len(code) > 4:
            human_read += " " + self.code_to_human_read(code[4:])
        return human_read
