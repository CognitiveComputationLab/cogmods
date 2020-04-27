"""Verification.

This module performs the lookup operations to check what value to return to
CCOBRA. This is possible because the DFT architecture always returns the same
value every time when given the same input.

Authors:
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>
    Rabea Turon
"""
import csv
import os


class Verification:
    def __init__(self):
        """Contains the methods required to return an answer based on a LUT.

        The LUT simply looks up the results based on the task since it contains
        all 48 tasks possible in the benchmark.
        """
        self.response_lut = {
            1: True,
            2: False,
            3: False,
            4: False,
            5: False,
            6: False,
            7: False,
            8: False,
            9: False,
            10: False,
            11: True,
            12: False,
            13: True,
            14: True,
            15: False,
            16: False,
            17: False,
            18: False,
            19: True,
            20: False,
            21: False,
            22: False,
            23: True,
            24: False,
            25: True,
            26: False,
            27: False,
            28: False,
            29: False,
            30: False,
            31: True,
            32: False,
            33: False,
            34: False,
            35: True,
            36: False,
            37: True,
            38: False,
            39: False,
            40: False,
            41: False,
            42: False,
            43: True,
            44: False,
            45: False,
            46: False,
            47: True,
            48: False,
        }  # Figure out result based on task_id

        # But to do that we need the task_id in the first place.
        # Build a LUT to figure out task_id based on task and choices
        self.task_lut = {}

        # Find the lut_csv
        lut_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
        lut_path = os.path.join(lut_path, "lookupTables")
        lut_path = os.path.join(lut_path, "verification_lut.csv")
        lut_csv = open(lut_path, mode="r")
        reader = csv.reader(lut_csv)

        for line in enumerate(reader):
            if (line[0]) != 0:
                self.task_lut[(line[1][0], line[1][1])] = int(line[1][2])

        lut_csv.close()

    def lookup(self, task_id):
        """Lookups the answer to a set of relations.

        Args:
            task_id : the unique task_id for a certain type of task.

        Returns:
            (bool): result from the model.
        """



        return self.response_lut[task_id]
