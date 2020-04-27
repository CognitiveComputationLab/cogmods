"""Single Choice.

This module performs the lookup operations to check what value to return to
CCOBRA. This is possible because the DFT architecture always returns the same
value every time when given the same input.

Authors:
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>
    Rabea Turon
"""


class SingleChoice:
    def __init__(self):
        """Contains the methods required to return an answer based on a LUT.

        The tables this contains are in the following form:
        {(R1, R2): R3, ...}
        where R1, R2, and R3 are spatial relations and the key is the tuple of
        relations in premise 1 and premise 2, and the value R3 is the chosen
        relation.

        The choices are always in the same order as follows:
        [south, north, east, west, south-east, south-west, north-east,
        north-west]
        
        Therefore we can simply return the index of the choice to select.
        """
        self.lut = {
            ("north", "north"): 0,
            ("north", "north-east"): 5,
            ("north", "east"): 5,
            ("north", "south-east"): 3,
            ("north", "south"): 1,
            ("north", "south-west"): 2,
            ("north", "west"): 4,
            ("north", "north-west"): 4,
            ("north-east", "north"): 5,
            ("north-east", "north-east"): 5,
            ("north-east", "east"): 5,
            ("north-east", "south-east"): 3,
            ("north-east", "south"): 3,
            ("north-east", "south-west"): 6,
            ("north-east", "west"): 0,
            ("north-east", "north-west"): 0,
            ("east", "north"): 5,
            ("east", "north-east"): 5,
            ("east", "east"): 3,
            ("east", "south-east"): 7,
            ("east", "south"): 7,
            ("east", "south-west"): 1,
            ("east", "west"): 2,
            ("east", "north-west"): 0,
            ("south-east", "north"): 3,
            ("south-east", "north-east"): 3,
            ("south-east", "east"): 7,
            ("south-east", "south-east"): 7,
            ("south-east", "south"): 7,
            ("south-east", "south-west"): 1,
            ("south-east", "west"): 1,
            ("south-east", "north-west"): 4,
            ("south", "north"): 0,
            ("south", "north-east"): 3,
            ("south", "east"): 7,
            ("south", "south-east"): 7,
            ("south", "south"): 1,
            ("south", "south-west"): 6,
            ("south", "west"): 6,
            ("south", "north-west"): 2,
            ("south-west", "north"): 2,
            ("south-west", "north-east"): 5,
            ("south-west", "east"): 0,
            ("south-west", "south-east"): 1,
            ("south-west", "south"): 6,
            ("south-west", "south-west"): 6,
            ("south-west", "west"): 6,
            ("south-west", "north-west"): 2,
            ("west", "north"): 4,
            ("west", "north-east"): 0,
            ("west", "east"): 3,
            ("west", "south-east"): 1,
            ("west", "south"): 6,
            ("west", "south-west"): 6,
            ("west", "west"): 2,
            ("west", "north-west"): 4,
            ("north-west", "north"): 4,
            ("north-west", "north-east"): 0,
            ("north-west", "east"): 0,
            ("north-west", "south-east"): 7,
            ("north-west", "south"): 2,
            ("north-west", "south-west"): 2,
            ("north-west", "west"): 4,
            ("north-west", "north-west"): 4,
        }

    def lookup(self, relation):
        """Lookups the answer to a set of relations.

        Args:
            relation (tuple): A tuple of two strings which are the two
                              relations.
        """
        return self.lut[relation]
