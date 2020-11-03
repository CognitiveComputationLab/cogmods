import ccobra
import random


class Matching_5(ccobra.CCobraModel):
    def __init__(self, name='Matching_5'):
        """ Model constructor  for the generalized Matching Hypothesis.
        Does not use any pre-training as the Model is "static".

        """

        # Call the super constructor to fully initialize the model
        supported_domains = ['syllogistic-generalized']
        supported_response_types = ['single-choice']
        self.mood_to_rank = {'No': 5, 'Most not': 4, 'Some': 3, 'Some not': 3, 'Few': 2, 'Most': 1, 'All': 0}
        self.rank_to_mood = {5: ['No'], 4: ['Most not'], 3: ['Some'],  2: ['Few'], 1: ['Most'], 0: ['All']}
        super(Matching_5, self).__init__(
                name, supported_domains, supported_response_types)

    def start_participant(self, **kwargs):
        """

        """
        pass

    def pre_train(self, dataset):
        """ Pre-trains the model based on one or more datasets.

        """
        pass

    def get_conclusion_mood(self, item):
        """computes the most conservative moods of a task."""
        most_conservative_rank = max(self.mood_to_rank[item.task[0][0]], self.mood_to_rank[item.task[1][0]])
        conclusion_mood = self.rank_to_mood[most_conservative_rank]
        return conclusion_mood

    @staticmethod
    def get_conclusion_terms(item):
        """extracts the two elements of the premises that are used for the conclusion, aka. removes the "connection"."""
        elements = [item.task[0][1], item.task[0][2], item.task[1][1], item.task[1][2]]
        connecting_element = None
        valid = True
        for i in range(1, 3):
            for j in range(1, 3):
                if item.task[0][i] == item.task[1][j]:
                    connecting_element = item.task[1][j]
                    for removals in range(2):
                        elements.remove(connecting_element)
        if not connecting_element:
            print("Found no connecting element in task {}".format(item.task))
            valid = False
        return elements, valid

    @staticmethod
    def build_conclusion(conclusion_mood, elements):
        """uses the given mood and elements to build all possible conclusions according to our Matching hypothesis"""
        possible_conclusions = []
        for mood in conclusion_mood:
            possible_conclusions.append([mood, elements[0], elements[1]])
            possible_conclusions.append([mood, elements[1], elements[0]])
        return possible_conclusions

    def predict(self, item, **kwargs):
        """Predict the responses based on our extension of the Matching hypothesis to generalized quantifiers"""

        """
        We need to know which of the elements in premise1 and premise2 is the term which connects the
         two premises (The term that appears in both). This term will not be part of the conclusions. If no such 
         element is found the two premises lead to no conclusion"""
        elements, is_valid = self.get_conclusion_terms(item)
        if not is_valid:
            # so far this never happened
            print("Found no connecting subject or predicate in task {}".format(item.task))
            return ['NVC']
        """
         We compute the most conservative moods as those are the moods in which the conclusion will be build
        """
        conclusion_mood = self.get_conclusion_mood(item)
        """As a last step we build the possible conclusions. As our Matching 
        (analogue to Theories of the Syllogism: A Meta-Analysis by Khemlani and Johnson-Laird) does always lead to
        conclusions "in both directions", we add two conclusions per mood.
        This technique does not exactly fit to the dataset we currently have, as not all of those conclusions are
         guaranteed to be in the set of possible choices for the task, so we again remove the unfitting ones.

        Currently the task is single_choice so we return a random element of remaining possible conclusions"""
        possible_conclusions = self.build_conclusion(conclusion_mood, elements)

        conclusion_list = []
        for poss in item.choices:
            conclusion_list.append(poss[0])
        for computed_conclusion in possible_conclusions:
            if computed_conclusion not in conclusion_list:
                possible_conclusions.remove(computed_conclusion)
        if len(possible_conclusions) == 0:
            # so far this never happened
            print("All computed conclusions got removed for task {}".format(item.task))
            return ['NVC']
        return random.choice(possible_conclusions)

    def adapt(self, item, target, **kwargs):
        """ Trains the model based on a given problem-target combination.
        """
        pass
