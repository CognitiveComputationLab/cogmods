import random
import ccobra


class ConversionImproved(ccobra.CCobraModel):
    def __init__(self, name='ConversionImproved'):
        self.params = {"reverse_first_premise": 0.2, "reverse_second_premise": 0.2, "All": 0.4, "No": 0,
                       "Some": 0, "Some not": 0.4,
                       "Most": 0.4, "Few": 0.4,
                       "Most not": 0.4, "Few not": 0.4}
        supported_domains = ['syllogistic-generalized']
        supported_response_types = ['single-choice']
        self.nvc_answered = False
        super(ConversionImproved, self).__init__(
            name, supported_domains, supported_response_types)

    @staticmethod
    def addall(s, elements):
        for e in elements:
            if not (e in s):
                s.append(e)

    def predict(self, item, **kwargs):
        syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(item)

        if self.nvc_answered and item.task[0][0] != 'All' and item.task[1][0] != 'All':
            return syl.decode_response('NVC')

        reverse_first_premise = True if random.random() < self.params["reverse_first_premise"] else False
        reverse_second_premise = True if random.random() < self.params["reverse_second_premise"] else False
        proposition1 = item.task[0]
        proposition2 = item.task[1]
        premises1 = [proposition1]
        premises2 = [proposition2]

        if reverse_first_premise and random.random() < self.params[proposition1[0]]:
            premises1.append([proposition1[0], proposition1[2], proposition1[1]])
        if reverse_second_premise and random.random() < self.params[proposition2[0]]:
            premises2.append([proposition2[0], proposition2[2], proposition2[1]])

        if item.task[0][1] == item.task[1][1]:
            a = item.task[0][2]
            b = item.task[0][1]
            c = item.task[1][2]
        elif item.task[0][1] == item.task[1][2]:
            a = item.task[0][2]
            b = item.task[0][1]
            c = item.task[1][1]
        elif item.task[0][2] == item.task[1][1]:
            a = item.task[0][1]
            b = item.task[0][2]
            c = item.task[1][2]
        else:
            a = item.task[0][1]
            b = item.task[0][2]
            c = item.task[1][1]

        predictions = []

        for p1 in premises1:
            for p2 in premises2:
                if p1 == ["All", a, b]:
                    if p2 == ["All", b, c]:
                        self.addall(predictions, [["All", a, c], ["Some", a, c], ["Some", c, a]])
                    elif p2 in [["No", b, c], ["No", c, b]]:
                        self.addall(predictions, [["No", a, c], ["No", c, a], ["Some not", a, c], ["Some not", c, a]])
                    elif p2 in [["Some not", c, b], ["Few", c, b], ["Most", c, b], ["Few not", c, b], ["Most not", c, b]]:
                        self.addall(predictions, [["Some not", c, a]])

                elif p1 == ["All", b, a]:
                    if p2 == ["All", c, b]:
                        self.addall(predictions, [["All", a, c], ["Some", a, c], ["Some", c, a]])
                    elif p2 in [["All", b, c], ["Some", c, b], ["Some", b, c]]:
                        self.addall(predictions, [["Some", a, c], ["Some", c, a]])
                    elif p2 in [["No", c, b], ["No", b, c], ["Some not", b, c]]:
                        self.addall(predictions, [["Some not", a, c]])
                    elif p2 in [["Few", b, c], ["Most", b, c], ["Few not", b, c], ["Most not", b, c]]:
                        self.addall(predictions, [["Some", a, c], ["Some", c, a], ["Some not", a, c]])
                    elif p2 in [["Few", c, b], ["Most not", c, b]]:
                        self.addall(predictions, [["Few", c, a], ["Some", a, c], ["Some", c, a], ["Most not", c, a],
                                                  ["Some not", c, a]])
                    elif p2 in [["Most", c, b], ["Few not", c, b]]:
                        self.addall(predictions, [["Most", c, a], ["Some", a, c], ["Some", c, a], ["Few not", c, a],
                                                  ["Some not", c, a]])

                elif p1 == ["Some", a, b]:
                    if p2 == ["All", b, c]:
                        self.addall(predictions, [["Some", a, c], ["Some", c, a]])
                    elif p2 in [["No", b, c], ["No", c, b]]:
                        self.addall(predictions, [["Some not", a, c]])

                elif p1 == ["Some", b, a]:
                    if p2 == ["All", b, c]:
                        self.addall(predictions, [["Some", a, c], ["Some", c, a]])
                    elif p2 in [["No", c, b], ["No", b, c]]:
                        self.addall(predictions, [["Some not", a, c]])

                elif p1[0] == "No":
                    if p2 == ["All", c, b]:
                        self.addall(predictions, [["No", c, a], ["No", a, c], ["Some not", a, c], ["Some not", c, a]])
                    elif p2 == ["All", b, c] or p2[0] in ["Some", "Few", "Most", "Most not", "Few not"]:
                        self.addall(predictions, [["Some not", c, a]])

                elif p1 == ["Some not", a, b]:
                    if p2 == ["All", c, b]:
                        self.addall(predictions, [["Some not", a, c]])

                elif p1 == ["Some not", b, a]:
                    if p2 == ["All", b, c]:
                        self.addall(predictions, [["Some not", c, a]])

                elif p1 in [["Few", a, b], ["Most Not", a, b]]:
                    if p2 == ['All', b, c]:
                        self.addall(predictions, [["Few", a, c], ["Some", a, c], ["Some", c, a], ["Some not", a, c],
                                                  ["Most not", a, c]])
                    elif p2 == ['All', c, b] or p2[0] == 'No':
                        self.addall(predictions, [["Some not", a, c]])

                elif p1 in [["Few", b, a], ["Most not", b, a]]:
                    if p2 == ["All", b, c]:
                        self.addall(predictions, [["Some", a, c], ["Some", c, a], ["Some not", c, a]])
                    elif p2 in [["Most", b, c], ["Few not", b, c]]:
                        self.addall(predictions, [["Some not", c, a]])

                elif p1 in [["Most", a, b], ["Few not", a, b]]:
                    if p2 == ['All', b, c]:
                        self.addall(predictions, [["Most", a, c], ["Some", a, c], ["Some", c, a], ["Few not", a, c],
                                                  ["Some not", a, c]])
                    elif p2 == ['All', c, b] or p2[0] == "No":
                        self.addall(predictions, [["Some not", a, c]])

                elif p1 == [["Most", b, a], ["Few not", b, a]]:
                    if p2 == ["All", b, c]:
                        self.addall(predictions, [["Some", a, c], ["Some", c, a], ["Some not", c, a]])
                    elif p2 in [["Most", b, c], ["Few not", b, c]]:
                        self.addall(predictions, [["Some", a, c], ["Some", c, a]])
                    elif p2 in [["Most not", b, c], ["Few", b, c]]:
                        self.addall(predictions, [["Some not", a, c]])

        for p in predictions:
            if item.task[0][0] in p[0] or item.task[1][0] in p[0]:
                return p

        for p in predictions:
            if p[0] == "Some":
                return p

        # NVC
        if [["NVC"]] in item.choices:
            return ["NVC"]
        else:
            return random.choices(item.choices)

    def adapt(self, item, truth, **kwargs):
            """ The Atmosphere model cannot adapt.

            """
            syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(item)
            task_enc = syl.encoded_task
            true = syl.encode_response(truth)

            if true == "NVC":
                self.nvc_answered = True