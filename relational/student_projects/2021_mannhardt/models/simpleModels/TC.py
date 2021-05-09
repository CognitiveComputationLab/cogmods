class TransitiveClosure():
    def __init__(self, premisses=list(), model=""):
        """creates transitive closure of premisses
            Parameters:
            argument1 (list): list of all premisses e.g. ["Left;A;B", "Right;C;B"]

            Returns:
            int:Returning value
        """
        self.premisses = premisses
        self.model = model
        self.closure = set()
        self.allItems = list()
        self.graph = dict()

        # for pathfinding
        self.visitedList = [[]]

    def create_graph(self):
        for premiss in self.premisses:
            # fill allItems
            if premiss[1] not in self.allItems:
                self.allItems.append(premiss[1])
            if premiss[2] not in self.allItems:
                self.allItems.append(premiss[2])

            # fill graph
            if premiss[0] == "Left":
                if premiss[1] not in self.graph:
                    self.graph[premiss[1]] = []
                self.graph[premiss[1]].append(premiss[2])
            if premiss[0] == "Right":
                if premiss[2] not in self.graph:
                    self.graph[premiss[2]] = []
                self.graph[premiss[2]].append(premiss[1])
        # print(self.graph)
    
    def create_closure(self):
        for item1 in self.allItems:
            for item2 in self.allItems:
                path = self.get_path(item1, item2)
                if len(path) != 0:
                    self.closure.add(path)

    def get_path(self, item1, item2):
        """checks whether path exists in self.graph from item1 to item2
            Parameters:
            argument1 (String): first premiss e.g. "A"
            argument2 (string): second premiss e.g. "B"

            Returns:
            list:either path from item1 to item2 if existing, else empty list
        """
        path = []

    def is_valid_model(self, model):
        """checks whether model is valid model. get_path needs to be called beforehand.
            Parameters:
            model (String): e.g. "ACB" or "ABC"

            Returns:
            bool: True if model exists, False else
        """
        self.closure.clear()
        self.depthFirst(list(self.graph.keys())[0], [])
        # print(self.closure)
        if model in self.closure:
            return True
        else:
            return False

    def check_counterfact(self, counterfact):
        """Check whether counterfact is in alignment with initial models. get_path needs to be called beforehand.
            Parameters:
            model (String): e.g. "Left;A;B" or "Right;C;B"

            Returns:
            bool: True if counterfact is in alignment with graph, False else
        """
        self.closure.clear()
        if len(self.graph) != 0:
            self.depthFirst(self.allItems[0], [])
        else:
            self.closure_from_model()
        if counterfact[0] == "Left":
            counterfact = counterfact[1]+counterfact[2]
        elif counterfact[0] == "Right":
            counterfact = counterfact[2]+counterfact[1]
        if counterfact in self.closure:
            return True
        else:
            return False

    def closure_from_model(self):
        for i in range(len(self.model)):
            for j in range(i+1, len(self.model)):
                self.closure.add(self.model[i]+self.model[j])

    def depthFirst(self, currentVertex, visited):
        visited.append(currentVertex)
        if currentVertex in self.graph:
            for vertex in self.graph[currentVertex]:
                if vertex not in visited:
                    self.depthFirst(vertex, visited.copy())
        self.closure.add("".join(visited))
