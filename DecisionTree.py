from math import log

class DecisionTree(object):

    def __init__(self, maxDepth, minSplits, splitEval):
        """
        Decision Tree constructor
        @param maxDepth: int of maximum depth of decision tree
        @param minSplits: int of minimum sample splits in decision tree
        @param splitEval: String of cost function to evaluate splits - 'gini' or 'entropy'
        """
        self.maxDepth = maxDepth
        self.minSplits = minSplits
        self.splitEval = splitEval

    # Takes the group of rows assigned to a node and returns the most common value in the group, used to make predictions
    def addToLeafNode(self, group):
        """
        Add comment
        @param group:
        @return:
        """
        outcome = [row[-1] for row in group]
        return max(set(outcome), key = outcome.count)


    def getBestSplit(self, data):
        """
        Find best split by computing the gini score
        @param data: list of train data
        @return result: best split information
        """
        classLabels = list(set(row[-1] for row in data))
        bestIndex = 9999
        bestValue = 999
        bestScore = 9999
        bestGroups = None
        
        for i in range(len(data[0]) - 1): # gets the number of attributes in each variety instance
            for row in data:
                groups = self.dataSplit(i, row[i], data)
                
                # if self.splitEval == 'entropy':
                #     computeScore = self.entropy(groups, classLabels)
                # else:
                #     computeScore = self.giniCompute(groups, classLabels)

                computeScore = self.entropy(groups, classLabels)

                if computeScore < bestScore:
                    bestIndex = i
                    bestValue = row[i]
                    bestScore = computeScore 
                    bestGroups = groups
        result = {}
        result['index'] = bestIndex
        result['value'] = bestValue
        result['groups'] = bestGroups
        return result


    def splitBranch(self, node, depth):
        """
        Recursively splitting the data
        @param node: list of 2 data sets - left & right nodes
        @param depth: int of the depth of current branch
        """
        leftNode, rightNode = node['groups']
        del(node['groups'])

        if not leftNode or not rightNode:
            node['left'] = self.addToLeafNode(leftNode + rightNode)
            node['right'] = self.addToLeafNode(leftNode + rightNode)
            return

        if depth >= self.maxDepth:
            node['left'] = self.addToLeafNode(leftNode)
            node['right'] = self.addToLeafNode(rightNode)
            return

        if len(leftNode) <= self.minSplits:
            node['left'] = self.addToLeafNode(leftNode)
        else:
            node['left'] = self.getBestSplit(leftNode)
            self.splitBranch(node['left'], depth + 1)

        if len(rightNode) <= self.minSplits:
            node['right'] = self.addToLeafNode(rightNode)
        else:
            node['right'] = self.getBestSplit(rightNode)
            self.splitBranch(node['right'], depth + 1)


    def entropy(self, groups, classLabel):
        """
        Entropy computed as our cost function used to evaluate splits in the dataset
        @param groups: list data set
        @param classLabel: list of possible classifications in data set
        @return entropyScore: float of entropy score
        """
        numberSamples = sum([len(group) for group in groups])     
        entropyScore = 0.0

        for group in groups:
            length = float(len(group))        
            if length == 0:
                continue

            entr = 0.0
            for label in classLabel:
                groupLabel = [row[-1] for row in group]
                proportion = groupLabel.count(label) / length      
                if proportion == 0.0:
                    continue
                entr = proportion * log(proportion, 2)
            entropyScore -= entr * (length / numberSamples)
        
        return entropyScore

    
     def giniCompute(self, groups, classLabel):
         """
         Gini Index computed as our cost function used to evaluate splits in the dataset
         @param groups: list data set
         @param classLabel: list of possible classifications in data set
         @return giniScore: float of gini score
         """
         numberSamples = sum([len(group) for group in groups])
         giniScore = 0.0
         for group in groups:
             length = float(len(group))     
             if length == 0:
                 continue
            
             score = 0.0
             for label in classLabel:
                 groupLabel = [row[-1] for row in group]
                 proportion = groupLabel.count(label) / length
                 score += proportion * proportion
             giniScore += (1.0 - score) * (length / numberSamples)
         return giniScore


    def dataSplit(self, index, value, data):
        """
        Split dataset into two groups based on attribute values
        @param index:
        @param value:
        @param data
        """
        leftData = list()
        rightData = list()
        for row in data:
            if row[index] <= value:
                leftData.append(row)
            else:
                rightData.append(row)
        return leftData, rightData


    def makeDecisionTree(self, train):
        """
        Generating tree recursively with getBestSplit() to get a root node and splitBranch()
        Only ran once to first build the tree
        @param train: list of train data
        @return root: dict i.e full decision tree returned
        """
        root = self.getBestSplit(train)
        self.splitBranch(root, 1)
        return root


    def predict(self, trainData, testData):
        """
        predicting set of data points based on train & test data
        @param trainData: list of train data
        @param testData: list of test data
        @return predictLabel: list of test data predictions
        """
        decisionTree = self.makeDecisionTree(trainData)
        
        predictLabel = list()
        for row in testData:
            predictLabel.append(self.testDataPrediction(decisionTree, row))
        return predictLabel


    def testDataPrediction(self, node, row):
        """
        Makes prediction of test data based on decision tree constructed from train data
        @param node: list of decision tree passed in
        @param row: attribute to predict passed in
        @return: prediction of row
        """
        if row[node['index']] < node['value']:
            # if the left node is a Python mapping of values ie. subtree(dictionary)
            if isinstance(node['left'], dict):
                return self.testDataPrediction(node['left'], row)
            else:
                return node['left']

        else:
            if isinstance(node['right'], dict):
                return self.testDataPrediction(node['right'], row)
            else:
                return node['right']
