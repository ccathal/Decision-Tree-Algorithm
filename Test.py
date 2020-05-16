from DecisionTree import DecisionTree as dt
from sklearn.tree import DecisionTreeClassifier
import csv
import time
import random

def readCSV(csvFile):
    '''
    Reads the csv file user inputs
    @param csvFile: string of csv file to be read
    @return data: list of csv file with heading removed
    '''
    file = open(csvFile)
    data = list(csv.reader(file))
    headings = data.pop(0)
    return data


def splitDataOnFolds(data, numberFolds):
    '''
    Splits data in csv file into equal number of folds for cross validation.
    If an overflow occurs rows are distributed evenly between folds
    @param data: list of csv file
    @param numberFolds: integer of the number of splits in list for k-fold cross validation
    @return list: with n-sublists
    '''
    random.Random(4).shuffle(data)
    indexNo = len(data) / numberFolds
    return [data[round(indexNo * i):round(indexNo * (i + 1))] for i in range(numberFolds)]


def trainTestSplit(data, testPercentage):
    '''
    Splits data based on percentage of test data
    @param data: list of csv file
    @param testPercentage: int for percentage of data split as test, remainder will be train data
    @return list: containing train data and test data split
    '''
    random.Random(4).shuffle(data)
    indexNo = round(len(data) * (testPercentage / 100))
    return data[indexNo:], data[:indexNo]


def predictionAccuracy(predict, actual):
    '''
    Calculates the percentage accuracy of training attributes that were correctly classified
    @param predict: list of predicted classified instances
    @param actual: list of actual classified instances
    @retrun float: correctly classified instances
    '''
    correctInstance = 0
    instances = len(actual)
    for variety in range(instances):
        if actual[variety] == predict[variety]:
            correctInstance += 1
    return (correctInstance / instances) * 100


def trainTestEvaluation(folds, decision_tree):
    '''
    Method build decision tree & test based on train-test data
    @param folds: list of 2 sets - train data & test data
    @param decision_tree: decision tree object passed in
    @retrun accuracy: float of correctly predicted values of test set based on decision tree formed
    '''
    trainData = list(folds[0])
    preTestData = list(folds[1])

    testData = list()
    for row in preTestData:
        testData.append(row)
        row = row[:-1]
    
    #decision_tree = dt(maxDepth, minSplits, splitEval)
    predict = decision_tree.predict(trainData, testData)

    actual = [row[-1] for row in preTestData]
    accuracy = predictionAccuracy(predict, actual)   
    return accuracy


def crossValidationEvaluation(folds, decision_tree):
    '''
    Method build decision tree & test based on k-fold cross validation
    @param folds: list of data folds for cross validation
    @param decision_tree: decision tree object passed in
    @retrun accuracyArray: list of floats of correctly predicted values of test sets based on decision tree formed
    '''
    accuracyArray = list()
    for fold in folds:
        trainData = list(folds)
        trainData.remove(fold)
        trainData = sum(trainData, []) ## try fix this x2
        
        testData = list()
        for row in fold:
            testData.append(row)
            row = row[:-1]
            
        predict = decision_tree.predict(trainData, testData)
        
        actual = [row[-1] for row in fold]
        accuracy = predictionAccuracy(predict, actual)
        accuracyArray.append(accuracy)
    return accuracyArray

def skLearnTrainTest(folds, sk_dt_model):
    '''
    Method build scikit learn decision tree based on train test split
    @param folds: list of 2 sets - train data & test data
    @param sk_dt_model: scikit learn decision tree object passed in
    @retrun accuracy: float of correctly predicted values of test set based on sklearn decision tree formed
    '''
    trainData = list(folds[0])
    preTestData = list(folds[1])

    X_train = [row[:-1] for row in trainData] ## try incorporate into own algorithm!!!!!!!!!!!
    y_train = [row[-1] for row in trainData]
    X_test = [row[:-1] for row in preTestData]
    y_test = [row[-1] for row in preTestData]

    sk_dt_model.fit(X_train, y_train)
    sk_dt_prediction = sk_dt_model.predict(X_test)
    accuracy = predictionAccuracy(sk_dt_prediction, y_test)
    return accuracy

def skLearnCrossValidation(folds, sk_dt_model):
    '''
    Method build scikit learn decision tree based on k-fold cross validation
    @param folds: list of 2 sets - train data & test data
    @param sk_dt_model: scikit learn decision tree object passed in
    @retrun skAccuracyArray: list of floats of correctly predicted values of test sets based on sklearn decision tree formed
    '''
    skAccuracyArray = list()
    for fold in folds:
        trainData = list(folds)
        trainData.remove(fold)
        trainData = sum(trainData, [])
        
        testData = list()
        for row in fold:
            rowCopy = list(row)
            testData.append(rowCopy)

        y_train = [r[-1] for r in trainData]
        X_train = [r[:-1] for r in trainData]
        y_test = [r[-1] for r in testData]
        X_test = [r[:-1] for r in testData]

        sk_dt_model.fit(X_train, y_train)
        sk_dt_prediction = sk_dt_model.predict(X_test)
        accuracy = predictionAccuracy(sk_dt_prediction, y_test)
        skAccuracyArray.append(accuracy)
    return skAccuracyArray


def main():

    csvFile = input('Enter the file to run on CART algorithm or press ENTER to be run on hazelnuts.csv: ') or 'hazelnuts.csv'
    data = readCSV(csvFile)
    random.seed(1)
    
    depth = (input('Enter maximum depth for decision tree or hit ENTER for 5: ') or 5)
    maxDepth = int(depth)
    
    splits = (input('Enter minumum sample splits for decision tree or hit ENTER for 8: ') or 8)
    minSplits = int(splits)

    # splitEval = input('Enter [1] if you would like to use entropy OR [2] to use gini index on data: ')
    # if splitEval == '1':
    #     splitEval = 'entropy'
    # elif splitEval == '2':
    #     splitEval = 'gini'
    # else :
    #     print('\n')
    #     print('***Invalid Input***')
    #     main()

    splitEval = 'entropy'

    decision_tree = dt(maxDepth, minSplits, splitEval)
    sk_dt_model = DecisionTreeClassifier(criterion = splitEval, max_depth = maxDepth, min_samples_split = minSplits)
    
    choice = input('Enter [1] if you would like to preform cross validation OR [2] to preform train test split on data: ')

    if choice == '1':

        choice = input('Enter number of folds in cross validation you would like or hit ENTER for 5 folds: ') or 5
        numberFolds = int(choice) # increasing the number of folds greatly increases the time to run the algorithm

        folds = splitDataOnFolds(data, numberFolds)

        print('\n')

        ### Own algorithm cross validation ###
        startTime = time.time()
        scores = crossValidationEvaluation(folds, decision_tree)
        finishTime = time.time()

        print('Accuracy of each fold --------------------> {}'.format(scores))
        print('Average accuracy of folds ----------------> %.2f%%' % (sum(scores)/(len(scores))))
        print('Time taken to run the algorithm ----------> %.2f seconds' % (finishTime - startTime))

        ### SK Learn cross validation ###
        refStartTime = time.time()
        skAccuracyArray = skLearnCrossValidation(folds, sk_dt_model)
        refFinishTime = time.time()

        print('Accuracy of each SK-Learn fold -----------> {}'.format(skAccuracyArray))
        print('SK-Learn Model accuracy ------------------> %.2f%%' % (sum(skAccuracyArray)/(len(skAccuracyArray))))
        print('Time taken to run SK-Learn algorithm -----> %.2f seconds' % (refFinishTime - refStartTime))

    elif choice == '2':

        test = (input('Enter the percentage of testing data (should be < 50) or hit ENTER for 20% testing data: ') or 20)
        testPercentage = int(test)

        folds = trainTestSplit(data, testPercentage)

        print('\n')

        ### Own algorithm Learn train test ###
        startTime = time.time()
        score = trainTestEvaluation(folds, decision_tree) ## testing &&& 3 methods should be in test class
        finishTime = time.time()

        print('Accuragy of train test CART algorithm -----------> %.2f%%' % score)
        print('Time taken to run the algorithm -----------------> %.2f seconds' % (finishTime - startTime))

        ### SK Learn train test ###
        refStartTime = time.time()
        accuracy = skLearnTrainTest(folds, sk_dt_model)
        refFinishTime = time.time()

        print('SK-Learn Model accuracy -------------------------> %.2f%%' % accuracy)
        print('Time taken to run SK-Learn algorithm ------------> %.2f seconds' % (refFinishTime - refStartTime))

    else:
        print('\n')
        print('***Invalid Input***')
        main()


if __name__ == '__main__':
    main()
