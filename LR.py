##########################
#
#  Author: Mitesh Khadgi
#    Date: 03/24/2019
#
##########################

import os
import re
import sys
import time
import string
from numpy import *

#learning rate value.
eta = 0.1
#number of iterations for weightMatrix computation.
numOfIterations = 100

def readAllFiles(path):
    allWordsList = []
    dicFileWords = {}
    files = os.listdir(path)
    for file in files:
        f = open(path + "/" + file, encoding="ISO-8859-1")
        words_list = f.read()
        words_list = re.sub('[^0-9a-zA-Z]', ' ', words_list)
        words_list = words_list.strip().split()
        dicFileWords[file] = words_list
        allWordsList.extend(words_list)
    return allWordsList, dicFileWords

def setLabels(numSpamFile, numHamFile):
    ClassLabel = []
    for i in range(numSpamFile):
        ClassLabel.append(0)
    for j in range(numHamFile):
        ClassLabel.append(1)
    return ClassLabel

def attributeValue(allWords, dict):
    attributeValList = []
    for i in dict:
        attrVal = [0] * (len(allWords)) #Initialize the size of the list - 'attrVal'.
        for word in allWords:
            if word in dict[i]:
                attrVal[allWords.index(word)] = 1
        attrVal.insert(0,1) #Considering x0 attribute equal to 1.
        attributeValList.append(attrVal)
    return attributeValList

def sigmoid(x):
    result = 1.0/(1 + exp(-x))
    return result

def updateWeights(lamb, trainList, trainLabels):
    attributeMatrix = matrix(trainList)
    finalLabelMatrix = matrix(trainLabels).transpose()
    columns = shape(attributeMatrix)[1]
    weightMatrix = zeros((columns,1)) #Initialize the weightMatrix with zeros with columns as number of rows.
    for i in range(numOfIterations):
        sigma = sigmoid(attributeMatrix*weightMatrix)
        actual = finalLabelMatrix
        predicted = sigma
        yerror = actual - predicted
        weightMatrix = weightMatrix + eta * (attributeMatrix.transpose()*yerror - lamb*weightMatrix)
    return weightMatrix


def classifyLabel(updatedWeight, testList, numTestSpam, numTestHam):
    attributeTestMatrix = matrix(testList)
    sum = attributeTestMatrix * updatedWeight
    totalTestSamples = numTestSpam + numTestHam
    correctPred = 0
    hamCorrectPred = 0
    spamCorrectPred = 0
    hamIncorrectPred = 0
    spamIncorrectPred = 0

    for i in range(numTestSpam):
        if sum[i][0] < 0.0:
            correctPred += 1
            spamCorrectPred += 1
        else:
            spamIncorrectPred += 1
    for j in range(numTestSpam+1,totalTestSamples):
        if sum[j][0] > 0.0:
            correctPred += 1
            hamCorrectPred += 1
        else:
            hamIncorrectPred += 1

    hamTotal = hamCorrectPred + hamIncorrectPred
    spamTotal = spamCorrectPred + spamIncorrectPred
    hamAccuracy = round(100.0 * hamCorrectPred/hamTotal, 2)
    spamAccuracy = round(100.0 * spamCorrectPred/spamTotal, 2)
    combinedAccuracy = round(100.0 * correctPred/totalTestSamples, 2)
    return sum, hamAccuracy, spamAccuracy, combinedAccuracy

def main():

    print("\nStarted simulation at 0 seconds\n")
    print("Please wait to compute the HAM and SPAM accuracies...\n")
    start = time.time()

    #Input 2 arguments as train folder and test folder.
    trainFolder = str(sys.argv[1])
    testFolder = str(sys.argv[2])
	
    #Get all the training folder filenames from the ham folder.
    trainHamPath = trainFolder+'/ham'

    #Get all the training folder filenames from the spam folder.
    trainSpamPath = trainFolder+'/spam'

    #Get all the test folder filenames from the ham folder.
    testHamPath = testFolder+'/ham'
	
    #Get all the test folder filenames from the spam folder.
    testSpamPath = testFolder+'/spam'

    #lambda value (typical value = 0.001).
    lamb = float(sys.argv[3])

    #Read the SPAM and HAM training data set and, extract each word in a list for each SPAM and HAM training data file
    trainSpamList, trainSpamDict = readAllFiles(trainSpamPath)
    trainHamList, trainHamDict = readAllFiles(trainHamPath)

    #Read the SPAM and HAM test data set and, extract each word in a list for each SPAM and HAM test data file
    testSpamList, testSpamDict = readAllFiles(testSpamPath)
    testHamList, testHamDict = readAllFiles(testHamPath)

    #Get all the unique words from SPAM and HAM in a list.
    allWords = list(set(trainSpamList)|set(trainHamList))
    #Get all the words from SPAM and HAM training data set in a dictionary with key values as filename.
    allWordsTrain = {**trainSpamDict, **trainHamDict}
    #Get all the words from SPAM and HAM test data set in a dictionary with key values as filename.
    allWordsTest = {**testSpamDict, **testHamDict}

    #Compute the number of SPAM and HAM - training and test data sets.
    [numTrainSpam, numTrainHam, numTestSpam, numTestHam] = [len(trainSpamDict), len(trainHamDict), len(testSpamDict), len(testHamDict)]
    print("Total number of training samples\t: ", numTrainSpam+numTrainHam)
    print("Total number of test samples\t\t: ", numTestSpam+numTestHam, "\n")
    #Set all the labels as SPAM first as '0' and HAM second as '1' in this particular order.
    trainLabels = setLabels(numTrainSpam, numTrainHam)

    #Compare each word from the allWords with the allWordsTrain word list, and if the word exists in the training words list, store '1' in the list, otherwise store '0' in the list.
    trainList = attributeValue(allWords, allWordsTrain)
    #Compare each word from the allWords with the allWordsTest word list, and if the word exists in the test words list, store '1' in the list, otherwise store '0' in the list.
    testList = attributeValue(allWords, allWordsTest)
	
    #Compute weights using learning rate as 'eta' and regularization parameter as 'lamb' using Gradient Ascent with L2 regularization.
    updatedWeight = updateWeights(lamb, trainList, trainLabels)
    #Classify label to the provided test data set and return the number of correct predictions with correct predictions for SPAM and HAM separately to calculate the accuracies for SPAM and HAM with combined accuracy.
    sum, hamAccuracy, spamAccuracy, combinedAccuracy = classifyLabel(updatedWeight, testList, numTestSpam, numTestHam)

    print("--------------------------------------------------")
    print("Accuracy of HAM\t\t\t\t: ", str(hamAccuracy) + " %")
    print("Accuracy of SPAM\t\t\t: ", str(spamAccuracy) + " %")
    print("Combined Accuracy of HAM and SPAM\t: ", str(combinedAccuracy) + " %")
    print("--------------------------------------------------")

    end = time.time()
    elapsed = end - start
    print("\nCurrent simulation took %f seconds to complete." % elapsed)

    print("\nThank You ! Program ran Successfully.")
	
if __name__ == "__main__":
    main()