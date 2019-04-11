##########################
#
#  Author: Mitesh Khadgi
#    Date: 03/24/2019
#
##########################

import sys
import time
import glob
import math
import string

def loadTrainingData(path):
    files = glob.glob(path)
    words = list()
    docCount = 0
    for file in files:
        f = open(file, 'r', encoding="ISO-8859-1")
        docCount = docCount + 1
        wordsFromFile = f.read().split()
        words.extend(wordsFromFile)
    return words, docCount

def countUniqueWords(dataFromFile):
    counter = dict()
    for i in dataFromFile:
        if not i in counter:
            counter[i] = 1
        else:
            counter[i] = counter[i] + 1
    return counter

def calcCondProb(condProb, dataFromFile, fLabel, uniqueWordsTrain):
    counterUniqueWords = countUniqueWords(dataFromFile)
    for word in uniqueWordsTrain:
        condProb[word + fLabel] = float(counterUniqueWords.get(word,0) + 1) / (len(dataFromFile) + len(uniqueWordsTrain))

def classifyLabel(condProb, prior, label, path, actualLabel):
    files = glob.glob(path)
    correctPred = 0
    incorrectPred = 0
    x = 0
    for file in files:
        f = open(file, 'r', encoding="ISO-8859-1")
        words = f.read().split()
        words = [''.join(c for c in s if c not in string.punctuation) for s in words]
        words = [s for s in words if s]
        #print(file, ": ", words)
        countOfWords = countUniqueWords(words)
        x = x + len(countOfWords)
        maxScore = -float('inf')
        predLabel = 0
        for fLabel in label:
            score = math.log(prior[fLabel])
            for word, count in countOfWords.items():
                score = score + count*(math.log(condProb.get(word + fLabel, 1)))
            if score > maxScore:
                maxScore = score
                predLabel = fLabel
        if (predLabel == actualLabel):
            correctPred = correctPred + 1
        else:
            incorrectPred = incorrectPred + 1
    return maxScore, correctPred, incorrectPred

def main():

    print("\nStarted simulation at 0 seconds\n")
    print("Please wait to compute the HAM and SPAM accuracies...\n")
    start = time.time()

    #Input 2 arguments as train folder and test folder.
    trainFolder = str(sys.argv[1])
    testFolder = str(sys.argv[2])

    #Get all the training folder filenames from the ham folder.
    hamTrainPath = trainFolder+'/ham/*.txt'

    #Get all the training folder filenames from the spam folder.
    spamTrainPath = trainFolder+'/spam/*.txt'

    #Get all the test folder filenames from the ham folder.
    hamTestPath = testFolder+'/ham/*.txt'
	
    #Get all the test folder filenames from the spam folder.
    spamTestPath = testFolder+'/spam/*.txt'

    #Store all possible final classifications for each sample data set.
    label = ['ham', 'spam']

    #Initialize priors and condProb as dictionaries.
    prior = dict()
    condProb = dict()
	
    #Load HAM and SPAM Training Data from HAM and SPAM folders.
    trainHam, trainHamCount = loadTrainingData(hamTrainPath)
    trainSpam, trainSpamCount = loadTrainingData(spamTrainPath)
    print("Total number of samples: ", trainSpamCount+trainHamCount, "\n")

    #Storing all the words read above from the Training Set.
    train = trainHam + trainSpam

    #Remove duplication of words from the Training Set.
    uniqueWordsTrain = list(set(train))

    #Calculate Prior Probabilities for HAM and SPAM data.
    prior[label[0]] = float(trainHamCount) / (trainHamCount + trainSpamCount)
    prior[label[1]] = float(trainSpamCount) / (trainHamCount + trainSpamCount)

    #Calculate Conditional Probabilities for HAM and SPAM data.
    calcCondProb(condProb, trainHam, label[0], uniqueWordsTrain)
    calcCondProb(condProb, trainSpam, label[1], uniqueWordsTrain)

    #Calculate Accuracy of HAM data.
    maxHamScore, hamCorrectPred, hamIncorrectPred = classifyLabel(condProb, prior, label, hamTestPath, label[0])
    hamAccuracy = round((hamCorrectPred / (hamCorrectPred + hamIncorrectPred)) * 100, 2)

    #Calculate Accuracy of SPAM data.
    maxSpamScore, spamCorrectPred, spamIncorrectPred = classifyLabel(condProb, prior, label, spamTestPath, label[1])
    spamAccuracy = round((spamCorrectPred / (spamCorrectPred + spamIncorrectPred)) * 100, 2)

    #Calculate Combined Accuracy of HAM and SPAM data.
    combinedAccuracy = round(float(hamCorrectPred + spamCorrectPred) / (hamCorrectPred + hamIncorrectPred + spamCorrectPred + spamIncorrectPred)*100, 2)

    print("--------------------------------------------------")
    print("Accuracy of HAM\t\t\t\t: ", str(hamAccuracy) + " %")
    print("Accuracy of SPAM\t\t\t: ", str(spamAccuracy) + " %")
    print("Combined Accuracy of HAM and SPAM\t: ", str(combinedAccuracy) + " %")
    print("--------------------------------------------------")

    end = time.time()
    elapsed = end - start
    print("\nCurrent simulation took %f seconds to complete." % elapsed)

    print("\nThank You ! Program ran Successfully.")
	
if __name__ == '__main__':
    main()