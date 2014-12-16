import numpy as np
import csv, util, time
from sklearn import svm
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

global trainingData 
global trainingDataLabels 
global testData 
global testDataLabels

# Feature Selection: multiple approaches #
def selectBestK():
	X = np.array(trainingData, dtype=float)
	X1 = np.array(testData, dtype=float)
	y = np.array(trainingDataLabels, dtype=float)
	model = SelectKBest(chi2, k=3)
	newX = model.fit_transform(X, y)
	newX1 = model.transform(X1)
	return (newX, newX1)

# Does not work #
def recursiveFeatureSelection():
	X = np.array(trainingData, dtype=float)
	y = np.array(trainingDataLabels, dtype=float)
	svc = SVC("linear", 1)
	rfe = RFE(svc, 1, 1)
	rfe.fit(X, y)
	print rfe

def l1FeatureSelection():
	X = np.array(trainingData, dtype=float)
	X1 = np.array(testData, dtype=float)
	y = np.array(trainingDataLabels, dtype=float)
	model = LinearSVC(C=0.01, penalty="l1", dual=False)
	newX = model.fit_transform(X, y)
	newX1 = model.transform(X1)
	return (newX, newX1)

    
def parseTrainingData(trainingData, trainingDataLabels): 
	with open('fordTrain.csv', 'rb') as f:
	    reader = csv.reader(f)
	    reader.next()
	    for row in reader:
	    	row[3] = float(row[3]) + 23
	    	row[4] = float(row[4]) + 46
	    	row[14] = float(row[14]) + 250
	    	row[23] = float(row[23]) + 5
	    	trainingDataLabels.append(row[2])
	    	trainingData.append(row[3:])
	# trainingData = np.array(trainingData, dtype=float)
	# trainingDataLabels = np.array(trainingDataLabels, dtype=float)


def parseTestingData(testData, testDataLabels):
	with open('fordTest.csv', 'rb') as f:
		reader = csv.reader(f)
		reader.next()
		for row in reader: 
			testData.append(row[3:])
	with open('Solution.csv', 'rb') as f:
		reader = csv.reader(f)
		reader.next()
		for row in reader: 
			testDataLabels.append(row[2])
	# testData = np.array(testData, dtype=float)
	# testDataLabels = np.array(testDataLabels, dtype=float)

def errorPercentage (prediction, actual) :
	vecSize = len(prediction)
	error = 0
	falsePositive = 0
	falseNegative = 0
	truePositive = 0
	trueNegative = 0
	for i in range (vecSize):
		if prediction[i] != actual[i] : 
			error += 1
			if prediction[i] == 1 : falsePositive += 1
			else : falseNegative += 1
		else:
			if prediction[i] == 1 : truePositive += 1
			else : trueNegative += 1
	correct = vecSize - error
	return [float(error) / float(vecSize) * 100, float(falsePositive) / float(error) * 100, float(falseNegative) / float(error) * 100, float(truePositive) / float(correct) * 100, float(trueNegative) / float(correct) * 100, float(sum(prediction))/float(vecSize) * 100, float(sum(actual))/float(vecSize) * 100]
			 

def runModel(trainingData, trainingDataLabels, testData, testDataLabels): 
	# Hinge-loss SGD #
	# clf = SGDClassifier(loss="hinge", penalty="l2", n_iter=5)

	# Logistic SGD #
	# clf = SGDClassifier(loss="log", penalty="l2", n_iter=5)

	# Naive Bayes #
	# clf = GaussianNB()

	# Randomized Forests #
	clf = RandomForestClassifier(n_estimators = 10)

	# Extremely Randomized Trees # 
	#clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state = 0)

	# Gradient Boosting Classifier #
	#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

	clf.fit(np.array(trainingData, dtype=float), np.array(trainingDataLabels, dtype=float))

	# Training Predictions Analyzed #
	print 'TRAIN PREDICTION'
	prediction = clf.predict(np.array(trainingData, dtype=float))
	errorData =  errorPercentage(prediction, np.array(trainingDataLabels, dtype=float))
	print "Train Error: " + str(errorData[0]) + '%.'
	print "False Positives: " + str(errorData[1]) + '%.'
	print "False Negatives: " + str(errorData[2]) + '%.'
	print "True Positives: " + str(errorData[3]) + '%.'
	print "True Negatives: " + str(errorData[4]) + '%.'
	print "Total positive predicitons: " + str(errorData[5]) + '%.'
	print "Total positive actual: " + str(errorData[6]) + '%.'

	# Testing Predictions Analyzed #
	print 'TEST PREDICTION'
	prediction = clf.predict(np.array(testData, dtype=float))
	errorData =  errorPercentage(prediction, np.array(testDataLabels, dtype=float))
	print "Test Error: " + str(errorData[0]) + '%.'
	print "False Positives: " + str(errorData[1]) + '%.'
	print "False Negatives: " + str(errorData[2]) + '%.'
	print "True Positives: " + str(errorData[3]) + '%.'
	print "True Negatives: " + str(errorData[4]) + '%.'
	print "Total positive predicitons: " + str(errorData[5]) + '%.'
	print "Total positive actual: " + str(errorData[6]) + '%.'


	return errorData[0]

def selectiveArray(oldVec, indeces):
    indeces = sorted(indeces)
    newVec = []
    for row in oldVec:
        newNewVec = []
        for index in indeces:
            newNewVec.append(row[index])
        newVec.append(newNewVec)
    return newVec


# Prepare data sets #
trainingData = []
trainingDataLabels = []
testData = []
testDataLabels = []
parseTrainingData(trainingData, trainingDataLabels)
parseTestingData(testData, testDataLabels)

#STARTING TIMING FEATURE SELECTION
# start_time_featureSelection = time.time()
# Select features and transform training data #
# data = l1FeatureSelection()
# trainingData = data[0]
# testData = data[1]

#STOP TIMING FEATURE SELECTIONG0
#print "---FEATURE SELECTION TIME: ",  time.time() - start_time_featureSelection

#STARTING TIMING LEARNING ALGORITHM
start_time_learningAlgorithm = time.time()


# Choose our own features #
featuresWanted = [15, 16, 29]
prediction = runModel(selectiveArray(trainingData, featuresWanted), trainingDataLabels, selectiveArray(testData, featuresWanted), testDataLabels)

# Choose all features #
#trainingData, testData, trainingDataLabels, testDataLabels = util.getDataset("fordTrain.csv", "fordTest.csv", "solution.csv", True, 50)
#prediction = runModel(trainingData, trainingDataLabels, testData, testDataLabels)
print "---LEARNING TIME: ", time.time() - start_time_learningAlgorithm

