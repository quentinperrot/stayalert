import numpy as np
import csv, util, time
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import load_iris


global trainingData 
global trainingDataLabels 
global testData 
global testDataLabels

# Feature Selection: multiple approaches #
def selectBestK():
	X = np.array(testData, dtype=float)
	y = np.array(testDataLabels, dtype=float)
	newX = SelectKBest(chi2, k=3).fit_transform(X, y)		
	print newX

 
def parseTrainingData(trainingData, trainingDataLabels): 
	with open('fordTrain.csv', 'rb') as f:
	    reader = csv.reader(f)
	    reader.next()
	    for row in reader:
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
	for i in range (vecSize):
		if prediction[i] != actual[i] : 
			error += 1
			if prediction[i] == 1 : falsePositive += 1
			else : falseNegative += 1
	return [float(error) / float(vecSize) * 100, float(falsePositive) / float(error) * 100, float(falseNegative) / float(error) * 100]
			 

def SVM(trainingData, trainingDataLabels, testData, testDataLabels): 
	clf = svm.SVC()
	print clf.fit(np.array(trainingData, dtype=float), np.array(trainingDataLabels, dtype=float))
	# SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
 #       fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
 #       loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5,
 #       random_state=None, shuffle=False, verbose=0, warm_start=False)
	print 'PREDICTION'
	prediction = clf.predict(np.array(testData, dtype=float))
	return prediction



trainingData = []
trainingDataLabels = []
testData = []
testDataLabels = []

parseTrainingData(trainingData, trainingDataLabels)
parseTestingData(testData, testDataLabels)
selectBestK()

# prediction = SVM(trainingData, trainingDataLabels, testData, testDataLabels)
# errorData =  errorPercentage(prediction, np.array(testDataLabels, dtype=float))

# print "Test Error: " + str(errorData[0]) + '%.'
# print "False Positives: " + str(errorData[1]) + '%.'
# print "False Negatives: " + str(errorData[2]) + '%.'

