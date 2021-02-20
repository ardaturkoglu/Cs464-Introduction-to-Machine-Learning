import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

test_set=pd.read_csv("q3_test_dataset.csv").to_numpy()
train_set=pd.read_csv("q3_train_dataset.csv").to_numpy()

for i in range(len(train_set)):
    if train_set[i][2] == 'male':
        train_set[i][2] = 1
    else:
        train_set[i][2] =0
        
    if train_set[i][7] == 'S':
        train_set[i][7] = 0
    elif train_set[i][7] == 'C':
        train_set[i][7] = 1
    else:
        train_set[i][7] = 2

for i in range(len(test_set)):
    if test_set[i][2] == 'male':
        test_set[i][2] = 1
    else:
        test_set[i][2] =0
    if test_set[i][7] == 'S':
        test_set[i][7] = 0
    elif test_set[i][7] == 'C':
        test_set[i][7] = 1
    else:
        test_set[i][7] = 2    

train_x = train_set[:,[1,2,3,4,5,6,7]]
train_x = train_x.astype(np.float64)
train_y = train_set[:,[0]].astype(np.float64)

test_x= test_set[:,[1,2,3,4,5,6,7]]
test_x = test_x.astype(np.float64)
test_y= test_set[:,[0]].astype(np.float64)

weights = np.random.normal(0,0.01,size=(7,1))
bias = 0
learning_rates = [10**-4,10**-3,10**-2]


def gradient(train_x, train_y, size, bias, weights):
    z = train_x.dot(weights) + bias
    sigmoid = 1 / (1 + np.exp(-z))
    gradient_weights = np.dot(train_x.T , (train_y - sigmoid)) / size #(x*y)/size
    gradient_bias   = np.sum(train_y - sigmoid) / size #y-sigmoid
    return  gradient_weights, gradient_bias


def confusion_matrix(weights,bias,batch_y,batch_x):
    z = test_x.dot(weights) + bias
    sigmoid = 1 / (1 + np.exp(-z))
    predict_survive = np.zeros((len(test_x), 1))
    truePositives  = 1 
    falsePositives = 1
    falseNegatives = 1
    trueNegatives  = 1

    #predict survive or not
    for i in range(len(test_x)):
        if sigmoid[i,0] > 0.5:
            predict_survive[i,0] = 1
        else:
            predict_survive[i,0] = 0

    for i in range(len(test_x)):
        if test_y[i] == 1 and predict_survive[i]==1:
            truePositives += 1
        if test_y[i] != 1 and predict_survive[i]==1:
            falsePositives += 1
        if test_y[i] == 1 and predict_survive[i]!=1:
            falseNegatives += 1
        if test_y[i] != 1 and predict_survive[i]!=1:
            trueNegatives += 1
 
    return truePositives,falsePositives,falseNegatives,trueNegatives

def showResults(truePositives,falsePositives,falseNegatives,trueNegatives,total):
    
    acurracy = ((truePositives + trueNegatives) / total) * 100
    precision = truePositives / (truePositives + falsePositives) 
    recall = truePositives / (truePositives + falseNegatives)
    negativePredictiveValue = trueNegatives / (falseNegatives + trueNegatives)
    falsePositiveRate = falsePositives / (falsePositives + trueNegatives)
    falseDiscoveryRate = falsePositives / (falsePositives + truePositives)
    f1Score = (2 * precision * recall) / (precision + recall)
    f2Score = (5 * precision * recall) / (4 * precision + recall)
    
    print("Accuracy:",acurracy)
    print("Precision:",precision)
    print("Recall:",recall)
    print("Negative Predictive Value(NPV):",negativePredictiveValue)
    print("False Positive Rate(FPR):",falsePositiveRate)
    print("False Discovery Rate(FDR):",falseDiscoveryRate)
    print("F1:",f1Score)
    print("F2:",f2Score)

def miniBatchGradient(learning_rate,weights,bias):
    start = time.time()
    weights = np.random.normal(0,0.01,size=(7,1))
    bias = 0
    for i in range(1,1001): 
        batch_index = np.random.randint(len(test_x), size=32)
        batch_y = train_y[batch_index]
        batch_x = train_x[batch_index,:]
        
        gradient_weights, gradient_bias = gradient(batch_x, batch_y, 32, bias, weights)
        
        weights +=  learning_rate * gradient_weights
        bias +=  learning_rate * gradient_bias
        if i % 100 == 0:
            print("Weights at i = ",i)
            print(weights.T)
    truePositives,falsePositives,falseNegatives,trueNegatives = confusion_matrix(weights,bias,batch_y,batch_x)
    print("Confusion Matrix(True Positives and False Positives) :",truePositives,falsePositives)
    print("Confusion Matrix(False Negatives and True Negatives) :",falseNegatives,trueNegatives)
    showResults(truePositives,falsePositives,falseNegatives,trueNegatives,len(test_x))
    end = time.time()
    print("Training time for Mini Batch Gradient:",end-start)
def sGradient(learning_rate,weights,bias):
    start = time.time()
    weights = np.random.normal(0,0.01,size=(7,1))
    bias = 0
    for i in range(1,1001): 
      for j in range(len(test_x)):
        index = np.array([j])
        labels_y = train_y[index]
        features_x = train_x[index,:]
        
        gradient_weights, gradient_bias = gradient(features_x, labels_y,len(test_x), bias, weights)

        weights += learning_rate * gradient_weights
        bias += learning_rate * gradient_bias
      if i % 100 == 0:
         print("Weights at i = ",i)
         print(weights.T)
    truePositives,falsePositives,falseNegatives,trueNegatives = confusion_matrix(weights,bias,labels_y,features_x)
    print("Confusion Matrix(True Positives and False Positives) :",truePositives,falsePositives)
    print("Confusion Matrix(False Negatives and True Negatives) :",falseNegatives,trueNegatives)
    showResults(truePositives,falsePositives,falseNegatives,trueNegatives,len(test_x))
    end = time.time()
    print("Training time for Stochastic Gradient:",end-start)
    
    
def fullBatchGradient(learning_rate,weights,bias):
    start = time.time()
    weights = np.random.normal(0,0.01,size=(7,1))
    bias = 0
    for i in range(1,1001):         
        gradient_weights, gradient_bias = gradient(train_x, train_y, len(test_x), bias, weights)

        weights +=  learning_rate * gradient_weights
        bias +=  learning_rate * gradient_bias
        if i % 100 == 0:
             print("Weights at i = ",i)
             print(weights.T)    
    truePositives,falsePositives,falseNegatives,trueNegatives = confusion_matrix(weights,bias,train_y,train_x)
    print("Confusion Matrix(True Positives and False Positives) :",truePositives,falsePositives)
    print("Confusion Matrix(False Negatives and True Negatives) :",falseNegatives,trueNegatives)
    showResults(truePositives,falsePositives,falseNegatives,trueNegatives,len(test_x))
    end = time.time()
    print("Training time for Full-Batch Gradient:",end-start)


print("Mini-Batch Gradient Ascent Algorithm (Size=32)")
print("Learning Rate =",learning_rates[0])
miniBatchGradient(learning_rates[0],weights,bias)
print("--------------------------------------------")
print("Learning Rate =",learning_rates[1])
miniBatchGradient(learning_rates[1],weights,bias)
print("--------------------------------------------")
print("Learning Rate =",learning_rates[2])
miniBatchGradient(learning_rates[2],weights,bias)
print("--------------------------------------------")

print("Stohastic Gradient Ascent Algorithm")
print("Learning Rate =",learning_rates[0])
sGradient(learning_rates[0],weights,bias)
print("--------------------------------------------")
print("Learning Rate =",learning_rates[1])
sGradient(learning_rates[1],weights,bias)
print("--------------------------------------------")
print("Learning Rate =",learning_rates[2])
sGradient(learning_rates[2],weights,bias)
print("--------------------------------------------")
print("Full Batch Gradient Ascent Algorithm")
print("Learning Rate =",learning_rates[0])
fullBatchGradient(learning_rates[0],weights,bias)
print("--------------------------------------------")
print("Learning Rate =",learning_rates[1])
fullBatchGradient(learning_rates[1],weights,bias)
print("--------------------------------------------")
print("Learning Rate =",learning_rates[2])
fullBatchGradient(learning_rates[2],weights,bias)


