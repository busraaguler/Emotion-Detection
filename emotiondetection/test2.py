import os
import cv2
import dlib
import numpy as np
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import time
import pandas as pd
import joblib


emotions = ["anger",  "disgust", "fear", "happiness",  "sadness", "surprise"] #Emotion liste
#emotions = ["anger", "contempt" ,"disgust", "fear", "happiness","neutral",  "sadness", "surprise"] #Emotion liste

data0 = np.load('test_dataset0.npz')
test_dataset0 = data0['test_data']
testing_labels_cat = data0['test_labels']

data1 = np.load('test_dataset1.npz')
test_dataset1 = data1['test_data']

data2 = np.load('test_dataset2.npz')
test_dataset2 = data2['test_data']

#svm0
#testing_dataset=test_dataset0
#svm1
#testing_dataset=np.concatenate((test_dataset0,test_dataset1,test_dataset2),axis=1)
#svm2
#testing_dataset=np.concatenate((test_dataset0,np.array((test_dataset2-test_dataset1))),axis=1)
#svm3
#testing_dataset=np.array((test_dataset2-test_dataset1))
#svm4
#testing_dataset=np.concatenate((test_dataset0 ,np.array((test_dataset2-2*(test_dataset1)+test_dataset0))),axis=1)
#svm5
#testing_dataset=np.concatenate((test_dataset0,np.array((test_dataset2-test_dataset1)),np.array((test_dataset2-2*(test_dataset1)+test_dataset0))),axis=1)
#svm6
#testing_dataset=np.array((test_dataset2-2*(test_dataset1)+test_dataset0))
#svm7
#testing_dataset=np.array((test_dataset2-test_dataset0))
#svm8
#testing_dataset=np.concatenate((test_dataset0,np.array((test_dataset2-test_dataset0))),axis=1)
#svm9

indices=np.concatenate((np.arange(0,10),np.arange(20,50),np.arange(60,80)),axis=0)
testing_dataset=test_dataset0[indices,:]
testing_labels_cat=np.concatenate((np.ones(10)*0,np.ones(10)*1,np.ones(10)*2,np.ones(10)*3,np.ones(10)*4,np.ones(10)*5),axis=0)
testing_labels_cat=testing_labels_cat.astype(int)



svm_model_name = "svm9_linear.model"
clf=joblib.load(svm_model_name)
prediction_accuracy = clf.score(testing_dataset, testing_labels_cat)
print("Accuracy for set:", prediction_accuracy*100)
prediction_labels = clf.predict(testing_dataset)
result = confusion_matrix(testing_labels_cat, prediction_labels)
print(result)
print(classification_report(testing_labels_cat, prediction_labels, target_names=emotions))





