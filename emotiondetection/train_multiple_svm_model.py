#import kivy
#from kivy.app import App
import os
import cv2
import dlib
import numpy as np
from sklearn.svm import SVC
from sklearn import svm
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import time
import pandas as pd
import seaborn as sn



emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"] #Emotion liste
#emotions = ["anger", "contempt" ,"disgust", "fear", "happiness","neutral",  "sadness", "surprise"] #Emotion liste
clf =svm.SVC(C=2.0, kernel='linear', decision_function_shape='ovr', random_state=1453)

data0 = np.load('train_dataset0.npz')
training_dataset0 = data0['train_data']
training_labels_cat = data0['train_labels']

data1 = np.load('train_dataset1.npz')
training_dataset1 = data1['train_data']

data2 = np.load('train_dataset2.npz')
training_dataset2 = data2['train_data']

#svm0
#training_dataset=training_dataset0
#svm1
#training_dataset=np.concatenate((training_dataset0,training_dataset1,training_dataset2),axis=1)
#svm2
#training_dataset=np.concatenate((training_dataset0,np.array((training_dataset2-training_dataset1))),axis=1)
#svm3
#training_dataset=np.array((training_dataset2-training_dataset1))
#svm4
#training_dataset=np.concatenate((training_dataset0 ,np.array((training_dataset2-2*(training_dataset1)+training_dataset0))),axis=1)
#svm5
#training_dataset=np.concatenate((training_dataset0,np.array((training_dataset2-training_dataset1)),np.array((training_dataset2-2*(training_dataset1)+training_dataset0))),axis=1)
#svm6
#training_dataset=np.array((training_dataset2-2*(training_dataset1)+training_dataset0))
#svm7
#training_dataset=np.array((training_dataset2-training_dataset0))
#svm8
#training_dataset=np.concatenate((training_dataset0,np.array((training_dataset2-training_dataset0))),axis=1)
#svm9

indices=np.concatenate((np.arange(0,10),np.arange(20,50),np.arange(60,80)),axis=0)
training_dataset=training_dataset0[indices,:]
training_labels_cat=np.concatenate((np.ones(10)*0,np.ones(10)*1,np.ones(10)*2,np.ones(10)*3,np.ones(10)*4,np.ones(10)*5),axis=0)
training_labels_cat=training_labels_cat.astype(int)

clf.fit(training_dataset,training_labels_cat)

accuracy_clf=[]
prediction_accuracy = clf.score(training_dataset, training_labels_cat)
print("Accuracy for set:", prediction_accuracy*100)
accuracy_clf.append(prediction_accuracy)
prediction_labels = clf.predict(training_dataset)
result = confusion_matrix(training_labels_cat, prediction_labels)

print(result)
print(classification_report(training_labels_cat, prediction_labels, target_names=emotions))

print("model kaydediliyor")
svm_model_name = "svm9_linear.model"
joblib.dump(clf, svm_model_name,compress = 3)