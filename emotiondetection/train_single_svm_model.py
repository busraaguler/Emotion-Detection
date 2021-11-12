import kivy
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager,Screen,SlideTransition
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



class Detection(GridLayout):
  def __init__(self, **kwargs):
        super(Detection, self).__init__(**kwargs)

        self.add_widget(Image(source='resim.jpg'))
        self.cols=1

    #def build(self):
        emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"] #Emotion liste
        clf =svm.SVC(C=2.0, kernel='poly', decision_function_shape='ovr', random_state=1453)


        data = np.load('train_dataset1.npz')
        training_dataset = data['train_data']
        training_labels_cat = data['train_labels']

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
        svm_model_name = "svm1.model"
        joblib.dump(clf, svm_model_name,compress = 3)
        #self.add_widget(Image(source='resim.jpg',size=self.size,pos=self.pos))
        self.add_widget(Label(text="Duygu Tanıma Doğruluğu: % "+str(prediction_accuracy*100),color=(0,0,0,1),font_name='Comic'))
        self.submit=Button(text=" Detection",font_size=40)
        self.add_widget(self.submit)


class emotionApp(App):
  def build(self):

        Window.clearcolor=(1,1,1,1)
        return Detection()



if __name__=="__main__":

    emotionApp().run()
 #result = confusion_matrix(training_labels_cat, prediction_labels)
#result=App()
 #print(result)








