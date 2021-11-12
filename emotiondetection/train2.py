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

data_path = "C:/Users/busraguler/PycharmProjects/emotiondetection/dataset1"
data_dir_list = os.listdir(data_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"] #Emotion liste

train_dataset=[]
train_labels=[]

def data_oku(data_path):
    for data in data_dir_list:
        img_list = os.listdir(data_path + '/' + data)
        print('' + '{}\n'.format(data))
        for img in img_list[0:10]:
            input_img = cv2.imread(data_path + '/' + data + '/' + img)
            print(img)
            imgGray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            faces = detector(imgGray)

            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                # img=cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                landmarks = predictor(imgGray, face)

            katilimci_landmarks = []

            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(input_img, (x, y), 3, (0, 0, 255), cv2.FILLED)
                katilimci_landmarks.append([x, y])

            landmark_sol = katilimci_landmarks[0]
            landmark_sag = katilimci_landmarks[16]
            landmark_sol_kas = katilimci_landmarks[19]
            landmark_sag_kas = katilimci_landmarks[24]

            landmark_ust = (np.array(landmark_sol_kas) + np.array(landmark_sag_kas)) / 2
            landmark_alt = katilimci_landmarks[8]

            x_kayma = float(landmark_sol[0])
            y_kayma = float(landmark_ust[1])

            x_olcek = float(landmark_sag[0] - landmark_sol[0])
            y_olcek = float(landmark_alt[1] - landmark_ust[1])

            katilimci_landmarks_np = np.array(katilimci_landmarks)
            katilimci_landmarks_np = katilimci_landmarks_np.astype(np.float32)
            katilimci_landmarks_np[:, 0] = 5.0 * (katilimci_landmarks_np[:, 0] - x_kayma) / x_olcek
            katilimci_landmarks_np[:, 1] = 5.0 * (katilimci_landmarks_np[:, 1] - y_kayma) / y_olcek
            # np.append(train_dataset, katilimci_landmarks_np.flatten(), axis=1)

            train_dataset.append(katilimci_landmarks_np.tolist())
            train_labels.append(data)

cv2.imshow("image", input_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

accuracy_clf=[]
#for i in range(0,10):    #svm training
#print("set olusturuluyor %s" % (i + 1))
#print("egitim SVM linear %s" %i)
training_dataset=np.array(train_dataset)
training_labels=np.array(train_labels)
#testing_dataset=np.array(test_dataset)
#testing_labels=np.array(test_labels)
# Kategorik etiketleri dönüştür
training_labels_df= pd.DataFrame(training_labels, columns=['Emotion_Types'])
labelencoder = LabelEncoder()
training_labels_df['Emotion_Types_Cat'] = labelencoder.fit_transform(training_labels_df['Emotion_Types'])
training_labels_cat = training_labels_df['Emotion_Types_Cat']
training_dataset = training_dataset.reshape(training_dataset.shape[0], (training_dataset.shape[1]*training_dataset.shape[2]))

np.savez('train_dataset0',train_data =training_dataset , train_labels = training_labels_cat);

