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


data_path = "C:/Users/busraguler/PycharmProjects/emotiondetection/dataset2"
data_dir_list = os.listdir(data_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"] #Emotion liste


test_dataset=[]
test_labels=[]

for data in data_dir_list:
    img_list=os.listdir(data_path+'/'+ data)
    print(''+'{}\n'.format(data))
    for img in img_list[-10:]:
        input_img=cv2.imread(data_path + '/'+ data + '/'+ img)
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

        katilimci_landmarks=[]

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(input_img, (x, y), 3, (0, 0, 255), cv2.FILLED)
            katilimci_landmarks.append([x,y])

        landmark_sol=katilimci_landmarks[0]
        landmark_sag=katilimci_landmarks[16]
        landmark_sol_kas=katilimci_landmarks[19]
        landmark_sag_kas=katilimci_landmarks[24]

        landmark_ust=np.array((landmark_sol_kas)+(landmark_sag_kas))/2
        landmark_alt=katilimci_landmarks[8]

        x_kayma=float(landmark_sol[0])
        y_kayma=float(landmark_ust[1])

        x_olcek=float(landmark_sag[0]-landmark_sol[0])
        y_olcek=float(landmark_alt[1]-landmark_ust[1])

        katilimci_landmarks_np = np.array(katilimci_landmarks)
        katilimci_landmarks_np = katilimci_landmarks_np.astype(np.float32)
        katilimci_landmarks_np[:, 0] = 5.0 * (katilimci_landmarks_np[:, 0] - x_kayma) / x_olcek
        katilimci_landmarks_np[:, 1] = 5.0 * (katilimci_landmarks_np[:, 1] - y_kayma) / y_olcek

        test_dataset.append(katilimci_landmarks_np.tolist())
        test_labels.append(data)




cv2.imshow("image", input_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
testing_dataset=np.array(test_dataset)
testing_labels=np.array(test_labels)
# Kategorik etiketleri dönüştür
testing_labels_df= pd.DataFrame(testing_labels, columns=['Emotion_Types'])
labelencoder = LabelEncoder()
testing_labels_df['Emotion_Types_Cat'] = labelencoder.fit_transform(testing_labels_df['Emotion_Types'])
testing_labels_cat = testing_labels_df['Emotion_Types_Cat']
testing_dataset = testing_dataset.reshape(testing_dataset.shape[0], (testing_dataset.shape[1]*testing_dataset.shape[2]))


np.savez('test_dataset2',test_data =testing_dataset , test_labels = testing_labels_cat);




