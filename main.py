import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np

cap = cv2.VideoCapture(2)  # Iniciando a captura de vídeo

hands = mp.solutions.hands.Hands(max_num_hands=1)  # Iniciando o modelo de detecção de mãos

classes = ['A','B','C','D','E']  # Classes que serão reconhecidas
model = load_model('keras_model.h5')  # Carregando o modelo de classificação
# Criando um array para armazenar a imagem
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    frameRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    handsPoints = results.multi_hand_landmarks
    h, w, _ = img.shape

    if handsPoints != None:
        for hand in handsPoints:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in hand.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max: x_max = x
                if x < x_min: x_min = x
                if y > y_max: y_max = y
                if y < y_min: y_min = y
            cv2.rectangle(img, (x_min-50, y_min-50), (x_max+50, y_max+50), (0, 255, 0), 2)
            try:
                imgCrop = img[y_min-50:y_max+50,x_min-50:x_max+50]
                imgCrop = cv2.resize(imgCrop,(224,224))
                imgArray = np.asarray(imgCrop)
                normalized_image_array = (imgArray.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array
                prediction = model.predict(data)
                indexVal = np.argmax(prediction)
                cv2.putText(img,classes[indexVal],(x_min-50,y_min-65),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),5)
            except: continue

    cv2.imshow('Imagem',img)
    cv2.waitKey(1)