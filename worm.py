#! python3
from __future__ import print_function
import pyautogui, sys
import numpy as np
from PIL import ImageGrab
import cv2
import os
import math
import keras
import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import TensorBoard
import random

WIDTH = 400
HEIGHT = 200
img_rows, img_cols = WIDTH, HEIGHT
input_shape = (WIDTH,HEIGHT,1)
LR = 1e-3
EPOCHS = 200
MODEL_NAME = 'new_slither-{}-{}-{}-epochs.model'.format(LR, 'convnets',EPOCHS)

def angle(x,y):
	center_x = 400
	center_y = 300
	h_x = 600
	h_y = 300
	x = x - center_x
	y = y - center_y
	magnitude = math.sqrt(x * x + y * y)
	if (magnitude > 0):
		angle1 = math.acos(x / magnitude)
	angle1 = angle1 * 180 / math.pi
	if (y < 0):
		angle1 = 360.0 - angle1
	return angle1

model = Sequential()
model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2), input_shape=input_shape,))
model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(9, activation='softmax'))
#model.summary()

model.load_weights(MODEL_NAME)

for i in list(range(4))[::-1]:
	print(i+1)
	time.sleep(1)

def zero():
	pyautogui.moveTo(500, 300,1)

def one():
	pyautogui.moveTo(500, 400,1)

def two():
	pyautogui.moveTo(500, 500,1)

def three():
	pyautogui.moveTo(300, 400,1)

def four():
	pyautogui.moveTo(300, 400,1)

def five():
	pyautogui.moveTo(300, 300,1)

def six():
	pyautogui.moveTo(400, 200,1)

def seven():
	pyautogui.moveTo(500, 200,1)

options = {0 : zero,
           1 : one,
           2 : two,
           3 : three,
           4 : four,
           5 : five,
           6 : six,
           7 : seven,
           8 : zero,
           9 : zero,		
}
file = 0
file_name = 'slither_data_new-{}.npy'.format(file)

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []

def recalculate(model):
     if os.path.isfile(file_name):
        tr_data = np.load(file_name)
        X = np.array([i[0] for i in tr_data]).reshape(-1, WIDTH, HEIGHT, 1)
        X = X.astype('float32')/127.0 - 1
        Y = [i[1] for i in tr_data]
        real_Y = keras.utils.to_categorical(Y, num_classes=9)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy',   metrics=['accuracy'])
        model.fit(X, real_Y, epochs=20, batch_size=10,
                verbose=1, validation_split=0.1)  
        model.save(MODEL_NAME)
        os.remove(file_name)
     return model

print('Press Ctrl-C to quit.')
prev_move = -1
time_same = 0
try:
    while True:
        printscreen =  np.array(ImageGrab.grab(bbox=(200,200,600,400)))
        printscreen = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
        printscreen = cv2.resize(printscreen, (WIDTH,HEIGHT))
        #cv2.imshow('window',printscreen)        
        moves = model.predict(printscreen.reshape(-1,WIDTH,HEIGHT,1))[0]
        final_move = -1        
        for _i in range(0, len(moves)):
            if moves[_i] > 0.8:
               final_move = _i
               break
        if final_move == -1:
            final_move = random.randint(0,8)
            print ('Random Move')
        training_data.append([printscreen,final_move])
        if final_move != prev_move:
            #print (final_move)
            options[final_move]()
            #x, y = pyautogui.position()
            #angle1 = angle(x,y)
            #positionStr = 'angle = ' + str(moves).rjust(4) + ' checksum ' + str(np.sum(moves)).rjust(4)
            #print(positionStr, end='')
            #print('\b' * len(positionStr), end='', flush=True)
            prev_move = final_move
        else:
           time_same += 1
           if time_same > 6:
              #model = recalculate(model)
              pyautogui.press('enter')
              time_same = 0
        if len(training_data) % 500 == 0:
            print (len(training_data))
            print (file_name)
            np.save(file_name,training_data)
            del training_data
            training_data = []
            file += 1
            file_name = 'slither_data_new-{}.npy'.format(file)

except KeyboardInterrupt:
    print('\n')




