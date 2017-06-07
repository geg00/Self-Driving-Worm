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

WIDTH = 80
HEIGHT = 60
img_rows, img_cols = WIDTH, HEIGHT
input_shape = (WIDTH,HEIGHT,3)
LR = 1e-3
EPOCHS = 1000
MODEL_NAME = 'slither-{}-{}-{}-epochs.model'.format(LR, 'convnets',EPOCHS)

def angle(x,y):
	center_x = 300
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
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(9, activation='softmax'))
model.summary()

model.load_weights(MODEL_NAME)

for i in list(range(4))[::-1]:
	print(i+1)
	time.sleep(1)

def zero():
	pyautogui.moveTo(300, 200,2)

def one():
	pyautogui.moveTo(600, 200,2)

def two():
	pyautogui.moveTo(600, 300,2)

def three():
	pyautogui.moveTo(600, 400,2)

def four():
	pyautogui.moveTo(300, 400,2)

def five():
	pyautogui.moveTo(200, 400,2)

def six():
	pyautogui.moveTo(200, 300,2)

def seven():
	pyautogui.moveTo(200, 200,2)

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

file_name = 'slither_data_new.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []


print('Press Ctrl-C to quit.')
prev_move = -1
time_same = 0
try:
    while True:
        printscreen =  np.array(ImageGrab.grab(bbox=(200,200,600,400)))
        printscreen = cv2.resize(printscreen, (80,60))
        moves = model.predict(printscreen.reshape(-1,80,60,3))[0]
        final_move = 0        
        for _i in range(0, len(moves)):
            if moves[_i] > 0.5:
               final_move = _i
               break
        training_data.append([printscreen,final_move])
        if final_move != prev_move:
            #print (final_move)
            options[final_move]()
            #x, y = pyautogui.position()
            #angle1 = angle(x,y)
            positionStr = 'angle = ' + str(moves).rjust(4) + ' checksum ' + str(np.sum(moves)).rjust(4)
            print(positionStr, end='')
            print('\b' * len(positionStr), end='', flush=True)
            prev_move = final_move
        else:
           time_same += 1
           if time_same > 6:
              pyautogui.press('enter')
              if len(training_data) > 20:
                 training_data.pop(20)
              time_same = 0
        if len(training_data) % 500 == 0:
           print (len(training_data))
           print (file_name)
           np.save(file_name,training_data)
except KeyboardInterrupt:
    print('\n')




