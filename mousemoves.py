#! python3
import pyautogui, sys
import numpy as np
from PIL import ImageGrab
import cv2
import os
import math

file_name = 'slither_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []

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
	return math.ceil(angle1/45)

print('Press Ctrl-C to quit.')
try:
    while True:
        printscreen =  np.array(ImageGrab.grab(bbox=(0,40,800,600)))
        x, y = pyautogui.position()
        angle1 = angle(x,y)
        positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4) + ' angle = ' + str(angle1).rjust(4)
        print(positionStr, end='')
        print('\b' * len(positionStr), end='', flush=True)
        printscreen = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
        printscreen = cv2.resize(printscreen, (400,200))
        #cv2.imshow('window',printscreen)
        training_data.append([printscreen,angle1])
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        if len(training_data) % 500 == 0:
           print (len(training_data))
           print (file_name)
           np.save(file_name,training_data)

except KeyboardInterrupt:
    print('\n')
