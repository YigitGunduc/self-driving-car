import numpy as np
import csv
import cv2
import time
from grabkeys import key_check
import os
from grabscreen import grabscreen
from PIL import Image


DATA_FOLDER = 'data/'

f = open('annotations.csv', 'a')

writer = csv.writer(f)

w  = [1,0,0,0,0,0,0,0,0]
s  = [0,1,0,0,0,0,0,0,0]
a  = [0,0,1,0,0,0,0,0,0]
d  = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]

def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    output = [0,0,0,0,0,0,0,0,0]

    if 'W' in keys and 'A' in keys:
        output = np.argmax(wa) #wa
    elif 'W' in keys and 'D' in keys:
        output = np.argmax(wd)
    elif 'S' in keys and 'A' in keys:
        output = np.argmax(sa) #sa
    elif 'S' in keys and 'D' in keys:
        output = np.argmax(sd) #sd
    elif 'W' in keys:
        output = np.argmax(w) # w
    elif 'S' in keys:
        output = np.argmax(s) #s
    elif 'A' in keys:
        output = np.argmax(a) #a
    elif 'D' in keys:
        output = np.argmax(d) #d
    else:
        output = np.argmax(w) #w 
    return output


def main():
    print('cap on')
    paused = False
    while True:
        if not paused:
            screen = grabscreen()
            image_name = f'train_{np.random.uniform()}.png' # unique image name
            cv2.imwrite(DATA_FOLDER + image_name, screen)
            keys = key_check()
            output = keys_to_output(keys)
            writer.writerow([image_name, output])
        keys = key_check()
        if 'T' in keys:
            if paused:
                print('paused')
                paused = False
            else:
                paused = True
    

if __name__ == "__main__":
    main()

