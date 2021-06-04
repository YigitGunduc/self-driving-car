from grabscreen import grabscreen
import torch
import numpy as np
from grabkeys import key_check
from grabscreen import grabscreen
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
import random

weights = np.array([1, 0.4, 0.5, 0.5, 0.5, 0.5, 1.5, 0.5, 0.2])

w  = [1,0,0,0,0,0,0,0,0]
s  = [0,1,0,0,0,0,0,0,0]
a  = [0,0,1,0,0,0,0,0,0]
d  = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)

def right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(S)

def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)

def no_keys():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


model = torch.load('driveNet-ResNetMini.pth')

def main():
    
    paused = False
    while(True):

        if not paused:
            screen = grabscreen()
            prediction = model(torch.tensor(np.expand_dims(screen.reshape(3, 128, 128), axis=0)).float())
            #prediction = model(torch.tensor([screen.reshape(3, 128 ,128)]))
            prediction = prediction.detach().numpy()
            prediction = prediction * np.array([1.3, 1, 1, 1, 1, 1, 1, 1, 1])
            #prediction = prediction * np.array([4.5, 0.1, 0.1, 0.1, 1.8, 1.8, 0.5, 0.5, 0.2])
            print(np.argmax(prediction))
            if np.argmax(prediction) == np.argmax(w):
                straight()

            elif np.argmax(prediction) == np.argmax(s):
                reverse()
            if np.argmax(prediction) == np.argmax(a):
                left()
            if np.argmax(prediction) == np.argmax(d):
                right()
            if np.argmax(prediction) == np.argmax(wa):
                forward_left()
            if np.argmax(prediction) == np.argmax(wd):
                forward_right()
            if np.argmax(prediction) == np.argmax(sa):
                reverse_left()
            if np.argmax(prediction) == np.argmax(sd):
                reverse_right()

main()

