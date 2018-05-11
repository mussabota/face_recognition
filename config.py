import select
import sys
import cv2
import numpy as np

"""Все необходимые конфигурации устанвливается в этом скрипте!"""

VIDEO_SOURCE = 1
#VIDEO_SOURCE = "rtsp://192.168.0.101:8080"

HAAR_FACES = 'haarcascades/haarcascade_frontalface_default.xml'

HAAR_FACE_CASCADE = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')


MODEL_DIR = './model/20170511-185253.pb'
CLASSIFIER_FILENAME = './class/classifier.pkl'
NPY = './npy'
TRAINING_DIR = 'training_dir/'
CHECK_FACE_FOLDER = 'check_face/'
FACES_DIR = '/faces_dir'
#"./training_dir"

esik_ashyk = False
#is_locked = True

def door_status():
    if esik_ashyk:
        return True
    else:
        return False

def set_status(status):
    if not status:
        esik_ashyk = True


def capturing():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    cap.set(3, 400)
    cap.set(4, 600)

    return cap

