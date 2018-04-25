import select
import sys
import cv2
import numpy as np

"""Все необходимые конфигурации устанвливается в этом скрипте!"""

VIDEO_SOURCE = 0
#VIDEO_SOURCE = "rtsp://192.168.0.101:8080"

HAAR_FACES = 'haarcascades/haarcascade_frontalface_default.xml'

HAAR_FACE_CASCADE = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')


MODEL_DIR = './model/20170511-185253.pb'
CLASSIFIER_FILENAME = './class/classifier.pkl'
NPY = './npy'
TRAINING_DIR = 'training_dir/'
FACES_DIR = '/faces_dir'
#"./training_dir"

