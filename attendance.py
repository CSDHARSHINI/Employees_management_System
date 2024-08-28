##package
import numpy as np
import cv2
import os
import csv
import time
import pickle

from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime


## xml file
video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default (1).xml')