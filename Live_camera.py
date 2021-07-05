import os
import cv2
import math
import pafy
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from moviepy.editor import *
from collections import deque
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import joblib

seed_constant = 23
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

image_height, image_width = 64, 64
max_images_per_class = 2000
dataset_directory = "UCF50"
classes_list = ['WalkingWithDog', 'HorseRace','JumpingJack','HulaHoop','YoYo','PizzaTossing']

model_output_size = len(classes_list)

# Using Keras's to_categorical method to convert labels into one-hot-encoded vectors
#one_hot_encoded_labels = to_categorical(labels)

loaded_model = tf.keras.models.load_model('har_new.h5')
print("Model loaded")

'''window_size = 25

predicted_labels_probabilities_deque = deque(maxlen = window_size)
video_reader = cv2.VideoCapture(0)

while True:
    status, frame = video_reader.read()
    if not status:
        break
    resized_frame = cv2.resize(frame, (image_height, image_width))
    normalized_frame = resized_frame / 255
    predicted_labels_probabilities = loaded_model.predict(np.expand_dims(normalized_frame, axis = 0))[0]
    predicted_labels_probabilities_deque.append(predicted_labels_probabilities)
    if len(predicted_labels_probabilities_deque) == window_size:
        predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)
        predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)
        predicted_label = np.argmax(predicted_labels_probabilities_averaged)
        print(predicted_label)
        predicted_class_name = classes_list[predicted_label]
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Predicted Frames', frame)
    key_pressed = cv2.waitKey(10)
    if key_pressed == ord('q'):
         break
cv2.destroyAllWindows()
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))
'''

window_size = 1
output_video_file_path = 'output/output_test_new_1.mp4'
video_file_path = 'test_videos/test_new_1.mp4'

def predict_on_live_video(video_file_path, output_file_path, window_size):

    predicted_labels_probabilities_deque = deque(maxlen = window_size)

    video_reader = cv2.VideoCapture(video_file_path)

    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_writer = cv2.VideoWriter(output_file_path, fourcc, 24, (original_video_width, original_video_height))

    while True:

        status, frame = video_reader.read()

        if not status:

            break

        resized_frame = cv2.resize(frame, (image_height, image_width))

        normalized_frame = resized_frame / 255

        predicted_labels_probabilities = loaded_model.predict(np.expand_dims(normalized_frame, axis = 0))[0]

        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)

        if len(predicted_labels_probabilities_deque) == window_size:

            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)

            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)

            predicted_label = np.argmax(predicted_labels_probabilities_averaged)

            predicted_class_name = classes_list[predicted_label]

            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        video_writer.write(frame)

    video_reader.release()
    video_writer.release()


predict_on_live_video(video_file_path, output_video_file_path, window_size)

print("done")